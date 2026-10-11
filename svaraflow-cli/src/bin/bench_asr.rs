//! Benchmark one ASR model on one audio file.
//!
//! Prints a single JSON object with timing and accuracy metrics.
//! `tools/bench-asr.ps1` calls this once per (model, mode, clip) triple and
//! aggregates the results into a Markdown table. Peak RAM and peak VRAM are
//! read by the PowerShell script from the OS, not here: this process cannot
//! see the memory whisper.cpp allocates on the C side.
//!
//! Metrics:
//!
//! - `load_s`            wall-clock time spent loading the model.
//! - `wall_transcribe_s` wall-clock time inside `Engine::transcribe`.
//! - `engine_elapsed_s`  what the engine itself reports (excludes load).
//! - `rtf_total`         `duration_s / (load_s + wall_transcribe_s)`.
//! - `engine_rtf`        `Transcript::real_time_factor`.
//! - `wer`               word error rate against `--reference`, if given.
//! - `median_word_offset_ms`  median |predicted - labeled| start time, in ms,
//!   over the words in `--timing`, if given.
//! - `estimated_words` / `total_words`  how many words fell back to
//!   proportional timing.
//! - `decoder_threads`   how many threads were handed to whisper.cpp.

use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use clap::Parser;
use serde::Deserialize;
use serde_json::json;

use svaraflow_asr::models::{ModelEntry, Tier};
use svaraflow_asr::transcribe::{DecodeStrategy, Engine, LanguageHint, TranscribeConfig};
use svaraflow_core::audio::{
    decode_native_mono, decode_via_ffmpeg, resample_to, MonoAudio, ANALYSIS_SAMPLE_RATE,
};

#[derive(Parser, Debug)]
#[command(
    name = "bench-asr",
    about = "Benchmark one ASR model on one audio file and print JSON"
)]
struct Args {
    /// Path to the GGML model file.
    #[arg(long)]
    model: PathBuf,
    /// Display name for the model, e.g. "Swift F16".
    #[arg(long)]
    model_name: String,
    /// Tier: `light` or `standard`. Informational; the model file decides
    /// what actually runs.
    #[arg(long, default_value = "light")]
    tier: String,
    /// Audio file: `.wav`, `.mp4`, or anything `symphonia` / `ffmpeg` can read.
    #[arg(long)]
    audio: PathBuf,
    /// Reference transcript for WER. Plain text.
    #[arg(long)]
    reference: Option<PathBuf>,
    /// Labeled word timings. JSON array of `{ "word", "start", "end" }`.
    #[arg(long)]
    timing: Option<PathBuf>,
    /// Language hint: `hi` or `auto`.
    #[arg(long, default_value = "hi")]
    language: String,
    /// Beam size. `1` means greedy (the CLI default).
    #[arg(long, default_value_t = 1)]
    beam: i32,
    /// Optional Silero VAD model. When absent, chunking is fixed-length.
    #[arg(long)]
    vad_model: Option<PathBuf>,
    /// Decoder threads. `0` means "all logical cores" (default).
    #[arg(long, default_value_t = 0)]
    threads: usize,
    /// Write JSON here instead of stdout.
    #[arg(long)]
    output: Option<PathBuf>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let value = run(&args)?;
    let text = serde_json::to_string_pretty(&value)?;
    match &args.output {
        Some(p) => {
            std::fs::write(p, &text).with_context(|| format!("writing {}", p.display()))?;
        }
        None => println!("{text}"),
    }
    Ok(())
}

fn run(args: &Args) -> Result<serde_json::Value> {
    let audio = load_audio(&args.audio)?;
    let duration_s = audio.duration_seconds();

    let tier = Tier::parse(&args.tier).map_err(|e| anyhow!("{e}"))?;
    let language = match args.language.trim().to_ascii_lowercase().as_str() {
        "hi" | "hindi" => LanguageHint::Hindi,
        "auto" | "" => LanguageHint::Auto,
        other => return Err(anyhow!("--language must be `hi` or `auto`; got `{other}`")),
    };

    let entry = ModelEntry {
        name: args.model_name.clone(),
        tier,
        description: String::new(),
        filename: args
            .model
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default(),
        url: String::new(),
        revision: String::new(),
        size_bytes: 0,
        sha256: String::new(),
        license: String::new(),
        quantization: String::new(),
        source_repo: String::new(),
    };

    let config = TranscribeConfig {
        tier,
        language,
        decode: if args.beam > 1 {
            DecodeStrategy::BeamSearch {
                beam_size: args.beam,
            }
        } else {
            DecodeStrategy::Greedy
        },
        use_dtw: false,
        initial_prompt: None,
        vocabulary: Vec::new(),
        temperature_fallback: true,
        no_speech_threshold: 0.6,
        repetition_guard: true,
        max_chunk_s: svaraflow_asr::DEFAULT_MAX_CHUNK_S,
        overlap_s: svaraflow_asr::DEFAULT_OVERLAP_S,
        threads: args.threads,
    };
    let decoder_threads = config.resolved_threads();

    let vad_path = args.vad_model.as_deref().filter(|p| p.exists());

    let load_start = Instant::now();
    let mut engine = Engine::load(&args.model, &entry, vad_path, config)
        .map_err(|e| anyhow!("loading model `{}`: {e}", args.model.display()))?;
    let load_s = load_start.elapsed().as_secs_f64();

    let cancel = AtomicBool::new(false);
    let mut progress = |_i: usize, _n: usize| {};

    let transcribe_start = Instant::now();
    let transcript = engine
        .transcribe(&audio, &mut progress, &cancel)
        .map_err(|e| anyhow!("transcription failed: {e}"))?;
    let wall_transcribe_s = transcribe_start.elapsed().as_secs_f64();

    let total_s = load_s + wall_transcribe_s;
    let rtf_total = if total_s > 0.0 {
        duration_s / total_s
    } else {
        f64::INFINITY
    };

    let wer = if let Some(p) = &args.reference {
        let reference = std::fs::read_to_string(p)
            .with_context(|| format!("reading reference {}", p.display()))?;
        let hypothesis: String = transcript
            .segments
            .iter()
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        Some(compute_wer(&reference, &hypothesis))
    } else {
        None
    };

    let median_word_offset_ms = if let Some(p) = &args.timing {
        compute_median_word_offset(p, &transcript)?
    } else {
        None
    };

    let words = transcript.words();
    let total_words = words.len();
    let estimated_words = words.iter().filter(|w| w.estimated).count();

    Ok(json!({
        "model": args.model_name,
        "audio": args.audio.display().to_string(),
        "duration_s": duration_s,
        "load_s": load_s,
        "wall_transcribe_s": wall_transcribe_s,
        "engine_elapsed_s": transcript.elapsed_s,
        "total_s": total_s,
        "rtf_total": rtf_total,
        "engine_rtf": transcript.real_time_factor,
        "wer": wer,
        "median_word_offset_ms": median_word_offset_ms,
        "estimated_words": estimated_words,
        "total_words": total_words,
        "decoder_threads": decoder_threads,
    }))
}

/// Decodes any media file into 16 kHz mono, matching `transcribe_cmd.rs`.
fn load_audio(path: &Path) -> Result<MonoAudio> {
    match decode_native_mono(path) {
        Ok(native) => {
            if native.sample_rate == ANALYSIS_SAMPLE_RATE {
                Ok(native)
            } else {
                resample_to(&native, ANALYSIS_SAMPLE_RATE).map_err(|e| anyhow!("resample: {e}"))
            }
        }
        Err(native_err) => decode_via_ffmpeg(path, ANALYSIS_SAMPLE_RATE)
            .map_err(|e| anyhow!("symphonia: {native_err}; ffmpeg: {e}")),
    }
}

/// Loads spelling-variant pairs from `tests/variants.txt`, if it exists.
///
/// Format: one `variant=canonical` pair per line. `#` starts a comment.
/// Absence of the file is not an error — it just means no collapsing.
fn load_variants() -> Vec<(String, String)> {
    let Ok(text) = std::fs::read_to_string("tests/variants.txt") else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some((a, b)) = line.split_once('=') {
            let a = a.trim().to_lowercase();
            let b = b.trim().to_lowercase();
            if !a.is_empty() && !b.is_empty() {
                out.push((a, b));
            }
        }
    }
    out
}

/// Lowercases, strips punctuation, collapses whitespace, and applies the
/// variants dictionary word-by-word.
fn normalize_for_wer(text: &str, variants: &[(String, String)]) -> String {
    let lowered: String = text
        .to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { ' ' })
        .collect();
    let tokens: Vec<&str> = lowered.split_whitespace().collect();
    let mapped: Vec<&str> = tokens
        .iter()
        .map(|t| {
            variants
                .iter()
                .find(|(from, _)| from.as_str() == *t)
                .map_or(*t, |(_, to)| to.as_str())
        })
        .collect();
    mapped.join(" ")
}

/// Levenshtein word error rate over two normalised strings.
fn compute_wer(reference: &str, hypothesis: &str) -> f64 {
    let variants = load_variants();
    let ref_norm = normalize_for_wer(reference, &variants);
    let hyp_norm = normalize_for_wer(hypothesis, &variants);
    let ref_words: Vec<&str> = ref_norm.split_whitespace().collect();
    let hyp_words: Vec<&str> = hyp_norm.split_whitespace().collect();
    if ref_words.is_empty() {
        return 0.0;
    }
    let n = ref_words.len();
    let m = hyp_words.len();

    // Two-row Levenshtein. u32 keeps a long clip well under any memory cap.
    let mut prev: Vec<u32> = (0..=m as u32).collect();
    let mut curr: Vec<u32> = vec![0; m + 1];
    for i in 1..=n {
        curr[0] = i as u32;
        for j in 1..=m {
            let cost = u32::from(ref_words[i - 1] != hyp_words[j - 1]);
            curr[j] = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    f64::from(prev[m]) / n as f64
}

#[derive(Deserialize)]
struct LabeledWord {
    word: String,
    start: f64,
}

/// Median absolute difference, in milliseconds, between the predicted start
/// time and the labeled start time for each matching word.
///
/// Greedy one-to-one matching in order; a labeled word consumes the first
/// unused predicted word with the same lowercase text. Returns `Ok(None)`
/// when the timing file exists but no word could be matched.
fn compute_median_word_offset(
    timing_path: &Path,
    transcript: &svaraflow_asr::transcribe::Transcript,
) -> Result<Option<f64>> {
    let raw = std::fs::read_to_string(timing_path)
        .with_context(|| format!("reading timing file {}", timing_path.display()))?;
    let labeled: Vec<LabeledWord> =
        serde_json::from_str(&raw).with_context(|| format!("parsing {}", timing_path.display()))?;

    let predicted = transcript.words();
    let mut used = vec![false; predicted.len()];
    let mut offsets_ms: Vec<f64> = Vec::new();
    for lw in &labeled {
        let target = lw.word.to_lowercase();
        for (i, pw) in predicted.iter().enumerate() {
            if used[i] {
                continue;
            }
            if pw.text.to_lowercase() == target {
                used[i] = true;
                offsets_ms.push((pw.start_s - lw.start).abs() * 1000.0);
                break;
            }
        }
    }
    if offsets_ms.is_empty() {
        return Ok(None);
    }
    offsets_ms.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = offsets_ms.len() / 2;
    let median = if offsets_ms.len() % 2 == 0 {
        (offsets_ms[mid - 1] + offsets_ms[mid]) / 2.0
    } else {
        offsets_ms[mid]
    };
    Ok(Some(median))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_strips_punctuation_and_lowercases() {
        let v = Vec::new();
        assert_eq!(normalize_for_wer("Hello, World!", &v), "hello world");
    }

    #[test]
    fn normalize_collapses_variants() {
        let v = vec![("nahi".to_string(), "nahin".to_string())];
        assert_eq!(normalize_for_wer("nahi yaar", &v), "nahin yaar");
    }

    #[test]
    fn wer_zero_for_identical_text() {
        let w = compute_wer("hello world", "hello world");
        assert!(w.abs() < 1e-12, "got {w}");
    }

    #[test]
    fn wer_counts_substitutions() {
        let w = compute_wer("a b c d", "a x c y");
        assert!((w - 0.5).abs() < 1e-12, "got {w}");
    }

    #[test]
    fn wer_counts_insertions() {
        let w = compute_wer("a b", "a b c d");
        assert!((w - 1.0).abs() < 1e-12, "got {w}");
    }

    #[test]
    fn wer_handles_empty_reference() {
        let w = compute_wer("", "a b c");
        assert!(w.abs() < 1e-12, "got {w}");
    }
}
