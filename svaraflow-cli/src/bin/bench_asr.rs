// svaraflow-cli/src/bin/bench_asr.rs
//! Benchmark one ASR model on one audio file.
//!
//! Prints a single JSON object with timing, accuracy and build metrics.
//! `tools/bench-asr.ps1` calls this once per (model, mode, clip, switch
//! set) and aggregates the results into Markdown tables.

use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use clap::Parser;
use serde::Deserialize;
use serde_json::json;

use svaraflow_asr::models::{ModelEntry, Tier};
use svaraflow_asr::transcribe::{
    parse_ranges, system_info, DecodeStrategy, Engine, LanguageHint, Timings, TranscribeConfig,
};
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
    ///
    /// Required for a benchmark run. Not required when `--system-info` is
    /// given: that path only reads compile-time and CPU probes and exits
    /// before any model is touched.
    #[arg(long, required_unless_present = "system_info")]
    model: Option<PathBuf>,
    /// Display name for the model, e.g. "Swift F16".
    #[arg(long, required_unless_present = "system_info")]
    model_name: Option<String>,
    /// Tier: `light` or `standard`. Informational.
    #[arg(long, default_value = "light")]
    tier: String,
    /// Audio file: `.wav`, `.mp4`, or anything `symphonia` / `ffmpeg` can read.
    #[arg(long, required_unless_present = "system_info")]
    audio: Option<PathBuf>,
    /// Reference transcript for WER. Plain text.
    #[arg(long)]
    reference: Option<PathBuf>,
    /// Labeled word timings. JSON array of `{ "word", "start", "end" }`.
    #[arg(long)]
    timing: Option<PathBuf>,
    /// Language hint: `hi` or `auto`.
    #[arg(long, default_value = "hi")]
    language: String,
    /// Beam size. `0` or `1` means greedy.
    #[arg(long, default_value_t = 1)]
    beam: i32,
    /// Optional Silero VAD model.
    #[arg(long)]
    vad_model: Option<PathBuf>,
    /// Decoder threads *per whisper state*. `0` means "all logical cores
    /// divided by --parallel".
    #[arg(long, default_value_t = 0)]
    threads: usize,
    /// Enable whisper.cpp's flash attention path.
    #[arg(long, default_value_t = false)]
    flash_attn: bool,
    /// Disable whisper.cpp's temperature fallback retries.
    #[arg(long, default_value_t = false)]
    no_temp_fallback: bool,
    /// Disable the DTW timestamp request. Currently the only mode.
    #[arg(long, default_value_t = false)]
    no_dtw: bool,
    /// Override whisper.cpp's audio context size, in frames.
    #[arg(long)]
    audio_ctx: Option<i32>,
    /// Number of concurrent whisper states. Each runs in its own thread
    /// with its own state.
    #[arg(long, default_value_t = 1)]
    parallel: usize,
    /// Transcribe only these time ranges. File: `start_s,end_s` per line.
    #[arg(long)]
    ranges: Option<PathBuf>,
    /// Drop silence and pack speech into full-length windows. Requires a VAD.
    #[arg(long, default_value_t = false)]
    vad_pack: bool,
    /// Print the build and CPU probes and exit. No transcription.
    #[arg(long, default_value_t = false)]
    system_info: bool,
    /// Write JSON here instead of stdout.
    #[arg(long)]
    output: Option<PathBuf>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.system_info {
        print!("{}", system_info());
        return Ok(());
    }
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
    // `--model`, `--model-name` and `--audio` are `Option` because
    // `--system-info` does not need them. clap's
    // `required_unless_present = "system_info"` already rejects a
    // benchmark run that omits them, so `expect` here is unreachable for
    // any argv that clap accepted. Reaching it would be a bug in the
    // clap attributes, and the message says so.
    let model = args
        .model
        .as_ref()
        .expect("clap guarantees --model when --system-info is unset");
    let model_name = args
        .model_name
        .as_deref()
        .expect("clap guarantees --model-name when --system-info is unset");
    let audio_path = args
        .audio
        .as_ref()
        .expect("clap guarantees --audio when --system-info is unset");

    let mut timings = Timings::default();
    let process_start = Instant::now();

    // 1. Decode.
    let decode_start = Instant::now();
    let (native, native_decode) = load_native(audio_path)?;
    timings.audio_decode_s = decode_start.elapsed().as_secs_f64();

    // 2. Resample.
    let (audio, resample_s) = if native.sample_rate == ANALYSIS_SAMPLE_RATE {
        (native, 0.0)
    } else {
        let rs = Instant::now();
        let out = resample_to(&native, ANALYSIS_SAMPLE_RATE)
            .map_err(|e| anyhow!("resample {}: {e}", audio_path.display()))?;
        (out, rs.elapsed().as_secs_f64())
    };
    timings.resample_s = resample_s;
    timings.audio_duration_s = audio.duration_seconds();

    let duration_s = audio.duration_seconds();

    let tier = Tier::parse(&args.tier).map_err(|e| anyhow!("{e}"))?;
    let language = match args.language.trim().to_ascii_lowercase().as_str() {
        "hi" | "hindi" => LanguageHint::Hindi,
        "auto" | "" => LanguageHint::Auto,
        other => return Err(anyhow!("--language must be `hi` or `auto`; got `{other}`")),
    };

    let ranges = if let Some(path) = &args.ranges {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading ranges {}", path.display()))?;
        Some(parse_ranges(&text).map_err(|e| anyhow!("{e}"))?)
    } else {
        None
    };

    let entry = ModelEntry {
        name: model_name.to_string(),
        tier,
        description: String::new(),
        filename: model
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
        use_dtw: !args.no_dtw,
        initial_prompt: None,
        vocabulary: Vec::new(),
        temperature_fallback: !args.no_temp_fallback,
        no_speech_threshold: 0.6,
        repetition_guard: true,
        max_chunk_s: svaraflow_asr::DEFAULT_MAX_CHUNK_S,
        overlap_s: svaraflow_asr::DEFAULT_OVERLAP_S,
        threads: args.threads,
        flash_attn: args.flash_attn,
        audio_ctx: args.audio_ctx,
        parallel: args.parallel.max(1),
        vad_pack: args.vad_pack,
        // Cloned because `ranges` is also read into the JSON below. The
        // vector is small (a handful of pairs), so the clone is free.
        ranges: ranges.clone(),
    };
    let decoder_threads = config.resolved_threads();
    let parallel = config.parallel;

    let vad_path = args.vad_model.as_deref().filter(|p| p.exists());

    // 3. Model load.
    let load_start = Instant::now();
    let mut engine = Engine::load(model, &entry, vad_path, config)
        .map_err(|e| anyhow!("loading model `{}`: {e}", model.display()))?;
    timings.model_load_s = load_start.elapsed().as_secs_f64();

    let cancel = AtomicBool::new(false);
    let mut progress = |_i: usize, _n: usize| {};

    // 4. Transcribe.
    let transcribe_start = Instant::now();
    let transcript = engine
        .transcribe_with_timings(&audio, &mut progress, &cancel, &mut timings)
        .map_err(|e| anyhow!("transcription failed: {e}"))?;
    let wall_transcribe_s = transcribe_start.elapsed().as_secs_f64();

    // 5. WER and word offset.
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

    // 6. Export (JSON shape only).
    let export_start = Instant::now();
    let _ = serde_json::to_value(&transcript)?;
    timings.export_s = export_start.elapsed().as_secs_f64();

    timings.finalize(process_start.elapsed());

    let words = transcript.words();
    let total_words = words.len();
    let estimated_words = words.iter().filter(|w| w.estimated).count();

    Ok(json!({
        "model": model_name,
        "audio": audio_path.display().to_string(),
        "duration_s": duration_s,
        "load_s": timings.model_load_s,
        "audio_decode_s": timings.audio_decode_s,
        "resample_s": timings.resample_s,
        "wall_transcribe_s": wall_transcribe_s,
        "engine_elapsed_s": transcript.elapsed_s,
        "total_s": timings.total_wall_s,
        "rtf_total": timings.real_time_factor,
        "engine_rtf": transcript.real_time_factor,
        "wer": wer,
        "median_word_offset_ms": median_word_offset_ms,
        "estimated_words": estimated_words,
        "total_words": total_words,
        "decoder_threads": decoder_threads,
        "parallel": parallel,
        "native_decode": native_decode,
        "vad_pack": args.vad_pack,
        "ranges": ranges
            .as_ref()
            .map(|r| r.iter().map(|(s, e)| json!({ "start_s": s, "end_s": e })).collect::<Vec<_>>()),
        "timings": timings,
    }))
}

/// Decodes to native-rate mono. Returns `(audio, decoded_natively)`.
fn load_native(path: &Path) -> Result<(MonoAudio, bool)> {
    match decode_native_mono(path) {
        Ok(a) => Ok((a, true)),
        Err(native_err) => {
            let a = decode_via_ffmpeg(path, ANALYSIS_SAMPLE_RATE)
                .map_err(|e| anyhow!("symphonia: {native_err}; ffmpeg: {e}"))?;
            Ok((a, false))
        }
    }
}

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
