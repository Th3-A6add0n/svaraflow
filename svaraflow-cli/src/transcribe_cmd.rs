// svaraflow-cli/src/transcribe_cmd.rs
//! `svaraflow-cli transcribe` and `svaraflow-cli models`.
//!
//! Every transcription run prints a diagnostics block on stderr:
//! whisper.cpp's system info, the backend, the CUDA device (when
//! applicable), the resolved thread count, and the full `FullParams`
//! set we hand to whisper.cpp. The block is emitted by
//! [`Engine::transcribe_with_timings`], so it appears whether the caller
//! uses `transcribe` or `transcribe_with_timings`.

use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use svaraflow_asr::{
    bench, cache, ensure_model, exports, models,
    transcribe::{DecodeStrategy, Engine, LanguageHint, Timings, TranscribeConfig},
    AsrError, Manifest, ModelEntry, Tier,
};
use svaraflow_core::audio::ANALYSIS_SAMPLE_RATE;
use svaraflow_core::audio::{decode_native_mono, decode_via_ffmpeg, resample_to, MonoAudio};

use crate::cli::{
    ModelsCli, ModelsCommand, ModelsDownloadArgs, ModelsHashArgs, ModelsRemoveArgs, TierArg,
    TranscribeArgs,
};

/// Default VAD model path, relative to the current directory.
const DEFAULT_VAD_MODEL: &str = "models/silero_vad_v5.onnx";

/// Runs `transcribe <args>`.
pub fn run_transcribe(args: TranscribeArgs) -> Result<()> {
    let manifest = Manifest::load_embedded().map_err(|e| anyhow!("{e}"))?;

    let light_path = models::path_for(manifest.for_tier(Tier::Light)?);
    let standard_path = models::path_for(manifest.for_tier(Tier::Standard)?);
    let light_present = light_path.exists();
    let standard_present = standard_path.exists();

    let tier = match args.tier {
        TierArg::Light => Tier::Light,
        TierArg::Standard => Tier::Standard,
        TierArg::Auto => {
            let vad_path = args
                .vad_model
                .as_deref()
                .or(Some(Path::new(DEFAULT_VAD_MODEL)));
            let choice = bench::pick_tier(&manifest, light_present, standard_present, vad_path)?;
            eprintln!("tier: {} ({})", choice.tier.as_str(), choice.reason);
            if let Some(rtf) = choice.rtf {
                eprintln!("tier: benchmark real-time factor {rtf:.2}x");
            }
            choice.tier
        }
    };

    let entry = manifest.for_tier(tier)?.clone();
    let model_path = models::path_for(&entry);
    if !model_path.exists() {
        return Err(anyhow!(
            "model `{}` is not on disk at {}. Download it with \
             `svaraflow-cli models download {}`, or place the file there by hand.",
            entry.name,
            model_path.display(),
            entry.name
        ));
    }

    eprintln!("reading {}", args.input.display());
    let audio = load_audio(&args.input)?;
    eprintln!(
        "audio: {:.2} s at {} Hz, {} samples",
        audio.duration_seconds(),
        audio.sample_rate,
        audio.len()
    );

    // `--threads` and `--flash-attn` are exposed on this CLI; the rest
    // of the perf switches (`audio_ctx`, `parallel`, `vad_pack`,
    // `ranges`) are at their defaults. See docs/PERF.md.
    let cfg = TranscribeConfig {
        tier,
        language: parse_language(&args.language)?,
        decode: match args.beam {
            Some(n) if n > 1 => DecodeStrategy::BeamSearch {
                beam_size: n as i32,
            },
            _ => DecodeStrategy::Greedy,
        },
        token_timestamps: !args.no_dtw,
        initial_prompt: None,
        vocabulary: args.vocabulary.clone(),
        temperature_fallback: true,
        no_speech_threshold: 0.6,
        repetition_guard: true,
        max_chunk_s: svaraflow_asr::DEFAULT_MAX_CHUNK_S,
        overlap_s: svaraflow_asr::DEFAULT_OVERLAP_S,
        threads: args.threads,
        flash_attn: args.flash_attn,
        audio_ctx: None,
        parallel: 1,
        vad_pack: false,
        ranges: None,
        junk_guard: Default::default(),
        max_phrase_s: svaraflow_asr::transcribe::DEFAULT_MAX_PHRASE_S,
        max_phrase_words: svaraflow_asr::transcribe::DEFAULT_MAX_PHRASE_WORDS,
    };

    let key = cache::key_for(&audio, &entry.name, &cfg);
    let mut transcript = if args.no_cache {
        eprintln!("cache: disabled (--no-cache)");
        None
    } else {
        let hit = cache::load(&key);
        if hit.is_some() {
            eprintln!("CACHE HIT — result loaded from disk, no transcription ran");
        }
        hit
    };

    if let Some(t) = &transcript {
        eprintln!(
            "cache: {} segments, {:.2} s audio transcribed in {:.2} s",
            t.segments.len(),
            t.audio_duration_s,
            t.elapsed_s
        );
    } else {
        let vad_path = args
            .vad_model
            .clone()
            .unwrap_or_else(|| PathBuf::from(DEFAULT_VAD_MODEL));
        let vad_arg = if vad_path.exists() {
            Some(vad_path.as_path())
        } else {
            eprintln!(
                "note: VAD model not found at {}; chunking without VAD",
                vad_path.display()
            );
            None
        };

        eprintln!("loading model `{}`", entry.name);
        let mut engine = Engine::load(&model_path, &entry, vad_arg, cfg.clone())?;

        eprintln!("transcribing...");
        let cancel = AtomicBool::new(false);
        let mut last_shown = 0usize;
        let mut progress = |i: usize, n: usize| {
            if n > 0 && (i == n || i != last_shown) {
                last_shown = i;
                let pct = (i * 100) / n;
                eprint!("\r  {i}/{n} chunks ({pct}%)");
                if i == n {
                    eprintln!();
                }
            }
        };

        // `transcribe_with_timings` emits the diagnostics block on
        // stderr and fills `timings`. The block is what makes a run
        // comparable to upstream whisper-cli at a glance.
        let process_start = Instant::now();
        let mut timings = Timings::default();
        match engine.transcribe_with_timings(&audio, &mut progress, &cancel, &mut timings) {
            Ok(t) => transcript = Some(t),
            Err(AsrError::Cancelled) => {
                eprintln!("\ncancelled");
                return Err(anyhow!("cancelled"));
            }
            Err(e) => return Err(anyhow!("transcription failed: {e}")),
        }
        timings.finalize(process_start.elapsed());

        // Skip the CUDA teardown path (Windows + redirected stdio aborts
        // with STATUS_STACK_BUFFER_OVERRUN). The process is about to exit.
        std::mem::forget(engine);

        eprintln!(
            "timings: total={:.3}s transcribe={:.3}s whisper_full={:.3}s chunks={}",
            timings.total_wall_s, timings.transcribe_s, timings.whisper_full_s, timings.chunks,
        );
        if !timings.chunk_reasons.is_empty() {
            eprintln!(
                "chunk reasons: {}",
                timings
                    .chunk_reasons
                    .iter()
                    .map(|r| format!("{r:?}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
        for w in &timings.windows {
            eprintln!(
                "  window {i}: {a:.2}s..{b:.2}s full={c:.3}s tokens={t} \
                 mean_p={p:.2} junk={j:.2} no_speech={ns:.2}{lc}",
                i = w.index,
                a = w.start_s,
                b = w.end_s,
                c = w.whisper_full_s,
                t = w.tokens,
                p = w.mean_probability,
                j = w.junk_fraction,
                ns = w.no_speech_prob,
                lc = if w.low_confidence {
                    " LOW_CONFIDENCE"
                } else {
                    ""
                },
            );
        }

        if let Some(t) = &transcript {
            if !args.no_cache {
                if let Err(e) = cache::save(&key, t) {
                    eprintln!("warning: could not write cache: {e}");
                }
            }
        }
    }

    let t = transcript.ok_or_else(|| anyhow!("no transcript produced"))?;

    println!(
        "real-time factor: {:.2}x ({:.2} s audio in {:.2} s)",
        t.real_time_factor, t.audio_duration_s, t.elapsed_s
    );

    let estimated = t.words().iter().filter(|w| w.estimated).count();
    if estimated > 0 {
        eprintln!(
            "warning: {estimated} of {} words have estimated (proportional) \
             timing because the model did not produce DTW timestamps",
            t.words().len()
        );
    }

    if args.json || args.out_json.is_some() {
        let json = exports::write_json(&t)?;
        if args.json && args.out_json.is_none() {
            println!("{json}");
        }
        if let Some(path) = &args.out_json {
            std::fs::write(path, &json).with_context(|| format!("writing {}", path.display()))?;
            eprintln!("wrote {}", path.display());
        }
    }

    if let Some(path) = &args.srt {
        let text = exports::write_srt(&t);
        std::fs::write(path, text).with_context(|| format!("writing {}", path.display()))?;
        eprintln!("wrote {}", path.display());
    }

    if let Some(path) = &args.vtt {
        let text = exports::write_vtt(&t);
        std::fs::write(path, text).with_context(|| format!("writing {}", path.display()))?;
        eprintln!("wrote {}", path.display());
    }

    if let Some(path) = &args.ass {
        let title = args
            .input
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "SvaraFlow".to_string());
        let text = exports::write_ass(&t, &title);
        std::fs::write(path, text).with_context(|| format!("writing {}", path.display()))?;
        eprintln!("wrote {}", path.display());
    }

    Ok(())
}

/// Runs `models <subcommand>`.
pub fn run_models(cli: ModelsCli) -> Result<()> {
    let manifest = Manifest::load_embedded().map_err(|e| anyhow!("{e}"))?;
    match cli.command {
        ModelsCommand::List => list_models(&manifest),
        ModelsCommand::Download(a) => download_model(&manifest, &a),
        ModelsCommand::Hash(a) => hash_model(&a),
        ModelsCommand::Remove(a) => remove_model(&manifest, &a),
    }
}

fn list_models(manifest: &Manifest) -> Result<()> {
    println!(
        "{:<36} {:<10} {:>10} {:<14} LICENSE",
        "NAME", "TIER", "ON DISK", "QUANT",
    );
    for m in &manifest.models {
        let path = models::path_for(m);
        let present = if path.exists() {
            match std::fs::metadata(&path) {
                Ok(meta) => format!("{:.1} MB", meta.len() as f64 / 1_048_576.0),
                Err(_) => "yes".to_string(),
            }
        } else {
            "-".to_string()
        };
        println!(
            "{:<36} {:<10} {:>10} {:<14} {}",
            m.name,
            m.tier.as_str(),
            present,
            m.quantization,
            m.license
        );
        if !m.has_real_url() {
            println!("  (url: TODO — fill in models.json or place the file by hand)");
        }
        if !m.has_real_hash() {
            println!("  (sha256: TODO — verification will be skipped)");
        }
    }
    println!();
    println!("Models folder: {}", models::storage_dir().display());
    Ok(())
}

fn download_model(manifest: &Manifest, args: &ModelsDownloadArgs) -> Result<()> {
    let entry: &ModelEntry = if let Ok(tier) = Tier::parse(&args.name) {
        manifest.for_tier(tier)?
    } else if let Some(e) = manifest.by_name(&args.name) {
        e
    } else {
        return Err(anyhow!(
            "no model named `{}` and no tier `{}` in the manifest; \
             run `svaraflow-cli models list`",
            args.name,
            args.name
        ));
    };

    if !entry.has_real_url() {
        return Err(anyhow!(
            "the manifest's `url` for `{}` is `TODO`. Edit models.json to fill \
             it in, or place the model by hand at {}.",
            entry.name,
            models::path_for(entry).display()
        ));
    }

    eprintln!("downloading `{}` from {}", entry.name, entry.url);
    if entry.size_bytes > 0 {
        eprintln!(
            "expected size: {:.1} MB",
            entry.size_bytes as f64 / 1_048_576.0
        );
    }

    let mut last_percent = -1i32;
    let mut progress = |written: u64, total: Option<u64>| {
        if let Some(t) = total {
            let pct = written
                .checked_mul(100)
                .and_then(|scaled| scaled.checked_div(t))
                .map_or(0, |v| v as i32);
            if pct != last_percent {
                last_percent = pct;
                eprint!(
                    "\r  {pct}% ({:.1} MB / {:.1} MB)",
                    written as f64 / 1_048_576.0,
                    t as f64 / 1_048_576.0
                );
            }
        } else {
            eprint!("\r  {:.1} MB", written as f64 / 1_048_576.0);
        }
    };

    let path = ensure_model(entry, &mut progress)?;
    eprintln!();
    println!("Downloaded to {}", path.display());
    Ok(())
}

fn hash_model(args: &ModelsHashArgs) -> Result<()> {
    if !args.file.exists() {
        return Err(anyhow!("file does not exist: {}", args.file.display()));
    }
    let h = models::hash_file(&args.file).map_err(|e| anyhow!("{e}"))?;
    println!("{h}  {}", args.file.display());
    Ok(())
}

fn remove_model(manifest: &Manifest, args: &ModelsRemoveArgs) -> Result<()> {
    let entry = manifest
        .by_name(&args.name)
        .or_else(|| {
            Tier::parse(&args.name)
                .ok()
                .and_then(|t| manifest.for_tier(t).ok())
        })
        .ok_or_else(|| anyhow!("no model named `{}`", args.name))?;
    let removed = models::remove_model(entry).map_err(|e| anyhow!("{e}"))?;
    if removed {
        println!("Removed `{}`.", entry.name);
    } else {
        println!("`{}` was not on disk.", entry.name);
    }
    Ok(())
}

/// Decodes any media file into 16 kHz mono.
fn load_audio(path: &Path) -> Result<MonoAudio> {
    match decode_native_mono(path) {
        Ok(native) => {
            if native.sample_rate == ANALYSIS_SAMPLE_RATE {
                Ok(native)
            } else {
                resample_to(&native, ANALYSIS_SAMPLE_RATE).map_err(|e| anyhow!("{e}"))
            }
        }
        Err(native_err) => {
            eprintln!(
                "note: symphonia could not decode {}; falling back to ffmpeg ({native_err})",
                path.display()
            );
            decode_via_ffmpeg(path, ANALYSIS_SAMPLE_RATE)
                .map_err(|e| anyhow!("symphonia: {native_err}; ffmpeg: {e}"))
        }
    }
}

/// Parses `--language`.
///
/// Accepts `auto` (default), `hi`, `en`, or any bare two-letter ASCII
/// code. Longer strings are rejected so a typo like `french` or `xyz`
/// does not silently become a bad whisper.cpp language hint.
fn parse_language(s: &str) -> Result<LanguageHint> {
    match s.trim().to_ascii_lowercase().as_str() {
        "hi" | "hindi" => Ok(LanguageHint::Hindi),
        "en" | "english" => Ok(LanguageHint::English),
        "auto" | "" => Ok(LanguageHint::Auto),
        other if other.len() == 2 && other.chars().all(|c| c.is_ascii_alphabetic()) => {
            Ok(LanguageHint::Other)
        }
        other => Err(anyhow!(
            "--language must be `auto`, `hi`, `en`, or a two-letter code; got `{other}`"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_language_accepts_known_values() {
        assert_eq!(parse_language("hi").unwrap(), LanguageHint::Hindi);
        assert_eq!(parse_language("Hindi").unwrap(), LanguageHint::Hindi);
        assert_eq!(parse_language("en").unwrap(), LanguageHint::English);
        assert_eq!(parse_language("English").unwrap(), LanguageHint::English);
        assert_eq!(parse_language("auto").unwrap(), LanguageHint::Auto);
        assert_eq!(parse_language("").unwrap(), LanguageHint::Auto);
    }

    #[test]
    fn parse_language_accepts_two_letter_codes() {
        assert_eq!(parse_language("fr").unwrap(), LanguageHint::Other);
        assert_eq!(parse_language("de").unwrap(), LanguageHint::Other);
        assert_eq!(parse_language("JA").unwrap(), LanguageHint::Other);
    }

    #[test]
    fn parse_language_rejects_long_unknown() {
        assert!(parse_language("french").is_err());
        assert!(parse_language("xyz").is_err());
        assert!(parse_language("hindi ").is_ok());
    }
}
