//! `svaraflow-cli transcribe` and `svaraflow-cli models`.

use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;

use anyhow::{anyhow, Context, Result};
use svaraflow_asr::{
    bench, cache, ensure_model, exports, models,
    transcribe::{DecodeStrategy, Engine, LanguageHint, TranscribeConfig},
    AsrError, Manifest, ModelEntry, Tier,
};
use svaraflow_core::audio::{decode_native_mono, decode_via_ffmpeg, resample_to, MonoAudio};
use svaraflow_core::audio::ANALYSIS_SAMPLE_RATE;

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

    let cfg = TranscribeConfig {
        tier,
        language: parse_language(&args.language)?,
        decode: match args.beam {
            Some(n) if n > 1 => DecodeStrategy::BeamSearch {
                beam_size: n as i32,
            },
            _ => DecodeStrategy::Greedy,
        },
        use_dtw: !args.no_dtw,
        initial_prompt: None,
        vocabulary: args.vocabulary.clone(),
        temperature_fallback: true,
        no_speech_threshold: 0.6,
        repetition_guard: true,
        max_chunk_s: svaraflow_asr::DEFAULT_MAX_CHUNK_S,
        overlap_s: svaraflow_asr::DEFAULT_OVERLAP_S,
    };
    if cfg.use_dtw {
        eprintln!("using token-level DTW timestamps");
    } else {
        eprintln!("DTW disabled; word timing will be proportional and flagged as estimated");
    }

    let key = cache::key_for(&audio, &entry.name, &cfg);
    let mut transcript = if args.no_cache {
        None
    } else {
        cache::load(&key)
    };

    if let Some(t) = &transcript {
        eprintln!(
            "cache hit: {} segments, {:.2} s transcribed in {:.2} s",
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

        match engine.transcribe(&audio, &mut progress, &cancel) {
            Ok(t) => transcript = Some(t),
            Err(AsrError::Cancelled) => {
                eprintln!("\ncancelled");
                return Err(anyhow!("cancelled"));
            }
            Err(e) => return Err(anyhow!("transcription failed: {e}")),
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

fn parse_language(s: &str) -> Result<LanguageHint> {
    match s.trim().to_ascii_lowercase().as_str() {
        "hi" | "hindi" => Ok(LanguageHint::Hindi),
        "auto" | "" => Ok(LanguageHint::Auto),
        other => Err(anyhow!(
            "--language must be `hi`, `hindi`, or `auto`; got `{other}`"
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
        assert_eq!(parse_language("auto").unwrap(), LanguageHint::Auto);
        assert_eq!(parse_language("").unwrap(), LanguageHint::Auto);
    }

    #[test]
    fn parse_language_rejects_unknown() {
        assert!(parse_language("fr").is_err());
    }
}