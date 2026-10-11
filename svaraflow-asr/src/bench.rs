// svaraflow-asr/src/bench.rs
//! Automatic tier selection.
//!
//! Two mechanisms, tried in order:
//!
//! 1. **Benchmark.** If `tests/audio/benchmark20s.wav` is present, run the
//!    Standard tier over it and measure the real-time factor. At
//!    [`AUTO_STANDARD_RTF`] or better, Standard is fast enough on this
//!    hardware and wins. Below that, Light wins.
//! 2. **Hardware heuristic.** If the benchmark file is missing, and only
//!    when both models are present, count CPU cores and look for an
//!    NVIDIA GPU. A GPU or 8+ cores implies Standard; otherwise Light.
//!
//! The caller supplies the manifest and the model directory. This module
//! never downloads anything: it only loads what is already on disk.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::AtomicBool;

use svaraflow_core::audio::{decode_native_mono, resample_to, MonoAudio, ANALYSIS_SAMPLE_RATE};

use crate::error::AsrError;
use crate::models::{path_for, Manifest, Tier};
use crate::transcribe::{Engine, TranscribeConfig};

/// Real-time factor at or above which Standard is preferred.
pub const AUTO_STANDARD_RTF: f64 = 2.0;

/// Sample rate the benchmark file must be, or will be converted to.
const BENCH_RATE: u32 = ANALYSIS_SAMPLE_RATE;

/// How the tier was chosen.
#[derive(Debug, Clone, PartialEq)]
pub struct TierChoice {
    /// The chosen tier.
    pub tier: Tier,
    /// A one-line explanation for the user.
    pub reason: String,
    /// Measured real-time factor, when a benchmark ran.
    pub rtf: Option<f64>,
}

impl TierChoice {
    /// A choice made without running the benchmark.
    fn heuristic(tier: Tier, reason: String) -> Self {
        Self {
            tier,
            reason,
            rtf: None,
        }
    }

    /// A choice made from a benchmark run.
    fn benchmarked(tier: Tier, rtf: f64) -> Self {
        let reason = format!(
            "Standard achieved {rtf:.2}x real time on the 20 s benchmark \
             (threshold: {AUTO_STANDARD_RTF:.1}x), so {} was chosen",
            match tier {
                Tier::Standard => "Standard",
                Tier::Light => "Light",
            }
        );
        Self {
            tier,
            reason,
            rtf: Some(rtf),
        }
    }
}

/// Picks a tier.
///
/// `light_present` and `standard_present` tell the function which models
/// are already on disk. `vad_path` is the Silero VAD ONNX file, required
/// when the `vad` feature is on.
///
/// # Errors
///
/// [`AsrError::WhisperContext`] when the benchmark model cannot be loaded.
/// [`AsrError::Io`] when the benchmark file exists but cannot be decoded.
pub fn pick_tier(
    manifest: &Manifest,
    light_present: bool,
    standard_present: bool,
    vad_path: Option<&Path>,
) -> Result<TierChoice, AsrError> {
    match (light_present, standard_present) {
        (false, false) => Err(AsrError::ModelMissing {
            name: "any".into(),
            path: "run `svaraflow-cli models download <name>` first".into(),
        }),
        (true, false) => Ok(TierChoice::heuristic(
            Tier::Light,
            "only the Light model is on disk; using it".into(),
        )),
        (false, true) => Ok(TierChoice::heuristic(
            Tier::Standard,
            "only the Standard model is on disk; using it".into(),
        )),
        (true, true) => {
            // Both present. Try the benchmark, fall back to hardware.
            let bench = find_benchmark();
            if let Some(path) = bench {
                match run_benchmark(manifest, &path, vad_path) {
                    Ok(rtf) => {
                        let tier = if rtf >= AUTO_STANDARD_RTF {
                            Tier::Standard
                        } else {
                            Tier::Light
                        };
                        return Ok(TierChoice::benchmarked(tier, rtf));
                    }
                    Err(e) => {
                        eprintln!(
                            "warning: benchmark failed ({e}); falling back to \
                             hardware heuristic"
                        );
                    }
                }
            }
            let tier = hardware_heuristic();
            Ok(TierChoice::heuristic(
                tier,
                format!(
                    "benchmark file not found at tests/audio/benchmark20s.wav; \
                     chose {} from CPU/GPU probe",
                    match tier {
                        Tier::Standard => "Standard",
                        Tier::Light => "Light",
                    }
                ),
            ))
        }
    }
}

/// Loads the Standard model over the benchmark clip and returns its RTF.
fn run_benchmark(
    manifest: &Manifest,
    audio_path: &Path,
    vad_path: Option<&Path>,
) -> Result<f64, AsrError> {
    let entry = manifest.for_tier(Tier::Standard)?;
    let model_path = path_for(entry);
    if !model_path.exists() {
        return Err(AsrError::ModelMissing {
            name: entry.name.clone(),
            path: model_path.display().to_string(),
        });
    }

    let audio = load_benchmark_audio(audio_path)?;
    let cfg = TranscribeConfig::default();
    let mut engine = Engine::load(&model_path, entry, vad_path, cfg)?;
    let cancel = AtomicBool::new(false);
    let mut progress = |_i: usize, _n: usize| {};
    let t = engine.transcribe(&audio, &mut progress, &cancel)?;
    Ok(t.real_time_factor)
}

/// Reads `tests/audio/benchmark20s.wav` and returns 16 kHz mono audio.
fn load_benchmark_audio(path: &Path) -> Result<MonoAudio, AsrError> {
    let native = decode_native_mono(path).map_err(|e| AsrError::WhisperState {
        reason: format!("decoding benchmark {}: {e}", path.display()),
    })?;
    if native.sample_rate == BENCH_RATE {
        return Ok(native);
    }
    resample_to(&native, BENCH_RATE).map_err(|e| AsrError::WhisperState {
        reason: format!("resampling benchmark: {e}"),
    })
}

/// The benchmark file's expected location.
///
/// Search order: relative to the current working directory, then relative
/// to the `svaraflow-asr` crate manifest (which is one level below the
/// workspace root).
#[must_use]
pub fn find_benchmark() -> Option<PathBuf> {
    let candidates = [
        PathBuf::from("tests/audio/benchmark20s.wav"),
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("tests")
            .join("audio")
            .join("benchmark20s.wav"),
    ];
    candidates.into_iter().find(|p| p.exists())
}

/// Picks a tier from CPU cores and NVIDIA GPU presence.
///
/// Deliberately conservative: without a benchmark, only a clear signal
/// (a CUDA-capable GPU, or eight or more logical cores) selects Standard.
#[must_use]
pub fn hardware_heuristic() -> Tier {
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let has_gpu = nvidia_gpu_present();
    if has_gpu || cores >= 8 {
        Tier::Standard
    } else {
        Tier::Light
    }
}

/// True when `nvidia-smi` runs and reports at least one GPU.
#[must_use]
pub fn nvidia_gpu_present() -> bool {
    Command::new("nvidia-smi")
        .args(["--query-gpu=name", "--format=csv,noheader"])
        .output()
        .map(|o| o.status.success() && !o.stdout.is_empty())
        .unwrap_or(false)
}

/// Number of logical CPU cores, or `1` when the query fails.
#[must_use]
pub fn logical_cores() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rtf_threshold_is_two() {
        assert_eq!(AUTO_STANDARD_RTF, 2.0);
    }

    #[test]
    fn choice_heuristic_has_no_rtf() {
        let c = TierChoice::heuristic(Tier::Light, "test".into());
        assert_eq!(c.tier, Tier::Light);
        assert!(c.rtf.is_none());
    }

    #[test]
    fn choice_benchmarked_has_rtf() {
        let c = TierChoice::benchmarked(Tier::Standard, 3.5);
        assert_eq!(c.tier, Tier::Standard);
        assert_eq!(c.rtf, Some(3.5));
        assert!(c.reason.contains("3.50"));
    }

    #[test]
    fn choice_benchmarked_reason_mentions_light_when_light_wins() {
        let c = TierChoice::benchmarked(Tier::Light, 0.8);
        assert!(c.reason.contains("Light"), "{}", c.reason);
    }

    #[test]
    fn pick_tier_errors_when_nothing_is_present() {
        let m = Manifest::load_embedded().unwrap();
        let r = pick_tier(&m, false, false, None);
        assert!(r.is_err());
        let msg = r.unwrap_err().to_string();
        assert!(msg.contains("models download"), "{msg}");
    }

    #[test]
    fn pick_tier_uses_light_when_only_light_present() {
        let m = Manifest::load_embedded().unwrap();
        let c = pick_tier(&m, true, false, None).unwrap();
        assert_eq!(c.tier, Tier::Light);
        assert!(c.rtf.is_none());
    }

    #[test]
    fn pick_tier_uses_standard_when_only_standard_present() {
        let m = Manifest::load_embedded().unwrap();
        let c = pick_tier(&m, false, true, None).unwrap();
        assert_eq!(c.tier, Tier::Standard);
    }

    #[test]
    fn hardware_heuristic_returns_a_tier() {
        let t = hardware_heuristic();
        assert!(matches!(t, Tier::Light | Tier::Standard));
    }

    #[test]
    fn logical_cores_is_at_least_one() {
        assert!(logical_cores() >= 1);
    }

    #[test]
    fn find_benchmark_does_not_panic() {
        let _ = find_benchmark();
    }
}
