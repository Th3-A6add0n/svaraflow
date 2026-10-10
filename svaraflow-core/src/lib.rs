//! Core audio, timeline and cut-engine primitives for SvaraFlow.
//!
//! This crate is host-agnostic. The CLI, the JSON-RPC server and the host
//! adapters all consume these types. Nothing here talks to Premiere Pro,
//! DaVinci Resolve or the network.

pub mod audio;
pub mod cut;
pub mod edl;
pub mod error;
pub mod silence;
pub mod timeline;
pub mod vad;
pub mod xmeml;

pub use audio::{
    compute_envelope, downsample_for_preview, resample_to, ClipDecoder, Envelope, EnvelopeParams,
    FileDecoder, MonoAudio, ANALYSIS_SAMPLE_RATE, DBFS_FLOOR,
};
pub use cut::{apply_cuts, CutMode, CutResult, CutStats};
pub use edl::{format_timecode, write_edl};
pub use error::CoreError;
pub use silence::{
    auto_noise_level, detect_silence, FrameRange, Method, NoiseSetting, PadSide, Preset,
    SilenceParams, SilenceRanges, SilenceStats, SilenceWarning, AUTO_NOISE_MAX_DB,
    AUTO_NOISE_MIN_DB, HYSTERESIS_DB, MIN_SAFE_SECONDS,
};
pub use timeline::{Clip, Fps, Timeline, Track, TrackKind};
pub use vad::{
    apply_vad_rules, hybrid_snap, StubVad, Vad, VadMask, VadParams, VadProbabilities,
    DEFAULT_MIN_SILENCE_S, DEFAULT_MIN_SPEECH_S, DEFAULT_PROB_THRESHOLD, DEFAULT_SNAP_RADIUS_S,
    DEFAULT_SPEECH_PAD_S, VAD_SAMPLE_RATE, VAD_WINDOW_SAMPLES,
};
pub use xmeml::{encode_path_url, write_xmeml, SEQUENCE_SUFFIX, XMEML_VERSION};

/// Returns the semver version of this crate as declared by Cargo.
#[must_use]
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_is_non_empty() {
        assert!(!version().is_empty());
    }

    #[test]
    fn placeholder_error_displays() {
        let err = CoreError::NotImplemented("silence");
        assert!(err.to_string().contains("silence"));
    }

    #[test]
    fn analysis_rate_is_whisper_rate() {
        assert_eq!(ANALYSIS_SAMPLE_RATE, 16_000);
    }

    #[test]
    fn xmeml_is_reachable_from_crate_root() {
        let _ = XMEML_VERSION;
        let _ = SEQUENCE_SUFFIX;
    }

    #[test]
    fn edl_is_reachable_from_crate_root() {
        let tc = format_timecode(0, Fps { num: 30, den: 1 }, false);
        assert_eq!(tc, "00:00:00:00");
    }

    #[test]
    fn vad_is_reachable_from_crate_root() {
        assert_eq!(VAD_WINDOW_SAMPLES, 512);
        assert_eq!(VAD_SAMPLE_RATE, 16_000);
        let p = VadParams::default();
        assert!((p.prob_threshold - 0.5).abs() < 1e-6);
    }
}
