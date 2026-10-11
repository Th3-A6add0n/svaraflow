//! Core audio, timeline and cut-engine primitives for SvaraFlow.

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
    auto_noise_level, auto_noise_level_from_vad, detect_silence, detect_silence_with_vad,
    AnalysisReport, BuiltinPreset, CutDetail, FrameRange, MarginSide, Method, NoiseSetting,
    NoiseSource, PresetValues, SilenceParams, SilenceRanges, SilenceStats, SilenceWarning,
    SnapMode, AUTO_NOISE_MAX_DB, AUTO_NOISE_MIN_DB, HYSTERESIS_DB, MIN_SAFE_MS,
};
pub use timeline::{Clip, Fps, Timeline, Track, TrackKind};
pub use vad::{
    apply_vad_rules, hybrid_snap, StubVad, Vad, VadMask, VadParams, VadProbabilities,
    DEFAULT_MIN_SILENCE_S, DEFAULT_MIN_SPEECH_S, DEFAULT_PROB_THRESHOLD, DEFAULT_SNAP_RADIUS_S,
    DEFAULT_SPEECH_PAD_S, VAD_SAMPLE_RATE, VAD_WINDOW_SAMPLES,
};
pub use xmeml::{
    encode_path_url, write_xmeml, SourceInfo, XmlProfile, SEQUENCE_SUFFIX, XMEML_VERSION,
};

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
        let _ = XmlProfile::Full;
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
    }

    #[test]
    fn source_info_default_is_mono() {
        let s = SourceInfo::default();
        assert_eq!(s.audio_channels, 1);
        assert_eq!(s.audio_sample_rate, 48_000);
    }

    #[test]
    fn silence_presets_are_reachable() {
        assert_eq!(BuiltinPreset::ALL.len(), 5);
        assert_eq!(BuiltinPreset::Natural.name(), "natural");
        let p = SilenceParams::from_preset(BuiltinPreset::Max);
        assert_eq!(p.snap, SnapMode::Nearest);
    }
}
