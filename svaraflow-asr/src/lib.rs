//! Speech-to-text and voice activity detection for SvaraFlow.
//!
//! Modules:
//!
//! - [`models`]: the pinned manifest of speech-to-text models, plus a
//!   resumable, SHA-256-verifying downloader.
//! - [`transcribe`]: the whisper.cpp engine. Loads a GGML model, chunks
//!   the audio with the VAD, and returns word-level timestamps.
//! - [`cache`]: transcript cache keyed on audio hash, model, and settings.
//! - [`exports`]: SRT, WebVTT, ASS with karaoke highlighting, and the
//!   transcript JSON.
//! - [`bench`]: automatic tier selection from a 20-second benchmark clip,
//!   with a CPU/GPU heuristic fallback.
//! - [`vad`] (feature-gated, default on): Silero VAD through ONNX Runtime.
//!
//! Everything runs locally. No audio, transcript, or model leaves the
//! machine.

pub mod bench;
pub mod cache;
pub mod error;
pub mod exports;
pub mod models;
pub mod transcribe;

#[cfg(feature = "vad")]
pub mod vad;

pub use bench::{hardware_heuristic, logical_cores, nvidia_gpu_present, pick_tier, TierChoice};
pub use cache::{CacheKey, CachedTranscript};
pub use error::AsrError;
pub use exports::{ass_time, srt_time, vtt_time, write_ass, write_json, write_srt, write_vtt};
pub use models::{
    ensure_model, ensure_storage_dir, hash_file, path_for, storage_dir, Manifest, ModelEntry,
    ProgressFn, Tier, TODO_SENTINEL,
};
pub use transcribe::{
    Chunk, DecodeStrategy, Engine, LanguageHint, Segment, TranscribeConfig, TranscribeProgressFn,
    Transcript, Word, DEFAULT_MAX_CHUNK_S, DEFAULT_OVERLAP_S, SAMPLE_RATE, SNAP_RADIUS_S,
};

#[cfg(feature = "vad")]
pub use vad::SileroVad;

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
        let err = AsrError::NotImplemented("whisper");
        assert!(err.to_string().contains("whisper"));
    }

    #[test]
    fn manifest_is_reachable_from_crate_root() {
        let m = Manifest::load_embedded().expect("manifest");
        assert_eq!(m.models.len(), 2);
        assert!(m.for_tier(Tier::Light).is_ok());
        assert!(m.for_tier(Tier::Standard).is_ok());
    }

    #[test]
    fn transcribe_config_is_reachable_from_crate_root() {
        let c = TranscribeConfig::default();
        assert_eq!(c.max_chunk_s, DEFAULT_MAX_CHUNK_S);
        assert_eq!(c.overlap_s, DEFAULT_OVERLAP_S);
    }

    #[test]
    fn export_helpers_are_reachable() {
        assert_eq!(srt_time(0.0), "00:00:00,000");
        assert_eq!(vtt_time(0.0), "00:00:00.000");
        assert_eq!(ass_time(0.0), "0:00:00.00");
    }

    #[test]
    fn cache_key_is_reachable() {
        let a = svaraflow_core::audio::MonoAudio::new(vec![0.0; 32], 16_000);
        let c = TranscribeConfig::default();
        let _ = cache::key_for(&a, "model", &c);
    }

    #[test]
    fn bench_helpers_are_reachable() {
        assert!(logical_cores() >= 1);
        let _ = hardware_heuristic();
    }

    #[cfg(feature = "vad")]
    #[test]
    fn silero_vad_is_reachable() {
        fn assert_impl<T: svaraflow_core::vad::Vad>() {}
        assert_impl::<SileroVad>();
    }

    #[test]
    fn hash_file_is_reachable() {
        let tmp = std::env::temp_dir().join("svaraflow-lib-hash-probe.bin");
        std::fs::write(&tmp, b"").unwrap();
        let h = hash_file(&tmp).unwrap();
        assert_eq!(h.len(), 64);
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn todo_sentinel_is_exported() {
        assert_eq!(TODO_SENTINEL, "TODO");
    }
}
