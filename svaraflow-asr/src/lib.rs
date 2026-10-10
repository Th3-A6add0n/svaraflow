//! Speech-to-text and voice activity detection for SvaraFlow.
//!
//! - `vad` (feature-gated, default on): Silero VAD v5 through ONNX Runtime.
//!   Implements [`svaraflow_core::vad::Vad`] so the detector in
//!   `svaraflow-core` does not need to know about ONNX.
//! - Speech-to-text through `whisper.cpp` is not implemented yet; the
//!   placeholder [`AsrError::NotImplemented`] covers it.

pub mod error;

#[cfg(feature = "vad")]
pub mod vad;

pub use error::AsrError;

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
    fn vad_feature_disabled_error_displays() {
        let err = AsrError::VadFeatureDisabled;
        let s = err.to_string();
        assert!(s.contains("vad") || s.contains("VAD"));
    }

    #[cfg(feature = "vad")]
    #[test]
    fn silero_vad_is_reachable() {
        // Compile-time check that the type exists and implements Vad.
        fn assert_impl<T: svaraflow_core::vad::Vad>() {}
        assert_impl::<SileroVad>();
    }
}
