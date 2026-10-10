//! Speech-to-text for SvaraFlow.
//!
//! Wraps `whisper.cpp` through Rust bindings and exposes two model tiers:
//! Light (Whisper-Hindi2Hinglish-Swift, whisper-base size) and Standard
//! (Whisper-Hindi2Hinglish-Apex Q5_0, ~547 MB). Both emit Roman Hinglish
//! directly. Silero VAD is used for speech-activity detection.

use thiserror::Error;

/// Errors produced by `svaraflow-asr`.
#[derive(Debug, Error)]
pub enum AsrError {
    /// Placeholder used while the scaffold is being built out.
    #[error("not yet implemented: {0}")]
    NotImplemented(&'static str),
}

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
        let err = AsrError::NotImplemented("vad");
        assert!(err.to_string().contains("vad"));
    }
}
