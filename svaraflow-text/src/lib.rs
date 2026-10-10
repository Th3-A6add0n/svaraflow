//! Text pipeline for SvaraFlow.
//!
//! Handles SRT/VTT import and export, the Roman Hinglish dictionary,
//! chunking of word timings into caption cues, style presets, and
//! translation. Version 1 targets Roman-script output only.

use thiserror::Error;

/// Errors produced by `svaraflow-text`.
#[derive(Debug, Error)]
pub enum TextError {
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
        let err = TextError::NotImplemented("chunker");
        assert!(err.to_string().contains("chunker"));
    }
}
