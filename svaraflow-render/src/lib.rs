//! Caption renderer for SvaraFlow.
//!
//! Turns styled, chunked cues from `svaraflow-text` into raster frames
//! (word-by-word animated captions in Roman Hinglish). Version 1 ships
//! style presets only; emoji and per-speaker styles are out of scope.

use thiserror::Error;

/// Errors produced by `svaraflow-render`.
#[derive(Debug, Error)]
pub enum RenderError {
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
        let err = RenderError::NotImplemented("style-preset");
        assert!(err.to_string().contains("style-preset"));
    }
}
