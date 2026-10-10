//! Core audio, timeline and cut-engine primitives for SvaraFlow.
//!
//! This crate is host-agnostic. The CLI, the JSON-RPC server and the host
//! adapters all consume these types. Nothing here talks to Premiere Pro,
//! DaVinci Resolve or the network.

use thiserror::Error;

/// Errors produced by `svaraflow-core`.
#[derive(Debug, Error)]
pub enum CoreError {
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
        let err = CoreError::NotImplemented("silence");
        assert!(err.to_string().contains("silence"));
    }
}
