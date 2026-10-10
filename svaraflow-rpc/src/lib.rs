//! JSON-RPC 2.0 server for SvaraFlow.
//!
//! Binds to `127.0.0.1` only and requires a one-time token in the
//! handshake so that host adapters (DaVinci Resolve Python bridge and
//! the Premiere Pro UXP plugin) can talk to the engine without exposing
//! it to the network.

use thiserror::Error;

/// Errors produced by `svaraflow-rpc`.
#[derive(Debug, Error)]
pub enum RpcError {
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
        let err = RpcError::NotImplemented("handshake");
        assert!(err.to_string().contains("handshake"));
    }
}
