//! Error type for `svaraflow-asr`.

use thiserror::Error;

/// Errors produced by `svaraflow-asr`.
#[derive(Debug, Error)]
pub enum AsrError {
    /// The VAD model file could not be found or read.
    #[error("vad model {path}: {reason}")]
    VadModelIo {
        /// Path to the model file.
        path: String,
        /// Underlying error message.
        reason: String,
    },

    /// The ONNX session could not be created.
    #[error("vad session: {0}")]
    VadSession(String),

    /// Inference failed.
    #[error("vad inference: {0}")]
    VadInference(String),

    /// A model output was missing or had the wrong shape.
    #[error("vad output `{name}`: {reason}")]
    VadOutputShape {
        /// Name of the offending input or output.
        name: String,
        /// What was expected and what was seen.
        reason: String,
    },

    /// The VAD feature was not compiled in.
    #[error(
        "Silero VAD is not available: build `svaraflow-asr` with the `vad` \
         feature (the default) to enable it"
    )]
    VadFeatureDisabled,

    /// Feature not yet implemented in this scaffold.
    #[error("not yet implemented: {0}")]
    NotImplemented(&'static str),
}