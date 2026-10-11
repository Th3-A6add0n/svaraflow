//! Error type for `svaraflow-asr`.

use thiserror::Error;

/// Errors produced by `svaraflow-asr`.
#[derive(Debug, Error)]
pub enum AsrError {
    // --- model management -------------------------------------------------
    /// The model manifest could not be parsed.
    #[error("model manifest: {0}")]
    Manifest(String),

    /// No model is declared for the requested tier.
    #[error("no model declared for tier `{0}`; check models.json")]
    UnknownTier(String),

    /// The model file is not present locally.
    #[error(
        "model `{name}` not found at {path}. Download it with \
         `svaraflow-cli models download {name}`, or place the file there \
         by hand."
    )]
    ModelMissing {
        /// Manifest name of the model.
        name: String,
        /// Where the loader expected to find it.
        path: String,
    },

    /// The model file's SHA-256 does not match the manifest.
    #[error(
        "model `{name}` at {path} has SHA-256 {actual}, expected {expected}. \
         The file is corrupt or the manifest is stale; delete it and \
         re-download."
    )]
    ModelHashMismatch {
        /// Manifest name of the model.
        name: String,
        /// Path that was hashed.
        path: String,
        /// Lowercase hex of what was on disk.
        actual: String,
        /// Lowercase hex from the manifest.
        expected: String,
    },

    /// A model download failed.
    #[error("downloading `{name}` from {url}: {reason}")]
    Download {
        /// Manifest name of the model.
        name: String,
        /// The URL that was tried.
        url: String,
        /// What went wrong.
        reason: String,
    },

    /// The model folder could not be located or created.
    #[error("model folder: {0}")]
    ModelFolder(String),

    // --- engine -----------------------------------------------------------
    /// The VAD model file could not be found or read.
    #[error("vad model {path}: {reason}")]
    VadModelIo {
        /// Path to the model file.
        path: String,
        /// Underlying error message.
        reason: String,
    },

    /// The ONNX session for the VAD could not be created.
    #[error("vad session: {0}")]
    VadSession(String),

    /// The VAD failed during inference.
    #[error("vad inference: {0}")]
    VadInference(String),

    /// A VAD model output was missing or had the wrong shape.
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

    /// whisper.cpp failed to load the model.
    #[error("whisper context from {path}: {reason}")]
    WhisperContext {
        /// Path to the model file.
        path: String,
        /// Underlying error message.
        reason: String,
    },

    /// whisper.cpp failed during transcription.
    #[error("whisper state: {reason}")]
    WhisperState {
        /// Underlying error message.
        reason: String,
    },

    /// whisper.cpp could not create an internal state object.
    #[error("whisper state creation: {0}")]
    WhisperStateCreate(String),

    /// Token-level timestamps were requested but the model produced none.
    #[error(
        "DTW timestamps requested but whisper.cpp returned none for segment \
         {segment}; falling back to proportional timing. Pass \
         `--no-dtw` to silence this."
    )]
    DtwUnavailable {
        /// Index of the segment where the fallback happened.
        segment: usize,
    },

    /// Transcription was cancelled.
    #[error("transcription cancelled")]
    Cancelled,

    // --- cross-crate ------------------------------------------------------
    /// An error from `svaraflow-core`.
    ///
    /// Most commonly raised by the VAD trait, whose
    /// `speech_probabilities` returns [`svaraflow_core::error::CoreError`].
    #[error("core: {0}")]
    Core(#[from] svaraflow_core::error::CoreError),

    // --- io ---------------------------------------------------------------
    /// Underlying I/O failure.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON parse or serialize failure.
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    /// Feature not yet implemented in this scaffold.
    #[error("not yet implemented: {0}")]
    NotImplemented(&'static str),
}
