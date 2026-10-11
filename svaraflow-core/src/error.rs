//! Error type for `svaraflow-core`.
//!
//! Every fallible operation in this crate returns [`CoreError`]. Variants are
//! grouped by subsystem (timeline parsing, audio decoding, detection, export)
//! so that callers can match narrowly. `#[from]` is used only for external
//! error types that are not part of SvaraFlow's own contract.

use thiserror::Error;

/// Errors produced by `svaraflow-core`.
#[derive(Debug, Error)]
pub enum CoreError {
    // --- timeline ---------------------------------------------------------
    /// Frame rate numerator or denominator was zero.
    #[error("invalid fps: numerator and denominator must be positive")]
    InvalidFps,

    /// A track index was not a positive integer.
    #[error("track index must start at 1 (got {0})")]
    InvalidTrackIndex(u32),

    /// A clip's `source_out` was not strictly greater than its `source_in`.
    #[error("clip `{id}`: source_out ({source_out}) must be greater than source_in ({source_in})")]
    InvalidClipRange {
        /// Clip identifier from the input JSON.
        id: String,
        /// Inclusive first source frame.
        source_in: i64,
        /// Exclusive last source frame.
        source_out: i64,
    },

    /// A clip's `speed` was zero, negative, or not finite.
    #[error("clip `{id}`: speed must be a finite value > 0 (got {speed})")]
    InvalidClipSpeed {
        /// Clip identifier from the input JSON.
        id: String,
        /// The offending speed.
        speed: f64,
    },

    /// Two clips in the same timeline shared an `id`.
    #[error("clip id `{0}` is used more than once")]
    DuplicateClipId(String),

    /// Two tracks in the same timeline shared an `index`.
    #[error("track index {0} is used more than once")]
    DuplicateTrackIndex(u32),

    /// A timeline-level invariant failed (dimensions, empty tracks, ...).
    #[error("timeline: {0}")]
    InvalidTimeline(String),

    // --- io / json --------------------------------------------------------
    /// Underlying I/O failure (files, pipes, stdin, stdout).
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON parse or serialize failure.
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    // --- audio ------------------------------------------------------------
    /// A clip's audio could not be decoded by any available backend.
    ///
    /// `id` always names the clip from `timeline.json`, so the CLI can report
    /// exactly which source failed and suggest `--audio <wav>`.
    #[error("clip `{id}`: audio decode failed: {reason}")]
    AudioDecode {
        /// Clip identifier from the input JSON.
        id: String,
        /// Human-readable reason, including every backend that was tried.
        reason: String,
    },

    /// A sample rate was zero or otherwise unusable.
    #[error("invalid sample rate: {0}")]
    InvalidSampleRate(u32),

    /// Sample-rate conversion failed.
    #[error("resample {from_hz} Hz -> {to_hz} Hz: {reason}")]
    Resample {
        /// Source rate.
        from_hz: u32,
        /// Target rate.
        to_hz: u32,
        /// Human-readable reason from the resampler.
        reason: String,
    },

    /// Mixing the analysis signal failed.
    #[error("mixdown: {0}")]
    Mixdown(String),

    // --- vad --------------------------------------------------------------
    /// The VAD engine failed during model loading or inference.
    ///
    /// The payload is a free-form message from the concrete engine
    /// (typically `svaraflow-asr`'s `AsrError`). It always names the
    /// operation that failed and, where available, the model path or
    /// ONNX Runtime variable to check.
    #[error("vad failed: {0}")]
    VadFailed(String),

    // --- scaffold ---------------------------------------------------------
    /// Feature not yet implemented.
    #[error("not yet implemented: {0}")]
    NotImplemented(&'static str),
}
