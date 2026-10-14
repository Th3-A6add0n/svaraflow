// svaraflow-asr/src/transcribe.rs
//! Transcription engine.
//!
//! Wraps `whisper-rs` behind a typed API. The engine:
//!
//! 1. Runs Silero VAD over the 16 kHz mono signal to find speech regions.
//! 2. Cuts the signal into chunks at pauses. Speech regions shorter than
//!    `max_chunk_s` become a single chunk; longer regions are split at the
//!    longest internal pause near the limit. When no pause is available,
//!    the region is split forcibly at `max_chunk_s` with `overlap_s` of
//!    padding on each side. Every chunk carries a [`ChunkReason`].
//! 3. Runs whisper.cpp on each chunk, with token-level timestamps if
//!    asked.
//! 4. Merges sub-word tokens into whole words, strips special tokens,
//!    drops duplicate words in the overlap, enforces increasing
//!    non-overlapping times, and snaps each boundary to a local energy
//!    minimum within [`SNAP_RADIUS_S`].
//! 5. Applies a repetition guard, clamps every time into
//!    `[0, audio_duration]`, merges any segment whose span is too short
//!    for its word count with the following segment, and regroups the
//!    words into caption-ready phrases of at most
//!    [`DEFAULT_MAX_PHRASE_S`] seconds and [`DEFAULT_MAX_PHRASE_WORDS`]
//!    words.
//!
//! # Per-window quality
//!
//! Every window's raw output is scored before post-processing. A window
//! that fails the guard is re-decoded with token timestamps off; whichever
//! pass produces the higher mean probability and lower junk fraction is
//! kept. The outcome is recorded in [`Timings::windows`].
//!
//! # Diagnostics
//!
//! [`Engine::transcribe_with_timings`] writes whisper.cpp's system info,
//! the effective `n_threads`, the compiled backend, the device index and
//! the full `FullParams` set to stderr, and mirrors them into [`Timings`].

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperState,
};

use svaraflow_core::audio::MonoAudio;

use crate::error::AsrError;
use crate::models::{ModelEntry, Tier};

/// Sample rate the engine expects.
pub const SAMPLE_RATE: u32 = 16_000;

/// Longest chunk fed to one `whisper.cpp` call when no pause is
/// available to split at. In practice, `plan_chunks_with_vad` splits at
/// a pause before this limit whenever it can.
pub const DEFAULT_MAX_CHUNK_S: f64 = 25.0;

/// Padding added on each side of a forced split.
pub const DEFAULT_OVERLAP_S: f64 = 0.3;

/// Radius searched when snapping a word boundary to a local energy minimum.
pub const SNAP_RADIUS_S: f64 = 0.060;

/// VAD probability above which a window counts as speech.
pub const VAD_THRESHOLD: f32 = 0.5;

/// Silence shorter than this merges two speech regions into one.
pub const VAD_MIN_SILENCE_S: f64 = 0.100;

/// Longest a caption phrase may be, in seconds.
pub const DEFAULT_MAX_PHRASE_S: f64 = 8.0;

/// Longest a caption phrase may be, in words.
pub const DEFAULT_MAX_PHRASE_WORDS: usize = 12;

/// Seconds of padding added on each side of every speech region when
/// planning chunks. Gives whisper.cpp a little silence context.
const CHUNK_PAD_S: f64 = 0.2;

/// Minimum silence, in seconds, that counts as a split point when a
/// speech region exceeds the chunk limit.
const MIN_PAUSE_FOR_SPLIT_S: f64 = 0.2;

/// Minimum silence, in seconds, that counts as a phrase boundary when
/// regrouping words into captions.
const MIN_GAP_FOR_PHRASE_S: f64 = 0.3;

/// Smallest plausible duration per word, in seconds.
///
/// whisper.cpp occasionally returns a segment whose `t0` and `t1` are
/// only a centisecond or two apart while containing a whole sentence's
/// tokens. Every word in the segment then inherits a fraction of a
/// millisecond, and the resulting caption cue is invisible on playback.
///
/// A segment with `n` words whose span is under `n * MIN_S_PER_WORD` is
/// **merged** with the following segment and the combined words
/// redistributed across the combined span. Merging rather than
/// stretching is required: the next segment starts within milliseconds
/// of the compressed one, so there is no room to stretch into.
const MIN_S_PER_WORD: f64 = 0.05;

/// Whether the whisper.cpp build passed the optimization guard.
fn build_optimized() -> Option<bool> {
    match option_env!("SVARAFLOW_BUILD_OPTIMIZED") {
        Some("1") => Some(true),
        Some("0") => Some(false),
        _ => None,
    }
}

/// Which backend this build was compiled for.
#[must_use]
pub fn compiled_backend() -> &'static str {
    if cfg!(feature = "cuda") {
        "CUDA"
    } else {
        "CPU"
    }
}

// ---------------------------------------------------------------------------
// Chunk reasons
// ---------------------------------------------------------------------------

/// Why the engine started a new chunk where it did.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChunkReason {
    /// The first chunk of the run.
    StartOfAudio,
    /// Chunk begins after a detected pause inside a long speech region.
    Pause,
    /// Chunk begins at the start of a speech region, i.e. after silence.
    RegionBoundary,
    /// Chunk was cut at `max_chunk_s` because no pause was available.
    /// `overlap_s` was added on both sides.
    ForcedSize,
    /// Chunk came from fixed-window chunking because VAD was not run.
    FixedWindow,
}

// ---------------------------------------------------------------------------
// VAD speech regions
// ---------------------------------------------------------------------------

/// Speech regions found by Silero VAD, in source seconds.
///
/// Empty when no VAD model was loaded. Used by both chunk planning and
/// phrase segmentation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SpeechRegions {
    /// Non-overlapping intervals `(start_s, end_s)` in source seconds.
    pub regions: Vec<(f64, f64)>,
}

impl SpeechRegions {
    /// True when there are no regions at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.regions.is_empty()
    }

    /// True when `t` falls inside any region.
    #[must_use]
    pub fn contains(&self, t: f64) -> bool {
        self.regions.iter().any(|(s, e)| t >= *s && t <= *e)
    }

    /// True when the interval `[a, b]` spans a silence of at least
    /// `min_gap_s` seconds between two consecutive speech regions.
    ///
    /// Used by phrase segmentation to break a phrase at a detected pause.
    #[must_use]
    pub fn crosses_silence(&self, a: f64, b: f64, min_gap_s: f64) -> bool {
        for pair in self.regions.windows(2) {
            let (_, prev_end) = pair[0];
            let (next_start, _) = pair[1];
            let gap = next_start - prev_end;
            if gap >= min_gap_s && a < prev_end && b > next_start {
                return true;
            }
        }
        false
    }
}

// ---------------------------------------------------------------------------
// Chunk plan
// ---------------------------------------------------------------------------

/// The chunk plan for one run: chunks, speech regions, and reasons.
#[derive(Debug, Clone)]
struct ChunkPlan {
    chunks: Vec<Chunk>,
    regions: SpeechRegions,
    reasons: Vec<ChunkReason>,
}

// ---------------------------------------------------------------------------
// Junk guard
// ---------------------------------------------------------------------------

/// Thresholds the per-window quality check uses.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct JunkGuard {
    /// Reject when non-Latin / non-Devanagari characters exceed this
    /// fraction of the non-whitespace output.
    pub max_junk_fraction: f32,
    /// Reject when the longest run of identical adjacent characters
    /// exceeds this fraction of the output length.
    pub max_repeat_ratio: f32,
    /// Reject when mean token probability is below this.
    pub min_mean_probability: f32,
    /// Reject when whisper.cpp's no-speech probability exceeds this.
    pub max_no_speech_prob: f32,
}

impl Default for JunkGuard {
    fn default() -> Self {
        Self {
            max_junk_fraction: 0.10,
            max_repeat_ratio: 0.40,
            min_mean_probability: 0.30,
            max_no_speech_prob: 0.75,
        }
    }
}

/// Quality of one window's raw output, before post-processing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindowTiming {
    /// Index in the chunk list, in order.
    pub index: usize,
    /// Window start in source seconds.
    pub start_s: f64,
    /// Window end in source seconds.
    pub end_s: f64,
    /// Wall-clock time inside `WhisperState::full`, summed over this
    /// window's attempts.
    pub whisper_full_s: f64,
    /// Number of tokens whisper.cpp produced for this window.
    pub tokens: usize,
    /// Number of segments after merging.
    pub segments: usize,
    /// Mean token probability.
    pub mean_probability: f32,
    /// Minimum token probability.
    pub min_probability: f32,
    /// Fraction of non-whitespace characters outside Basic Latin,
    /// Devanagari and ASCII punctuation.
    pub junk_fraction: f32,
    /// Longest run of identical adjacent characters, divided by length.
    pub repeat_ratio: f32,
    /// whisper.cpp's no-speech probability for this window.
    pub no_speech_prob: f32,
    /// `true` when either pass failed the guard.
    pub low_confidence: bool,
    /// `Some(reason)` when the guard tripped and a fallback was tried.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retry_reason: Option<String>,
    /// Language hint the winning pass used.
    pub language_requested: String,
}

// ---------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------

/// A wall-clock accumulator for one phase.
#[derive(Debug, Default, Clone)]
pub struct Section {
    start: Option<Instant>,
    total: Duration,
}

impl Section {
    /// Begins timing. Overwrites a running start, if any.
    pub fn start(&mut self) {
        self.start = Some(Instant::now());
    }

    /// Ends timing and adds the elapsed wall-clock time.
    pub fn stop(&mut self) {
        if let Some(s) = self.start.take() {
            self.total += s.elapsed();
        }
    }

    /// Total wall-clock time recorded so far.
    #[must_use]
    pub fn total(&self) -> Duration {
        self.total
    }
}

/// Wall-clock breakdown of one transcription run.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Timings {
    /// Decoding the source file into PCM. Caller-owned.
    #[serde(default)]
    pub audio_decode_s: f64,
    /// Sample-rate conversion to 16 kHz. Caller-owned.
    #[serde(default)]
    pub resample_s: f64,
    /// Silero VAD. Zero when VAD did not run.
    pub vad_s: f64,
    /// Choosing which samples go into which whisper.cpp window.
    pub chunk_plan_s: f64,
    /// `WhisperContext::new_with_params`, i.e. loading the GGML model.
    #[serde(default)]
    pub model_load_s: f64,
    /// `WhisperContext::create_state`, i.e. allocating KV and compute
    /// buffers. GPU-heavy: on CUDA this is the `cudaMalloc` storm that
    /// whisper.cpp resets between full calls.
    #[serde(default)]
    pub state_create_s: f64,
    /// Everything inside `Engine::transcribe_with_timings` after load.
    pub transcribe_s: f64,
    /// Sum of `WhisperState::full` calls across all workers, including
    /// junk-guard fallbacks.
    pub whisper_full_s: f64,
    /// Merging tokens, deduping overlaps, monotonic ordering, snapping,
    /// and phrase segmentation.
    pub postprocess_s: f64,
    /// Writing JSON / SRT / VTT / ASS. Caller-owned.
    #[serde(default)]
    pub export_s: f64,
    /// Wall time from process start to process exit.
    pub total_wall_s: f64,
    /// Input duration from the decoded signal.
    pub audio_duration_s: f64,
    /// `audio_duration_s / total_wall_s`. `0.0` when `total_wall_s` is 0.
    pub real_time_factor: f64,
    /// CPU and build probes collected at load time.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_info: Option<String>,
    /// Number of windows that failed the junk guard on the first pass.
    #[serde(default)]
    pub decode_retries: usize,
    /// `true` when every word carries a DTW-derived timestamp.
    #[serde(default)]
    pub dtw_usable: bool,
    /// `true` when the run used VAD packing.
    #[serde(default)]
    pub vad_pack: bool,
    /// Number of source chunks actually fed to the engine.
    #[serde(default)]
    pub chunks: usize,
    /// Why each chunk started where it did. Parallel to the chunk list
    /// up to the point where `--ranges` filtering runs; cleared when
    /// ranges are applied because the reasons no longer apply.
    #[serde(default)]
    pub chunk_reasons: Vec<ChunkReason>,
    /// Number of whisper states used concurrently.
    #[serde(default)]
    pub parallel: usize,
    /// Compile-time backend: `"CPU"` or `"CUDA"`.
    #[serde(default)]
    pub backend: String,
    /// CUDA device index when the backend is CUDA, `"-"` on CPU.
    #[serde(default)]
    pub device: String,
    /// Threads handed to whisper.cpp per state.
    #[serde(default)]
    pub threads: i32,
    /// The `FullParams` set we actually passed.
    #[serde(default)]
    pub params: String,
    /// The `WHISPER : …` line from whisper.cpp's system info.
    #[serde(default)]
    pub build_flags: String,
    /// Whether the whisper.cpp build passed the optimization guard.
    #[serde(default)]
    pub build_optimized: bool,
    /// Per-window quality and timing.
    #[serde(default)]
    pub windows: Vec<WindowTiming>,
}

impl Timings {
    /// Records the total wall time and derives `real_time_factor`.
    pub fn finalize(&mut self, total_wall: Duration) {
        self.total_wall_s = total_wall.as_secs_f64();
        self.real_time_factor = if self.total_wall_s > 0.0 {
            self.audio_duration_s / self.total_wall_s
        } else {
            0.0
        };
    }
}

/// Collects what we can honestly say about the CPU and the build.
#[must_use]
pub fn system_info() -> String {
    let mut out = String::new();
    out.push_str("--- svaraflow ---\n");
    out.push_str(&format!("arch={}\n", std::env::consts::ARCH));
    out.push_str(&format!("os={}\n", std::env::consts::OS));
    out.push_str(&format!("target_avx2={}\n", cfg!(target_feature = "avx2")));
    out.push_str(&format!("target_fma={}\n", cfg!(target_feature = "fma")));
    out.push_str(&format!("target_f16c={}\n", cfg!(target_feature = "f16c")));
    #[cfg(target_arch = "x86_64")]
    {
        out.push_str(&format!(
            "runtime_avx2={}\n",
            std::arch::is_x86_feature_detected!("avx2")
        ));
        out.push_str(&format!(
            "runtime_fma={}\n",
            std::arch::is_x86_feature_detected!("fma")
        ));
        out.push_str(&format!(
            "runtime_f16c={}\n",
            std::arch::is_x86_feature_detected!("f16c")
        ));
        out.push_str(&format!(
            "runtime_avx512f={}\n",
            std::arch::is_x86_feature_detected!("avx512f")
        ));
    }
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    out.push_str(&format!("logical_cores={threads}\n"));
    out.push_str("--- whisper.cpp ---\n");
    out.push_str(whisper_rs::print_system_info());
    if !out.ends_with('\n') {
        out.push('\n');
    }
    out
}

/// Pulls the `WHISPER : …` line out of [`system_info`] output.
#[must_use]
pub fn extract_build_flags(system_info: &str) -> String {
    for line in system_info.lines() {
        if line.trim_start().starts_with("WHISPER :") {
            return line.trim().to_string();
        }
    }
    String::new()
}

/// Formats the `FullParams` set we hand whisper.cpp.
#[must_use]
pub fn params_summary(config: &TranscribeConfig, n_threads: i32) -> String {
    let strategy = match config.decode {
        DecodeStrategy::Greedy => "greedy(best_of=1)".to_string(),
        DecodeStrategy::BeamSearch { beam_size } => {
            format!("beam(size={beam_size}, patience=-1)")
        }
    };
    let temperature_inc = if config.temperature_fallback {
        0.2
    } else {
        0.0
    };
    let lang = match config.language {
        LanguageHint::Hindi => "hi",
        LanguageHint::English => "en",
        LanguageHint::Auto => "auto",
        LanguageHint::Other => "other",
    };
    format!(
        "strategy={strategy} threads={n_threads} language={lang} \
         token_timestamps={ts} temperature_inc={ti:.2} \
         no_speech_thold={ns:.2} temperature_fallback={tf} \
         flash_attn={fa} audio_ctx={ac:?} \
         [upstream defaults: threads=4 strategy=greedy/beam(5) \
         token_timestamps=false temperature_inc=0.2 \
         no_speech_thold=0.6 flash_attn=off audio_ctx=model-default]",
        ts = config.token_timestamps,
        ti = temperature_inc,
        ns = config.no_speech_threshold,
        tf = config.temperature_fallback,
        fa = config.flash_attn,
        ac = config.audio_ctx,
    )
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Language hint passed to whisper.cpp.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LanguageHint {
    /// Hindi.
    Hindi,
    /// English.
    English,
    /// Let whisper.cpp detect. This is the default: our fine-tuned models
    /// write Roman script regardless of the language token, and forcing
    /// `hi` on English speech causes the model to collapse.
    Auto,
    /// Any other two-letter code; passed through verbatim.
    Other,
}

impl LanguageHint {
    /// The string `whisper.cpp` expects, or `None` for auto.
    #[must_use]
    pub fn as_whisper_str(self) -> Option<&'static str> {
        match self {
            LanguageHint::Hindi => Some("hi"),
            LanguageHint::English => Some("en"),
            LanguageHint::Auto => None,
            LanguageHint::Other => None,
        }
    }

    /// Short tag for logging and JSON.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            LanguageHint::Hindi => "hi",
            LanguageHint::English => "en",
            LanguageHint::Auto => "auto",
            LanguageHint::Other => "other",
        }
    }

    /// Parses `hi`, `en`, `auto`, or a bare 2-letter code.
    #[must_use]
    pub fn parse(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "hi" | "hindi" => LanguageHint::Hindi,
            "en" | "english" => LanguageHint::English,
            "auto" | "" => LanguageHint::Auto,
            _ => LanguageHint::Other,
        }
    }
}

/// Decoding strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeStrategy {
    /// Greedy. The default.
    Greedy,
    /// Beam search with this beam width.
    BeamSearch {
        /// Number of beams.
        beam_size: i32,
    },
}

/// Everything the engine needs to know about one run.
#[derive(Debug, Clone)]
pub struct TranscribeConfig {
    /// Which tier is loaded.
    pub tier: Tier,
    /// Language hint.
    pub language: LanguageHint,
    /// Decoding strategy.
    pub decode: DecodeStrategy,
    /// Ask whisper.cpp for token-level timestamps.
    pub token_timestamps: bool,
    /// Optional initial prompt.
    pub initial_prompt: Option<String>,
    /// Domain vocabulary to bias the decoder with.
    pub vocabulary: Vec<String>,
    /// Whether to let whisper.cpp fall back to higher temperatures.
    pub temperature_fallback: bool,
    /// Segments with a no-speech probability above this are dropped.
    pub no_speech_threshold: f32,
    /// Whether to drop consecutive repeated n-grams.
    pub repetition_guard: bool,
    /// Longest chunk fed to one whisper.cpp call.
    pub max_chunk_s: f64,
    /// Overlap added on each side of a forced split.
    pub overlap_s: f64,
    /// Number of CPU threads *per whisper state*.
    pub threads: usize,
    /// Enable whisper.cpp's flash attention path. Off by default.
    pub flash_attn: bool,
    /// Override whisper.cpp's audio context size, in frames.
    pub audio_ctx: Option<i32>,
    /// Number of whisper states running concurrently.
    pub parallel: usize,
    /// Drop silence and pack speech into full-length windows.
    pub vad_pack: bool,
    /// Restrict transcription to these source time ranges, in seconds.
    pub ranges: Option<Vec<(f64, f64)>>,
    /// Junk-guard thresholds.
    pub junk_guard: JunkGuard,
    /// Longest caption phrase, in seconds.
    pub max_phrase_s: f64,
    /// Longest caption phrase, in words.
    pub max_phrase_words: usize,
}

impl Default for TranscribeConfig {
    fn default() -> Self {
        Self {
            tier: Tier::Standard,
            // Auto by default: the fine-tuned models write Roman
            // Hinglish regardless of the language token, and forcing
            // `hi` on English speech collapses Apex into repeated
            // Devanagari tokens. See docs/PERF.md.
            language: LanguageHint::Auto,
            decode: DecodeStrategy::Greedy,
            token_timestamps: false,
            initial_prompt: None,
            vocabulary: Vec::new(),
            temperature_fallback: true,
            no_speech_threshold: 0.6,
            repetition_guard: true,
            max_chunk_s: DEFAULT_MAX_CHUNK_S,
            overlap_s: DEFAULT_OVERLAP_S,
            threads: 0,
            flash_attn: false,
            audio_ctx: None,
            parallel: 1,
            vad_pack: false,
            ranges: None,
            junk_guard: JunkGuard::default(),
            max_phrase_s: DEFAULT_MAX_PHRASE_S,
            max_phrase_words: DEFAULT_MAX_PHRASE_WORDS,
        }
    }
}

impl TranscribeConfig {
    /// The initial prompt to pass to whisper.cpp, if any.
    #[must_use]
    pub fn effective_prompt(&self) -> Option<String> {
        if let Some(p) = &self.initial_prompt {
            return Some(p.clone());
        }
        if self.vocabulary.is_empty() {
            return None;
        }
        let joined = self.vocabulary.join(", ");
        Some(format!("Vocabulary: {joined}."))
    }

    /// Resolves [`Self::threads`] into a concrete `i32` for whisper.cpp,
    /// dividing evenly across [`Self::parallel`] states.
    #[must_use]
    pub fn resolved_threads(&self) -> i32 {
        let parallel = self.parallel.max(1);
        if self.threads > 0 {
            return (self.threads / parallel).max(1).min(i32::MAX as usize) as i32;
        }
        let n = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        (n / parallel).max(1) as i32
    }
}

/// Parses a ranges file.
///
/// # Errors
///
/// [`AsrError::WhisperState`] wrapping the failing line number on any
/// malformed line.
pub fn parse_ranges(text: &str) -> Result<Vec<(f64, f64)>, AsrError> {
    let mut out = Vec::new();
    for (i, raw) in text.lines().enumerate() {
        let line = raw.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let Some((a, b)) = line.split_once(',') else {
            return Err(AsrError::WhisperState {
                reason: format!("ranges line {}: expected `start,end`, got `{line}`", i + 1),
            });
        };
        let s: f64 = a.trim().parse().map_err(|e| AsrError::WhisperState {
            reason: format!("ranges line {}: bad start `{}`: {e}", i + 1, a.trim()),
        })?;
        let e: f64 = b.trim().parse().map_err(|e| AsrError::WhisperState {
            reason: format!("ranges line {}: bad end `{}`: {e}", i + 1, b.trim()),
        })?;
        if !(s.is_finite() && e.is_finite()) {
            return Err(AsrError::WhisperState {
                reason: format!("ranges line {}: non-finite value", i + 1),
            });
        }
        if e <= s {
            return Err(AsrError::WhisperState {
                reason: format!(
                    "ranges line {}: end ({e}) must be greater than start ({s})",
                    i + 1
                ),
            });
        }
        out.push((s, e));
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Results
// ---------------------------------------------------------------------------

/// One word with its timing and probability.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Word {
    /// The word's text, without leading or trailing whitespace and
    /// without any special token.
    pub text: String,
    /// Start time in seconds from the start of the source.
    pub start_s: f64,
    /// End time in seconds from the start of the source.
    pub end_s: f64,
    /// Whisper's probability for the token that produced this word.
    pub probability: f32,
    /// True when the timing came from proportional fallback, not DTW.
    pub estimated: bool,
}

/// One segment (roughly one caption phrase) with its words.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Segment {
    /// The segment's text, rebuilt from the words.
    pub text: String,
    /// Start time in seconds, clamped to `[0, audio_duration]`.
    pub start_s: f64,
    /// End time in seconds, clamped to `[0, audio_duration]`.
    pub end_s: f64,
    /// The words in this segment, in order.
    pub words: Vec<Word>,
    /// `true` when any window that contributed words to this segment
    /// failed the junk guard on its first pass.
    #[serde(default)]
    pub low_confidence: bool,
}

/// A finished transcription.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Transcript {
    /// The segments, in order.
    pub segments: Vec<Segment>,
    /// Detected or requested language.
    pub language: String,
    /// Which tier produced this.
    pub tier: Tier,
    /// Manifest name of the model.
    pub model: String,
    /// Duration of the input audio.
    pub audio_duration_s: f64,
    /// Wall-clock time spent inside `whisper.cpp` and VAD.
    pub elapsed_s: f64,
    /// `audio_duration_s / elapsed_s`. Above 1 is faster than real time.
    pub real_time_factor: f64,
}

impl Transcript {
    /// Every word across every segment, in order.
    #[must_use]
    pub fn words(&self) -> Vec<&Word> {
        self.segments.iter().flat_map(|s| s.words.iter()).collect()
    }
}

/// Progress callback signature for transcription.
pub type TranscribeProgressFn<'a> = dyn FnMut(usize, usize) + 'a;

/// The engine. Loaded once, reused across calls.
pub struct Engine {
    ctx: WhisperContext,
    config: TranscribeConfig,
    model_name: String,
    #[cfg(feature = "vad")]
    vad: Option<crate::vad::SileroVad>,
}

impl Engine {
    /// Loads a model and (optionally) a VAD.
    ///
    /// # Errors
    ///
    /// - [`AsrError::ModelMissing`] when the model file does not exist.
    /// - [`AsrError::WhisperContext`] when whisper.cpp rejects it.
    /// - [`AsrError::VadModelIo`] and friends when the VAD cannot load.
    /// - [`AsrError::WhisperState`] when `vad_pack` is set without a VAD
    ///   model, or when the `vad` feature is not compiled in.
    pub fn load(
        model_path: &Path,
        entry: &ModelEntry,
        vad_path: Option<&Path>,
        config: TranscribeConfig,
    ) -> Result<Self, AsrError> {
        if !model_path.exists() {
            return Err(AsrError::ModelMissing {
                name: entry.name.clone(),
                path: model_path.display().to_string(),
            });
        }
        if config.vad_pack {
            #[cfg(not(feature = "vad"))]
            {
                return Err(AsrError::WhisperState {
                    reason: "--vad-pack requires the `vad` feature on \
                             svaraflow-asr."
                        .into(),
                });
            }
            let vad_present = vad_path.map(|p| p.exists()).unwrap_or(false);
            #[cfg(feature = "vad")]
            if !vad_present {
                return Err(AsrError::WhisperState {
                    reason: "--vad-pack requires a VAD model. Pass \
                             --vad-model <path> or place \
                             models/silero_vad_v5.onnx."
                        .into(),
                });
            }
            let _ = vad_present;
        }

        let mut ctx_params = WhisperContextParameters::default();
        ctx_params.flash_attn(config.flash_attn);

        let model_str = model_path
            .to_str()
            .ok_or_else(|| AsrError::WhisperContext {
                path: model_path.display().to_string(),
                reason: "path is not valid UTF-8".into(),
            })?;

        let ctx = WhisperContext::new_with_params(model_str, ctx_params).map_err(|e| {
            AsrError::WhisperContext {
                path: model_path.display().to_string(),
                reason: e.to_string(),
            }
        })?;

        #[cfg(feature = "vad")]
        let vad = match vad_path {
            Some(p) => Some(crate::vad::SileroVad::from_file(p)?),
            None => None,
        };

        Ok(Self {
            ctx,
            config,
            model_name: entry.name.clone(),
            #[cfg(feature = "vad")]
            vad,
        })
    }

    /// The loaded model's manifest name.
    #[must_use]
    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    /// The configuration this engine was built with.
    #[must_use]
    pub fn config(&self) -> &TranscribeConfig {
        &self.config
    }

    /// Transcribes a 16 kHz mono signal, discarding the timing breakdown.
    pub fn transcribe(
        &mut self,
        audio: &MonoAudio,
        progress: &mut TranscribeProgressFn<'_>,
        cancel: &AtomicBool,
    ) -> Result<Transcript, AsrError> {
        let mut discard = Timings::default();
        self.transcribe_with_timings(audio, progress, cancel, &mut discard)
    }

    /// Transcribes a 16 kHz mono signal and fills `timings`.
    pub fn transcribe_with_timings(
        &mut self,
        audio: &MonoAudio,
        progress: &mut TranscribeProgressFn<'_>,
        cancel: &AtomicBool,
        timings: &mut Timings,
    ) -> Result<Transcript, AsrError> {
        if audio.sample_rate != SAMPLE_RATE {
            return Err(AsrError::WhisperState {
                reason: format!(
                    "engine expects {SAMPLE_RATE} Hz audio, got {} Hz",
                    audio.sample_rate
                ),
            });
        }

        let started = Instant::now();
        let audio_duration_s = audio.duration_seconds();
        let backend = compiled_backend();
        let threads = self.config.resolved_threads();

        let sys = system_info();
        let params = params_summary(&self.config, threads);
        let build_flags = extract_build_flags(&sys);
        let device = if backend == "CUDA" { "0" } else { "-" };

        timings.audio_duration_s = audio_duration_s;
        timings.system_info = Some(sys.clone());
        timings.backend = backend.to_string();
        timings.device = device.to_string();
        timings.threads = threads;
        timings.params = params.clone();
        timings.build_flags = build_flags.clone();
        let optimized = build_optimized();
        timings.build_optimized = optimized.unwrap_or(false);
        timings.vad_pack = self.config.vad_pack;
        timings.parallel = self.config.parallel.max(1);

        eprintln!("--- svaraflow transcribe ---");
        eprintln!("backend={backend} device={device} threads={threads}");
        eprintln!("params: {params}");
        eprintln!("build_flags: {build_flags}");
        eprintln!(
            "build_optimized: {}",
            match optimized {
                Some(true) => "yes",
                Some(false) => "no",
                None => "unknown (CMakeCache not yet on disk)",
            }
        );
        eprintln!("{sys}");

        if optimized == Some(false) {
            eprintln!(
                "WARNING: whisper.cpp was built WITHOUT optimization. \
                 Expect 8-10x slower transcription. See docs/BUILD.md."
            );
        }

        // 1. Plan chunks: VAD-based pause cuts when a VAD is present,
        //    fixed windows otherwise.
        let plan_start = Instant::now();
        let ChunkPlan {
            mut chunks,
            regions: speech_regions,
            reasons: chunk_reasons,
        } = self.plan_chunks(audio, timings)?;
        timings.chunk_reasons = chunk_reasons;

        // 2. Optional --ranges clipping.
        if let Some(ranges) = &self.config.ranges {
            chunks = apply_ranges(chunks, ranges, audio.sample_rate);
            if chunks.is_empty() {
                return Err(AsrError::WhisperState {
                    reason: "--ranges left no chunks to transcribe".into(),
                });
            }
            // The reasons no longer correspond to the surviving chunks.
            timings.chunk_reasons.clear();
        }
        timings.chunk_plan_s = plan_start.elapsed().as_secs_f64();
        timings.chunks = chunks.len();

        // 3. Optional VAD packing.
        let (packed_audio, packed_map): (Option<MonoAudio>, Option<Vec<PackedMap>>) =
            if self.config.vad_pack && chunks.len() >= 2 {
                let (p, m) = pack_chunks(audio, &chunks);
                (Some(p), Some(m))
            } else {
                (None, None)
            };

        let working_audio: &MonoAudio = packed_audio.as_ref().unwrap_or(audio);
        let effective_chunks: Vec<Chunk> = if packed_audio.is_some() {
            vec![Chunk {
                start_sample: 0,
                end_sample: working_audio.samples.len(),
                start_s: 0.0,
            }]
        } else {
            chunks
        };

        // 4. Transcribe.
        let (all_segments, detected_language) = if self.config.parallel > 1
            && effective_chunks.len() > 1
        {
            self.transcribe_parallel(working_audio, &effective_chunks, progress, cancel, timings)?
        } else {
            self.transcribe_sequential(working_audio, &effective_chunks, progress, cancel, timings)?
        };

        let mut segments = all_segments;
        if let Some(map) = &packed_map {
            remap_segments(&mut segments, map);
        }

        // 5. Post-process.
        //
        // `snap_word` is deliberately not run here. Without DTW, word
        // timings are proportional within each segment; snapping each
        // word independently to a local energy minimum can move a run
        // of consecutive words to the same time, and `enforce_monotonic`
        // then stacks them at 1 ms apiece — which is how a nine-word
        // cue ends up with a 9 ms span. `snap_word` is retained in the
        // source for when the DTW alignment-heads preset is available,
        // at which point it will run against real word times.
        let post_start = Instant::now();
        dedup_overlaps(&mut segments, self.config.overlap_s);
        enforce_monotonic(&mut segments);
        if self.config.repetition_guard {
            drop_repeated_ngrams(&mut segments);
        }
        clamp_times(&mut segments, audio_duration_s);
        segments = merge_compressed_segments(segments, MIN_S_PER_WORD);

        // 6. Regroup into caption phrases.
        let segments = segment_into_phrases(
            segments,
            &speech_regions,
            self.config.max_phrase_s,
            self.config.max_phrase_words,
        );

        timings.postprocess_s = post_start.elapsed().as_secs_f64();

        timings.decode_retries = timings
            .windows
            .iter()
            .filter(|w| w.retry_reason.is_some())
            .count();

        let (total_words, estimated_words) = {
            let mut n = 0usize;
            let mut est = 0usize;
            for seg in &segments {
                for w in &seg.words {
                    n += 1;
                    if w.estimated {
                        est += 1;
                    }
                }
            }
            (n, est)
        };
        timings.dtw_usable = total_words > 0 && estimated_words == 0;

        let elapsed_s = started.elapsed().as_secs_f64();
        timings.transcribe_s = elapsed_s;
        let real_time_factor = if elapsed_s > 0.0 {
            audio_duration_s / elapsed_s
        } else {
            f64::INFINITY
        };

        Ok(Transcript {
            segments,
            language: detected_language,
            tier: self.config.tier,
            model: self.model_name.clone(),
            audio_duration_s,
            elapsed_s,
            real_time_factor,
        })
    }

    /// Sequential path: one state, one chunk at a time.
    fn transcribe_sequential(
        &self,
        audio: &MonoAudio,
        chunks: &[Chunk],
        progress: &mut TranscribeProgressFn<'_>,
        cancel: &AtomicBool,
        timings: &mut Timings,
    ) -> Result<(Vec<Segment>, String), AsrError> {
        let state_start = Instant::now();
        let mut state: WhisperState = self
            .ctx
            .create_state()
            .map_err(|e| AsrError::WhisperStateCreate(e.to_string()))?;
        timings.state_create_s = state_start.elapsed().as_secs_f64();

        let mut total_full = Duration::ZERO;
        let mut all_segments: Vec<Segment> = Vec::new();
        let total = chunks.len();

        for (i, chunk) in chunks.iter().enumerate() {
            if cancel.load(Ordering::Relaxed) {
                return Err(AsrError::Cancelled);
            }
            progress(i + 1, total);

            let slice = &audio.samples[chunk.start_sample..chunk.end_sample];
            let window = run_window(&mut state, &self.config, slice, chunk.start_s, i, chunk)?;
            total_full += Duration::from_secs_f64(window.whisper_full_s);
            all_segments.extend(window.segments);
            timings.windows.push(window.timing);
        }

        timings.whisper_full_s = total_full.as_secs_f64();
        Ok((all_segments, self.config.language.as_str().to_string()))
    }

    /// Parallel path: up to `parallel` states, each in its own scoped
    /// worker thread. States are created inside the worker because
    /// `WhisperState` is `!Send`.
    fn transcribe_parallel(
        &self,
        audio: &MonoAudio,
        chunks: &[Chunk],
        progress: &mut TranscribeProgressFn<'_>,
        cancel: &AtomicBool,
        timings: &mut Timings,
    ) -> Result<(Vec<Segment>, String), AsrError> {
        let n = self.config.parallel.max(1);
        let total = chunks.len();
        let mut all_segments: Vec<Segment> = Vec::new();
        let mut total_full = Duration::ZERO;
        let mut windows: Vec<(usize, WindowTiming)> = Vec::new();
        let mut state_create_total = 0.0f64;

        for batch_start in (0..chunks.len()).step_by(n) {
            if cancel.load(Ordering::Relaxed) {
                return Err(AsrError::Cancelled);
            }
            let batch_end = (batch_start + n).min(chunks.len());
            let batch: Vec<(usize, Chunk)> = chunks[batch_start..batch_end]
                .iter()
                .copied()
                .enumerate()
                .map(|(i, c)| (batch_start + i, c))
                .collect();

            let results: Vec<(Result<WindowResult, String>, f64)> = std::thread::scope(|s| {
                let handles: Vec<_> = batch
                    .iter()
                    .map(|(idx, chunk)| {
                        let ctx = &self.ctx;
                        let config = &self.config;
                        let samples = &audio.samples[chunk.start_sample..chunk.end_sample];
                        let i = *idx;
                        let c = *chunk;
                        s.spawn(move || -> (Result<WindowResult, String>, f64) {
                            let sc_start = Instant::now();
                            let mut state = match ctx.create_state() {
                                Ok(s) => s,
                                Err(e) => {
                                    return (
                                        Err(format!("create_state: {e}")),
                                        sc_start.elapsed().as_secs_f64(),
                                    );
                                }
                            };
                            let sc_s = sc_start.elapsed().as_secs_f64();
                            let r = run_window(&mut state, config, samples, c.start_s, i, &c)
                                .map_err(|e| e.to_string());
                            (r, sc_s)
                        })
                    })
                    .collect();
                handles
                    .into_iter()
                    .map(|h| {
                        h.join()
                            .unwrap_or_else(|_| (Err("worker thread panicked".to_string()), 0.0))
                    })
                    .collect()
            });

            for (i, (r, sc_s)) in results.into_iter().enumerate() {
                state_create_total += sc_s;
                let outcome = r.map_err(|reason| AsrError::WhisperState { reason })?;
                total_full += Duration::from_secs_f64(outcome.whisper_full_s);
                all_segments.extend(outcome.segments);
                windows.push((batch[i].0, outcome.timing));
                progress(batch[i].0 + 1, total);
            }
        }

        timings.state_create_s = state_create_total;
        windows.sort_by_key(|(i, _)| *i);
        timings.windows = windows.into_iter().map(|(_, w)| w).collect();
        timings.whisper_full_s = total_full.as_secs_f64();
        Ok((all_segments, self.config.language.as_str().to_string()))
    }

    /// Plans chunks and returns them with the VAD speech regions.
    fn plan_chunks(
        &mut self,
        audio: &MonoAudio,
        timings: &mut Timings,
    ) -> Result<ChunkPlan, AsrError> {
        #[cfg(feature = "vad")]
        {
            if let Some(vad) = self.vad.as_mut() {
                let vad_start = Instant::now();
                let plan = plan_chunks_with_vad(
                    audio,
                    vad,
                    self.config.max_chunk_s,
                    self.config.overlap_s,
                )?;
                timings.vad_s += vad_start.elapsed().as_secs_f64();
                return Ok(plan);
            }
        }
        #[cfg(not(feature = "vad"))]
        {
            let _ = audio;
        }
        Ok(plan_chunks_fixed(
            audio,
            self.config.max_chunk_s,
            self.config.overlap_s,
        ))
    }
}

/// Decides whether a window's output is junk, and re-decodes with token
/// timestamps off if so.
fn run_window(
    state: &mut WhisperState,
    config: &TranscribeConfig,
    samples: &[f32],
    offset_s: f64,
    index: usize,
    chunk: &Chunk,
) -> Result<WindowResult, AsrError> {
    let first = run_whisper_full(state, config, samples, offset_s, config.token_timestamps)?;
    let first_quality = assess(&first, config.junk_guard);

    if !first_quality.low_confidence {
        return Ok(WindowResult {
            segments: first.segments,
            whisper_full_s: first.full_s,
            timing: WindowTiming {
                index,
                start_s: chunk.start_s,
                end_s: chunk.start_s
                    + (chunk.end_sample - chunk.start_sample) as f64 / f64::from(SAMPLE_RATE),
                whisper_full_s: first.full_s,
                tokens: first_quality.tokens,
                segments: first_quality.segments,
                mean_probability: first_quality.mean_probability,
                min_probability: first_quality.min_probability,
                junk_fraction: first_quality.junk_fraction,
                repeat_ratio: first_quality.repeat_ratio,
                no_speech_prob: first_quality.no_speech_prob,
                low_confidence: false,
                retry_reason: None,
                language_requested: config.language.as_str().to_string(),
            },
        });
    }

    // Fallback pass: token timestamps off.
    let second = run_whisper_full(state, config, samples, offset_s, false)?;
    let second_quality = assess(&second, config.junk_guard);

    let use_second = second_quality.mean_probability > first_quality.mean_probability
        && second_quality.junk_fraction <= first_quality.junk_fraction;

    let reason = format!(
        "first pass: junk={:.2} mean_prob={:.2} no_speech={:.2}",
        first_quality.junk_fraction, first_quality.mean_probability, first_quality.no_speech_prob
    );

    let (chosen, chosen_quality) = if use_second {
        (second, second_quality)
    } else {
        (first, first_quality)
    };

    let total_s = chosen.full_s;

    Ok(WindowResult {
        segments: chosen.segments,
        whisper_full_s: total_s,
        timing: WindowTiming {
            index,
            start_s: chunk.start_s,
            end_s: chunk.start_s
                + (chunk.end_sample - chunk.start_sample) as f64 / f64::from(SAMPLE_RATE),
            whisper_full_s: total_s,
            tokens: chosen_quality.tokens,
            segments: chosen_quality.segments,
            mean_probability: chosen_quality.mean_probability,
            min_probability: chosen_quality.min_probability,
            junk_fraction: chosen_quality.junk_fraction,
            repeat_ratio: chosen_quality.repeat_ratio,
            no_speech_prob: chosen_quality.no_speech_prob,
            low_confidence: true,
            retry_reason: Some(reason),
            language_requested: config.language.as_str().to_string(),
        },
    })
}

/// The raw result of one `whisper_full` call before quality scoring.
struct RawWindow {
    segments: Vec<Segment>,
    full_s: f64,
    tokens: Vec<RawToken>,
    mean_probability: f32,
    min_probability: f32,
    no_speech_prob: f32,
    text: String,
}

/// The outcome of one window after the junk guard has run.
struct WindowResult {
    segments: Vec<Segment>,
    whisper_full_s: f64,
    timing: WindowTiming,
}

/// Junk-guard verdict for one raw window.
struct QualityVerdict {
    tokens: usize,
    segments: usize,
    mean_probability: f32,
    min_probability: f32,
    junk_fraction: f32,
    repeat_ratio: f32,
    no_speech_prob: f32,
    low_confidence: bool,
}

fn assess(raw: &RawWindow, guard: JunkGuard) -> QualityVerdict {
    let junk = junk_fraction(&raw.text);
    let repeat = repeat_ratio(&raw.text);
    let low_confidence = junk > guard.max_junk_fraction
        || repeat > guard.max_repeat_ratio
        || raw.mean_probability < guard.min_mean_probability
        || raw.no_speech_prob > guard.max_no_speech_prob;

    QualityVerdict {
        tokens: raw.tokens.len(),
        segments: raw.segments.len(),
        mean_probability: raw.mean_probability,
        min_probability: raw.min_probability,
        junk_fraction: junk,
        repeat_ratio: repeat,
        no_speech_prob: raw.no_speech_prob,
        low_confidence,
    }
}

/// Runs one `whisper.cpp` call over `samples`.
fn run_whisper_full(
    state: &mut WhisperState,
    config: &TranscribeConfig,
    samples: &[f32],
    offset_s: f64,
    token_timestamps: bool,
) -> Result<RawWindow, AsrError> {
    let strategy = match config.decode {
        DecodeStrategy::Greedy => SamplingStrategy::Greedy { best_of: 1 },
        DecodeStrategy::BeamSearch { beam_size } => SamplingStrategy::BeamSearch {
            beam_size,
            patience: -1.0,
        },
    };
    let mut params = FullParams::new(strategy);

    params.set_n_threads(config.resolved_threads());
    if let Some(ctx) = config.audio_ctx {
        params.set_audio_ctx(ctx);
    }
    params.set_language(config.language.as_whisper_str());
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);
    params.set_no_speech_thold(config.no_speech_threshold);
    params.set_token_timestamps(token_timestamps);

    if config.temperature_fallback {
        params.set_temperature_inc(0.2);
    } else {
        params.set_temperature_inc(0.0);
    }
    if let Some(prompt) = config.effective_prompt() {
        params.set_initial_prompt(&prompt);
    }

    let full_start = Instant::now();
    let full_result = state.full(params, samples);
    let full_s = full_start.elapsed().as_secs_f64();
    full_result.map_err(|e| AsrError::WhisperState {
        reason: e.to_string(),
    })?;

    let mut segments: Vec<Segment> = Vec::new();
    let mut all_tokens: Vec<RawToken> = Vec::new();
    let mut text_buf = String::new();
    let mut min_prob = f32::INFINITY;
    let mut prob_sum = 0.0f32;
    let mut prob_count = 0usize;
    let mut max_no_speech = 0.0f32;

    // whisper.cpp pads every input to a 30 s encoder window. Its segment
    // timestamps reflect that padded window, not the audio we handed it.
    // A segment inside a 22 s chunk will report `end_timestamp` near 30 s.
    // Left unchecked, words are distributed across 30 s and drift by
    // `30 - chunk_duration` at the end of each chunk. Clamp every segment
    // to the chunk's real span.
    let chunk_duration_s = samples.len() as f64 / f64::from(SAMPLE_RATE);
    let chunk_end_s = offset_s + chunk_duration_s;

    for segment in state.as_iter() {
        let seg_text_raw = segment.to_str_lossy().unwrap_or_default().to_string();
        if seg_text_raw.trim().is_empty() {
            continue;
        }
        let seg_start =
            (segment.start_timestamp() as f64 / 100.0 + offset_s).clamp(offset_s, chunk_end_s);
        let seg_end =
            (segment.end_timestamp() as f64 / 100.0 + offset_s).clamp(seg_start, chunk_end_s);
        if seg_end - seg_start < 0.005 {
            // A zero-length segment cannot carry timing. Skip it rather
            // than let merge_tokens_into_words produce a compressed run.
            continue;
        }
        let no_speech = segment.no_speech_probability();
        if no_speech > max_no_speech {
            max_no_speech = no_speech;
        }

        let n_tokens = segment.n_tokens();
        let mut raw_tokens: Vec<RawToken> = Vec::with_capacity(n_tokens.max(0) as usize);
        for ti in 0..n_tokens {
            let Some(token) = segment.get_token(ti) else {
                continue;
            };
            let Ok(token_text) = token.to_str() else {
                continue;
            };
            if is_special_token(token_text) {
                continue;
            }
            let data = token.token_data();
            prob_sum += data.p;
            prob_count += 1;
            if data.p < min_prob {
                min_prob = data.p;
            }
            raw_tokens.push(RawToken {
                text: token_text.to_string(),
                t_dtw_cs: data.t_dtw,
                probability: data.p,
            });
        }

        let dtw_usable = !raw_tokens.is_empty() && raw_tokens.iter().all(|t| t.t_dtw_cs >= 0);
        let words = merge_tokens_into_words(&raw_tokens, dtw_usable, seg_start, seg_end);

        let rebuilt_text: String = words
            .iter()
            .map(|w| w.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");

        text_buf.push_str(&rebuilt_text);
        text_buf.push(' ');
        let token_count = raw_tokens.len();
        all_tokens.extend(raw_tokens);

        eprintln!(
            "seg: offset={offset_s:.2}s start={seg_start:.2}s end={seg_end:.2}s \
             tokens={token_count}",
        );
        segments.push(Segment {
            text: rebuilt_text,
            start_s: seg_start,
            end_s: seg_end,
            words,
            low_confidence: false,
        });
    }

    let mean_probability = if prob_count > 0 {
        prob_sum / prob_count as f32
    } else {
        0.0
    };
    let min_probability = if min_prob.is_finite() { min_prob } else { 0.0 };

    Ok(RawWindow {
        segments,
        full_s,
        tokens: all_tokens,
        mean_probability,
        min_probability,
        no_speech_prob: max_no_speech,
        text: text_buf,
    })
}

/// A chunk of the input signal, in source or packed coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Chunk {
    /// Inclusive first sample.
    pub start_sample: usize,
    /// Exclusive last sample.
    pub end_sample: usize,
    /// Start time in seconds, in the same coordinate system as `samples`.
    pub start_s: f64,
}

/// One token's raw timings from whisper.cpp.
#[derive(Debug, Clone)]
struct RawToken {
    text: String,
    /// DTW timestamp in centiseconds, or `-1`.
    t_dtw_cs: i64,
    /// Probability.
    probability: f32,
}

// ---------------------------------------------------------------------------
// Special-token and quality helpers
// ---------------------------------------------------------------------------

/// `true` when a token's text is a whisper.cpp control token.
///
/// whisper.cpp emits two shapes:
///
/// - `[_BEG_]`, `[_EOT_]`, `[_LANG_hi_]` — bracket, underscore, name,
///   optional trailing underscore, closing bracket.
/// - `[_TT_1380]` — bracket, underscore, `TT_`, digits, closing bracket,
///   **no trailing underscore**.
///
/// The common closing character is `]`, so that is what we check. The
/// leading `[_` keeps ordinary bracket content like `[music]` safe.
#[must_use]
pub fn is_special_token(text: &str) -> bool {
    let t = text.trim();
    if t.is_empty() {
        return false;
    }
    if t.starts_with("[_") && t.ends_with(']') {
        return true;
    }
    if t.starts_with("<|") && t.ends_with("|>") {
        return true;
    }
    false
}

/// Removes any `[_…]` or `<|…|>` substring from `s`.
///
/// This handles the case where whisper.cpp emitted a control token as
/// its own token and the leading-space heuristic then appended it to the
/// preceding word. The whole-token check in [`is_special_token`] cannot
/// catch that shape; the substring scan can.
#[must_use]
pub fn strip_special_substrings(s: &str) -> String {
    let mut out = s.to_string();
    loop {
        let before = out.len();
        if let Some(i) = out.find("[_") {
            if let Some(rel) = out[i + 2..].find(']') {
                out.replace_range(i..i + 2 + rel + 1, "");
            }
        }
        if let Some(i) = out.find("<|") {
            if let Some(rel) = out[i + 2..].find("|>") {
                out.replace_range(i..i + 2 + rel + 2, "");
            }
        }
        if out.len() == before {
            break;
        }
    }
    out.trim().to_string()
}

/// Fraction of non-whitespace characters outside Basic Latin, Devanagari
/// and ASCII punctuation.
#[must_use]
pub fn junk_fraction(text: &str) -> f32 {
    let mut total = 0usize;
    let mut junk = 0usize;
    for c in text.chars() {
        if c.is_whitespace() {
            continue;
        }
        total += 1;
        let allowed = c.is_ascii() || ('\u{0900}'..='\u{097F}').contains(&c);
        if !allowed {
            junk += 1;
        }
    }
    if total == 0 {
        0.0
    } else {
        junk as f32 / total as f32
    }
}

/// Longest run of identical adjacent characters divided by length.
#[must_use]
pub fn repeat_ratio(text: &str) -> f32 {
    let chars: Vec<char> = text.chars().filter(|c| !c.is_whitespace()).collect();
    if chars.len() < 4 {
        return 0.0;
    }
    let mut max_run = 1usize;
    let mut run = 1usize;
    for i in 1..chars.len() {
        if chars[i] == chars[i - 1] {
            run += 1;
            if run > max_run {
                max_run = run;
            }
        } else {
            run = 1;
        }
    }
    max_run as f32 / chars.len() as f32
}

/// True when the text ends with a sentence-ending punctuation mark.
fn ends_with_sentence_punct(s: &str) -> bool {
    matches!(
        s.chars().last(),
        Some('.') | Some('?') | Some('!') | Some(';')
    )
}

// ---------------------------------------------------------------------------
// Token merging and post-processing
// ---------------------------------------------------------------------------

fn merge_tokens_into_words(
    tokens: &[RawToken],
    dtw_usable: bool,
    seg_start_s: f64,
    seg_end_s: f64,
) -> Vec<Word> {
    let mut words: Vec<Word> = Vec::new();
    let mut current: Option<Word> = None;
    let filtered: Vec<&RawToken> = tokens
        .iter()
        .filter(|t| !is_special_token(&t.text))
        .collect();
    let total_tokens = filtered.len().max(1);

    for (i, tok) in filtered.iter().enumerate() {
        let raw = tok.text.as_str();
        let starts_word = raw.starts_with(' ')
            || raw.starts_with('\u{0120}')
            || (words.is_empty() && current.is_none());
        let piece = strip_special_substrings(raw.trim_start_matches([' ', '\u{0120}']));
        if piece.is_empty() {
            continue;
        }

        let (start_s, end_s) = if dtw_usable {
            (tok.t_dtw_cs as f64 / 100.0, tok.t_dtw_cs as f64 / 100.0)
        } else {
            let frac_start = i as f64 / total_tokens as f64;
            let frac_end = (i + 1) as f64 / total_tokens as f64;
            (
                seg_start_s + frac_start * (seg_end_s - seg_start_s),
                seg_start_s + frac_end * (seg_end_s - seg_start_s),
            )
        };

        if starts_word {
            if let Some(w) = current.take() {
                words.push(w);
            }
            current = Some(Word {
                text: piece,
                start_s,
                end_s,
                probability: tok.probability,
                estimated: !dtw_usable,
            });
        } else if let Some(w) = current.as_mut() {
            w.text.push_str(&piece);
            w.end_s = end_s;
            w.probability = w.probability.min(tok.probability);
        } else {
            current = Some(Word {
                text: piece,
                start_s,
                end_s,
                probability: tok.probability,
                estimated: !dtw_usable,
            });
        }
    }
    if let Some(w) = current.take() {
        words.push(w);
    }

    if dtw_usable {
        for i in 0..words.len() {
            if i + 1 < words.len() {
                words[i].end_s = words[i + 1].start_s;
            } else {
                words[i].end_s = words[i].start_s.max(seg_end_s.min(words[i].start_s + 0.5));
            }
        }
    }
    words
}

/// Clamps every segment and word time into `[0, audio_duration_s]` and
/// enforces `end >= start`.
fn clamp_times(segments: &mut [Segment], audio_duration_s: f64) {
    for seg in segments.iter_mut() {
        seg.start_s = seg.start_s.clamp(0.0, audio_duration_s);
        seg.end_s = seg.end_s.clamp(seg.start_s, audio_duration_s);
        for w in seg.words.iter_mut() {
            w.start_s = w.start_s.clamp(0.0, audio_duration_s);
            w.end_s = w.end_s.clamp(w.start_s, audio_duration_s);
        }
    }
}

/// Merges segments whose span is too short for their word count.
///
/// whisper.cpp occasionally returns a segment whose `t0` and `t1` are
/// only a centisecond or two apart while containing a whole sentence's
/// tokens. Every word in the segment then inherits a fraction of a
/// millisecond, and the resulting caption cue is invisible on playback.
///
/// A segment with `n` words is compressed when its span is under
/// `n * min_s_per_word`. It absorbs following segments until the merged
/// span is plausible; the combined words are then redistributed evenly
/// across it. Absorbing rather than stretching is required: the next
/// segment starts within milliseconds of the compressed one, so there
/// is no room to stretch into.
///
/// `enforce_monotonic` has already run, so absorbing preserves order.
fn merge_compressed_segments(segments: Vec<Segment>, min_s_per_word: f64) -> Vec<Segment> {
    eprintln!("merge: called with {} segments", segments.len());
    for (i, s) in segments.iter().enumerate() {
        eprintln!(
            "merge in[{i}]: span={span:.4}s words={w} min_span={ms:.4}s text={txt:?}",
            span = s.end_s - s.start_s,
            w = s.words.len(),
            ms = s.words.len() as f64 * min_s_per_word,
            txt = s.text.chars().take(40).collect::<String>(),
        );
    }
    let mut out: Vec<Segment> = Vec::with_capacity(segments.len());
    let mut iter = segments.into_iter().peekable();
    while let Some(mut seg) = iter.next() {
        if seg.words.is_empty() {
            out.push(seg);
            continue;
        }
        let min_span = seg.words.len() as f64 * min_s_per_word;
        if seg.end_s - seg.start_s >= min_span {
            out.push(seg);
            continue;
        }
        let original_start = seg.start_s;
        let original_span = seg.end_s - seg.start_s;
        while let Some(next) = iter.peek() {
            seg.words.extend(next.words.iter().cloned());
            seg.end_s = next.end_s;
            if seg.text.is_empty() {
                seg.text = next.text.clone();
            } else if !next.text.is_empty() {
                seg.text = format!("{} {}", seg.text, next.text);
            }
            seg.low_confidence = seg.low_confidence || next.low_confidence;
            iter.next();
            if seg.end_s - seg.start_s >= seg.words.len() as f64 * min_s_per_word {
                break;
            }
        }
        let span = seg.end_s - seg.start_s;
        let wc = seg.words.len() as f64;
        for (j, w) in seg.words.iter_mut().enumerate() {
            w.start_s = original_start + (j as f64 / wc) * span;
            w.end_s = original_start + ((j + 1) as f64 / wc) * span;
        }
        eprintln!(
            "warn: merged compressed segment at {original_start:.3}s: \
             {original_span:.3}s span merged into {span:.3}s, {} words",
            seg.words.len(),
        );
        out.push(seg);
    }
    out
}

// ---------------------------------------------------------------------------
// Phrase segmentation
// ---------------------------------------------------------------------------

/// Regroups whisper's raw segments into caption-ready phrases.
///
/// Every phrase is at most `max_duration_s` long and at most `max_words`
/// words. A phrase is broken before a word when the previous word ends
/// with sentence punctuation, when the span so far exceeds
/// `max_duration_s`, when the phrase would exceed `max_words`, or when
/// the phrase's span crosses a VAD silence of at least
/// [`MIN_GAP_FOR_PHRASE_S`].
///
/// Word-level timings are preserved from the input. Inside a phrase they
/// remain whatever whisper.cpp produced (usually proportional fallback,
/// flagged `estimated`). Only the phrase's own start and end are set from
/// its words.
///
/// `low_confidence` propagates: a phrase is low-confidence if any word
/// in it came from a low-confidence window.
fn segment_into_phrases(
    segments: Vec<Segment>,
    speech_regions: &SpeechRegions,
    max_duration_s: f64,
    max_words: usize,
) -> Vec<Segment> {
    let words_with_flag: Vec<(Word, bool)> = segments
        .into_iter()
        .flat_map(|s| {
            let flag = s.low_confidence;
            s.words.into_iter().map(move |w| (w, flag))
        })
        .collect();
    if words_with_flag.is_empty() {
        return Vec::new();
    }

    let mut out: Vec<Segment> = Vec::new();
    let mut current: Vec<(Word, bool)> = Vec::new();

    for (w, flag) in words_with_flag {
        let should_break = if current.is_empty() {
            false
        } else {
            let start_s = current.first().map(|(w, _)| w.start_s).unwrap_or(w.start_s);
            let span = w.end_s - start_s;
            let words_after = current.len() + 1;
            let prev_ends_sentence = current
                .last()
                .map(|(pw, _)| ends_with_sentence_punct(&pw.text))
                .unwrap_or(false);
            let crosses_silence =
                speech_regions.crosses_silence(start_s, w.end_s, MIN_GAP_FOR_PHRASE_S);
            span > max_duration_s
                || words_after > max_words
                || prev_ends_sentence
                || crosses_silence
        };

        if should_break {
            out.push(build_phrase(std::mem::take(&mut current)));
        }
        current.push((w, flag));
    }
    if !current.is_empty() {
        out.push(build_phrase(current));
    }
    out
}

/// Builds one phrase from a run of words.
fn build_phrase(words_with_flag: Vec<(Word, bool)>) -> Segment {
    let start_s = words_with_flag
        .first()
        .map(|(w, _)| w.start_s)
        .unwrap_or(0.0);
    let end_s = words_with_flag
        .last()
        .map(|(w, _)| w.end_s)
        .unwrap_or(start_s);
    let low_confidence = words_with_flag.iter().any(|(_, flag)| *flag);
    let words: Vec<Word> = words_with_flag.into_iter().map(|(w, _)| w).collect();
    let text = words
        .iter()
        .map(|w| w.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    Segment {
        text,
        start_s,
        end_s,
        words,
        low_confidence,
    }
}

/// Plans chunks using Silero VAD to cut at pauses.
///
/// Algorithm:
///
/// 1. Compute speech probabilities and extract speech regions separated
///    by at least [`VAD_MIN_SILENCE_S`] of silence.
/// 2. Pad each region by [`CHUNK_PAD_S`] on each side.
/// 3. Greedily pack adjacent regions into one chunk until the next region
///    would push the chunk past `max_chunk_s`. The chunk boundary falls at
///    the midpoint of the gap between the last included region and the
///    next one, so it is centred in the pause.
/// 4. A single region longer than `max_chunk_s` is split internally at the
///    longest pause in `[cursor + max_chunk_s / 2, cursor + max_chunk_s]`,
///    or forcibly at `max_chunk_s` with `overlap_s` padding when no pause
///    is available.
///
/// The result is a small number of chunks, each at most `max_chunk_s`
/// long, cut at pauses. Pause-aligned *phrases* are a separate granularity:
/// [`segment_into_phrases`] uses the speech regions directly, so caption
/// cues still break at every detected pause even when a chunk contains
/// several of them.
fn plan_chunks_with_vad(
    audio: &MonoAudio,
    vad: &mut crate::vad::SileroVad,
    max_chunk_s: f64,
    overlap_s: f64,
) -> Result<ChunkPlan, AsrError> {
    use svaraflow_core::vad::Vad;

    let probs = vad.speech_probabilities(&audio.samples)?;
    let spw = probs.seconds_per_window();
    eprintln!(
        "vad: windows={} spw={:.4}s mean={:.3} max={:.3} min={:.3}",
        probs.values.len(),
        spw,
        probs.values.iter().sum::<f32>() / probs.values.len().max(1) as f32,
        probs
            .values
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max),
        probs.values.iter().cloned().fold(f32::INFINITY, f32::min),
    );
    if spw <= 0.0 || probs.values.is_empty() {
        return Ok(plan_chunks_fixed(audio, max_chunk_s, overlap_s));
    }

    // Speech regions: runs of speech separated by >= VAD_MIN_SILENCE_S.
    let min_silence_windows = ((VAD_MIN_SILENCE_S / spw).round() as usize).max(1);
    let mut regions: Vec<(f64, f64)> = Vec::new();
    let mut run_start: Option<usize> = None;
    let mut silence_run = 0usize;
    for (i, p) in probs.values.iter().enumerate() {
        let speech = *p >= VAD_THRESHOLD;
        if speech {
            silence_run = 0;
            if run_start.is_none() {
                run_start = Some(i);
            }
        } else if let Some(s) = run_start {
            silence_run += 1;
            if silence_run >= min_silence_windows {
                regions.push((s as f64 * spw, (i + 1 - silence_run) as f64 * spw));
                run_start = None;
                silence_run = 0;
            }
        }
    }
    if let Some(s) = run_start {
        regions.push((s as f64 * spw, probs.values.len() as f64 * spw));
    }

    eprintln!("vad: {} speech regions extracted", regions.len());
    if regions.is_empty() {
        return Ok(plan_chunks_fixed(audio, max_chunk_s, overlap_s));
    }

    let speech_regions = SpeechRegions {
        regions: regions.clone(),
    };

    let duration = audio.duration_seconds();
    let rate = audio.sample_rate as f64;

    // Padded regions, used only for chunk packing.
    let padded: Vec<(f64, f64)> = regions
        .iter()
        .map(|(s, e)| ((s - CHUNK_PAD_S).max(0.0), (e + CHUNK_PAD_S).min(duration)))
        .collect();

    let mut chunks: Vec<Chunk> = Vec::new();
    let mut reasons: Vec<ChunkReason> = Vec::new();
    let mut is_first = true;

    let mut idx = 0usize;
    while idx < padded.len() {
        let chunk_start = padded[idx].0;

        // Extend the chunk as far as we can without exceeding the limit.
        let mut last = idx;
        while last + 1 < padded.len() && padded[last + 1].1 - chunk_start <= max_chunk_s {
            last += 1;
        }

        // A single region that is itself longer than max_chunk_s gets
        // split at pauses, or forcibly when no pause is available.
        if last == idx && padded[idx].1 - chunk_start > max_chunk_s {
            let region_end = padded[idx].1;
            let mut cursor = chunk_start;
            loop {
                let remaining = region_end - cursor;
                if remaining <= max_chunk_s {
                    chunks.push(Chunk {
                        start_sample: (cursor * rate).round() as usize,
                        end_sample: (region_end * rate).round() as usize,
                        start_s: cursor,
                    });
                    reasons.push(if is_first {
                        ChunkReason::StartOfAudio
                    } else {
                        ChunkReason::Pause
                    });
                    is_first = false;
                    break;
                }
                let split = find_longest_pause(
                    &probs.values,
                    spw,
                    cursor + max_chunk_s / 2.0,
                    cursor + max_chunk_s,
                );
                let (cut_s, reason) = match split {
                    Some(mid) => (
                        mid,
                        if is_first {
                            ChunkReason::StartOfAudio
                        } else {
                            ChunkReason::Pause
                        },
                    ),
                    None => (
                        cursor + max_chunk_s,
                        if is_first {
                            ChunkReason::StartOfAudio
                        } else {
                            ChunkReason::ForcedSize
                        },
                    ),
                };
                chunks.push(Chunk {
                    start_sample: (cursor * rate).round() as usize,
                    end_sample: (cut_s * rate).round() as usize,
                    start_s: cursor,
                });
                reasons.push(reason);
                cursor = cut_s;
                is_first = false;
            }
            idx += 1;
            continue;
        }

        // Otherwise, one chunk spanning regions idx..=last. Cut at the
        // midpoint of the gap before the next region when there is one.
        let chunk_end = if last + 1 < padded.len() {
            let gap_start = regions[last].1;
            let gap_end = regions[last + 1].0;
            (gap_start + gap_end) / 2.0
        } else {
            padded[last].1
        };

        chunks.push(Chunk {
            start_sample: (chunk_start * rate).round() as usize,
            end_sample: (chunk_end * rate).round() as usize,
            start_s: chunk_start,
        });
        reasons.push(if is_first {
            ChunkReason::StartOfAudio
        } else {
            ChunkReason::Pause
        });
        is_first = false;

        idx = last + 1;
    }

    Ok(ChunkPlan {
        chunks,
        regions: speech_regions,
        reasons,
    })
}

/// Finds the midpoint of the longest silence in
/// `[window_start, window_end]`. Returns `None` when no silence of at
/// least [`MIN_PAUSE_FOR_SPLIT_S`] is present.
fn find_longest_pause(probs: &[f32], spw: f64, window_start: f64, window_end: f64) -> Option<f64> {
    if spw <= 0.0 || probs.is_empty() {
        return None;
    }
    let i0 = ((window_start / spw).ceil() as usize).min(probs.len());
    let i1 = ((window_end / spw).floor() as usize).min(probs.len());
    if i1 <= i0 {
        return None;
    }
    let mut best_run = 0usize;
    let mut best_mid: Option<f64> = None;
    let mut run = 0usize;
    let mut run_start = i0;
    for (i, p) in probs.iter().enumerate().take(i1).skip(i0) {
        if *p < VAD_THRESHOLD {
            if run == 0 {
                run_start = i;
            }
            run += 1;
            if run > best_run {
                best_run = run;
                best_mid = Some((run_start + run / 2) as f64 * spw);
            }
        } else {
            run = 0;
        }
    }
    if (best_run as f64) * spw < MIN_PAUSE_FOR_SPLIT_S {
        None
    } else {
        best_mid
    }
}

/// Fixed-window chunking with no VAD.
fn plan_chunks_fixed(audio: &MonoAudio, max_chunk_s: f64, overlap_s: f64) -> ChunkPlan {
    let samples_per_s = audio.sample_rate as f64;
    let max_samples = (max_chunk_s * samples_per_s) as usize;
    let overlap_samples = (overlap_s * samples_per_s) as usize;
    let step = max_samples.saturating_sub(2 * overlap_samples).max(1);

    let mut chunks = Vec::new();
    let mut start = 0usize;
    while start < audio.samples.len() {
        let end = (start + max_samples).min(audio.samples.len());
        chunks.push(Chunk {
            start_sample: start,
            end_sample: end,
            start_s: start as f64 / samples_per_s,
        });
        if end == audio.samples.len() {
            break;
        }
        start += step;
    }
    if chunks.is_empty() {
        chunks.push(Chunk {
            start_sample: 0,
            end_sample: audio.samples.len(),
            start_s: 0.0,
        });
    }

    let reasons: Vec<ChunkReason> = chunks
        .iter()
        .enumerate()
        .map(|(i, _)| {
            if i == 0 {
                ChunkReason::StartOfAudio
            } else {
                ChunkReason::FixedWindow
            }
        })
        .collect();

    ChunkPlan {
        chunks,
        regions: SpeechRegions::default(),
        reasons,
    }
}

/// Removes words that appear in the overlap between two consecutive chunks.
fn dedup_overlaps(segments: &mut Vec<Segment>, overlap_s: f64) {
    if segments.len() < 2 {
        return;
    }
    let mut i = 1;
    while i < segments.len() {
        let boundary = segments[i - 1].end_s;
        let mut to_remove: Vec<usize> = Vec::new();
        for (j, w) in segments[i].words.iter().enumerate() {
            if w.start_s > boundary - overlap_s
                && segments[i - 1]
                    .words
                    .iter()
                    .any(|prev| prev.text == w.text && (prev.start_s - w.start_s).abs() < 0.5)
            {
                to_remove.push(j);
            }
        }
        for j in to_remove.into_iter().rev() {
            segments[i].words.remove(j);
        }
        if segments[i].words.is_empty() {
            segments.remove(i);
        } else {
            i += 1;
        }
    }
}

/// Enforces strictly increasing, non-overlapping word times.
fn enforce_monotonic(segments: &mut [Segment]) {
    let mut last_end = 0.0f64;
    for segment in segments.iter_mut() {
        for word in segment.words.iter_mut() {
            if word.start_s < last_end {
                word.start_s = last_end;
            }
            if word.end_s <= word.start_s {
                word.end_s = word.start_s + 0.001;
            }
            last_end = word.end_s;
        }
        if let (Some(first), Some(last)) = (segment.words.first(), segment.words.last()) {
            segment.start_s = first.start_s;
            segment.end_s = last.end_s;
        }
    }
}

/// Snaps a word's start and end to the lowest-energy sample within
/// [`SNAP_RADIUS_S`].
#[allow(dead_code)]
fn snap_word(word: &mut Word, samples: &[f32], rate: u32) {
    let radius = (SNAP_RADIUS_S * f64::from(rate)) as usize;
    word.start_s = snap_to_min_energy(word.start_s, samples, rate, radius);
    word.end_s = snap_to_min_energy(word.end_s, samples, rate, radius).max(word.start_s + 0.001);
}

/// Finds the lowest-RMS position within `radius` samples of `t`.
#[allow(dead_code)]
fn snap_to_min_energy(t: f64, samples: &[f32], rate: u32, radius: usize) -> f64 {
    let center = (t * f64::from(rate)) as isize;
    let lo = (center - radius as isize).max(0) as usize;
    let hi = ((center + radius as isize) as usize).min(samples.len().saturating_sub(1));
    if hi <= lo {
        return t;
    }
    let mut best = center as usize;
    let mut best_energy = f32::INFINITY;
    let win = 80usize;
    let mut i = lo;
    while i + win <= hi {
        let rms: f32 = samples[i..i + win].iter().map(|s| s * s).sum();
        if rms < best_energy {
            best_energy = rms;
            best = i + win / 2;
        }
        i += win / 2;
    }
    best as f64 / f64::from(rate)
}

/// Drops consecutive repeated n-grams.
fn drop_repeated_ngrams(segments: &mut [Segment]) {
    for segment in segments.iter_mut() {
        for n in 2..=5usize {
            if segment.words.len() < n * 3 {
                continue;
            }
            let mut i = 0;
            while i + n * 3 <= segment.words.len() {
                let first: Vec<&str> = segment.words[i..i + n]
                    .iter()
                    .map(|w| w.text.as_str())
                    .collect();
                let mut repeat_count = 1;
                let mut j = i + n;
                while j + n <= segment.words.len() {
                    let next: Vec<&str> = segment.words[j..j + n]
                        .iter()
                        .map(|w| w.text.as_str())
                        .collect();
                    if next == first {
                        repeat_count += 1;
                        j += n;
                    } else {
                        break;
                    }
                }
                if repeat_count >= 3 {
                    let remove_from = i + n;
                    let remove_to = j;
                    segment.words.drain(remove_from..remove_to);
                } else {
                    i += 1;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// VAD packing
// ---------------------------------------------------------------------------

/// Where one packed chunk came from in the source signal.
#[derive(Debug, Clone, Copy)]
struct PackedMap {
    packed_start_s: f64,
    packed_end_s: f64,
    source_start_s: f64,
}

/// Concatenates the samples of `chunks` into one buffer and returns the
/// buffer plus a map from packed time back to source time.
fn pack_chunks(audio: &MonoAudio, chunks: &[Chunk]) -> (MonoAudio, Vec<PackedMap>) {
    let rate = f64::from(audio.sample_rate);
    let mut samples: Vec<f32> = Vec::with_capacity(audio.samples.len());
    let mut map: Vec<PackedMap> = Vec::with_capacity(chunks.len());
    for c in chunks {
        let start = samples.len();
        samples.extend_from_slice(&audio.samples[c.start_sample..c.end_sample]);
        let end = samples.len();
        map.push(PackedMap {
            packed_start_s: start as f64 / rate,
            packed_end_s: end as f64 / rate,
            source_start_s: c.start_s,
        });
    }
    (MonoAudio::new(samples, audio.sample_rate), map)
}

/// Maps a time in packed space to a time in the source signal.
fn packed_to_source(t: f64, map: &[PackedMap]) -> f64 {
    if map.is_empty() {
        return t;
    }
    for m in map {
        if t >= m.packed_start_s && t < m.packed_end_s {
            return m.source_start_s + (t - m.packed_start_s);
        }
    }
    if let Some(last) = map.last() {
        return last.source_start_s + (last.packed_end_s - last.packed_start_s);
    }
    t
}

/// Rewrites every segment and word time from packed space to source space.
fn remap_segments(segments: &mut [Segment], map: &[PackedMap]) {
    for seg in segments.iter_mut() {
        seg.start_s = packed_to_source(seg.start_s, map);
        seg.end_s = packed_to_source(seg.end_s, map);
        for w in seg.words.iter_mut() {
            w.start_s = packed_to_source(w.start_s, map);
            w.end_s = packed_to_source(w.end_s, map);
        }
    }
}

// ---------------------------------------------------------------------------
// Ranges
// ---------------------------------------------------------------------------

/// Restricts `chunks` to those intersecting a set of source time ranges.
fn apply_ranges(chunks: Vec<Chunk>, ranges: &[(f64, f64)], rate: u32) -> Vec<Chunk> {
    let rate_f = f64::from(rate);
    let sample_ranges: Vec<(usize, usize)> = ranges
        .iter()
        .map(|(s, e)| {
            let s = (s * rate_f).round().max(0.0) as usize;
            let e = (e * rate_f).round().max(0.0) as usize;
            (s, e)
        })
        .collect();

    let mut out: Vec<Chunk> = Vec::new();
    for chunk in chunks {
        for (rs, re) in &sample_ranges {
            let s = chunk.start_sample.max(*rs);
            let e = chunk.end_sample.min(*re);
            if s < e {
                out.push(Chunk {
                    start_sample: s,
                    end_sample: e,
                    start_s: chunk.start_s + (s - chunk.start_sample) as f64 / rate_f,
                });
            }
        }
    }
    out.sort_by_key(|c| (c.start_sample, c.end_sample));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn word(text: &str, start: f64, end: f64) -> Word {
        Word {
            text: text.to_string(),
            start_s: start,
            end_s: end,
            probability: 0.9,
            estimated: false,
        }
    }

    fn segment(words: Vec<Word>) -> Segment {
        let start = words.first().map(|w| w.start_s).unwrap_or(0.0);
        let end = words.last().map(|w| w.end_s).unwrap_or(0.0);
        Segment {
            text: words
                .iter()
                .map(|w| w.text.as_str())
                .collect::<Vec<_>>()
                .join(" "),
            start_s: start,
            end_s: end,
            words,
            low_confidence: false,
        }
    }

    #[test]
    fn default_config_uses_auto_language() {
        let cfg = TranscribeConfig::default();
        assert_eq!(cfg.language, LanguageHint::Auto);
    }

    #[test]
    fn default_config_uses_standard_tier() {
        let cfg = TranscribeConfig::default();
        assert_eq!(cfg.tier, Tier::Standard);
    }

    #[test]
    fn default_config_has_token_timestamps_off() {
        let cfg = TranscribeConfig::default();
        assert!(!cfg.token_timestamps);
    }

    #[test]
    fn default_config_uses_all_cores() {
        let cfg = TranscribeConfig::default();
        assert_eq!(cfg.threads, 0);
        assert_eq!(cfg.parallel, 1);
        let n = cfg.resolved_threads();
        let expected = std::thread::available_parallelism()
            .map(|n| n.get() as i32)
            .unwrap_or(4);
        assert_eq!(n, expected);
        assert!(n >= 1);
    }

    #[test]
    fn resolved_threads_divides_by_parallel() {
        let cfg = TranscribeConfig {
            threads: 8,
            parallel: 4,
            ..TranscribeConfig::default()
        };
        assert_eq!(cfg.resolved_threads(), 2);
    }

    #[test]
    fn default_config_keeps_perf_switches_off() {
        let cfg = TranscribeConfig::default();
        assert!(!cfg.flash_attn);
        assert!(cfg.audio_ctx.is_none());
        assert_eq!(cfg.parallel, 1);
        assert!(!cfg.vad_pack);
        assert!(cfg.ranges.is_none());
        assert_eq!(cfg.max_phrase_s, DEFAULT_MAX_PHRASE_S);
        assert_eq!(cfg.max_phrase_words, DEFAULT_MAX_PHRASE_WORDS);
    }

    #[test]
    fn language_hint_parses_known_values() {
        assert_eq!(LanguageHint::parse("hi"), LanguageHint::Hindi);
        assert_eq!(LanguageHint::parse("Hindi"), LanguageHint::Hindi);
        assert_eq!(LanguageHint::parse("en"), LanguageHint::English);
        assert_eq!(LanguageHint::parse("EN"), LanguageHint::English);
        assert_eq!(LanguageHint::parse("auto"), LanguageHint::Auto);
        assert_eq!(LanguageHint::parse(""), LanguageHint::Auto);
        assert_eq!(LanguageHint::parse("fr"), LanguageHint::Other);
    }

    #[test]
    fn params_summary_mentions_upstream_defaults() {
        let cfg = TranscribeConfig::default();
        let s = params_summary(&cfg, 12);
        assert!(s.contains("threads=12"), "{s}");
        assert!(s.contains("upstream defaults"), "{s}");
        assert!(s.contains("threads=4"), "{s}");
        assert!(s.contains("language=auto"), "{s}");
    }

    #[test]
    fn extract_build_flags_finds_whisper_line() {
        let info = "--- whisper.cpp ---\nWHISPER : COREML = 0 | CPU : AVX2 = 1 |\n";
        let flags = extract_build_flags(info);
        assert!(flags.starts_with("WHISPER :"), "{flags}");
        assert!(flags.contains("AVX2 = 1"), "{flags}");
    }

    #[test]
    fn extract_build_flags_returns_empty_when_absent() {
        assert_eq!(extract_build_flags("nothing here"), "");
    }

    // --- special tokens -------------------------------------------------

    #[test]
    fn is_special_token_recognises_bracketed_underscores() {
        assert!(is_special_token("[_BEG_]"));
        assert!(is_special_token("[_EOT_]"));
        assert!(is_special_token("[_TT_123_]"));
        // Real timestamp token shape: no trailing underscore.
        assert!(is_special_token("[_TT_1380]"));
    }

    #[test]
    fn is_special_token_recognises_angle_pipe() {
        assert!(is_special_token("<|startoftranscript|>"));
        assert!(is_special_token("<|hi|>"));
    }

    #[test]
    fn is_special_token_leaves_ordinary_brackets_alone() {
        assert!(!is_special_token("[music]"));
        assert!(!is_special_token("hello"));
        assert!(!is_special_token(""));
    }

    #[test]
    fn merge_tokens_drops_special_tokens() {
        let tokens = vec![
            RawToken {
                text: "[_BEG_]".into(),
                t_dtw_cs: 0,
                probability: 0.9,
            },
            RawToken {
                text: " hello".into(),
                t_dtw_cs: 0,
                probability: 0.9,
            },
            RawToken {
                text: "<|hi|>".into(),
                t_dtw_cs: 0,
                probability: 0.9,
            },
            RawToken {
                text: " world".into(),
                t_dtw_cs: 50,
                probability: 0.9,
            },
            RawToken {
                text: "[_EOT_]".into(),
                t_dtw_cs: 80,
                probability: 0.9,
            },
        ];
        let words = merge_tokens_into_words(&tokens, true, 0.0, 1.0);
        let texts: Vec<&str> = words.iter().map(|w| w.text.as_str()).collect();
        assert_eq!(texts, vec!["hello", "world"]);
        assert!(words.iter().all(|w| !w.text.contains("[_")));
    }

    #[test]
    fn strip_special_substrings_removes_trailing_control_token() {
        assert_eq!(strip_special_substrings("chaahie.[_TT_1380]"), "chaahie.");
    }

    #[test]
    fn strip_special_substrings_removes_leading_control_token() {
        assert_eq!(strip_special_substrings("[_BEG_]hello"), "hello");
    }

    #[test]
    fn strip_special_substrings_removes_angle_pipe() {
        assert_eq!(strip_special_substrings("a<|hi|>b"), "ab");
    }

    #[test]
    fn strip_special_substrings_leaves_plain_text_alone() {
        assert_eq!(strip_special_substrings("hello world"), "hello world");
    }

    #[test]
    fn strip_special_substrings_handles_unclosed_pattern() {
        assert_eq!(strip_special_substrings("hello [_world"), "hello [_world");
    }

    // --- quality helpers -------------------------------------------------

    #[test]
    fn junk_fraction_is_zero_for_latin() {
        assert!((junk_fraction("hello world")).abs() < 1e-6);
    }

    #[test]
    fn junk_fraction_is_zero_for_devanagari() {
        assert!((junk_fraction("नमस्ते")).abs() < 1e-6);
    }

    #[test]
    fn junk_fraction_flags_unrelated_scripts() {
        let f = junk_fraction("日本語");
        assert!(f > 0.9, "got {f}");
    }

    #[test]
    fn junk_fraction_ignores_whitespace() {
        assert!((junk_fraction("   ")).abs() < 1e-6);
    }

    #[test]
    fn repeat_ratio_is_zero_for_short_strings() {
        assert!((repeat_ratio("abc")).abs() < 1e-6);
    }

    #[test]
    fn repeat_ratio_flags_long_runs() {
        let f = repeat_ratio("aaaaaaaaaa");
        assert!(f > 0.9, "got {f}");
    }

    // --- clamping --------------------------------------------------------

    #[test]
    fn clamp_times_caps_to_audio_duration() {
        let mut segs = vec![segment(vec![word("a", 0.0, 10.0), word("b", 5.0, 500.0)])];
        clamp_times(&mut segs, 100.0);
        assert!(segs[0].end_s <= 100.0);
        assert!(segs[0].words[1].end_s <= 100.0);
    }

    #[test]
    fn clamp_times_enforces_end_after_start() {
        let mut segs = vec![segment(vec![word("a", 5.0, 5.0)])];
        clamp_times(&mut segs, 100.0);
        assert!(segs[0].end_s >= segs[0].start_s);
    }

    // --- compressed-segment merging -------------------------------------

    #[test]
    fn merge_compressed_segments_absorbs_next_segment() {
        // Nine words in a 9 ms span, followed by ten words in 1.9 s.
        let first = Segment {
            text: "one two three four five six seven eight nine".into(),
            start_s: 30.806,
            end_s: 30.815,
            words: (0..9)
                .map(|i| word(&format!("w{i}"), 30.806, 30.815))
                .collect(),
            low_confidence: false,
        };
        let second = Segment {
            text: "ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen"
                .into(),
            start_s: 30.815,
            end_s: 32.744,
            words: (0..10)
                .map(|i| {
                    word(
                        &format!("x{i}"),
                        30.815 + i as f64 * 0.19,
                        30.815 + (i as f64 + 1.0) * 0.19,
                    )
                })
                .collect(),
            low_confidence: false,
        };
        let merged = merge_compressed_segments(vec![first, second], MIN_S_PER_WORD);
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].words.len(), 19);
        assert!((merged[0].start_s - 30.806).abs() < 1e-9);
        assert!((merged[0].end_s - 32.744).abs() < 1e-9);
        // Words must be strictly ordered and inside the merged span.
        let mut prev_end = merged[0].start_s;
        for w in &merged[0].words {
            assert!(w.start_s >= prev_end - 1e-9, "{w:?}");
            assert!(w.end_s > w.start_s);
            prev_end = w.end_s;
        }
    }

    #[test]
    fn merge_compressed_segments_leaves_normal_alone() {
        let seg = segment(vec![word("a", 0.0, 0.5), word("b", 0.5, 1.0)]);
        let merged = merge_compressed_segments(vec![seg.clone()], MIN_S_PER_WORD);
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].words.len(), 2);
        assert!((merged[0].end_s - 1.0).abs() < 1e-9);
    }

    #[test]
    fn merge_compressed_segments_keeps_trailing_normal_segment() {
        let compressed = Segment {
            text: "a b c d e".into(),
            start_s: 10.0,
            end_s: 10.005,
            words: (0..5)
                .map(|i| word(&format!("a{i}"), 10.0, 10.005))
                .collect(),
            low_confidence: false,
        };
        let filler = Segment {
            text: "x y z".into(),
            start_s: 10.005,
            end_s: 10.5,
            words: (0..3)
                .map(|i| {
                    word(
                        &format!("x{i}"),
                        10.005 + i as f64 * 0.16,
                        10.005 + (i as f64 + 1.0) * 0.16,
                    )
                })
                .collect(),
            low_confidence: false,
        };
        let trailing = segment(vec![word("q", 10.5, 11.0)]);
        let merged = merge_compressed_segments(vec![compressed, filler, trailing], MIN_S_PER_WORD);
        // First two merged, third stays.
        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].words.len(), 8);
        assert_eq!(merged[1].words.len(), 1);
    }

    // --- phrase segmentation --------------------------------------------

    #[test]
    fn phrases_split_at_max_words() {
        let words: Vec<Word> = (0..30)
            .map(|i| word(&format!("w{i}"), i as f64 * 0.1, i as f64 * 0.1 + 0.05))
            .collect();
        let segs = vec![segment(words)];
        let phrases = segment_into_phrases(segs, &SpeechRegions::default(), 30.0, 10);
        // 30 words / 10 per phrase = 3 phrases.
        assert_eq!(phrases.len(), 3);
        for p in &phrases {
            assert!(p.words.len() <= 10);
        }
    }

    #[test]
    fn phrases_split_at_max_duration() {
        let words: Vec<Word> = (0..20)
            .map(|i| word(&format!("w{i}"), i as f64 * 1.0, i as f64 * 1.0 + 0.5))
            .collect();
        let segs = vec![segment(words)];
        let phrases = segment_into_phrases(segs, &SpeechRegions::default(), 4.0, 100);
        for p in &phrases {
            let dur = p.end_s - p.start_s;
            assert!(dur <= 4.5, "dur {dur} too long on {p:?}");
        }
    }

    #[test]
    fn phrases_split_at_sentence_punctuation() {
        let words = vec![
            word("Hello", 0.0, 0.3),
            word("world.", 0.3, 0.6),
            word("Another", 0.6, 0.9),
            word("sentence.", 0.9, 1.2),
        ];
        let segs = vec![segment(words)];
        let phrases = segment_into_phrases(segs, &SpeechRegions::default(), 30.0, 100);
        // Each `.` should force a break.
        assert_eq!(phrases.len(), 2);
        assert!(phrases[0].text.ends_with('.'));
        assert!(phrases[1].text.ends_with('.'));
    }

    #[test]
    fn phrases_split_at_vad_silence() {
        let words = vec![
            word("before", 0.0, 0.5),
            word("silence", 0.5, 1.0),
            word("after", 5.0, 5.5),
            word("later", 5.5, 6.0),
        ];
        let segs = vec![segment(words)];
        let vad = SpeechRegions {
            regions: vec![(0.0, 1.0), (5.0, 6.0)],
        };
        let phrases = segment_into_phrases(segs, &vad, 30.0, 100);
        // The 4 s gap crosses the VAD silence; expect two phrases.
        assert_eq!(phrases.len(), 2);
        assert_eq!(phrases[0].words.len(), 2);
        assert_eq!(phrases[1].words.len(), 2);
    }

    #[test]
    fn phrases_carry_low_confidence_from_source() {
        let mut seg = segment(vec![word("a", 0.0, 0.5)]);
        seg.low_confidence = true;
        let phrases = segment_into_phrases(vec![seg], &SpeechRegions::default(), 30.0, 10);
        assert!(phrases[0].low_confidence);
    }

    // --- misc ------------------------------------------------------------

    #[test]
    fn merge_tokens_uses_leading_space_heuristic() {
        let tokens = vec![
            RawToken {
                text: " Hel".into(),
                t_dtw_cs: 0,
                probability: 0.9,
            },
            RawToken {
                text: "lo".into(),
                t_dtw_cs: 0,
                probability: 0.85,
            },
            RawToken {
                text: " world".into(),
                t_dtw_cs: 50,
                probability: 0.88,
            },
        ];
        let words = merge_tokens_into_words(&tokens, true, 0.0, 1.0);
        assert_eq!(words.len(), 2);
        assert_eq!(words[0].text, "Hello");
        assert_eq!(words[1].text, "world");
    }

    #[test]
    fn repetition_guard_drops_three_peats() {
        let mut segs = vec![segment(vec![
            word("hello", 0.0, 0.1),
            word("there", 0.1, 0.2),
            word("hello", 0.2, 0.3),
            word("there", 0.3, 0.4),
            word("hello", 0.4, 0.5),
            word("there", 0.5, 0.6),
            word("goodbye", 0.6, 0.7),
        ])];
        drop_repeated_ngrams(&mut segs);
        let texts: Vec<&str> = segs[0].words.iter().map(|w| w.text.as_str()).collect();
        assert_eq!(texts, vec!["hello", "there", "goodbye"]);
    }

    #[test]
    fn timings_serde_round_trip_includes_new_fields() {
        let mut t = Timings {
            backend: "CUDA".into(),
            device: "0".into(),
            threads: 12,
            params: "strategy=greedy".into(),
            build_flags: "WHISPER : COREML = 0 | CPU : AVX2 = 1".into(),
            build_optimized: true,
            state_create_s: 0.42,
            chunk_reasons: vec![ChunkReason::StartOfAudio, ChunkReason::Pause],
            ..Timings::default()
        };
        t.windows.push(WindowTiming {
            index: 0,
            start_s: 0.0,
            end_s: 30.0,
            whisper_full_s: 0.5,
            tokens: 10,
            segments: 1,
            mean_probability: 0.9,
            min_probability: 0.8,
            junk_fraction: 0.0,
            repeat_ratio: 0.0,
            no_speech_prob: 0.01,
            low_confidence: false,
            retry_reason: None,
            language_requested: "auto".into(),
        });
        let json = serde_json::to_string(&t).unwrap();
        let back: Timings = serde_json::from_str(&json).unwrap();
        assert_eq!(back.backend, "CUDA");
        assert_eq!(back.device, "0");
        assert_eq!(back.threads, 12);
        assert!(back.build_optimized);
        assert!((back.state_create_s - 0.42).abs() < 1e-9);
        assert_eq!(back.chunk_reasons.len(), 2);
        assert_eq!(back.chunk_reasons[0], ChunkReason::StartOfAudio);
        assert_eq!(back.windows.len(), 1);
    }

    #[test]
    fn chunk_reason_serde_round_trip() {
        for r in [
            ChunkReason::StartOfAudio,
            ChunkReason::Pause,
            ChunkReason::RegionBoundary,
            ChunkReason::ForcedSize,
            ChunkReason::FixedWindow,
        ] {
            let s = serde_json::to_string(&r).unwrap();
            let back: ChunkReason = serde_json::from_str(&s).unwrap();
            assert_eq!(back, r);
        }
    }

    #[test]
    fn find_longest_pause_picks_middle_of_longest_run() {
        // 100 windows of 10 ms = 1 s. Silence at indices 10..40, i.e.
        // 30 windows = 0.3 s, which is above MIN_PAUSE_FOR_SPLIT_S.
        let mut probs: Vec<f32> = vec![1.0; 100];
        for p in probs.iter_mut().take(40).skip(10) {
            *p = 0.0;
        }
        let mid = find_longest_pause(&probs, 0.010, 0.0, 1.0);
        assert!(mid.is_some(), "expected a pause in a 0.3 s silence");
        let m = mid.unwrap();
        // Midpoint of indices 10..40 is 25, at 0.010 s per window.
        assert!((m - 0.25).abs() < 0.02, "mid = {m}");
    }

    #[test]
    fn find_longest_pause_returns_none_for_short_silence() {
        // All speech: no silence.
        let probs: Vec<f32> = vec![1.0; 20];
        assert!(find_longest_pause(&probs, 0.010, 0.0, 0.200).is_none());
    }

    #[test]
    fn speech_regions_crosses_silence_detects_gap() {
        let v = SpeechRegions {
            regions: vec![(0.0, 1.0), (3.0, 4.0)],
        };
        // Spanning the 2 s gap.
        assert!(v.crosses_silence(0.5, 3.5, 0.3));
        // Entirely inside region 0.
        assert!(!v.crosses_silence(0.2, 0.8, 0.3));
        // Entirely inside region 1.
        assert!(!v.crosses_silence(3.2, 3.8, 0.3));
    }

    #[test]
    fn speech_regions_crosses_silence_rejects_small_gap() {
        let v = SpeechRegions {
            regions: vec![(0.0, 1.0), (1.1, 2.0)],
        };
        // 100 ms gap is below the 300 ms threshold.
        assert!(!v.crosses_silence(0.5, 1.5, 0.3));
    }

    #[test]
    fn plan_chunks_fixed_has_reasons() {
        let audio = MonoAudio::new(vec![0.0; SAMPLE_RATE as usize * 90], SAMPLE_RATE);
        let plan = plan_chunks_fixed(&audio, 30.0, 0.3);
        assert!(plan.chunks.len() >= 3);
        assert_eq!(plan.reasons.len(), plan.chunks.len());
        assert_eq!(plan.reasons[0], ChunkReason::StartOfAudio);
        assert!(plan
            .reasons
            .iter()
            .skip(1)
            .all(|r| *r == ChunkReason::FixedWindow));
    }
}
