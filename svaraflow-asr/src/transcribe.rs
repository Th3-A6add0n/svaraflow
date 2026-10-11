// svaraflow-asr/src/transcribe.rs
//! Transcription engine.
//!
//! Wraps `whisper-rs` behind a typed API. The engine:
//!
//! 1. Runs Silero VAD over the 16 kHz mono signal to find speech regions.
//! 2. Merges regions into chunks of at most [`DEFAULT_MAX_CHUNK_S`] with
//!    [`DEFAULT_OVERLAP_S`] of overlap on each side.
//! 3. Runs whisper.cpp on each chunk with token-level timestamps.
//! 4. Merges sub-word tokens into whole words, drops duplicate words that
//!    fall in the overlap, enforces increasing non-overlapping times, and
//!    snaps each boundary to a local energy minimum within
//!    [`SNAP_RADIUS_S`].
//! 5. Applies a repetition guard: consecutive repeated n-grams are dropped.
//!
//! # Thread count
//!
//! whisper.cpp's compiled-in default for `n_threads` is
//! `min(4, hardware_concurrency)`. On most modern CPUs that leaves the
//! majority of cores idle. [`TranscribeConfig::threads`] overrides it:
//! `0` means "use every logical core the process can see", any other
//! value is passed through unchanged. When [`TranscribeConfig::parallel`]
//! is greater than one the value is divided by the number of workers, so
//! each whisper state gets an equal share of the cores.
//!
//! # Parallel windows
//!
//! When [`TranscribeConfig::parallel`] is greater than 1, the engine
//! dispatches up to N chunks at a time, each to its own worker thread.
//! The workers share one [`WhisperContext`] (which is `Send + Sync`) and
//! each creates its own [`WhisperState`] inside its own thread, because
//! `WhisperState` is `!Send` and may not cross a thread boundary. Worker
//! threads are scoped, so they cannot outlive the engine call. Results
//! are collected in chunk order and the post-process pass runs once at
//! the end, so `--parallel` does not change the transcript bytes.
//!
//! # Timing
//!
//! [`Engine::transcribe_with_timings`] fills a [`Timings`] struct with a
//! wall-clock breakdown of every phase the engine can see. Phases the
//! caller owns — audio decode, resample, model load, export — are timed
//! by the caller.
//!
//! # Ranges
//!
//! [`TranscribeConfig::ranges`] restricts transcription to a set of source
//! time ranges. It is applied after VAD chunking and before VAD packing.
//!
//! # VAD packing
//!
//! When [`TranscribeConfig::vad_pack`] is on, the planned chunks are
//! concatenated into one buffer that contains only speech, plus
//! [`DEFAULT_OVERLAP_S`] of padding at each chunk boundary. Every segment
//! and word time is in packed space and is mapped back to source space
//! before post-processing.
//!
//! # Timestamp quality
//!
//! DTW alignment is not usable through `whisper-rs` 0.16 because the
//! alignment-heads preset is not exposed. Word timing falls back to
//! proportional and every word carries `estimated = true`.

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

/// Longest chunk fed to one `whisper.cpp` call.
pub const DEFAULT_MAX_CHUNK_S: f64 = 30.0;

/// Overlap added on each side of every chunk.
pub const DEFAULT_OVERLAP_S: f64 = 0.3;

/// Radius searched when snapping a word boundary to a local energy minimum.
pub const SNAP_RADIUS_S: f64 = 0.060;

/// VAD probability above which a window counts as speech.
pub const VAD_THRESHOLD: f32 = 0.5;

/// Silence shorter than this merges two speech regions into one.
pub const VAD_MIN_SILENCE_S: f64 = 0.100;

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
    /// Everything inside `Engine::transcribe_with_timings` after load.
    pub transcribe_s: f64,
    /// Sum of `WhisperState::full` calls across all workers.
    pub whisper_full_s: f64,
    /// Merging tokens, deduping overlaps, monotonic ordering, snapping.
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
    /// Number of chunks whose decoder temperature was above zero.
    /// `None` when the binding does not expose per-segment temperature.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decode_retries: Option<usize>,
    /// `true` when every word carries a DTW-derived timestamp.
    #[serde(default)]
    pub dtw_usable: bool,
    /// `true` when the run used VAD packing.
    #[serde(default)]
    pub vad_pack: bool,
    /// Number of source chunks actually fed to the engine.
    #[serde(default)]
    pub chunks: usize,
    /// Number of whisper states used concurrently.
    #[serde(default)]
    pub parallel: usize,
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
    // `standalone` is a private module in whisper-rs; its items are
    // re-exported at the crate root via `pub use standalone::*;`, so the
    // call path is `whisper_rs::print_system_info()`.
    out.push_str(whisper_rs::print_system_info());
    if !out.ends_with('\n') {
        out.push('\n');
    }
    out
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Language hint passed to whisper.cpp.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LanguageHint {
    /// Hindi. The model writes Roman Hinglish regardless.
    Hindi,
    /// Let whisper.cpp detect.
    Auto,
}

impl LanguageHint {
    /// The string `whisper.cpp` expects, or `None` for auto.
    #[must_use]
    pub fn as_whisper_str(self) -> Option<&'static str> {
        match self {
            LanguageHint::Hindi => Some("hi"),
            LanguageHint::Auto => None,
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
    /// Request token-level DTW timestamps.
    pub use_dtw: bool,
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
    /// Overlap on each side of every chunk.
    pub overlap_s: f64,
    /// Number of CPU threads *per whisper state*.
    ///
    /// `0` means "all logical cores divided by `parallel`". Any other
    /// value is divided by `parallel` too, then floored at 1, so
    /// `--threads 24 --parallel 3` gives each of the three states
    /// `24 / 3 = 8` threads and does not oversubscribe the CPU.
    pub threads: usize,
    /// Enable whisper.cpp's flash attention path. Off by default.
    pub flash_attn: bool,
    /// Override whisper.cpp's audio context size, in frames.
    pub audio_ctx: Option<i32>,
    /// Number of whisper states running concurrently.
    ///
    /// `1` (the default) runs windows sequentially. Values above 1
    /// dispatch that many chunks at a time. See the module docs for the
    /// `!Send` reason each state is created inside its own thread.
    pub parallel: usize,
    /// Drop silence and pack speech into full-length windows.
    pub vad_pack: bool,
    /// Restrict transcription to these source time ranges, in seconds.
    pub ranges: Option<Vec<(f64, f64)>>,
}

impl Default for TranscribeConfig {
    fn default() -> Self {
        Self {
            tier: Tier::Standard,
            language: LanguageHint::Hindi,
            decode: DecodeStrategy::Greedy,
            use_dtw: false,
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
/// Format: one `start_s,end_s` per line, `#` starts a comment, blank lines
/// are ignored. Whitespace around the numbers is trimmed. A line that
/// cannot be parsed is an error, not a silent skip.
///
/// # Errors
///
/// [`AsrError::WhisperState`] wrapping the failing line number on any
/// malformed line, and on any range where `end <= start`.
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
    /// The word's text, without leading or trailing whitespace.
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

/// One segment (roughly one sentence) with its words.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Segment {
    /// The segment's text.
    pub text: String,
    /// Start time in seconds.
    pub start_s: f64,
    /// End time in seconds.
    pub end_s: f64,
    /// The words in this segment, in order.
    pub words: Vec<Word>,
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

    /// Transcribes a 16 kHz mono signal and fills `timings` with a
    /// wall-clock breakdown of every phase the engine can see.
    ///
    /// # Errors
    ///
    /// - [`AsrError::Cancelled`] when `cancel` becomes `true` between
    ///   batches.
    /// - [`AsrError::WhisperState`] and [`AsrError::WhisperStateCreate`]
    ///   on whisper.cpp failures.
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
        timings.audio_duration_s = audio_duration_s;
        timings.system_info = Some(system_info());
        timings.vad_pack = self.config.vad_pack;
        timings.parallel = self.config.parallel.max(1);

        // 1. Plan chunks.
        let plan_start = Instant::now();
        let mut chunks = self.plan_chunks(audio, timings)?;

        // 2. Apply --ranges.
        if let Some(ranges) = &self.config.ranges {
            chunks = apply_ranges(chunks, ranges, audio.sample_rate);
            if chunks.is_empty() {
                return Err(AsrError::WhisperState {
                    reason: "--ranges left no chunks to transcribe".into(),
                });
            }
        }
        timings.chunk_plan_s = plan_start.elapsed().as_secs_f64();

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
        timings.chunks = effective_chunks.len();

        // 4. Transcribe (sequential or parallel).
        let (mut all_segments, detected_language) = if self.config.parallel > 1
            && effective_chunks.len() > 1
        {
            self.transcribe_parallel(working_audio, &effective_chunks, progress, cancel, timings)?
        } else {
            self.transcribe_sequential(working_audio, &effective_chunks, progress, cancel, timings)?
        };

        // 5. Remap packed times back to source times.
        if let Some(map) = &packed_map {
            remap_segments(&mut all_segments, map);
        }

        // 6. Post-process.
        let post_start = Instant::now();
        dedup_overlaps(&mut all_segments, self.config.overlap_s);
        enforce_monotonic(&mut all_segments);
        for segment in &mut all_segments {
            for word in &mut segment.words {
                snap_word(word, &audio.samples, SAMPLE_RATE);
            }
        }
        if self.config.repetition_guard {
            drop_repeated_ngrams(&mut all_segments);
        }
        for segment in &mut all_segments {
            if let (Some(first), Some(last)) = (segment.words.first(), segment.words.last()) {
                segment.start_s = first.start_s;
                segment.end_s = last.end_s;
            }
        }
        timings.postprocess_s = post_start.elapsed().as_secs_f64();

        // decode_retries: not exposed by whisper-rs 0.16. See docs/PERF.md.
        timings.decode_retries = None;

        let (total_words, estimated_words) = {
            let mut n = 0usize;
            let mut est = 0usize;
            for seg in &all_segments {
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
            segments: all_segments,
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
        let mut state: WhisperState = self
            .ctx
            .create_state()
            .map_err(|e| AsrError::WhisperStateCreate(e.to_string()))?;

        let mut total_full = Duration::ZERO;
        let mut all_segments: Vec<Segment> = Vec::new();
        let mut detected_language = String::new();
        let total = chunks.len();

        for (i, chunk) in chunks.iter().enumerate() {
            if cancel.load(Ordering::Relaxed) {
                return Err(AsrError::Cancelled);
            }
            progress(i + 1, total);

            let slice = &audio.samples[chunk.start_sample..chunk.end_sample];
            let (segments, dur) = run_whisper_full(&mut state, &self.config, slice, chunk.start_s)?;
            total_full += dur;
            if detected_language.is_empty() {
                detected_language = match self.config.language {
                    LanguageHint::Hindi => "hi".to_string(),
                    LanguageHint::Auto => "auto".to_string(),
                };
            }
            all_segments.extend(segments);
        }

        timings.whisper_full_s = total_full.as_secs_f64();
        Ok((all_segments, detected_language))
    }

    /// Parallel path: up to `parallel` states, each in its own scoped
    /// worker thread. The thread bodies create their states, because
    /// `WhisperState` is `!Send` and may not cross a thread boundary.
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
        let mut detected_language = String::new();

        for batch_start in (0..chunks.len()).step_by(n) {
            if cancel.load(Ordering::Relaxed) {
                return Err(AsrError::Cancelled);
            }
            let batch_end = (batch_start + n).min(chunks.len());
            let batch = &chunks[batch_start..batch_end];

            // Scoped threads: they cannot outlive this closure, so no
            // dangling pointers even with `!Send` states inside.
            let results: Vec<Result<(Vec<Segment>, Duration), String>> = std::thread::scope(|s| {
                let handles: Vec<_> = batch
                    .iter()
                    .map(|chunk| {
                        let ctx = &self.ctx;
                        let config = &self.config;
                        let samples = &audio.samples[chunk.start_sample..chunk.end_sample];
                        let offset = chunk.start_s;
                        s.spawn(move || -> Result<(Vec<Segment>, Duration), String> {
                            let mut state = ctx
                                .create_state()
                                .map_err(|e| format!("create_state: {e}"))?;
                            run_whisper_full(&mut state, config, samples, offset)
                                .map_err(|e| e.to_string())
                        })
                    })
                    .collect();
                handles
                    .into_iter()
                    .map(|h| {
                        h.join()
                            .unwrap_or_else(|_| Err("worker thread panicked".to_string()))
                    })
                    .collect()
            });

            for (i, r) in results.into_iter().enumerate() {
                let (segs, dur) = r.map_err(|reason| AsrError::WhisperState { reason })?;
                total_full += dur;
                all_segments.extend(segs);
                if detected_language.is_empty() {
                    detected_language = match self.config.language {
                        LanguageHint::Hindi => "hi".to_string(),
                        LanguageHint::Auto => "auto".to_string(),
                    };
                }
                progress(batch_start + i + 1, total);
            }
        }

        timings.whisper_full_s = total_full.as_secs_f64();
        Ok((all_segments, detected_language))
    }

    /// Splits the audio into chunks based on VAD.
    fn plan_chunks(
        &mut self,
        audio: &MonoAudio,
        timings: &mut Timings,
    ) -> Result<Vec<Chunk>, AsrError> {
        #[cfg(feature = "vad")]
        {
            if let Some(vad) = self.vad.as_mut() {
                let vad_start = Instant::now();
                let chunks = plan_chunks_with_vad(
                    audio,
                    vad,
                    self.config.max_chunk_s,
                    self.config.overlap_s,
                )?;
                timings.vad_s += vad_start.elapsed().as_secs_f64();
                return Ok(chunks);
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

/// Runs one `whisper.cpp` call over `samples` using `state` and returns
/// the resulting segments plus the wall-clock time spent inside
/// `WhisperState::full`.
///
/// Free function, not a method, so the parallel worker closures can call
/// it on a state they created inside their own thread.
fn run_whisper_full(
    state: &mut WhisperState,
    config: &TranscribeConfig,
    samples: &[f32],
    offset_s: f64,
) -> Result<(Vec<Segment>, Duration), AsrError> {
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

    // DTW alignment-heads preset is not exposed by whisper-rs 0.16, so
    // token timestamps stay off regardless of `config.use_dtw`.
    let _ = config.use_dtw;
    params.set_token_timestamps(false);

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
    let full_dur = full_start.elapsed();
    full_result.map_err(|e| AsrError::WhisperState {
        reason: e.to_string(),
    })?;

    let mut out = Vec::new();
    for segment in state.as_iter() {
        let text = match segment.to_str_lossy() {
            Ok(s) => s.to_string(),
            Err(_) => continue,
        };
        if text.trim().is_empty() {
            continue;
        }
        let seg_start = segment.start_timestamp() as f64 / 100.0 + offset_s;
        let seg_end = segment.end_timestamp() as f64 / 100.0 + offset_s;

        let n_tokens = segment.n_tokens();
        let mut raw: Vec<RawToken> = Vec::with_capacity(n_tokens.max(0) as usize);
        for ti in 0..n_tokens {
            let Some(token) = segment.get_token(ti) else {
                continue;
            };
            let Ok(token_text) = token.to_str() else {
                continue;
            };
            let data = token.token_data();
            raw.push(RawToken {
                text: token_text.to_string(),
                t_dtw_cs: data.t_dtw,
                probability: data.p,
            });
        }

        let dtw_usable = raw.iter().all(|t| t.t_dtw_cs >= 0);
        let words = merge_tokens_into_words(&raw, dtw_usable, seg_start, seg_end);
        out.push(Segment {
            text: text.trim().to_string(),
            start_s: seg_start,
            end_s: seg_end,
            words,
        });
    }
    Ok((out, full_dur))
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
///
/// Chunk boundaries are shared points: the `packed_end_s` of one chunk
/// equals the `packed_start_s` of the next. The mapping therefore uses
/// **half-open** intervals `[packed_start, packed_end)` for every chunk,
/// which partitions packed time with no gaps and no overlaps, and makes
/// the boundary belong to the later chunk. A time at or beyond the end of
/// the final chunk is clamped to that chunk's source end, because there
/// is no later chunk to hand it to.
///
/// The choice matters for `remap_segments`: a segment that begins exactly
/// on a chunk boundary is almost always a segment that began in that
/// later chunk's source window, not a segment that began at the previous
/// chunk's very last sample.
fn packed_to_source(t: f64, map: &[PackedMap]) -> f64 {
    if map.is_empty() {
        return t;
    }
    for m in map {
        if t >= m.packed_start_s && t < m.packed_end_s {
            return m.source_start_s + (t - m.packed_start_s);
        }
    }
    // At or past the end of the last chunk. Clamp to the last chunk's
    // source-time end.
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

// ---------------------------------------------------------------------------
// Token merging and post-processing
// ---------------------------------------------------------------------------

/// Groups tokens into words using the leading-space heuristic.
fn merge_tokens_into_words(
    tokens: &[RawToken],
    dtw_usable: bool,
    seg_start_s: f64,
    seg_end_s: f64,
) -> Vec<Word> {
    let mut words: Vec<Word> = Vec::new();
    let mut current: Option<Word> = None;
    let total_tokens = tokens.len().max(1);

    for (i, tok) in tokens.iter().enumerate() {
        let raw = tok.text.as_str();
        let starts_word = raw.starts_with(' ')
            || raw.starts_with('\u{0120}')
            || (words.is_empty() && current.is_none());
        let piece = raw.trim_start_matches([' ', '\u{0120}']).to_string();
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

/// Runs VAD and returns chunks with overlap.
#[cfg(feature = "vad")]
fn plan_chunks_with_vad(
    audio: &MonoAudio,
    vad: &mut crate::vad::SileroVad,
    max_chunk_s: f64,
    overlap_s: f64,
) -> Result<Vec<Chunk>, AsrError> {
    use svaraflow_core::vad::Vad;

    let probs = vad.speech_probabilities(&audio.samples)?;
    let spw = probs.seconds_per_window();
    if spw <= 0.0 || probs.values.is_empty() {
        return Ok(plan_chunks_fixed(audio, max_chunk_s, overlap_s));
    }

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
    if regions.is_empty() {
        return Ok(plan_chunks_fixed(audio, max_chunk_s, overlap_s));
    }

    let mut split: Vec<(f64, f64)> = Vec::new();
    for (s, e) in regions {
        if e - s <= max_chunk_s {
            split.push((s, e));
            continue;
        }
        let mut cursor = s;
        while cursor < e {
            let next = (cursor + max_chunk_s).min(e);
            split.push((cursor, next));
            cursor = next;
        }
    }

    let samples_per_s = audio.sample_rate as f64;
    let duration = audio.duration_seconds();
    let chunks: Vec<Chunk> = split
        .into_iter()
        .map(|(s, e)| Chunk {
            start_sample: (((s - overlap_s).max(0.0)) * samples_per_s) as usize,
            end_sample: (((e + overlap_s).min(duration)) * samples_per_s) as usize,
            start_s: (s - overlap_s).max(0.0),
        })
        .collect();
    Ok(chunks)
}

/// Fallback chunking with no VAD: fixed-length windows.
fn plan_chunks_fixed(audio: &MonoAudio, max_chunk_s: f64, overlap_s: f64) -> Vec<Chunk> {
    let samples_per_s = audio.sample_rate as f64;
    let max_samples = (max_chunk_s * samples_per_s) as usize;
    let overlap_samples = (overlap_s * samples_per_s) as usize;
    let step = max_samples.saturating_sub(2 * overlap_samples).max(1);

    let mut out = Vec::new();
    let mut start = 0usize;
    while start < audio.samples.len() {
        let end = (start + max_samples).min(audio.samples.len());
        out.push(Chunk {
            start_sample: start,
            end_sample: end,
            start_s: start as f64 / samples_per_s,
        });
        if end == audio.samples.len() {
            break;
        }
        start += step;
    }
    if out.is_empty() {
        out.push(Chunk {
            start_sample: 0,
            end_sample: audio.samples.len(),
            start_s: 0.0,
        });
    }
    out
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
fn snap_word(word: &mut Word, samples: &[f32], rate: u32) {
    let radius = (SNAP_RADIUS_S * f64::from(rate)) as usize;
    word.start_s = snap_to_min_energy(word.start_s, samples, rate, radius);
    word.end_s = snap_to_min_energy(word.end_s, samples, rate, radius).max(word.start_s + 0.001);
}

/// Finds the lowest-RMS position within `radius` samples of `t`.
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
        }
    }

    #[test]
    fn default_config_uses_standard_tier() {
        let cfg = TranscribeConfig::default();
        assert_eq!(cfg.tier, Tier::Standard);
        assert_eq!(cfg.language, LanguageHint::Hindi);
    }

    #[test]
    fn default_config_has_dtw_off() {
        let cfg = TranscribeConfig::default();
        assert!(!cfg.use_dtw);
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
    fn resolved_threads_honours_explicit_value() {
        let cfg = TranscribeConfig {
            threads: 3,
            parallel: 1,
            ..TranscribeConfig::default()
        };
        assert_eq!(cfg.resolved_threads(), 3);
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
    fn resolved_threads_auto_divides_by_parallel() {
        let cfg = TranscribeConfig {
            threads: 0,
            parallel: 2,
            ..TranscribeConfig::default()
        };
        let cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        let expected = (cores / 2).max(1) as i32;
        assert_eq!(cfg.resolved_threads(), expected);
    }

    #[test]
    fn resolved_threads_floor_at_one() {
        let cfg = TranscribeConfig {
            threads: 4,
            parallel: 16,
            ..TranscribeConfig::default()
        };
        assert_eq!(cfg.resolved_threads(), 1);
    }

    #[test]
    fn default_config_keeps_perf_switches_off() {
        let cfg = TranscribeConfig::default();
        assert!(!cfg.flash_attn, "flash_attn must be off by default");
        assert!(cfg.audio_ctx.is_none(), "audio_ctx must be None by default");
        assert_eq!(cfg.parallel, 1, "parallel must default to 1");
        assert!(!cfg.vad_pack, "vad_pack must be off by default");
        assert!(cfg.ranges.is_none(), "ranges must be None by default");
    }

    #[test]
    fn effective_prompt_from_vocabulary() {
        let cfg = TranscribeConfig {
            vocabulary: vec!["SvaraFlow".into(), "Th3 A6add0n".into()],
            ..TranscribeConfig::default()
        };
        let p = cfg.effective_prompt().unwrap();
        assert!(p.contains("SvaraFlow"));
        assert!(p.contains("Th3 A6add0n"));
    }

    #[test]
    fn effective_prompt_explicit_wins() {
        let cfg = TranscribeConfig {
            vocabulary: vec!["x".into()],
            initial_prompt: Some("custom".into()),
            ..TranscribeConfig::default()
        };
        assert_eq!(cfg.effective_prompt().unwrap(), "custom");
    }

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
    fn merge_tokens_marks_proportional_as_estimated() {
        let tokens = vec![
            RawToken {
                text: " a".into(),
                t_dtw_cs: -1,
                probability: 0.9,
            },
            RawToken {
                text: " b".into(),
                t_dtw_cs: -1,
                probability: 0.9,
            },
        ];
        let words = merge_tokens_into_words(&tokens, false, 0.0, 1.0);
        assert!(words.iter().all(|w| w.estimated));
    }

    #[test]
    fn monotonic_enforcement_fixes_overlaps() {
        let mut segs = vec![segment(vec![
            word("a", 0.0, 1.0),
            word("b", 0.5, 1.5),
            word("c", 1.4, 1.4),
        ])];
        enforce_monotonic(&mut segs);
        let w = &segs[0].words;
        assert!(w[0].end_s <= w[1].start_s);
        assert!(w[1].end_s <= w[2].start_s);
        assert!(w[2].end_s > w[2].start_s);
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
            word("hello", 0.6, 0.7),
            word("there", 0.7, 0.8),
            word("goodbye", 0.8, 0.9),
        ])];
        drop_repeated_ngrams(&mut segs);
        let texts: Vec<&str> = segs[0].words.iter().map(|w| w.text.as_str()).collect();
        assert_eq!(texts, vec!["hello", "there", "goodbye"]);
    }

    #[test]
    fn repetition_guard_leaves_two_repeats_alone() {
        let mut segs = vec![segment(vec![
            word("a", 0.0, 0.1),
            word("b", 0.1, 0.2),
            word("a", 0.2, 0.3),
            word("b", 0.3, 0.4),
        ])];
        drop_repeated_ngrams(&mut segs);
        assert_eq!(segs[0].words.len(), 4);
    }

    #[test]
    fn plan_chunks_fixed_splits_long_audio() {
        let audio = MonoAudio::new(vec![0.0; SAMPLE_RATE as usize * 90], SAMPLE_RATE);
        let chunks = plan_chunks_fixed(&audio, 30.0, 0.3);
        assert!(chunks.len() >= 3, "got {} chunks", chunks.len());
        for c in &chunks {
            assert!(c.end_sample > c.start_sample);
        }
    }

    #[test]
    fn plan_chunks_fixed_short_audio_is_one_chunk() {
        let audio = MonoAudio::new(vec![0.0; SAMPLE_RATE as usize * 5], SAMPLE_RATE);
        let chunks = plan_chunks_fixed(&audio, 30.0, 0.3);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].start_sample, 0);
        assert_eq!(chunks[0].end_sample, audio.samples.len());
    }

    #[test]
    fn transcript_words_flattens_segments() {
        let t = Transcript {
            segments: vec![
                segment(vec![word("a", 0.0, 0.1), word("b", 0.1, 0.2)]),
                segment(vec![word("c", 0.2, 0.3)]),
            ],
            language: "hi".into(),
            tier: Tier::Standard,
            model: "test".into(),
            audio_duration_s: 1.0,
            elapsed_s: 0.5,
            real_time_factor: 2.0,
        };
        assert_eq!(t.words().len(), 3);
    }

    #[test]
    fn default_config_has_spec_values() {
        let c = TranscribeConfig::default();
        assert_eq!(c.max_chunk_s, 30.0);
        assert_eq!(c.overlap_s, 0.3);
        assert!(c.temperature_fallback);
        assert!(c.repetition_guard);
        assert_eq!(c.threads, 0);
    }

    #[test]
    fn section_accumulates_across_starts() {
        let mut s = Section::default();
        s.start();
        std::thread::sleep(Duration::from_millis(2));
        s.stop();
        let first = s.total();
        s.start();
        std::thread::sleep(Duration::from_millis(2));
        s.stop();
        assert!(s.total() > first, "second stop added time");
    }

    #[test]
    fn section_stop_without_start_is_a_noop() {
        let mut s = Section::default();
        s.stop();
        assert_eq!(s.total(), Duration::ZERO);
    }

    #[test]
    fn timings_finalize_computes_rtf() {
        let mut t = Timings {
            audio_duration_s: 60.0,
            ..Timings::default()
        };
        t.finalize(Duration::from_secs(10));
        assert!((t.total_wall_s - 10.0).abs() < 1e-9);
        assert!((t.real_time_factor - 6.0).abs() < 1e-9);
    }

    #[test]
    fn timings_finalize_zero_wall_is_zero_rtf() {
        let mut t = Timings {
            audio_duration_s: 60.0,
            ..Timings::default()
        };
        t.finalize(Duration::ZERO);
        assert_eq!(t.total_wall_s, 0.0);
        assert_eq!(t.real_time_factor, 0.0);
    }

    #[test]
    fn system_info_mentions_arch_and_cores() {
        let s = system_info();
        assert!(
            s.contains(&format!("arch={}", std::env::consts::ARCH)),
            "{s}"
        );
        assert!(s.contains("logical_cores="), "{s}");
        assert!(s.contains("--- whisper.cpp ---"), "{s}");
    }

    #[test]
    fn timings_serde_round_trip() {
        let t = Timings {
            audio_decode_s: 1.5,
            vad_s: 0.25,
            whisper_full_s: 12.0,
            dtw_usable: false,
            vad_pack: true,
            chunks: 3,
            parallel: 2,
            system_info: Some("arch=x86_64\n".into()),
            ..Timings::default()
        };
        let json = serde_json::to_string(&t).unwrap();
        let back: Timings = serde_json::from_str(&json).unwrap();
        assert!((back.audio_decode_s - 1.5).abs() < 1e-12);
        assert!(back.vad_pack);
        assert_eq!(back.chunks, 3);
        assert_eq!(back.parallel, 2);
        assert_eq!(back.system_info.as_deref(), Some("arch=x86_64\n"));
    }

    #[test]
    fn parse_ranges_accepts_two_pairs_and_ignores_comments() {
        let text = "\
# keep list from silence removal
1.5,3.25
  10 , 12
";
        let r = parse_ranges(text).unwrap();
        assert_eq!(r, vec![(1.5, 3.25), (10.0, 12.0)]);
    }

    #[test]
    fn parse_ranges_rejects_bad_line() {
        let err = parse_ranges("1.0\n").unwrap_err();
        assert!(format!("{err}").contains("expected `start,end`"), "{err}");
    }

    #[test]
    fn parse_ranges_rejects_non_finite() {
        let err = parse_ranges("nan,3\n").unwrap_err();
        assert!(
            format!("{err}").contains("non-finite") || format!("{err}").contains("bad start"),
            "{err}"
        );
    }

    #[test]
    fn parse_ranges_rejects_end_before_start() {
        let err = parse_ranges("5,3\n").unwrap_err();
        assert!(format!("{err}").contains("end (3)"), "{err}");
    }

    #[test]
    fn apply_ranges_clips_chunks() {
        let chunks = vec![Chunk {
            start_sample: 0,
            end_sample: 160_000,
            start_s: 0.0,
        }];
        let ranges = vec![(1.0, 2.0), (4.0, 6.0), (20.0, 30.0)];
        let out = apply_ranges(chunks, &ranges, 16_000);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].start_sample, 16_000);
        assert_eq!(out[0].end_sample, 32_000);
        assert!((out[0].start_s - 1.0).abs() < 1e-9);
        assert_eq!(out[1].start_sample, 64_000);
        assert_eq!(out[1].end_sample, 96_000);
    }

    #[test]
    fn apply_ranges_empty_when_nothing_overlaps() {
        let chunks = vec![Chunk {
            start_sample: 0,
            end_sample: 16_000,
            start_s: 0.0,
        }];
        let out = apply_ranges(chunks, &[(10.0, 20.0)], 16_000);
        assert!(out.is_empty());
    }

    #[test]
    fn pack_chunks_concatenates_in_order() {
        let audio = MonoAudio::new((0..16_000).map(|i| i as f32).collect(), 16_000);
        let chunks = vec![
            Chunk {
                start_sample: 0,
                end_sample: 1600,
                start_s: 0.0,
            },
            Chunk {
                start_sample: 8000,
                end_sample: 9600,
                start_s: 0.5,
            },
        ];
        let (packed, map) = pack_chunks(&audio, &chunks);
        assert_eq!(packed.samples.len(), 3200);
        assert_eq!(packed.sample_rate, 16_000);
        assert_eq!(map.len(), 2);
        assert!((map[0].packed_start_s - 0.0).abs() < 1e-9);
        assert!((map[0].packed_end_s - 0.1).abs() < 1e-9);
        assert!((map[0].source_start_s - 0.0).abs() < 1e-9);
        assert!((map[1].packed_start_s - 0.1).abs() < 1e-9);
        assert!((map[1].packed_end_s - 0.2).abs() < 1e-9);
        assert!((map[1].source_start_s - 0.5).abs() < 1e-9);
    }

    #[test]
    fn packed_to_source_maps_inside_each_chunk() {
        let map = vec![
            PackedMap {
                packed_start_s: 0.0,
                packed_end_s: 1.0,
                source_start_s: 10.0,
            },
            PackedMap {
                packed_start_s: 1.0,
                packed_end_s: 2.0,
                source_start_s: 30.0,
            },
        ];
        // Strictly inside chunk 0.
        assert!((packed_to_source(0.5, &map) - 10.5).abs() < 1e-9);
        // Strictly inside chunk 1.
        assert!((packed_to_source(1.5, &map) - 30.5).abs() < 1e-9);
        // Exactly on the shared boundary: half-open intervals make the
        // later chunk win.
        assert!((packed_to_source(1.0, &map) - 30.0).abs() < 1e-9);
        // At the start of the first chunk.
        assert!((packed_to_source(0.0, &map) - 10.0).abs() < 1e-9);
        // Past the end of the last chunk: clamp to the last chunk's
        // source-time end.
        assert!((packed_to_source(2.5, &map) - 31.0).abs() < 1e-9);
        // Exactly at the end of the last chunk.
        assert!((packed_to_source(2.0, &map) - 31.0).abs() < 1e-9);
    }

    #[test]
    fn remap_segments_rewrites_words_too() {
        let map = vec![PackedMap {
            packed_start_s: 0.0,
            packed_end_s: 10.0,
            source_start_s: 100.0,
        }];
        let mut segs = vec![segment(vec![word("hi", 1.0, 1.5), word("there", 2.0, 2.5)])];
        remap_segments(&mut segs, &map);
        assert!((segs[0].words[0].start_s - 101.0).abs() < 1e-9);
        assert!((segs[0].words[0].end_s - 101.5).abs() < 1e-9);
        assert!((segs[0].words[1].start_s - 102.0).abs() < 1e-9);
        assert!((segs[0].start_s - 101.0).abs() < 1e-9);
        assert!((segs[0].end_s - 102.5).abs() < 1e-9);
    }
}
