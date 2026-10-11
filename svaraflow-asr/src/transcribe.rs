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
//! # Whisper state
//!
//! One `WhisperState` is created per `transcribe` call and reused across
//! all chunks. `whisper.cpp` resets its internal state between `full`
//! calls, so reusing the object is safe and avoids a `cudaMalloc` /
//! `malloc` storm for every 30-second chunk.
//!
//! # Timestamp quality
//!
//! Word timing falls into two modes:
//!
//! - **DTW.** whisper.cpp can compute per-token timestamps via dynamic
//!   time warping against a preset of "alignment heads". That preset must
//!   match the loaded model. `whisper-rs` 0.16 does not expose
//!   `WhisperContextParameters::with_dtw_aheads_preset` (nor a
//!   `FullParams` equivalent), so the engine cannot declare a preset.
//!   Without one, whisper.cpp's DTW pass produces nothing usable and, on
//!   some builds, does not terminate quickly.
//! - **Proportional.** The engine divides each segment's span evenly
//!   across its tokens. Coarse, but it always terminates and every word is
//!   flagged `estimated = true`.
//!
//! Because DTW is currently unusable, [`TranscribeConfig::use_dtw`] is
//! accepted for compatibility but ignored: the engine requests token
//! timestamps only when a future change wires the alignment-heads preset.
//! See [`Engine::transcribe_chunk`].
//!
//! # Cancellation
//!
//! Cancellation is checked between chunks, not within a chunk.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

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
    ///
    /// Currently ignored. See the module docs: the DTW alignment-heads
    /// preset is not exposed by `whisper-rs` 0.16, so requesting token
    /// timestamps costs decoder time and produces no usable output. The
    /// flag is retained so callers can set it today and get DTW as soon
    /// as the preset is wired.
    pub use_dtw: bool,
    /// Optional initial prompt.
    pub initial_prompt: Option<String>,
    /// Domain vocabulary to bias the decoder with.
    pub vocabulary: Vec<String>,
    /// Whether to let whisper.cpp fall back to higher temperatures when a
    /// segment fails its thresholds.
    pub temperature_fallback: bool,
    /// Segments with a no-speech probability above this are dropped.
    pub no_speech_threshold: f32,
    /// Whether to drop consecutive repeated n-grams.
    pub repetition_guard: bool,
    /// Longest chunk fed to one whisper.cpp call.
    pub max_chunk_s: f64,
    /// Overlap on each side of every chunk.
    pub overlap_s: f64,
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
}

/// One word with its timing and probability.
#[derive(Debug, Clone, PartialEq)]
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
#[derive(Debug, Clone, PartialEq)]
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
#[derive(Debug, Clone, PartialEq)]
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

        // Do not set `use_gpu` explicitly. `WhisperContextParameters::default()`
        // already defaults it to `cfg!(feature = "_gpu")` inside the
        // `whisper-rs` crate, which is `true` whenever a GPU backend
        // feature is enabled on `whisper-rs`.
        //
        // Flash attention must be off for token timestamps to be
        // produced at all, and it interacts badly with the DTW pass on
        // some builds.
        let mut ctx_params = WhisperContextParameters::default();
        ctx_params.flash_attn(false);

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

    /// Transcribes a 16 kHz mono signal.
    ///
    /// # Errors
    ///
    /// - [`AsrError::Cancelled`] when `cancel` becomes `true` between
    ///   chunks.
    /// - [`AsrError::WhisperState`] and [`AsrError::WhisperStateCreate`]
    ///   on whisper.cpp failures.
    pub fn transcribe(
        &mut self,
        audio: &MonoAudio,
        progress: &mut TranscribeProgressFn<'_>,
        cancel: &AtomicBool,
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

        let chunks = self.plan_chunks(audio)?;
        let total = chunks.len();
        let mut all_segments: Vec<Segment> = Vec::new();
        let mut detected_language = String::new();

        // Create the state once. `whisper.cpp` resets internal state
        // between `full` calls, so the same object serves every chunk and
        // avoids re-allocating ~400 MB of buffers per chunk.
        let mut state: WhisperState = self
            .ctx
            .create_state()
            .map_err(|e| AsrError::WhisperStateCreate(e.to_string()))?;

        for (i, chunk) in chunks.iter().enumerate() {
            if cancel.load(Ordering::Relaxed) {
                return Err(AsrError::Cancelled);
            }
            progress(i + 1, total);

            let slice = &audio.samples[chunk.start_sample..chunk.end_sample];
            let segments = self.transcribe_chunk(&mut state, slice, chunk.start_s)?;
            if detected_language.is_empty() {
                detected_language = match self.config.language {
                    LanguageHint::Hindi => "hi".to_string(),
                    LanguageHint::Auto => "auto".to_string(),
                };
            }
            all_segments.extend(segments);
        }

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

        let elapsed_s = started.elapsed().as_secs_f64();
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

    /// Runs whisper.cpp over one chunk using a caller-provided state.
    fn transcribe_chunk(
        &self,
        state: &mut WhisperState,
        samples: &[f32],
        offset_s: f64,
    ) -> Result<Vec<Segment>, AsrError> {
        let strategy = match self.config.decode {
            DecodeStrategy::Greedy => SamplingStrategy::Greedy { best_of: 1 },
            DecodeStrategy::BeamSearch { beam_size } => SamplingStrategy::BeamSearch {
                beam_size,
                patience: -1.0,
            },
        };
        let mut params = FullParams::new(strategy);

        params.set_language(self.config.language.as_whisper_str());
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        params.set_no_speech_thold(self.config.no_speech_threshold);

        // Token timestamps enable whisper.cpp's DTW alignment pass. That
        // pass needs an alignment-heads preset that matches the loaded
        // model, and `whisper-rs` 0.16 does not expose the setter for it.
        // Without the preset the pass consumes decoder time and produces
        // nothing: every token comes back with `t_dtw == -1`, and every
        // word gets `estimated = true` regardless. Requesting the pass
        // is therefore worse than useless until the alignment-heads path
        // is wired.
        //
        // `self.config.use_dtw` is retained as an API field so that when
        // the preset is available this line becomes
        // `params.set_token_timestamps(self.config.use_dtw);`. For now,
        // off.
        let _ = self.config.use_dtw;
        params.set_token_timestamps(false);

        if self.config.temperature_fallback {
            params.set_temperature_inc(0.2);
        } else {
            params.set_temperature_inc(0.0);
        }
        if let Some(prompt) = self.config.effective_prompt() {
            params.set_initial_prompt(&prompt);
        }

        state
            .full(params, samples)
            .map_err(|e| AsrError::WhisperState {
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
        Ok(out)
    }

    /// Splits the audio into chunks based on VAD.
    fn plan_chunks(&mut self, audio: &MonoAudio) -> Result<Vec<Chunk>, AsrError> {
        #[cfg(feature = "vad")]
        {
            if let Some(vad) = self.vad.as_mut() {
                return plan_chunks_with_vad(
                    audio,
                    vad,
                    self.config.max_chunk_s,
                    self.config.overlap_s,
                );
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

/// A chunk of the input signal.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Chunk {
    /// Inclusive first sample.
    pub start_sample: usize,
    /// Exclusive last sample.
    pub end_sample: usize,
    /// Start time in seconds.
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
        // The DTW alignment pass is disabled until the alignment-heads
        // preset is wired. Requesting it costs decoder time and produces
        // no usable output.
        let cfg = TranscribeConfig::default();
        assert!(!cfg.use_dtw);
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
    }
}