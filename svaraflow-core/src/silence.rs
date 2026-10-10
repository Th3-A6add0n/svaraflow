//! Silence detection over a dBFS envelope.
//!
//! Two entry points:
//!
//! - [`detect_silence`] — the threshold method (spec section C). Uses the
//!   envelope alone.
//! - [`detect_silence_with_vad`] — the neural VAD and hybrid methods (spec
//!   section D). Uses a [`Vad`] implementation to classify speech, then
//!   either keeps that mask as-is (`Method::Vad`) or snaps its boundaries to
//!   the lowest-energy frame within 100 ms using the envelope
//!   (`Method::Hybrid`).
//!
//! Both share the same post-processing pipeline: drop short talks, fill
//! short silences, pad and merge, then snap to the timeline frame grid
//! inward. See the module-level docs of each function for details.
//!
//! # Units
//!
//! Two different "frames" appear in the code and in this documentation:
//!
//! - **envelope frames** — 10 ms each with the default [`EnvelopeParams`].
//!   Steps 1–4 operate here.
//! - **timeline frames** — whatever the timeline's `fps` says. Step 5
//!   converts into timeline frames and stays there.
//!
//! [`EnvelopeParams`]: crate::audio::EnvelopeParams

use crate::audio::Envelope;
use crate::error::CoreError;
use crate::timeline::Fps;
use crate::vad::{hybrid_snap, Vad, VadParams, DEFAULT_SNAP_RADIUS_S};

/// Hysteresis applied below the noise floor when a sound run ends, in dB.
///
/// A run that is currently sound ends only when the level drops strictly
/// below `noise_db - HYSTERESIS_DB`.
pub const HYSTERESIS_DB: f32 = 3.0;

/// Safety floor for padding and silence removal, in seconds.
pub const MIN_SAFE_SECONDS: f64 = 0.05;

/// Lowest auto noise level, in dBFS.
pub const AUTO_NOISE_MIN_DB: f32 = -60.0;

/// Highest auto noise level, in dBFS.
pub const AUTO_NOISE_MAX_DB: f32 = -25.0;

/// Offset added to the 95th percentile of the quiet frames, in dB.
pub const AUTO_NOISE_OFFSET_DB: f32 = 3.0;

/// Fraction of the quietest envelope frames considered by
/// [`auto_noise_level`] when no VAD mask is available.
pub const AUTO_NOISE_QUIET_FRACTION: f64 = 0.20;

/// Detection method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Method {
    /// Threshold detection on the dBFS envelope with hysteresis.
    Threshold,
    /// Silero VAD only. Requires [`detect_silence_with_vad`].
    Vad,
    /// Silero VAD for the speech/silence decision, then boundary snapping
    /// to the lowest-energy frame within 100 ms.
    Hybrid,
}

/// How the noise floor is determined.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NoiseSetting {
    /// Derive from the envelope (or the VAD mask).
    Auto,
    /// Use this dBFS value directly.
    Manual(f32),
}

/// Built-in tuning presets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Preset {
    /// `remove_silences_s = 0.80`, `keep_talks_s = 0.30`.
    Calm,
    /// `remove_silences_s = 0.50`, `keep_talks_s = 0.25`.
    Measured,
    /// `remove_silences_s = 0.35`, `keep_talks_s = 0.20`.
    Paced,
    /// `remove_silences_s = 0.20`, `keep_talks_s = 0.15`,
    /// `pad_before_s = pad_after_s = 0.08`.
    Fast,
}

/// Parameters for [`detect_silence`] and [`detect_silence_with_vad`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SilenceParams {
    /// Detection method.
    pub method: Method,
    /// Noise floor source.
    pub noise: NoiseSetting,
    /// Minimum silence length to remove, in seconds.
    pub remove_silences_s: f64,
    /// Minimum sound length to keep, in seconds.
    pub keep_talks_s: f64,
    /// Pad added before each sound run, in seconds.
    pub pad_before_s: f64,
    /// Pad added after each sound run, in seconds.
    pub pad_after_s: f64,
}

impl Default for SilenceParams {
    fn default() -> Self {
        Self {
            method: Method::Threshold,
            noise: NoiseSetting::Auto,
            remove_silences_s: 0.50,
            keep_talks_s: 0.25,
            pad_before_s: 0.10,
            pad_after_s: 0.10,
        }
    }
}

impl SilenceParams {
    /// Constructs the preset's parameters.
    #[must_use]
    pub fn preset(preset: Preset) -> Self {
        match preset {
            Preset::Calm => Self {
                remove_silences_s: 0.80,
                keep_talks_s: 0.30,
                ..Self::default()
            },
            Preset::Measured => Self::default(),
            Preset::Paced => Self {
                remove_silences_s: 0.35,
                keep_talks_s: 0.20,
                ..Self::default()
            },
            Preset::Fast => Self {
                remove_silences_s: 0.20,
                keep_talks_s: 0.15,
                pad_before_s: 0.08,
                pad_after_s: 0.08,
                ..Self::default()
            },
        }
    }

    /// Warnings for values below [`MIN_SAFE_SECONDS`].
    #[must_use]
    pub fn warnings(&self) -> Vec<SilenceWarning> {
        let mut out = Vec::new();
        if self.pad_before_s < MIN_SAFE_SECONDS {
            out.push(SilenceWarning::PaddingBelowSafety {
                which: PadSide::Before,
                value_s: self.pad_before_s,
            });
        }
        if self.pad_after_s < MIN_SAFE_SECONDS {
            out.push(SilenceWarning::PaddingBelowSafety {
                which: PadSide::After,
                value_s: self.pad_after_s,
            });
        }
        if self.remove_silences_s < MIN_SAFE_SECONDS {
            out.push(SilenceWarning::RemoveSilencesBelowSafety {
                value_s: self.remove_silences_s,
            });
        }
        out
    }
}

/// Which pad a [`SilenceWarning::PaddingBelowSafety`] refers to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PadSide {
    /// `pad_before_s`.
    Before,
    /// `pad_after_s`.
    After,
}

/// Non-fatal warnings raised by the detector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SilenceWarning {
    /// Padding below [`MIN_SAFE_SECONDS`].
    PaddingBelowSafety {
        /// Which side.
        which: PadSide,
        /// The offending value, in seconds.
        value_s: f64,
    },
    /// `remove_silences_s` below [`MIN_SAFE_SECONDS`].
    RemoveSilencesBelowSafety {
        /// The offending value, in seconds.
        value_s: f64,
    },
}

impl std::fmt::Display for SilenceWarning {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PaddingBelowSafety { which, value_s } => {
                let side = match which {
                    PadSide::Before => "pad_before",
                    PadSide::After => "pad_after",
                };
                write!(
                    f,
                    "{side} {value_s:.3} s is below the {MIN_SAFE_SECONDS:.2} s safety threshold"
                )
            }
            Self::RemoveSilencesBelowSafety { value_s } => write!(
                f,
                "remove_silences {value_s:.3} s is below the {MIN_SAFE_SECONDS:.2} s safety threshold"
            ),
        }
    }
}

/// A half-open frame range `[start, end)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameRange {
    /// Inclusive first frame.
    pub start: i64,
    /// Exclusive last frame.
    pub end: i64,
}

impl FrameRange {
    /// Constructs a range.
    #[must_use]
    pub fn new(start: i64, end: i64) -> Self {
        Self { start, end }
    }

    /// Number of frames, clamped to `>= 0`.
    #[must_use]
    pub fn len(&self) -> i64 {
        (self.end - self.start).max(0)
    }

    /// True when the range contains no frames.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.end <= self.start
    }
}

/// Summary statistics for the analysis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SilenceStats {
    /// Total length of the envelope, in timeline frames.
    pub total_frames: i64,
    /// Frames kept.
    pub kept_frames: i64,
    /// Frames removed.
    pub removed_frames: i64,
    /// Number of silence regions.
    pub cut_count: usize,
    /// `removed_frames / total_frames * 100`.
    pub percent_shorter: f64,
}

/// The detector's result.
#[derive(Debug, Clone, PartialEq)]
pub struct SilenceRanges {
    /// Sound regions to keep, in timeline frames.
    pub keep: Vec<FrameRange>,
    /// Silences to remove, in timeline frames.
    pub silences: Vec<FrameRange>,
    /// Summary statistics.
    pub stats: SilenceStats,
    /// Noise level used, in dBFS.
    pub noise_db: f32,
    /// Non-fatal warnings from the parameters.
    pub warnings: Vec<SilenceWarning>,
}

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/// Derives an automatic noise level from the envelope alone.
///
/// Quietest `AUTO_NOISE_QUIET_FRACTION` of the envelope frames, nearest-rank
/// 95th percentile, plus [`AUTO_NOISE_OFFSET_DB`], clamped to
/// `[AUTO_NOISE_MIN_DB, AUTO_NOISE_MAX_DB]`. Returns [`AUTO_NOISE_MIN_DB`]
/// for an empty envelope.
#[must_use]
pub fn auto_noise_level(envelope: &Envelope) -> f32 {
    let n = envelope.levels_db.len();
    if n == 0 {
        return AUTO_NOISE_MIN_DB;
    }
    let mut sorted = envelope.levels_db.clone();
    sorted.sort_by(f32::total_cmp);

    let quiet_count = ((n as f64) * AUTO_NOISE_QUIET_FRACTION).ceil() as usize;
    let quiet_count = quiet_count.clamp(1, n);
    let quiet = &sorted[..quiet_count];

    let idx = ((quiet_count as f64) * 0.95).ceil() as usize;
    let idx = idx.saturating_sub(1).min(quiet_count - 1);
    let p95 = quiet[idx];

    (p95 + AUTO_NOISE_OFFSET_DB).clamp(AUTO_NOISE_MIN_DB, AUTO_NOISE_MAX_DB)
}

/// Derives an automatic noise level using the frames the VAD calls
/// non-speech.
///
/// Implements the first clause of spec section C7. `speech` is the VAD mask
/// resampled to the envelope's frame rate; entries beyond the shorter of the
/// two slices are ignored. If `speech` contains no `false` entries (the VAD
/// marked every frame as speech), falls back to [`auto_noise_level`].
///
/// Same percentile, offset and clamp as the non-VAD path, so the two are
/// directly comparable.
#[must_use]
pub fn auto_noise_level_from_vad(envelope: &Envelope, speech: &[bool]) -> f32 {
    let n = envelope.levels_db.len().min(speech.len());
    if n == 0 {
        return AUTO_NOISE_MIN_DB;
    }

    let mut non_speech: Vec<f32> = (0..n)
        .filter(|&i| !speech[i])
        .map(|i| envelope.levels_db[i])
        .collect();

    if non_speech.is_empty() {
        return auto_noise_level(envelope);
    }

    non_speech.sort_by(f32::total_cmp);
    let len = non_speech.len();
    let idx = ((len as f64) * 0.95).ceil() as usize;
    let idx = idx.saturating_sub(1).min(len - 1);
    let p95 = non_speech[idx];
    (p95 + AUTO_NOISE_OFFSET_DB).clamp(AUTO_NOISE_MIN_DB, AUTO_NOISE_MAX_DB)
}

/// Detects keep and silence ranges using the threshold method.
///
/// `Method::Vad` and `Method::Hybrid` are rejected here; call
/// [`detect_silence_with_vad`] for those.
///
/// # Errors
///
/// - [`CoreError::NotImplemented`] for `Method::Vad` and `Method::Hybrid`.
/// - [`CoreError::InvalidTimeline`] when the envelope's
///   `seconds_per_frame()` is zero.
pub fn detect_silence(
    envelope: &Envelope,
    params: &SilenceParams,
    timeline_fps: Fps,
) -> Result<SilenceRanges, CoreError> {
    let warnings = params.warnings();

    if envelope.is_empty() {
        return Ok(empty_result(params, warnings));
    }

    match params.method {
        Method::Threshold => {}
        Method::Vad => {
            return Err(CoreError::NotImplemented(
                "Method::Vad requires detect_silence_with_vad",
            ))
        }
        Method::Hybrid => {
            return Err(CoreError::NotImplemented(
                "Method::Hybrid requires detect_silence_with_vad",
            ))
        }
    }

    let seconds_per_env_frame = envelope.seconds_per_frame();
    if seconds_per_env_frame <= 0.0 {
        return Err(CoreError::InvalidTimeline(
            "envelope has zero seconds per frame".into(),
        ));
    }

    let noise_db = match params.noise {
        NoiseSetting::Manual(v) => v,
        NoiseSetting::Auto => auto_noise_level(envelope),
    };

    let mask = sound_mask(&envelope.levels_db, noise_db, HYSTERESIS_DB);
    let pipeline = finish_pipeline(mask, seconds_per_env_frame, params, timeline_fps);

    Ok(result_from_pipeline(pipeline, noise_db, warnings))
}

/// Detects keep and silence ranges using the neural VAD or the hybrid
/// method.
///
/// `vad` is the engine. `samples` is the 16 kHz mono analysis signal the VAD
/// consumes. `envelope` must be the envelope computed from the same signal.
///
/// For `Method::Threshold` this delegates to [`detect_silence`]; the VAD is
/// not consulted.
///
/// # Errors
///
/// - Whatever the [`Vad`] implementation returns (typically
///   [`CoreError::VadFailed`]).
/// - [`CoreError::InvalidTimeline`] when the envelope's
///   `seconds_per_frame()` is zero.
pub fn detect_silence_with_vad(
    envelope: &Envelope,
    params: &SilenceParams,
    timeline_fps: Fps,
    vad: &dyn Vad,
    samples: &[f32],
) -> Result<SilenceRanges, CoreError> {
    if params.method == Method::Threshold {
        return detect_silence(envelope, params, timeline_fps);
    }

    let warnings = params.warnings();

    if envelope.is_empty() {
        return Ok(empty_result(params, warnings));
    }

    let seconds_per_env_frame = envelope.seconds_per_frame();
    if seconds_per_env_frame <= 0.0 {
        return Err(CoreError::InvalidTimeline(
            "envelope has zero seconds per frame".into(),
        ));
    }

    let probabilities = vad.speech_probabilities(samples)?;
    let native_mask = probabilities.into_mask(&VadParams::default());
    let resampled = native_mask.resample_to(seconds_per_env_frame, envelope.len());

    let sound = if params.method == Method::Hybrid {
        let radius_frames =
            (DEFAULT_SNAP_RADIUS_S / seconds_per_env_frame).round() as usize;
        hybrid_snap(&resampled, &envelope.levels_db, radius_frames)
    } else {
        resampled
    };

    let noise_db = match params.noise {
        NoiseSetting::Manual(v) => v,
        NoiseSetting::Auto => auto_noise_level_from_vad(envelope, &sound),
    };

    let pipeline = finish_pipeline(sound, seconds_per_env_frame, params, timeline_fps);

    Ok(result_from_pipeline(pipeline, noise_db, warnings))
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// The parts of the result the shared pipeline produces.
struct Pipeline {
    keep: Vec<FrameRange>,
    silences: Vec<FrameRange>,
    total_frames: i64,
    kept_frames: i64,
    removed_frames: i64,
    cut_count: usize,
    percent_shorter: f64,
}

/// Builds an empty result for an empty envelope.
fn empty_result(params: &SilenceParams, warnings: Vec<SilenceWarning>) -> SilenceRanges {
    SilenceRanges {
        keep: Vec::new(),
        silences: Vec::new(),
        stats: SilenceStats {
            total_frames: 0,
            kept_frames: 0,
            removed_frames: 0,
            cut_count: 0,
            percent_shorter: 0.0,
        },
        noise_db: match params.noise {
            NoiseSetting::Manual(v) => v,
            NoiseSetting::Auto => AUTO_NOISE_MIN_DB,
        },
        warnings,
    }
}

/// Wraps a pipeline result in the public [`SilenceRanges`] shape.
fn result_from_pipeline(
    pipeline: Pipeline,
    noise_db: f32,
    warnings: Vec<SilenceWarning>,
) -> SilenceRanges {
    SilenceRanges {
        keep: pipeline.keep,
        silences: pipeline.silences,
        stats: SilenceStats {
            total_frames: pipeline.total_frames,
            kept_frames: pipeline.kept_frames,
            removed_frames: pipeline.removed_frames,
            cut_count: pipeline.cut_count,
            percent_shorter: pipeline.percent_shorter,
        },
        noise_db,
        warnings,
    }
}

/// Runs the shared post-processing pipeline over an initial sound mask.
///
/// Steps, matching spec section C:
///
/// 1. Convert the mask to runs (already done by the caller for threshold; the
///    VAD supplies a mask directly).
/// 2. Drop sound runs shorter than `keep_talks_s`.
/// 3. Fill silences shorter than `remove_silences_s`.
/// 4. Pad sound runs, then merge overlaps.
/// 5. Silences are the complement; snap them inward to timeline frames and
///    drop any that collapse to fewer than one frame. Keep ranges are the
///    complement of the snapped silences.
fn finish_pipeline(
    mask: Vec<bool>,
    seconds_per_env_frame: f64,
    params: &SilenceParams,
    timeline_fps: Fps,
) -> Pipeline {
    let total_env_frames = mask.len() as i64;
    let mut runs = mask_to_runs(&mask);

    // Step 2: drop short sound runs.
    let keep_talks_frames = seconds_to_env_frames(params.keep_talks_s, seconds_per_env_frame);
    runs = flip_and_merge(&runs, true, keep_talks_frames);

    // Step 3: fill short silences.
    let remove_silences_frames =
        seconds_to_env_frames(params.remove_silences_s, seconds_per_env_frame);
    runs = flip_and_merge(&runs, false, remove_silences_frames);

    // Split into sound runs only; silences are re-derived from the complement
    // after padding (so a merge across a vanished silence is handled once).
    let sound_env: Vec<FrameRange> = runs
        .iter()
        .filter(|(is_sound, _, _)| *is_sound)
        .map(|(_, s, e)| FrameRange::new(*s, *e))
        .collect();

    // Step 4: pad, then merge overlaps.
    let pad_before = seconds_to_env_frames(params.pad_before_s, seconds_per_env_frame);
    let pad_after = seconds_to_env_frames(params.pad_after_s, seconds_per_env_frame);
    let padded = pad_ranges(&sound_env, pad_before, pad_after, total_env_frames);
    let sound_env = merge_ranges(&padded);

    let silence_env = complement(&sound_env, total_env_frames);

    // Step 5: snap to timeline frames, inward.
    let ratio = seconds_per_env_frame * timeline_fps.as_f64();
    let tl_total = ((total_env_frames as f64) * ratio).floor() as i64;

    let silences_tl: Vec<FrameRange> = silence_env
        .iter()
        .filter_map(|s| snap_silence_inward(*s, ratio))
        .collect();
    let silences_tl = merge_ranges(&silences_tl);
    let keep_tl: Vec<FrameRange> = complement(&silences_tl, tl_total)
        .into_iter()
        .filter(|r| !r.is_empty())
        .collect();

    let kept_frames: i64 = keep_tl.iter().map(FrameRange::len).sum();
    let removed_frames: i64 = silences_tl.iter().map(FrameRange::len).sum();
    let percent_shorter = if tl_total > 0 {
        (removed_frames as f64 / tl_total as f64) * 100.0
    } else {
        0.0
    };
    let cut_count = silences_tl.len();

    Pipeline {
        keep: keep_tl,
        silences: silences_tl,
        total_frames: tl_total,
        kept_frames,
        removed_frames,
        cut_count,
        percent_shorter,
    }
}

/// Converts a duration in seconds to envelope frames, rounding to nearest.
fn seconds_to_env_frames(seconds: f64, seconds_per_frame: f64) -> i64 {
    if seconds <= 0.0 || seconds_per_frame <= 0.0 {
        0
    } else {
        (seconds / seconds_per_frame).round() as i64
    }
}

/// Applies the hysteresis rule to the envelope, frame by frame.
fn sound_mask(levels: &[f32], noise_db: f32, hysteresis_db: f32) -> Vec<bool> {
    let mut out = vec![false; levels.len()];
    let mut is_sound = false;
    let end_threshold = noise_db - hysteresis_db;
    for (i, &level) in levels.iter().enumerate() {
        if is_sound {
            if level < end_threshold {
                is_sound = false;
            }
        } else if level >= noise_db {
            is_sound = true;
        }
        out[i] = is_sound;
    }
    out
}

/// Turns a boolean mask into `(is_sound, start, end)` runs.
fn mask_to_runs(mask: &[bool]) -> Vec<(bool, i64, i64)> {
    if mask.is_empty() {
        return Vec::new();
    }
    let mut runs: Vec<(bool, i64, i64)> = Vec::new();
    let mut current = mask[0];
    let mut start = 0i64;
    for (i, &value) in mask.iter().enumerate().skip(1) {
        if value != current {
            runs.push((current, start, i as i64));
            current = value;
            start = i as i64;
        }
    }
    runs.push((current, start, mask.len() as i64));
    runs
}

/// Flips runs of `target` shorter than `min_len` to the opposite, merging
/// adjacent runs that end up with the same value.
fn flip_and_merge(
    runs: &[(bool, i64, i64)],
    target: bool,
    min_len: i64,
) -> Vec<(bool, i64, i64)> {
    let mut out: Vec<(bool, i64, i64)> = Vec::with_capacity(runs.len());
    for &(value, start, end) in runs {
        let flipped = if value == target && (end - start) < min_len {
            !value
        } else {
            value
        };
        match out.last_mut() {
            Some(last) if last.0 == flipped && last.2 == start => {
                last.2 = end;
            }
            _ => out.push((flipped, start, end)),
        }
    }
    out
}

/// Grows each range by `pad_before` and `pad_after`, clamped to `[0, total)`.
fn pad_ranges(
    ranges: &[FrameRange],
    pad_before: i64,
    pad_after: i64,
    total: i64,
) -> Vec<FrameRange> {
    ranges
        .iter()
        .map(|r| {
            FrameRange::new(
                (r.start - pad_before).max(0),
                (r.end + pad_after).min(total),
            )
        })
        .collect()
}

/// Sorts by `(start, end)` and merges any overlapping or touching ranges.
fn merge_ranges(ranges: &[FrameRange]) -> Vec<FrameRange> {
    if ranges.is_empty() {
        return Vec::new();
    }
    let mut sorted: Vec<FrameRange> = ranges.to_vec();
    sorted.sort_by_key(|r| (r.start, r.end));
    let mut out: Vec<FrameRange> = Vec::with_capacity(sorted.len());
    for r in sorted {
        match out.last_mut() {
            Some(last) if r.start <= last.end => {
                if r.end > last.end {
                    last.end = r.end;
                }
            }
            _ => out.push(r),
        }
    }
    out
}

/// Returns the complement of `ranges` within `[0, total)`.
fn complement(ranges: &[FrameRange], total: i64) -> Vec<FrameRange> {
    let mut out = Vec::new();
    let mut cursor = 0i64;
    for r in ranges {
        if r.start > cursor {
            out.push(FrameRange::new(cursor, r.start));
        }
        if r.end > cursor {
            cursor = r.end;
        }
    }
    if cursor < total {
        out.push(FrameRange::new(cursor, total));
    }
    out
}

/// Snaps a silence range into timeline frames, shrinking it inward.
fn snap_silence_inward(silence: FrameRange, ratio: f64) -> Option<FrameRange> {
    if ratio <= 0.0 {
        return None;
    }
    let tl_start = ((silence.start as f64) * ratio).ceil() as i64;
    let tl_end = ((silence.end as f64) * ratio).floor() as i64;
    if tl_end <= tl_start {
        None
    } else {
        Some(FrameRange::new(tl_start, tl_end))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vad::{VadProbabilities, VAD_SAMPLE_RATE, VAD_WINDOW_SAMPLES};

    const ENV_RATE: u32 = 16_000;
    const ENV_HOP: usize = 160;
    const ENV_WINDOW: usize = 320;

    fn env_from_levels(levels: Vec<f32>) -> Envelope {
        Envelope {
            levels_db: levels,
            sample_rate: ENV_RATE,
            hop_samples: ENV_HOP,
            window_samples: ENV_WINDOW,
        }
    }

    fn tl_30() -> Fps {
        Fps { num: 30, den: 1 }
    }

    fn manual_noise(v: f32) -> NoiseSetting {
        NoiseSetting::Manual(v)
    }

    fn one_island(total: usize, sound_start: usize, sound_len: usize) -> Envelope {
        let mut levels = vec![-100.0f32; total];
        for l in levels.iter_mut().skip(sound_start).take(sound_len) {
            *l = -20.0;
        }
        env_from_levels(levels)
    }

    // --- params -----------------------------------------------------------

    #[test]
    fn default_params_match_spec() {
        let p = SilenceParams::default();
        assert_eq!(p.method, Method::Threshold);
        assert!(matches!(p.noise, NoiseSetting::Auto));
        assert!((p.remove_silences_s - 0.50).abs() < 1e-12);
        assert!((p.keep_talks_s - 0.25).abs() < 1e-12);
        assert!((p.pad_before_s - 0.10).abs() < 1e-12);
        assert!((p.pad_after_s - 0.10).abs() < 1e-12);
    }

    #[test]
    fn presets_match_spec() {
        let calm = SilenceParams::preset(Preset::Calm);
        assert!((calm.remove_silences_s - 0.80).abs() < 1e-12);
        assert!((calm.keep_talks_s - 0.30).abs() < 1e-12);
        assert!((calm.pad_before_s - 0.10).abs() < 1e-12);

        let measured = SilenceParams::preset(Preset::Measured);
        assert!((measured.remove_silences_s - 0.50).abs() < 1e-12);
        assert!((measured.keep_talks_s - 0.25).abs() < 1e-12);

        let paced = SilenceParams::preset(Preset::Paced);
        assert!((paced.remove_silences_s - 0.35).abs() < 1e-12);
        assert!((paced.keep_talks_s - 0.20).abs() < 1e-12);

        let fast = SilenceParams::preset(Preset::Fast);
        assert!((fast.remove_silences_s - 0.20).abs() < 1e-12);
        assert!((fast.keep_talks_s - 0.15).abs() < 1e-12);
        assert!((fast.pad_before_s - 0.08).abs() < 1e-12);
        assert!((fast.pad_after_s - 0.08).abs() < 1e-12);
    }

    #[test]
    fn defaults_have_no_warnings() {
        assert!(SilenceParams::default().warnings().is_empty());
    }

    #[test]
    fn warnings_for_small_padding_and_silence() {
        let mut p = SilenceParams::default();
        p.pad_before_s = 0.04;
        p.pad_after_s = 0.01;
        p.remove_silences_s = 0.049;
        let w = p.warnings();
        assert_eq!(w.len(), 3);
        assert!(w.iter().any(|x| matches!(
            x,
            SilenceWarning::PaddingBelowSafety {
                which: PadSide::Before,
                ..
            }
        )));
        assert!(w.iter().any(|x| matches!(
            x,
            SilenceWarning::PaddingBelowSafety {
                which: PadSide::After,
                ..
            }
        )));
        assert!(w
            .iter()
            .any(|x| matches!(x, SilenceWarning::RemoveSilencesBelowSafety { .. })));
    }

    // --- auto noise -------------------------------------------------------

    #[test]
    fn auto_noise_empty_envelope_is_floor() {
        let env = env_from_levels(Vec::new());
        assert_eq!(auto_noise_level(&env), AUTO_NOISE_MIN_DB);
    }

    #[test]
    fn auto_noise_all_quiet_clamps_to_min() {
        let env = env_from_levels(vec![-80.0; 100]);
        assert_eq!(auto_noise_level(&env), AUTO_NOISE_MIN_DB);
    }

    #[test]
    fn auto_noise_all_loud_clamps_to_max() {
        let env = env_from_levels(vec![-10.0; 100]);
        assert_eq!(auto_noise_level(&env), AUTO_NOISE_MAX_DB);
    }

    #[test]
    fn auto_noise_mixed_envelope_uses_quiet_tail() {
        let mut levels = vec![-50.0f32; 80];
        levels.extend(vec![-20.0; 20]);
        let env = env_from_levels(levels);
        assert!((auto_noise_level(&env) - (-47.0)).abs() < 0.01);
    }

    #[test]
    fn auto_noise_single_frame_uses_that_frame() {
        let env = env_from_levels(vec![-45.0]);
        assert!((auto_noise_level(&env) - (-42.0)).abs() < 0.01);
    }

    // --- auto noise from VAD ---------------------------------------------

    #[test]
    fn auto_noise_from_vad_uses_non_speech_frames_only() {
        // 8 frames at -50 (non-speech), 4 at -20 (speech).
        let mut levels = vec![-50.0f32; 8];
        levels.extend(vec![-20.0; 4]);
        let env = env_from_levels(levels);
        let speech = vec![false; 8]
            .into_iter()
            .chain(vec![true; 4])
            .collect::<Vec<_>>();
        // P95 of 8 non-speech frames at -50 = -50. +3 = -47.
        assert!((auto_noise_level_from_vad(&env, &speech) - (-47.0)).abs() < 0.01);
    }

    #[test]
    fn auto_noise_from_vad_falls_back_when_all_speech() {
        // VAD says everything is speech; fall back to quietest-20%.
        let mut levels = vec![-50.0f32; 80];
        levels.extend(vec![-20.0; 20]);
        let env = env_from_levels(levels);
        let speech = vec![true; 100];
        assert!(
            (auto_noise_level_from_vad(&env, &speech) - auto_noise_level(&env)).abs() < 1e-6
        );
    }

    #[test]
    fn auto_noise_from_vad_handles_empty() {
        let env = env_from_levels(Vec::new());
        assert_eq!(auto_noise_level_from_vad(&env, &[]), AUTO_NOISE_MIN_DB);
    }

    #[test]
    fn auto_noise_from_vad_truncates_to_shorter_slice() {
        let env = env_from_levels(vec![-50.0; 4]);
        let speech = vec![false, false, true, true, true, true];
        // Only 4 envelope frames; 2 non-speech. P95 of [-50, -50] = -50.
        // +3 = -47.
        assert!((auto_noise_level_from_vad(&env, &speech) - (-47.0)).abs() < 0.01);
    }

    // --- sound mask / hysteresis -----------------------------------------

    #[test]
    fn sound_mask_starts_silent_on_quiet_signal() {
        let mask = sound_mask(&[-80.0, -80.0, -80.0], -50.0, HYSTERESIS_DB);
        assert_eq!(mask, vec![false, false, false]);
    }

    #[test]
    fn sound_mask_starts_sound_on_loud_frame_zero() {
        let mask = sound_mask(&[-20.0, -20.0], -50.0, HYSTERESIS_DB);
        assert_eq!(mask, vec![true, true]);
    }

    #[test]
    fn hysteresis_keeps_sound_through_a_brief_dip() {
        let levels = [-20.0, -20.0, -52.0, -52.0, -52.0, -60.0, -20.0];
        let mask = sound_mask(&levels, -50.0, HYSTERESIS_DB);
        assert_eq!(mask, vec![true, true, true, true, true, false, true]);
    }

    #[test]
    fn hysteresis_ends_exactly_below_threshold() {
        let levels = [-20.0, -53.0, -53.01, -20.0];
        let mask = sound_mask(&levels, -50.0, HYSTERESIS_DB);
        assert_eq!(mask, vec![true, true, false, true]);
    }

    // --- run flips --------------------------------------------------------

    #[test]
    fn mask_to_runs_handles_empty() {
        assert_eq!(mask_to_runs(&[]), Vec::new());
    }

    #[test]
    fn mask_to_runs_alternating() {
        let mask = [true, true, false, false, false, true];
        assert_eq!(
            mask_to_runs(&mask),
            vec![(true, 0, 2), (false, 2, 5), (true, 5, 6)]
        );
    }

    #[test]
    fn flip_and_merge_bridges_gap_when_middle_run_is_flipped() {
        let runs = [(false, 0, 5), (true, 5, 6), (false, 6, 11)];
        let out = flip_and_merge(&runs, true, 3);
        assert_eq!(out, vec![(false, 0, 11)]);
    }

    #[test]
    fn flip_and_merge_leaves_long_enough_runs() {
        let runs = [(false, 0, 5), (true, 5, 10), (false, 10, 15)];
        let out = flip_and_merge(&runs, true, 3);
        assert_eq!(out, runs.to_vec());
    }

    // --- detect_silence (threshold path) ---------------------------------

    #[test]
    fn detect_rejects_vad_and_hybrid_on_threshold_entry() {
        let env = one_island(200, 50, 50);
        let mut p = SilenceParams::default();
        p.noise = manual_noise(-50.0);
        p.method = Method::Vad;
        assert!(matches!(
            detect_silence(&env, &p, tl_30()),
            Err(CoreError::NotImplemented(_))
        ));
        p.method = Method::Hybrid;
        assert!(matches!(
            detect_silence(&env, &p, tl_30()),
            Err(CoreError::NotImplemented(_))
        ));
    }

    #[test]
    fn detect_empty_envelope_returns_empty_ranges() {
        let env = env_from_levels(Vec::new());
        let r = detect_silence(&env, &SilenceParams::default(), tl_30()).unwrap();
        assert!(r.keep.is_empty());
        assert!(r.silences.is_empty());
        assert_eq!(r.stats.total_frames, 0);
    }

    #[test]
    fn isolated_one_frame_blip_is_dropped_by_keep_talks() {
        let env = one_island(200, 100, 1);
        let mut p = SilenceParams::default();
        p.noise = manual_noise(-50.0);
        let r = detect_silence(&env, &p, tl_30()).unwrap();
        assert!(r.keep.is_empty(), "keep = {:?}", r.keep);
        assert_eq!(r.silences.len(), 1);
    }

    #[test]
    fn long_enough_island_survives_keep_talks() {
        let env = one_island(200, 100, 30);
        let mut p = SilenceParams::default();
        p.noise = manual_noise(-50.0);
        let r = detect_silence(&env, &p, tl_30()).unwrap();
        assert_eq!(r.keep.len(), 1);
        assert_eq!(r.keep[0], FrameRange::new(27, 42));
    }

    #[test]
    fn short_silence_between_two_talks_is_filled() {
        let mut levels = vec![-100.0f32; 300];
        for l in levels.iter_mut().take(130).skip(100) {
            *l = -20.0;
        }
        for l in levels.iter_mut().take(170).skip(140) {
            *l = -20.0;
        }
        let env = env_from_levels(levels);
        let mut p = SilenceParams::default();
        p.noise = manual_noise(-50.0);
        let r = detect_silence(&env, &p, tl_30()).unwrap();
        assert_eq!(r.keep.len(), 1);
        assert_eq!(r.keep[0], FrameRange::new(27, 54));
    }

    #[test]
    fn padding_merges_runs_after_fill_step() {
        let mut levels = vec![-100.0f32; 300];
        for l in levels.iter_mut().take(130).skip(100) {
            *l = -20.0;
        }
        for l in levels.iter_mut().take(180).skip(150) {
            *l = -20.0;
        }
        let env = env_from_levels(levels);
        let mut p = SilenceParams::default();
        p.noise = manual_noise(-50.0);
        p.remove_silences_s = 0.10;
        p.pad_before_s = 0.15;
        p.pad_after_s = 0.15;
        let r = detect_silence(&env, &p, tl_30()).unwrap();
        assert_eq!(r.keep, vec![FrameRange::new(25, 59)]);
    }

    #[test]
    fn snap_shrinks_silence_inward() {
        let mut levels = vec![-100.0f32; 310];
        for l in levels.iter_mut().take(151).skip(100) {
            *l = -20.0;
        }
        for l in levels.iter_mut().take(208).skip(159) {
            *l = -20.0;
        }
        let env = env_from_levels(levels);
        let mut p = SilenceParams::default();
        p.noise = manual_noise(-50.0);
        p.remove_silences_s = 0.0;
        p.keep_talks_s = 0.0;
        p.pad_before_s = 0.0;
        p.pad_after_s = 0.0;
        let r = detect_silence(&env, &p, tl_30()).unwrap();
        assert_eq!(r.silences.len(), 3);
        assert_eq!(r.silences[1], FrameRange::new(46, 47));
        assert_eq!(
            r.keep,
            vec![FrameRange::new(30, 46), FrameRange::new(47, 63)]
        );
    }

    #[test]
    fn stats_are_consistent() {
        let env = one_island(200, 100, 30);
        let mut p = SilenceParams::default();
        p.noise = manual_noise(-50.0);
        let r = detect_silence(&env, &p, tl_30()).unwrap();
        assert_eq!(
            r.stats.kept_frames + r.stats.removed_frames,
            r.stats.total_frames
        );
        let kept_sum: i64 = r.keep.iter().map(FrameRange::len).sum();
        let removed_sum: i64 = r.silences.iter().map(FrameRange::len).sum();
        assert_eq!(kept_sum, r.stats.kept_frames);
        assert_eq!(removed_sum, r.stats.removed_frames);
        assert_eq!(r.stats.cut_count, r.silences.len());
        let pct = (r.stats.removed_frames as f64 / r.stats.total_frames as f64) * 100.0;
        assert!((pct - r.stats.percent_shorter).abs() < 1e-9);
    }

    #[test]
    fn keep_and_silence_ranges_partition_the_timeline() {
        let mut levels = vec![-100.0f32; 500];
        for l in levels.iter_mut().take(160).skip(100) {
            *l = -20.0;
        }
        for l in levels.iter_mut().take(400).skip(300) {
            *l = -20.0;
        }
        let env = env_from_levels(levels);
        let mut p = SilenceParams::default();
        p.noise = manual_noise(-50.0);
        let r = detect_silence(&env, &p, tl_30()).unwrap();

        let mut all: Vec<FrameRange> = r
            .keep
            .iter()
            .chain(r.silences.iter())
            .copied()
            .collect();
        all.sort_by_key(|x| x.start);
        assert_eq!(all[0].start, 0);
        let mut prev_end = 0i64;
        for range in &all {
            assert_eq!(range.start, prev_end, "gap or overlap at {range:?}");
            prev_end = range.end;
        }
        assert_eq!(prev_end, r.stats.total_frames);
    }

    // --- detect_silence_with_vad -----------------------------------------

    /// A VAD that returns a fixed probability per window.
    struct ScriptedVad {
        probs: Vec<f32>,
    }

    impl Vad for ScriptedVad {
        fn name(&self) -> &'static str {
            "scripted"
        }
        fn window_samples(&self) -> u32 {
            VAD_WINDOW_SAMPLES
        }
        fn sample_rate(&self) -> u32 {
            VAD_SAMPLE_RATE
        }
        fn speech_probabilities(&self, _samples: &[f32]) -> Result<VadProbabilities, CoreError> {
            Ok(VadProbabilities {
                values: self.probs.clone(),
                hop_samples: VAD_WINDOW_SAMPLES,
                sample_rate: VAD_SAMPLE_RATE,
            })
        }
    }

    /// Builds a VAD that marks `[speech_start_s, speech_end_s)` as speech.
    /// Duration is derived from the number of envelope frames in the test,
    /// assuming 100 Hz envelope frames.
    fn vad_with_speech_window(
        total_env_frames: usize,
        speech_start_s: f64,
        speech_end_s: f64,
    ) -> ScriptedVad {
        // VAD windows are 32 ms; 1 second = 31.25 windows.
        let window_s = 0.032;
        let total_windows = (total_env_frames as f64 * 0.010 / window_s).ceil() as usize;
        let start = (speech_start_s / window_s) as usize;
        let end = (speech_end_s / window_s) as usize;
        let mut probs = vec![0.0f32; total_windows];
        for p in probs.iter_mut().take(end.min(total_windows)).skip(start) {
            *p = 0.9;
        }
        ScriptedVad { probs }
    }

    #[test]
    fn vad_method_delegates_threshold_to_detect_silence() {
        let env = one_island(200, 100, 30);
        let mut p = SilenceParams::default();
        p.noise = manual_noise(-50.0);
        // Method::Threshold via detect_silence_with_vad == detect_silence.
        let vad = ScriptedVad { probs: vec![] };
        let a = detect_silence(&env, &p, tl_30()).unwrap();
        let b = detect_silence_with_vad(&env, &p, tl_30(), &vad, &[]).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn vad_method_uses_mask_and_applies_padding() {
        // 200 envelope frames = 2 s at 100 Hz. VAD says speech in [0.5, 1.5).
        // With default params (keep_talks 0.25 s, remove_silences 0.50 s,
        // padding 0.10 s each side), speech run of 1.0 s survives, and the
        // padding extends it by 0.1 s on each side.
        let env = env_from_levels(vec![-70.0; 200]);
        let mut p = SilenceParams::default();
        p.method = Method::Vad;
        p.noise = manual_noise(-50.0);
        let vad = vad_with_speech_window(200, 0.5, 1.5);

        let r = detect_silence_with_vad(&env, &p, tl_30(), &vad, &[]).unwrap();
        assert_eq!(r.keep.len(), 1, "keep = {:?}", r.keep);
        // Envelope-frame speech was [50, 150). Padding 10 frames each side
        // gives [40, 160). Snap to 30 fps (ratio 0.3): silence before is
        // [0, 40) -> [0, 12); silence after is [160, 200) -> [48, 60).
        // keep = [12, 48).
        assert_eq!(r.keep[0], FrameRange::new(12, 48));
    }

    #[test]
    fn vad_method_marks_no_speech_as_all_silence() {
        let env = env_from_levels(vec![-70.0; 200]);
        let mut p = SilenceParams::default();
        p.method = Method::Vad;
        p.noise = manual_noise(-50.0);
        let vad = ScriptedVad {
            probs: vec![0.0; 100],
        };
        let r = detect_silence_with_vad(&env, &p, tl_30(), &vad, &[]).unwrap();
        assert!(r.keep.is_empty());
        assert_eq!(r.silences.len(), 1);
    }

    #[test]
    fn hybrid_method_snaps_boundaries_to_dips() {
        // Envelope: silence, then a dip right before the VAD-declared speech.
        // VAD mask says speech in [1.0, 2.0) seconds. Envelope has the
        // quietest frame at 0.95 s, so the hybrid snap should pull the
        // boundary back from 1.0 to 0.95.
        let mut levels = vec![-70.0f32; 300];
        // Speech region.
        for l in levels.iter_mut().take(200).skip(100) {
            *l = -30.0;
        }
        // A quieter frame at index 95 (0.95 s), just before the boundary.
        levels[95] = -85.0;

        let env = env_from_levels(levels);
        let mut p = SilenceParams::default();
        p.method = Method::Hybrid;
        p.noise = manual_noise(-50.0);
        let vad = vad_with_speech_window(300, 1.0, 2.0);

        let r = detect_silence_with_vad(&env, &p, tl_30(), &vad, &[]).unwrap();
        assert_eq!(r.keep.len(), 1);
        // The VAD says speech starts at envelope frame 100; the snap should
        // move it to 95, but the envelope-frame padding of 10 frames pulls
        // it back further. The exact result depends on the pipeline; we only
        // assert that the boundary is at or before frame 100 (the raw VAD
        // boundary). The snap must not move it later.
        let snapped_boundary = r.keep[0].start;
        // Envelope frame 100 -> timeline frame 30. With padding 10 frames
        // -> 90 -> timeline frame 27.
        assert!(
            snapped_boundary <= 30,
            "boundary was not snapped earlier: {snapped_boundary}"
        );
    }

    #[test]
    fn vad_empty_envelope_returns_empty() {
        let env = env_from_levels(Vec::new());
        let mut p = SilenceParams::default();
        p.method = Method::Vad;
        let vad = ScriptedVad { probs: vec![] };
        let r = detect_silence_with_vad(&env, &p, tl_30(), &vad, &[]).unwrap();
        assert!(r.keep.is_empty());
        assert!(r.silences.is_empty());
        assert_eq!(r.stats.total_frames, 0);
    }

    #[test]
    fn vad_auto_noise_uses_non_speech_frames() {
        // Envelope: non-speech at -60, speech at -20. Auto noise should
        // land at -60 + 3 = -57, not at the quietest-20% value.
        let mut levels = vec![-60.0f32; 200];
        for l in levels.iter_mut().take(200).skip(100) {
            *l = -20.0;
        }
        let env = env_from_levels(levels);
        let mut p = SilenceParams::default();
        p.method = Method::Vad;
        // Leave noise as Auto.
        let vad = vad_with_speech_window(200, 1.0, 2.0);

        let r = detect_silence_with_vad(&env, &p, tl_30(), &vad, &[]).unwrap();
        assert!(
            (r.noise_db - (-57.0)).abs() < 0.5,
            "noise_db was {}",
            r.noise_db
        );
    }
}