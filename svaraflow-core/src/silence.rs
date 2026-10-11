//! Silence detection over a dBFS envelope.
//!
//! Two entry points:
//!
//! - [`detect_silence`] — the threshold method (spec section C). Uses the
//!   envelope alone.
//! - [`detect_silence_with_vad`] — the neural VAD and hybrid methods (spec
//!   section D).
//!
//! Both share the same post-processing pipeline: drop short talks, fill
//! short silences, pad and merge, then snap to the timeline frame grid.
//!
//! # Units
//!
//! User-facing durations are **integer milliseconds**. Internal math is
//! exact: milliseconds are converted to seconds only where the pipeline
//! needs them, and the conversion is `ms / 1000.0`.
//!
//! [`EnvelopeParams`]: crate::audio::EnvelopeParams

use crate::audio::Envelope;
use crate::error::CoreError;
use crate::timeline::Fps;
use crate::vad::{hybrid_snap, Vad, VadParams, DEFAULT_SNAP_RADIUS_S};

/// Hysteresis applied below the noise floor when a sound run ends, in dB.
pub const HYSTERESIS_DB: f32 = 3.0;

/// Safety floor. Values below this raise a warning but are accepted.
pub const MIN_SAFE_MS: u32 = 50;

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
    /// Silero VAD only.
    Vad,
    /// Silero VAD for the speech/silence decision, then boundary snapping.
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

/// Where the noise floor in a report came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoiseSource {
    /// Derived from the envelope or the VAD mask.
    Auto,
    /// Supplied by the user.
    Manual,
}

/// How silence boundaries snap to the timeline's frame grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SnapMode {
    /// Shrink inward. Default; never clips a word.
    Inward,
    /// Round to the nearest frame. Can clip a word at an internal boundary.
    Nearest,
}

/// The five built-in pacing presets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BuiltinPreset {
    /// Long, gentle cuts.
    Relaxed,
    /// Balanced. The default.
    #[default]
    Natural,
    /// Tighter cuts.
    Brisk,
    /// Tight cuts for dense speech.
    Tight,
    /// Maximum removal; uses [`SnapMode::Nearest`] by default.
    Max,
}

/// The numeric values a preset resolves to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PresetValues {
    /// Silence removal threshold, milliseconds.
    pub remove_silences_ms: u32,
    /// Minimum kept speech, milliseconds.
    pub keep_talks_ms: u32,
    /// Padding before each kept run, milliseconds.
    pub margin_before_ms: u32,
    /// Padding after each kept run, milliseconds.
    pub margin_after_ms: u32,
    /// Minimum gap between cuts. `0` disables the rule.
    pub min_between_cuts_ms: u32,
}

impl BuiltinPreset {
    /// Every built-in, in the order they print in.
    pub const ALL: [BuiltinPreset; 5] = [
        BuiltinPreset::Relaxed,
        BuiltinPreset::Natural,
        BuiltinPreset::Brisk,
        BuiltinPreset::Tight,
        BuiltinPreset::Max,
    ];

    /// Canonical lowercase name.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            BuiltinPreset::Relaxed => "relaxed",
            BuiltinPreset::Natural => "natural",
            BuiltinPreset::Brisk => "brisk",
            BuiltinPreset::Tight => "tight",
            BuiltinPreset::Max => "max",
        }
    }

    /// The preset's numeric values.
    #[must_use]
    pub fn values(self) -> PresetValues {
        match self {
            BuiltinPreset::Relaxed => PresetValues {
                remove_silences_ms: 500,
                keep_talks_ms: 250,
                margin_before_ms: 150,
                margin_after_ms: 150,
                min_between_cuts_ms: 0,
            },
            BuiltinPreset::Natural => PresetValues {
                remove_silences_ms: 300,
                keep_talks_ms: 150,
                margin_before_ms: 100,
                margin_after_ms: 100,
                min_between_cuts_ms: 0,
            },
            BuiltinPreset::Brisk => PresetValues {
                remove_silences_ms: 180,
                keep_talks_ms: 100,
                margin_before_ms: 80,
                margin_after_ms: 80,
                min_between_cuts_ms: 0,
            },
            BuiltinPreset::Tight => PresetValues {
                remove_silences_ms: 100,
                keep_talks_ms: 70,
                margin_before_ms: 60,
                margin_after_ms: 60,
                min_between_cuts_ms: 0,
            },
            BuiltinPreset::Max => PresetValues {
                remove_silences_ms: 50,
                keep_talks_ms: 50,
                margin_before_ms: 50,
                margin_after_ms: 50,
                min_between_cuts_ms: 0,
            },
        }
    }

    /// The snap mode this preset uses by default.
    #[must_use]
    pub fn default_snap(self) -> SnapMode {
        match self {
            BuiltinPreset::Max => SnapMode::Nearest,
            _ => SnapMode::Inward,
        }
    }

    /// Parses a name. Case-insensitive.
    pub fn parse(s: &str) -> Result<Self, String> {
        match s.to_ascii_lowercase().as_str() {
            "relaxed" => Ok(BuiltinPreset::Relaxed),
            "natural" => Ok(BuiltinPreset::Natural),
            "brisk" => Ok(BuiltinPreset::Brisk),
            "tight" => Ok(BuiltinPreset::Tight),
            "max" => Ok(BuiltinPreset::Max),
            _ => Err(format!(
                "unknown preset `{s}`; try relaxed, natural, brisk, tight, max"
            )),
        }
    }
}

/// Parameters for [`detect_silence`] and [`detect_silence_with_vad`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SilenceParams {
    /// Detection method.
    pub method: Method,
    /// Noise floor source.
    pub noise: NoiseSetting,
    /// Silence removal threshold, milliseconds.
    pub remove_silences_ms: u32,
    /// Minimum kept speech, milliseconds.
    pub keep_talks_ms: u32,
    /// Padding before each kept run, milliseconds.
    pub margin_before_ms: u32,
    /// Padding after each kept run, milliseconds.
    pub margin_after_ms: u32,
    /// Minimum gap between cuts, milliseconds. `0` disables the rule.
    pub min_between_cuts_ms: u32,
    /// Boundary snap mode.
    pub snap: SnapMode,
}

impl Default for SilenceParams {
    fn default() -> Self {
        Self::from_preset(BuiltinPreset::Natural)
    }
}

impl SilenceParams {
    /// Builds parameters from a built-in preset.
    #[must_use]
    pub fn from_preset(preset: BuiltinPreset) -> Self {
        let v = preset.values();
        Self {
            method: Method::Threshold,
            noise: NoiseSetting::Auto,
            remove_silences_ms: v.remove_silences_ms,
            keep_talks_ms: v.keep_talks_ms,
            margin_before_ms: v.margin_before_ms,
            margin_after_ms: v.margin_after_ms,
            min_between_cuts_ms: v.min_between_cuts_ms,
            snap: preset.default_snap(),
        }
    }

    /// Builds parameters from explicit values. Snap mode is
    /// [`SnapMode::Inward`].
    #[must_use]
    pub fn from_values(v: PresetValues) -> Self {
        Self {
            method: Method::Threshold,
            noise: NoiseSetting::Auto,
            remove_silences_ms: v.remove_silences_ms,
            keep_talks_ms: v.keep_talks_ms,
            margin_before_ms: v.margin_before_ms,
            margin_after_ms: v.margin_after_ms,
            min_between_cuts_ms: v.min_between_cuts_ms,
            snap: SnapMode::Inward,
        }
    }

    /// The numeric values this parameter set carries.
    #[must_use]
    pub fn values(&self) -> PresetValues {
        PresetValues {
            remove_silences_ms: self.remove_silences_ms,
            keep_talks_ms: self.keep_talks_ms,
            margin_before_ms: self.margin_before_ms,
            margin_after_ms: self.margin_after_ms,
            min_between_cuts_ms: self.min_between_cuts_ms,
        }
    }

    /// Warnings for values below [`MIN_SAFE_MS`].
    #[must_use]
    pub fn warnings(&self) -> Vec<SilenceWarning> {
        let mut out = Vec::new();
        if self.margin_before_ms < MIN_SAFE_MS {
            out.push(SilenceWarning::MarginBelowSafety {
                which: MarginSide::Before,
                value_ms: self.margin_before_ms,
            });
        }
        if self.margin_after_ms < MIN_SAFE_MS {
            out.push(SilenceWarning::MarginBelowSafety {
                which: MarginSide::After,
                value_ms: self.margin_after_ms,
            });
        }
        if self.remove_silences_ms < MIN_SAFE_MS {
            out.push(SilenceWarning::RemoveSilencesBelowSafety {
                value_ms: self.remove_silences_ms,
            });
        }
        out
    }
}

/// Which margin a [`SilenceWarning::MarginBelowSafety`] refers to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarginSide {
    /// `margin_before_ms`.
    Before,
    /// `margin_after_ms`.
    After,
}

/// Non-fatal warnings raised by the detector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SilenceWarning {
    /// Margin below [`MIN_SAFE_MS`].
    MarginBelowSafety {
        /// Which side.
        which: MarginSide,
        /// The offending value, in milliseconds.
        value_ms: u32,
    },
    /// `remove_silences_ms` below [`MIN_SAFE_MS`].
    RemoveSilencesBelowSafety {
        /// The offending value, in milliseconds.
        value_ms: u32,
    },
}

impl std::fmt::Display for SilenceWarning {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MarginBelowSafety { which, value_ms } => {
                let side = match which {
                    MarginSide::Before => "margin_before_ms",
                    MarginSide::After => "margin_after_ms",
                };
                write!(
                    f,
                    "{side} {value_ms} ms is below the {MIN_SAFE_MS} ms safety threshold"
                )
            }
            Self::RemoveSilencesBelowSafety { value_ms } => write!(
                f,
                "remove_silences_ms {value_ms} ms is below the {MIN_SAFE_MS} ms safety threshold"
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

/// One cut in the report.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CutDetail {
    /// Inclusive first timeline frame.
    pub start: i64,
    /// Exclusive last timeline frame.
    pub end: i64,
}

impl CutDetail {
    /// Frames removed by this cut.
    #[must_use]
    pub fn length_frames(&self) -> i64 {
        (self.end - self.start).max(0)
    }

    /// Milliseconds removed by this cut at `fps`.
    #[must_use]
    pub fn length_ms(&self, fps: Fps) -> u32 {
        (fps.frames_to_seconds(self.length_frames()) * 1000.0).round() as u32
    }
}

/// Explainability data for one analysis.
#[derive(Debug, Clone, PartialEq)]
pub struct AnalysisReport {
    /// Noise level used, in dBFS.
    pub noise_db: f32,
    /// Whether the noise level was auto-derived or supplied.
    pub noise_source: NoiseSource,
    /// Percentage of envelope frames strictly below `noise_db`.
    pub frames_below_noise_pct: f64,
    /// Silences found by detection, before `min_between_cuts`.
    pub silences_found: usize,
    /// Silences that survived to be cut.
    pub silences_cut: usize,
    /// Cuts prevented by `min_between_cuts_ms`. Two per merged keep.
    pub cuts_prevented: usize,
    /// Average length of a cut, in timeline frames.
    pub average_cut_frames: f64,
    /// Longest cut, in timeline frames.
    pub max_cut_frames: i64,
    /// Shortest cut, in timeline frames.
    pub shortest_cut_frames: i64,
    /// Up to five longest cuts, in descending length.
    pub longest_cuts: Vec<FrameRange>,
    /// Every cut, in timeline order.
    pub per_cut: Vec<CutDetail>,
}

impl AnalysisReport {
    /// An empty report.
    #[must_use]
    pub fn empty(noise_db: f32, noise_source: NoiseSource) -> Self {
        Self {
            noise_db,
            noise_source,
            frames_below_noise_pct: 0.0,
            silences_found: 0,
            silences_cut: 0,
            cuts_prevented: 0,
            average_cut_frames: 0.0,
            max_cut_frames: 0,
            shortest_cut_frames: 0,
            longest_cuts: Vec::new(),
            per_cut: Vec::new(),
        }
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
    /// Explainability data.
    pub report: AnalysisReport,
}

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/// Derives an automatic noise level from the envelope alone.
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
pub fn detect_silence(
    envelope: &Envelope,
    params: &SilenceParams,
    timeline_fps: Fps,
) -> Result<SilenceRanges, CoreError> {
    if params.method != Method::Threshold {
        return Err(CoreError::NotImplemented(
            "Method::Vad and Method::Hybrid require detect_silence_with_vad",
        ));
    }
    detect_impl(envelope, params, timeline_fps, None)
}

/// Detects keep and silence ranges using the neural VAD or the hybrid
/// method.
pub fn detect_silence_with_vad(
    envelope: &Envelope,
    params: &SilenceParams,
    timeline_fps: Fps,
    vad: &mut dyn Vad,
    samples: &[f32],
) -> Result<SilenceRanges, CoreError> {
    if params.method == Method::Threshold {
        return detect_silence(envelope, params, timeline_fps);
    }
    detect_impl(envelope, params, timeline_fps, Some((vad, samples)))
}

fn detect_impl(
    envelope: &Envelope,
    params: &SilenceParams,
    timeline_fps: Fps,
    vad_and_samples: Option<(&mut dyn Vad, &[f32])>,
) -> Result<SilenceRanges, CoreError> {
    let warnings = params.warnings();

    if envelope.is_empty() {
        let (noise_db, source) = resolve_noise_empty(params);
        return Ok(empty_result(noise_db, source, warnings));
    }

    let seconds_per_env_frame = envelope.seconds_per_frame();
    if seconds_per_env_frame <= 0.0 {
        return Err(CoreError::InvalidTimeline(
            "envelope has zero seconds per frame".into(),
        ));
    }

    let (mask, noise_db, noise_source) = match vad_and_samples {
        None => {
            let (noise_db, source) = match params.noise {
                NoiseSetting::Manual(v) => (v, NoiseSource::Manual),
                NoiseSetting::Auto => (auto_noise_level(envelope), NoiseSource::Auto),
            };
            let mask = sound_mask(&envelope.levels_db, noise_db, HYSTERESIS_DB);
            (mask, noise_db, source)
        }
        Some((vad, samples)) => {
            let probs = vad.speech_probabilities(samples)?;
            let native_mask = probs.into_mask(&VadParams::default());
            let resampled = native_mask.resample_to(seconds_per_env_frame, envelope.len());
            let sound = if params.method == Method::Hybrid {
                let radius = (DEFAULT_SNAP_RADIUS_S / seconds_per_env_frame).round() as usize;
                hybrid_snap(&resampled, &envelope.levels_db, radius)
            } else {
                resampled
            };
            let (noise_db, source) = match params.noise {
                NoiseSetting::Manual(v) => (v, NoiseSource::Manual),
                NoiseSetting::Auto => (
                    auto_noise_level_from_vad(envelope, &sound),
                    NoiseSource::Auto,
                ),
            };
            (sound, noise_db, source)
        }
    };

    let pct_below = frames_below_noise_pct(&envelope.levels_db, noise_db);
    let pipeline = finish_pipeline(mask, seconds_per_env_frame, params, timeline_fps);
    let report = build_report(&pipeline, noise_db, noise_source, pct_below, timeline_fps);

    Ok(SilenceRanges {
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
        report,
    })
}

fn resolve_noise_empty(params: &SilenceParams) -> (f32, NoiseSource) {
    match params.noise {
        NoiseSetting::Manual(v) => (v, NoiseSource::Manual),
        NoiseSetting::Auto => (AUTO_NOISE_MIN_DB, NoiseSource::Auto),
    }
}

fn empty_result(
    noise_db: f32,
    noise_source: NoiseSource,
    warnings: Vec<SilenceWarning>,
) -> SilenceRanges {
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
        noise_db,
        warnings,
        report: AnalysisReport::empty(noise_db, noise_source),
    }
}

/// Percentage of envelope frames strictly below `noise_db`.
fn frames_below_noise_pct(levels: &[f32], noise_db: f32) -> f64 {
    if levels.is_empty() {
        return 0.0;
    }
    let below = levels.iter().filter(|l| **l < noise_db).count();
    (below as f64 / levels.len() as f64) * 100.0
}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

struct Pipeline {
    keep: Vec<FrameRange>,
    silences: Vec<FrameRange>,
    total_frames: i64,
    kept_frames: i64,
    removed_frames: i64,
    cut_count: usize,
    percent_shorter: f64,
    silences_found: usize,
    cuts_prevented: usize,
}

fn finish_pipeline(
    mask: Vec<bool>,
    seconds_per_env_frame: f64,
    params: &SilenceParams,
    timeline_fps: Fps,
) -> Pipeline {
    let total_env_frames = mask.len() as i64;
    let mut runs = mask_to_runs(&mask);

    let keep_talks_frames = ms_to_env_frames(params.keep_talks_ms, seconds_per_env_frame);
    runs = flip_and_merge(&runs, true, keep_talks_frames);

    let remove_silences_frames = ms_to_env_frames(params.remove_silences_ms, seconds_per_env_frame);
    runs = flip_and_merge(&runs, false, remove_silences_frames);

    let sound_env: Vec<FrameRange> = runs
        .iter()
        .filter(|(is_sound, _, _)| *is_sound)
        .map(|(_, s, e)| FrameRange::new(*s, *e))
        .collect();

    let pad_before = ms_to_env_frames(params.margin_before_ms, seconds_per_env_frame);
    let pad_after = ms_to_env_frames(params.margin_after_ms, seconds_per_env_frame);
    let padded = pad_ranges(&sound_env, pad_before, pad_after, total_env_frames);
    let sound_env = merge_ranges(&padded);

    let silences_found = complement(&sound_env, total_env_frames).len();

    let min_between_frames = ms_to_env_frames(params.min_between_cuts_ms, seconds_per_env_frame);
    let (sound_env, cuts_prevented) =
        apply_min_between_cuts(&sound_env, total_env_frames, min_between_frames);

    let silence_env = complement(&sound_env, total_env_frames);

    let ratio = seconds_per_env_frame * timeline_fps.as_f64();
    let tl_total = ((total_env_frames as f64) * ratio).ceil() as i64;

    let silences_tl: Vec<FrameRange> = silence_env
        .iter()
        .filter_map(|s| snap_silence(*s, ratio, total_env_frames, tl_total, params.snap))
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
        silences_found,
        cuts_prevented,
    }
}

fn build_report(
    pipeline: &Pipeline,
    noise_db: f32,
    noise_source: NoiseSource,
    frames_below_noise_pct: f64,
    _fps: Fps,
) -> AnalysisReport {
    let per_cut: Vec<CutDetail> = pipeline
        .silences
        .iter()
        .map(|r| CutDetail {
            start: r.start,
            end: r.end,
        })
        .collect();

    let lengths: Vec<i64> = pipeline.silences.iter().map(FrameRange::len).collect();
    let n = lengths.len();
    let total: i64 = lengths.iter().sum();

    let average_cut_frames = if n > 0 { total as f64 / n as f64 } else { 0.0 };
    let max_cut_frames = lengths.iter().copied().max().unwrap_or(0);
    let shortest_cut_frames = lengths.iter().copied().min().unwrap_or(0);

    let mut sorted = pipeline.silences.clone();
    sorted.sort_by_key(|r| std::cmp::Reverse(r.len()));
    let longest_cuts: Vec<FrameRange> = sorted.into_iter().take(5).collect();

    AnalysisReport {
        noise_db,
        noise_source,
        frames_below_noise_pct,
        silences_found: pipeline.silences_found,
        silences_cut: n,
        cuts_prevented: pipeline.cuts_prevented,
        average_cut_frames,
        max_cut_frames,
        shortest_cut_frames,
        longest_cuts,
        per_cut,
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn ms_to_env_frames(ms: u32, seconds_per_frame: f64) -> i64 {
    if ms == 0 || seconds_per_frame <= 0.0 {
        return 0;
    }
    (f64::from(ms) / 1000.0 / seconds_per_frame).round() as i64
}

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

fn flip_and_merge(runs: &[(bool, i64, i64)], target: bool, min_len: i64) -> Vec<(bool, i64, i64)> {
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

/// Applies the min-between-cuts rule.
fn apply_min_between_cuts(
    keeps: &[FrameRange],
    total_env_frames: i64,
    min_frames: i64,
) -> (Vec<FrameRange>, usize) {
    if min_frames <= 0 || keeps.is_empty() || total_env_frames <= 0 {
        return (keeps.to_vec(), 0);
    }

    let n = total_env_frames as usize;
    let mut mask = vec![false; n];
    for r in keeps {
        let start = r.start.max(0) as usize;
        let end = (r.end.max(0) as usize).min(n);
        if start < end {
            mask[start..end].fill(true);
        }
    }

    let mut prevented = 0usize;
    loop {
        let runs = mask_to_runs(&mask);
        let mut merged = false;
        for i in 0..runs.len() {
            let (kind, s, e) = runs[i];
            if !kind {
                continue;
            }
            if (e - s) >= min_frames {
                continue;
            }
            if i == 0 || runs[i - 1].0 {
                continue;
            }
            if i + 1 >= runs.len() || runs[i + 1].0 {
                continue;
            }
            let before = runs[i - 1];
            let after = runs[i + 1];
            let before_start = before.1.max(0) as usize;
            let before_end = (before.2.max(0) as usize).min(n);
            if before_start < before_end {
                mask[before_start..before_end].fill(true);
            }
            let after_start = after.1.max(0) as usize;
            let after_end = (after.2.max(0) as usize).min(n);
            if after_start < after_end {
                mask[after_start..after_end].fill(true);
            }
            prevented += 2;
            merged = true;
            break;
        }
        if !merged {
            break;
        }
    }

    let out: Vec<FrameRange> = mask_to_runs(&mask)
        .into_iter()
        .filter(|(k, _, _)| *k)
        .map(|(_, s, e)| FrameRange::new(s, e))
        .collect();
    (out, prevented)
}

/// Snaps a silence range into timeline frames.
fn snap_silence(
    silence: FrameRange,
    ratio: f64,
    env_total: i64,
    tl_total: i64,
    mode: SnapMode,
) -> Option<FrameRange> {
    if ratio <= 0.0 {
        return None;
    }
    let start_f = silence.start as f64 * ratio;
    let end_f = silence.end as f64 * ratio;

    let tl_start = if silence.start <= 0 {
        0
    } else {
        match mode {
            SnapMode::Inward => start_f.ceil() as i64,
            SnapMode::Nearest => start_f.round() as i64,
        }
    };
    let tl_end = if silence.end >= env_total {
        tl_total
    } else {
        match mode {
            SnapMode::Inward => end_f.floor() as i64,
            SnapMode::Nearest => end_f.round() as i64,
        }
    };
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

    /// `SilenceParams::default()` with a manual noise floor.
    fn manual_params(noise: f32) -> SilenceParams {
        SilenceParams {
            noise: manual_noise(noise),
            ..SilenceParams::default()
        }
    }

    /// `SilenceParams::default()` with `Method::Vad` and a manual noise
    /// floor.
    fn vad_params() -> SilenceParams {
        SilenceParams {
            method: Method::Vad,
            noise: manual_noise(-50.0),
            ..SilenceParams::default()
        }
    }

    /// `SilenceParams::default()` with `Method::Hybrid` and a manual noise
    /// floor.
    fn hybrid_params() -> SilenceParams {
        SilenceParams {
            method: Method::Hybrid,
            noise: manual_noise(-50.0),
            ..SilenceParams::default()
        }
    }

    fn one_island(total: usize, sound_start: usize, sound_len: usize) -> Envelope {
        let mut levels = vec![-100.0f32; total];
        for l in levels.iter_mut().skip(sound_start).take(sound_len) {
            *l = -20.0;
        }
        env_from_levels(levels)
    }

    // --- preset table -----------------------------------------------------

    #[test]
    fn builtin_presets_match_documented_values() {
        let relaxed = BuiltinPreset::Relaxed.values();
        assert_eq!(relaxed.remove_silences_ms, 500);
        assert_eq!(relaxed.keep_talks_ms, 250);
        assert_eq!(relaxed.margin_before_ms, 150);
        assert_eq!(relaxed.margin_after_ms, 150);
        assert_eq!(relaxed.min_between_cuts_ms, 0);

        let natural = BuiltinPreset::Natural.values();
        assert_eq!(natural.remove_silences_ms, 300);
        assert_eq!(natural.keep_talks_ms, 150);
        assert_eq!(natural.margin_before_ms, 100);
        assert_eq!(natural.margin_after_ms, 100);
        assert_eq!(natural.min_between_cuts_ms, 0);

        let brisk = BuiltinPreset::Brisk.values();
        assert_eq!(brisk.remove_silences_ms, 180);
        assert_eq!(brisk.keep_talks_ms, 100);
        assert_eq!(brisk.margin_before_ms, 80);
        assert_eq!(brisk.margin_after_ms, 80);

        let tight = BuiltinPreset::Tight.values();
        assert_eq!(tight.remove_silences_ms, 100);
        assert_eq!(tight.keep_talks_ms, 70);
        assert_eq!(tight.margin_before_ms, 60);
        assert_eq!(tight.margin_after_ms, 60);

        let max = BuiltinPreset::Max.values();
        assert_eq!(max.remove_silences_ms, 50);
        assert_eq!(max.keep_talks_ms, 50);
        assert_eq!(max.margin_before_ms, 50);
        assert_eq!(max.margin_after_ms, 50);
    }

    #[test]
    fn max_preset_uses_nearest_snap_by_default() {
        assert_eq!(BuiltinPreset::Max.default_snap(), SnapMode::Nearest);
        assert_eq!(BuiltinPreset::Natural.default_snap(), SnapMode::Inward);
        let p = SilenceParams::from_preset(BuiltinPreset::Max);
        assert_eq!(p.snap, SnapMode::Nearest);
    }

    #[test]
    fn default_params_are_natural_preset() {
        let p = SilenceParams::default();
        assert_eq!(p.remove_silences_ms, 300);
        assert_eq!(p.keep_talks_ms, 150);
        assert_eq!(p.margin_before_ms, 100);
        assert_eq!(p.margin_after_ms, 100);
        assert_eq!(p.min_between_cuts_ms, 0);
        assert_eq!(p.snap, SnapMode::Inward);
    }

    #[test]
    fn default_builtin_preset_is_natural() {
        assert_eq!(BuiltinPreset::default(), BuiltinPreset::Natural);
    }

    // --- warnings ---------------------------------------------------------

    #[test]
    fn fifty_ms_is_accepted_without_warning() {
        let p = SilenceParams {
            remove_silences_ms: 50,
            keep_talks_ms: 50,
            margin_before_ms: 50,
            margin_after_ms: 50,
            ..SilenceParams::default()
        };
        assert!(p.warnings().is_empty(), "{:?}", p.warnings());
    }

    #[test]
    fn below_fifty_ms_warns() {
        let p = SilenceParams {
            remove_silences_ms: 49,
            keep_talks_ms: 50,
            margin_before_ms: 50,
            margin_after_ms: 50,
            ..SilenceParams::default()
        };
        let w = p.warnings();
        assert_eq!(w.len(), 1);
        assert!(matches!(
            w[0],
            SilenceWarning::RemoveSilencesBelowSafety { value_ms: 49 }
        ));

        let p = SilenceParams {
            remove_silences_ms: 50,
            keep_talks_ms: 50,
            margin_before_ms: 40,
            margin_after_ms: 30,
            ..SilenceParams::default()
        };
        let w = p.warnings();
        assert_eq!(w.len(), 2);
    }

    // --- snap modes -------------------------------------------------------

    #[test]
    fn nearest_and_inward_differ_on_a_fractional_boundary() {
        let ratio = 0.3;
        let silence = FrameRange::new(101, 200);
        let inward = snap_silence(silence, ratio, 1000, 60, SnapMode::Inward).unwrap();
        let nearest = snap_silence(silence, ratio, 1000, 60, SnapMode::Nearest).unwrap();
        assert_eq!(inward.start, 31);
        assert_eq!(nearest.start, 30);
        assert_ne!(inward.start, nearest.start);
    }

    // --- min_between_cuts -------------------------------------------------

    #[test]
    fn min_between_cuts_off_leaves_short_keeps() {
        let mut levels = vec![-100.0f32; 200];
        for l in levels.iter_mut().take(50).skip(30) {
            *l = -20.0;
        }
        for l in levels.iter_mut().take(100).skip(80) {
            *l = -20.0;
        }
        for l in levels.iter_mut().take(170).skip(140) {
            *l = -20.0;
        }
        let env = env_from_levels(levels);
        let p = SilenceParams {
            remove_silences_ms: 300,
            keep_talks_ms: 100,
            margin_before_ms: 0,
            margin_after_ms: 0,
            min_between_cuts_ms: 0,
            ..SilenceParams::default()
        };
        let r = detect_silence(&env, &p, tl_30()).unwrap();
        assert_eq!(r.report.cuts_prevented, 0);
        assert!(r.silences.len() >= 2);
    }

    #[test]
    fn min_between_cuts_merges_short_interior_keep() {
        let mut levels = vec![-100.0f32; 200];
        for l in levels.iter_mut().take(50).skip(30) {
            *l = -20.0;
        }
        for l in levels.iter_mut().take(100).skip(80) {
            *l = -20.0;
        }
        for l in levels.iter_mut().take(170).skip(140) {
            *l = -20.0;
        }
        let env = env_from_levels(levels);
        let p = SilenceParams {
            remove_silences_ms: 300,
            keep_talks_ms: 100,
            margin_before_ms: 0,
            margin_after_ms: 0,
            min_between_cuts_ms: 2000,
            ..SilenceParams::default()
        };
        let r = detect_silence(&env, &p, tl_30()).unwrap();
        assert!(r.report.cuts_prevented >= 4);
        assert!(r.silences.len() <= 1);
    }

    /// With the `max` preset (50 ms remove, 50 ms keep, 50 ms pad each
    /// side), a gap of 90 ms is not cut: padding of 50 ms on each side
    /// closes it during the merge step. A gap of 200 ms is cut: padding
    /// of 100 ms total leaves 100 ms of silence, which survives inward
    /// and nearest snapping as at least one timeline frame.
    #[test]
    fn max_preset_does_not_cut_a_90ms_gap_but_cuts_a_larger_gap() {
        let make = |gap_ms: u32| {
            let gap_frames = ((gap_ms as f64) / 10.0).round() as usize;
            let mut levels = vec![-20.0f32; 200];
            for l in levels.iter_mut().take(100 + gap_frames).skip(100) {
                *l = -100.0;
            }
            env_from_levels(levels)
        };
        let p = SilenceParams::from_preset(BuiltinPreset::Max);

        let r90 = detect_silence(&make(90), &p, tl_30()).unwrap();
        assert!(
            r90.silences.is_empty(),
            "90 ms gap should be closed by padding: {:?}",
            r90.silences
        );

        let r200 = detect_silence(&make(200), &p, tl_30()).unwrap();
        assert_eq!(r200.silences.len(), 1, "{:?}", r200.silences);
    }

    // --- detect_silence (threshold path) ---------------------------------

    #[test]
    fn detect_rejects_vad_and_hybrid_on_threshold_entry() {
        let env = one_island(200, 50, 50);
        let p_vad = vad_params();
        assert!(matches!(
            detect_silence(&env, &p_vad, tl_30()),
            Err(CoreError::NotImplemented(_))
        ));
        let p_hybrid = hybrid_params();
        assert!(matches!(
            detect_silence(&env, &p_hybrid, tl_30()),
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
        let p = manual_params(-50.0);
        let r = detect_silence(&env, &p, tl_30()).unwrap();
        assert!(r.keep.is_empty(), "keep = {:?}", r.keep);
        assert_eq!(r.silences.len(), 1);
    }

    #[test]
    fn long_enough_island_survives_keep_talks() {
        let env = one_island(200, 100, 30);
        let p = manual_params(-50.0);
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
        let p = manual_params(-50.0);
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
        let p = SilenceParams {
            noise: manual_noise(-50.0),
            remove_silences_ms: 100,
            margin_before_ms: 150,
            margin_after_ms: 150,
            ..SilenceParams::default()
        };
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
        let p = SilenceParams {
            noise: manual_noise(-50.0),
            remove_silences_ms: 0,
            keep_talks_ms: 0,
            margin_before_ms: 0,
            margin_after_ms: 0,
            ..SilenceParams::default()
        };
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
        let p = manual_params(-50.0);
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
        let p = manual_params(-50.0);
        let r = detect_silence(&env, &p, tl_30()).unwrap();

        let mut all: Vec<FrameRange> = r.keep.iter().chain(r.silences.iter()).copied().collect();
        all.sort_by_key(|x| x.start);
        assert_eq!(all[0].start, 0);
        let mut prev_end = 0i64;
        for range in &all {
            assert_eq!(range.start, prev_end, "gap or overlap at {range:?}");
            prev_end = range.end;
        }
        assert_eq!(prev_end, r.stats.total_frames);
    }

    // --- outer edges ------------------------------------------------------

    #[test]
    fn outer_silences_cover_the_full_timeline() {
        let env = one_island(299, 100, 100);
        let p = manual_params(-50.0);
        let r = detect_silence(&env, &p, tl_30()).unwrap();
        assert_eq!(r.stats.total_frames, 90);
        assert_eq!(r.silences[0].start, 0);
        assert_eq!(r.silences.last().unwrap().end, 90);
        let all: i64 = r.keep.iter().map(FrameRange::len).sum::<i64>()
            + r.silences.iter().map(FrameRange::len).sum::<i64>();
        assert_eq!(all, 90);
    }

    #[test]
    fn outer_edges_round_up_never_down() {
        let env = one_island(251, 100, 50);
        let p = manual_params(-50.0);
        let r = detect_silence(&env, &p, tl_30()).unwrap();
        assert_eq!(r.stats.total_frames, 76);
    }

    // --- detect_silence_with_vad -----------------------------------------

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
        fn speech_probabilities(
            &mut self,
            _samples: &[f32],
        ) -> Result<VadProbabilities, CoreError> {
            Ok(VadProbabilities {
                values: self.probs.clone(),
                hop_samples: VAD_WINDOW_SAMPLES,
                sample_rate: VAD_SAMPLE_RATE,
            })
        }
    }

    fn vad_with_speech_window(
        total_env_frames: usize,
        speech_start_s: f64,
        speech_end_s: f64,
    ) -> ScriptedVad {
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
        let p = manual_params(-50.0);
        let mut vad = ScriptedVad { probs: vec![] };
        let a = detect_silence(&env, &p, tl_30()).unwrap();
        let b = detect_silence_with_vad(&env, &p, tl_30(), &mut vad, &[]).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn vad_method_uses_mask_and_applies_padding() {
        let env = env_from_levels(vec![-70.0; 200]);
        let p = vad_params();
        let mut vad = vad_with_speech_window(200, 0.5, 1.5);

        let r = detect_silence_with_vad(&env, &p, tl_30(), &mut vad, &[]).unwrap();
        assert_eq!(r.stats.total_frames, 60);
        let kept: i64 = r.keep.iter().map(FrameRange::len).sum();
        let removed: i64 = r.silences.iter().map(FrameRange::len).sum();
        assert_eq!(kept + removed, 60);
        if let Some(range) = r.keep.first() {
            assert!(range.start >= 0);
            assert!(range.end <= 60);
        }
    }

    #[test]
    fn vad_method_marks_no_speech_as_all_silence() {
        let env = env_from_levels(vec![-70.0; 200]);
        let p = vad_params();
        let mut vad = ScriptedVad {
            probs: vec![0.0; 100],
        };
        let r = detect_silence_with_vad(&env, &p, tl_30(), &mut vad, &[]).unwrap();
        assert!(r.keep.is_empty());
        assert_eq!(r.silences.len(), 1);
    }

    #[test]
    fn hybrid_method_snaps_boundaries_to_dips() {
        let mut levels = vec![-70.0f32; 300];
        for l in levels.iter_mut().take(200).skip(100) {
            *l = -30.0;
        }
        levels[95] = -85.0;

        let env = env_from_levels(levels);
        let p = hybrid_params();
        let mut vad = vad_with_speech_window(300, 1.0, 2.0);

        let r = detect_silence_with_vad(&env, &p, tl_30(), &mut vad, &[]).unwrap();
        assert_eq!(r.keep.len(), 1);
        let snapped_boundary = r.keep[0].start;
        assert!(
            snapped_boundary <= 30,
            "boundary was not snapped earlier: {snapped_boundary}"
        );
    }

    #[test]
    fn vad_empty_envelope_returns_empty() {
        let env = env_from_levels(Vec::new());
        let p = vad_params();
        let mut vad = ScriptedVad { probs: vec![] };
        let r = detect_silence_with_vad(&env, &p, tl_30(), &mut vad, &[]).unwrap();
        assert!(r.keep.is_empty());
        assert!(r.silences.is_empty());
        assert_eq!(r.stats.total_frames, 0);
    }

    #[test]
    fn vad_auto_noise_uses_non_speech_frames() {
        let mut levels = vec![-60.0f32; 200];
        for l in levels.iter_mut().take(200).skip(100) {
            *l = -20.0;
        }
        let env = env_from_levels(levels);
        let p = SilenceParams {
            method: Method::Vad,
            ..SilenceParams::default()
        };
        let mut vad = vad_with_speech_window(200, 1.0, 2.0);

        let r = detect_silence_with_vad(&env, &p, tl_30(), &mut vad, &[]).unwrap();
        assert!(
            (r.noise_db - (-57.0)).abs() < 0.5,
            "noise_db was {}",
            r.noise_db
        );
    }
}
