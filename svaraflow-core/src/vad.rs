//! Engine-agnostic half of the neural VAD described in section D of the spec.
//!
//! This module defines the mask type, the parameters, the [`Vad`] trait, and
//! the hybrid boundary-snap rule. It has **no ONNX dependency** and compiles
//! with the base crate. The concrete Silero implementation lives in
//! `svaraflow-asr`.
//!
//! # Data flow
//!
//! 1. A [`Vad`] implementation produces one probability per fixed-size window
//!    (Silero: 512 samples at 16 kHz, i.e. 32 ms windows).
//! 2. [`VadProbabilities::into_mask`] thresholds at `prob_threshold`, then
//!    applies min-speech, min-silence and speech-pad to the binary mask.
//! 3. [`VadMask::resample_to`] converts the mask to the envelope's frame
//!    rate for use by [`crate::silence::detect_silence_with_vad`].
//! 4. [`hybrid_snap`] refines each boundary in the resampled mask to the
//!    lowest-energy frame within [`DEFAULT_SNAP_RADIUS_S`] (100 ms), using
//!    the dBFS envelope as the energy signal.

use crate::error::CoreError;
use crate::silence::FrameRange;

/// Samples per Silero VAD window.
pub const VAD_WINDOW_SAMPLES: u32 = 512;

/// Sample rate Silero expects.
pub const VAD_SAMPLE_RATE: u32 = 16_000;

/// Probability threshold for speech. Values at or above are speech.
pub const DEFAULT_PROB_THRESHOLD: f32 = 0.5;

/// Minimum speech run to keep, in seconds.
pub const DEFAULT_MIN_SPEECH_S: f64 = 0.250;

/// Minimum silence run to keep, in seconds.
pub const DEFAULT_MIN_SILENCE_S: f64 = 0.100;

/// Speech padding applied on both sides of each run, in seconds.
pub const DEFAULT_SPEECH_PAD_S: f64 = 0.030;

/// Radius used by [`hybrid_snap`] to search for a boundary, in seconds.
pub const DEFAULT_SNAP_RADIUS_S: f64 = 0.100;

/// One probability per VAD window, before thresholding.
#[derive(Debug, Clone, PartialEq)]
pub struct VadProbabilities {
    /// Probabilities in `[0, 1]`.
    pub values: Vec<f32>,
    /// Samples between window starts.
    pub hop_samples: u32,
    /// Sample rate the probabilities were computed at.
    pub sample_rate: u32,
}

impl VadProbabilities {
    /// Seconds per window.
    #[must_use]
    pub fn seconds_per_window(&self) -> f64 {
        if self.sample_rate == 0 {
            0.0
        } else {
            f64::from(self.hop_samples) / f64::from(self.sample_rate)
        }
    }

    /// Thresholds at `params.prob_threshold`, then applies min-speech,
    /// min-silence and speech-pad rules.
    #[must_use]
    pub fn into_mask(&self, params: &VadParams) -> VadMask {
        let raw: Vec<bool> = self
            .values
            .iter()
            .map(|p| *p >= params.prob_threshold)
            .collect();
        let cleaned = apply_vad_rules(&raw, self.seconds_per_window(), params);
        VadMask {
            speech: cleaned,
            hop_samples: self.hop_samples,
            sample_rate: self.sample_rate,
        }
    }
}

/// A binary speech/silence mask at the VAD's native window rate.
#[derive(Debug, Clone, PartialEq)]
pub struct VadMask {
    /// `speech[i]` is true when window `i` contains speech.
    pub speech: Vec<bool>,
    /// Samples between window starts.
    pub hop_samples: u32,
    /// Sample rate the mask was computed at.
    pub sample_rate: u32,
}

impl VadMask {
    /// An empty mask at the given window size and rate.
    #[must_use]
    pub fn empty(sample_rate: u32, hop_samples: u32) -> Self {
        Self {
            speech: Vec::new(),
            hop_samples,
            sample_rate,
        }
    }

    /// Number of windows.
    #[must_use]
    pub fn len(&self) -> usize {
        self.speech.len()
    }

    /// True when there are no windows.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.speech.is_empty()
    }

    /// Seconds per window.
    #[must_use]
    pub fn seconds_per_window(&self) -> f64 {
        if self.sample_rate == 0 {
            0.0
        } else {
            f64::from(self.hop_samples) / f64::from(self.sample_rate)
        }
    }

    /// Total duration covered by the mask.
    #[must_use]
    pub fn duration_seconds(&self) -> f64 {
        self.speech.len() as f64 * self.seconds_per_window()
    }

    /// Resamples the mask to `target_frames` values, each covering
    /// `target_seconds_per_frame` seconds.
    ///
    /// A target frame is speech when at least half of the VAD windows that
    /// overlap it are speech. Returns a vector of length `target_frames`.
    /// Empty source masks and non-positive rates yield all-false.
    #[must_use]
    pub fn resample_to(&self, target_seconds_per_frame: f64, target_frames: usize) -> Vec<bool> {
        if target_frames == 0 {
            return Vec::new();
        }
        let src_spf = self.seconds_per_window();
        if src_spf <= 0.0 || target_seconds_per_frame <= 0.0 || self.speech.is_empty() {
            return vec![false; target_frames];
        }

        let mut out = Vec::with_capacity(target_frames);
        for i in 0..target_frames {
            let t0 = i as f64 * target_seconds_per_frame;
            let t1 = t0 + target_seconds_per_frame;
            let src_start = (t0 / src_spf).floor() as usize;
            let src_end = ((t1 / src_spf).ceil() as usize).min(self.speech.len());
            if src_end <= src_start {
                out.push(false);
                continue;
            }
            let speech_count = self.speech[src_start..src_end]
                .iter()
                .filter(|x| **x)
                .count();
            out.push(speech_count * 2 >= src_end - src_start);
        }
        out
    }
}

/// Parameters for thresholding and cleaning a VAD probability series.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VadParams {
    /// Probability threshold; windows at or above this are speech.
    pub prob_threshold: f32,
    /// Minimum speech run to keep, in seconds.
    pub min_speech_s: f64,
    /// Minimum silence run to keep, in seconds.
    pub min_silence_s: f64,
    /// Speech padding applied on both sides, in seconds.
    pub speech_pad_s: f64,
}

impl Default for VadParams {
    fn default() -> Self {
        Self {
            prob_threshold: DEFAULT_PROB_THRESHOLD,
            min_speech_s: DEFAULT_MIN_SPEECH_S,
            min_silence_s: DEFAULT_MIN_SILENCE_S,
            speech_pad_s: DEFAULT_SPEECH_PAD_S,
        }
    }
}

/// The engine-agnostic VAD interface.
///
/// Implementations produce one probability per fixed-size window. The window
/// size and sample rate are the implementor's concern; callers that use
/// [`VadMask::resample_to`] do not need to match the envelope's hop.
pub trait Vad: Send + Sync {
    /// Human-readable name for logs and error messages.
    fn name(&self) -> &'static str;

    /// Samples per window this VAD expects.
    fn window_samples(&self) -> u32;

    /// Sample rate this VAD expects.
    fn sample_rate(&self) -> u32;

    /// Runs the VAD over `samples` and returns one probability per window.
    ///
    /// `samples` must be at [`Vad::sample_rate`]. Implementations should
    /// zero-pad the final partial window rather than dropping it.
    ///
    /// # Errors
    ///
    /// [`CoreError::VadFailed`] if the underlying model cannot be run.
    fn speech_probabilities(&self, samples: &[f32]) -> Result<VadProbabilities, CoreError>;
}

/// A no-op VAD that classifies every window as silence.
///
/// Used by tests and as the default when no VAD model is configured. It is
/// deliberately not a stand-in for Silero: it exists so the pipeline can be
/// exercised without an ONNX runtime.
#[derive(Debug, Clone, Copy, Default)]
pub struct StubVad;

impl Vad for StubVad {
    fn name(&self) -> &'static str {
        "stub"
    }

    fn window_samples(&self) -> u32 {
        VAD_WINDOW_SAMPLES
    }

    fn sample_rate(&self) -> u32 {
        VAD_SAMPLE_RATE
    }

    fn speech_probabilities(&self, samples: &[f32]) -> Result<VadProbabilities, CoreError> {
        let windows = samples.len() / self.window_samples() as usize;
        Ok(VadProbabilities {
            values: vec![0.0; windows],
            hop_samples: self.window_samples(),
            sample_rate: self.sample_rate(),
        })
    }
}

/// Applies min-speech, min-silence and speech-pad rules to a raw binary mask.
///
/// Mirrors the shape of the threshold detector's rules:
///
/// 1. Drop speech runs shorter than `min_speech_s`.
/// 2. Fill silence runs shorter than `min_silence_s`.
/// 3. Grow each speech run by `speech_pad_s` on both sides, then merge.
///
/// `seconds_per_window` is the duration of one element of `raw`. Non-positive
/// values, or an empty `raw`, return `raw` unchanged.
#[must_use]
pub fn apply_vad_rules(
    raw: &[bool],
    seconds_per_window: f64,
    params: &VadParams,
) -> Vec<bool> {
    if raw.is_empty() || seconds_per_window <= 0.0 {
        return raw.to_vec();
    }

    let min_speech = (params.min_speech_s / seconds_per_window).round() as i64;
    let min_silence = (params.min_silence_s / seconds_per_window).round() as i64;
    let pad = (params.speech_pad_s / seconds_per_window).round() as i64;

    let mut runs = mask_to_runs(raw);
    runs = flip_runs(&runs, true, min_speech);
    runs = flip_runs(&runs, false, min_silence);

    let padded: Vec<FrameRange> = runs
        .iter()
        .filter(|(is_speech, _, _)| *is_speech)
        .map(|(_, s, e)| {
            FrameRange::new((*s - pad).max(0), (*e + pad).min(raw.len() as i64))
        })
        .collect();
    let merged = merge_ranges(&padded);

    let mut out = vec![false; raw.len()];
    for r in merged {
        for i in r.start..r.end {
            out[i as usize] = true;
        }
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
    for (i, &v) in mask.iter().enumerate().skip(1) {
        if v != current {
            runs.push((current, start, i as i64));
            current = v;
            start = i as i64;
        }
    }
    runs.push((current, start, mask.len() as i64));
    runs
}

fn flip_runs(runs: &[(bool, i64, i64)], target: bool, min_len: i64) -> Vec<(bool, i64, i64)> {
    let mut out: Vec<(bool, i64, i64)> = Vec::with_capacity(runs.len());
    for &(value, s, e) in runs {
        let flipped = if value == target && (e - s) < min_len {
            !value
        } else {
            value
        };
        match out.last_mut() {
            Some(last) if last.0 == flipped && last.2 == s => last.2 = e,
            _ => out.push((flipped, s, e)),
        }
    }
    out
}

fn merge_ranges(ranges: &[FrameRange]) -> Vec<FrameRange> {
    if ranges.is_empty() {
        return Vec::new();
    }
    let mut sorted = ranges.to_vec();
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

/// Snaps every speech/silence boundary in `mask` to the lowest-energy frame
/// within `radius_frames`, using the dBFS envelope as the energy signal.
///
/// This implements the hybrid rule from section D: the VAD decides speech
/// versus silence, then each boundary snaps to the lowest-energy frame
/// within 100 ms.
///
/// `mask` and `envelope_db` must be at the same frame rate. `radius_frames`
/// is expressed in the same unit (with the default envelope that is 10 ms
/// per frame, so 100 ms = 10 frames). The two slices are truncated to their
/// common length; an empty common length yields an empty vector.
#[must_use]
pub fn hybrid_snap(mask: &[bool], envelope_db: &[f32], radius_frames: usize) -> Vec<bool> {
    let n = mask.len().min(envelope_db.len());
    if n < 2 {
        return mask[..n].to_vec();
    }

    // Boundaries are indices where mask[i] != mask[i-1].
    let mut boundaries: Vec<usize> = (1..n).filter(|&i| mask[i] != mask[i - 1]).collect();
    // Process from the end so earlier indices remain valid as we move later
    // boundaries first.
    boundaries.sort_unstable_by(|a, b| b.cmp(a));

    let mut snapped = mask[..n].to_vec();
    for b in boundaries {
        let lo = b.saturating_sub(radius_frames).max(1);
        let hi = (b + radius_frames).min(n - 1);
        if hi <= lo {
            continue;
        }

        let mut best_idx = b;
        let mut best_level = envelope_db[b];
        for i in lo..=hi {
            if envelope_db[i] < best_level {
                best_level = envelope_db[i];
                best_idx = i;
            }
        }
        if best_idx == b {
            continue;
        }

        let before = mask[b - 1];
        let after = mask[b];
        if best_idx < b {
            // The new value (after) extends earlier over [best_idx, b).
            for i in best_idx..b {
                snapped[i] = after;
            }
        } else {
            // The old value (before) extends later over [b, best_idx).
            for i in b..best_idx {
                snapped[i] = before;
            }
        }
    }

    snapped
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spf(hz: u32) -> f64 {
        1.0 / f64::from(hz)
    }

    // --- VadProbabilities::into_mask -------------------------------------

    #[test]
    fn threshold_50_keeps_values_at_or_above() {
        let probs = VadProbabilities {
            values: vec![0.4, 0.5, 0.6],
            hop_samples: VAD_WINDOW_SAMPLES,
            sample_rate: VAD_SAMPLE_RATE,
        };
        let p = VadParams {
            min_speech_s: 0.0,
            min_silence_s: 0.0,
            speech_pad_s: 0.0,
            ..VadParams::default()
        };
        let mask = probs.into_mask(&p);
        assert_eq!(mask.speech, vec![false, true, true]);
    }

    #[test]
    fn default_params_match_spec() {
        let p = VadParams::default();
        assert!((p.prob_threshold - 0.5).abs() < 1e-6);
        assert!((p.min_speech_s - 0.250).abs() < 1e-12);
        assert!((p.min_silence_s - 0.100).abs() < 1e-12);
        assert!((p.speech_pad_s - 0.030).abs() < 1e-12);
    }

    // --- apply_vad_rules --------------------------------------------------

    #[test]
    fn apply_vad_rules_drops_short_speech() {
        // 10 windows at 32 ms each. One true at index 5 (32 ms of speech).
        // min_speech = 250 ms -> 8 windows; the run is dropped.
        let raw = vec![false; 5].into_iter().chain(vec![true]).chain(vec![false; 4]).collect::<Vec<_>>();
        let out = apply_vad_rules(&raw, 0.032, &VadParams::default());
        assert!(out.iter().all(|x| !x), "{out:?}");
    }

    #[test]
    fn apply_vad_rules_keeps_long_speech() {
        // 10 true windows = 320 ms, above the 250 ms min_speech.
        let raw = vec![false, false]
            .into_iter()
            .chain(vec![true; 10])
            .chain(vec![false; 5])
            .collect::<Vec<_>>();
        let out = apply_vad_rules(&raw, 0.032, &VadParams::default());
        assert!(out.iter().filter(|x| **x).count() >= 10);
    }

    #[test]
    fn apply_vad_rules_fills_short_silence() {
        // Two long speech runs separated by a single silence window.
        // min_silence = 100 ms -> 4 windows at 32 ms. The 1-window gap is
        // filled.
        let mut raw = vec![true; 8];
        raw.push(false);
        raw.extend(vec![true; 8]);
        let out = apply_vad_rules(&raw, 0.032, &VadParams::default());
        assert!(out.iter().all(|x| *x), "{out:?}");
    }

    #[test]
    fn apply_vad_rules_pads_boundaries() {
        // 20 false, 10 true, 20 false. min_speech = 250 ms -> 8 windows,
        // the 10-true run survives. min_silence = 100 ms -> 4 windows.
        // speech_pad = 30 ms -> 1 window each side.
        let mut raw = vec![false; 20];
        raw.extend(vec![true; 10]);
        raw.extend(vec![false; 20]);
        let out = apply_vad_rules(&raw, 0.032, &VadParams::default());
        let first_true = out.iter().position(|x| *x).unwrap();
        let last_true = out.iter().rposition(|x| *x).unwrap();
        assert_eq!(first_true, 19);
        assert_eq!(last_true, 30);
    }

    // --- VadMask::resample_to --------------------------------------------

    #[test]
    fn resample_upsamples_32ms_to_10ms() {
        // 1 second of Silero mask at 32 ms: 31 windows (0..30).
        // First 16 windows true, rest false.
        let mut speech = vec![true; 16];
        speech.extend(vec![false; 15]);
        let mask = VadMask {
            speech,
            hop_samples: VAD_WINDOW_SAMPLES,
            sample_rate: VAD_SAMPLE_RATE,
        };
        // Resample to 100 Hz (10 ms frames), 100 frames.
        let out = mask.resample_to(0.010, 100);
        assert_eq!(out.len(), 100);
        // Windows 0..16 cover 0..512 ms. Frames 0..50 are speech-dominant.
        assert!(out[0], "frame 0 should be speech");
        assert!(out[49], "frame 49 should be speech");
        assert!(!out[52], "frame 52 should be silence");
        assert!(!out[99], "last frame should be silence");
    }

    #[test]
    fn resample_handles_empty_source() {
        let mask = VadMask::empty(VAD_SAMPLE_RATE, VAD_WINDOW_SAMPLES);
        let out = mask.resample_to(0.010, 100);
        assert_eq!(out.len(), 100);
        assert!(out.iter().all(|x| !x));
    }

    #[test]
    fn resample_zero_target_returns_empty() {
        let mask = VadMask {
            speech: vec![true; 10],
            hop_samples: VAD_WINDOW_SAMPLES,
            sample_rate: VAD_SAMPLE_RATE,
        };
        assert!(mask.resample_to(0.010, 0).is_empty());
    }

    // --- StubVad ----------------------------------------------------------

    #[test]
    fn stub_vad_reports_all_silence() {
        let vad = StubVad;
        let probs = vad.speech_probabilities(&vec![0.5f32; 1024]).unwrap();
        assert_eq!(probs.values.len(), 2);
        assert!(probs.values.iter().all(|p| *p == 0.0));
    }

    #[test]
    fn stub_vad_handles_short_input() {
        // Fewer than one window: zero complete windows.
        let probs = StubVad.speech_probabilities(&[0.0; 100]).unwrap();
        assert!(probs.values.is_empty());
    }

    // --- hybrid_snap ------------------------------------------------------

    #[test]
    fn hybrid_snap_moves_boundary_to_lower_energy() {
        // Mask [F F T T T F F] with a boundary at 2 (F->T) and 5 (T->F).
        // Envelope has a dip at index 1 and a dip at index 6, both within
        // radius 1, so the boundaries should snap outward.
        let mask = vec![false, false, true, true, true, false, false];
        let env: Vec<f32> = vec![-30.0, -80.0, -20.0, -20.0, -20.0, -30.0, -80.0];
        let out = hybrid_snap(&mask, &env, 1);
        // The F->T transition moves from 2 to 1.
        assert!(!out[0]);
        assert!(out[1], "transition should have moved to index 1");
        assert!(out[4]);
        // The T->F transition moves from 5 to 6.
        assert!(out[5], "frame 5 stays speech");
        assert!(!out[6], "frame 6 becomes silence");
    }

    #[test]
    fn hybrid_snap_is_noop_when_radius_is_zero() {
        let mask = vec![false, true, true, false];
        let env = vec![-10.0, -20.0, -30.0, -40.0];
        let out = hybrid_snap(&mask, &env, 0);
        assert_eq!(out, mask);
    }

    #[test]
    fn hybrid_snap_empty_input() {
        assert!(hybrid_snap(&[], &[], 5).is_empty());
    }

    #[test]
    fn hybrid_snap_truncates_to_shorter_input() {
        let mask = vec![true, true, true];
        let env = vec![-10.0, -20.0];
        let out = hybrid_snap(&mask, &env, 5);
        assert_eq!(out.len(), 2);
    }
}