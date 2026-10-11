// svaraflow-core/src/audio/envelope.rs
//! dBFS envelope for the analysis signal.
//!
//! The envelope reduces a `MonoAudio` to one dBFS level per hop. It is the
//! input to every silence-detector rule: threshold detection, hysteresis,
//! auto noise floor, and the preview chart. Nothing else in the crate reads
//! the raw samples after this point.
//!
//! # Parameters
//!
//! The defaults come from the spec: a 20 ms RMS window, a 10 ms hop, and a
//! 3-frame median filter. At [`ANALYSIS_SAMPLE_RATE`] that is 320 samples per
//! window, 160 samples per hop, 100 frames per second.

use super::MonoAudio;

/// Lowest reported dBFS value. Digital silence maps here; the value is
/// chosen so downstream arithmetic never touches `-inf`.
pub const DBFS_FLOOR: f32 = -120.0;

/// Parameters for [`compute_envelope`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EnvelopeParams {
    /// RMS window length, in milliseconds.
    pub window_ms: u32,
    /// Hop between consecutive windows, in milliseconds.
    pub hop_ms: u32,
    /// Number of frames for the median filter. Values `>= 3` apply the
    /// 3-frame filter the spec defines; `0`, `1` and `2` skip it.
    pub median_frames: u32,
}

impl Default for EnvelopeParams {
    fn default() -> Self {
        Self {
            window_ms: 20,
            hop_ms: 10,
            median_frames: 3,
        }
    }
}

/// A sequence of dBFS levels, one per hop.
#[derive(Debug, Clone, PartialEq)]
pub struct Envelope {
    /// The levels, in dBFS. `DBFS_FLOOR` for digital silence.
    pub levels_db: Vec<f32>,
    /// Sample rate of the signal the envelope was computed from.
    pub sample_rate: u32,
    /// Hop in samples (the distance between consecutive level values).
    pub hop_samples: usize,
    /// RMS window length in samples.
    pub window_samples: usize,
}

impl Envelope {
    /// An empty envelope at the given rate.
    #[must_use]
    pub fn empty(sample_rate: u32) -> Self {
        Self {
            levels_db: Vec::new(),
            sample_rate,
            hop_samples: 0,
            window_samples: 0,
        }
    }

    /// Number of level values.
    #[must_use]
    pub fn len(&self) -> usize {
        self.levels_db.len()
    }

    /// True when there are no levels.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.levels_db.is_empty()
    }

    /// Duration represented by one level value, in seconds.
    ///
    /// Returns `0.0` when the envelope is empty or the rate is zero.
    #[must_use]
    pub fn seconds_per_frame(&self) -> f64 {
        if self.sample_rate == 0 {
            0.0
        } else {
            self.hop_samples as f64 / f64::from(self.sample_rate)
        }
    }

    /// Sample index of the start of `frame`.
    #[must_use]
    pub fn frame_to_sample(&self, frame: usize) -> usize {
        frame * self.hop_samples
    }

    /// Level at `seconds`, or `None` when out of range.
    #[must_use]
    pub fn level_at_seconds(&self, seconds: f64) -> Option<f32> {
        if seconds < 0.0 || self.is_empty() || self.sample_rate == 0 || self.hop_samples == 0 {
            return None;
        }
        let frame = (seconds * f64::from(self.sample_rate) / self.hop_samples as f64) as usize;
        self.levels_db.get(frame).copied()
    }

    /// Lowest level, or `None` for an empty envelope.
    #[must_use]
    pub fn min_level(&self) -> Option<f32> {
        self.levels_db.iter().copied().reduce(f32::min)
    }

    /// Highest level, or `None` for an empty envelope.
    #[must_use]
    pub fn max_level(&self) -> Option<f32> {
        self.levels_db.iter().copied().reduce(f32::max)
    }
}

/// Converts milliseconds to samples, rounding to nearest.
fn ms_to_samples(ms: u32, sample_rate: u32) -> usize {
    (f64::from(ms) * f64::from(sample_rate) / 1000.0).round() as usize
}

/// RMS of `samples` expressed in dBFS, floored at [`DBFS_FLOOR`].
///
/// Returns [`DBFS_FLOOR`] for an empty slice, an all-zero slice, or a
/// non-finite RMS.
fn rms_dbfs(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return DBFS_FLOOR;
    }
    let sum_sq: f64 = samples.iter().map(|s| f64::from(*s) * f64::from(*s)).sum();
    let mean = sum_sq / samples.len() as f64;
    let rms = mean.sqrt();
    if rms <= 0.0 || !rms.is_finite() {
        return DBFS_FLOOR;
    }
    let db = 20.0 * rms.log10();
    (db as f32).max(DBFS_FLOOR)
}

/// 3-frame median filter with boundary replication.
///
/// For `input[i]` the neighbourhood is `[input[i-1], input[i], input[i+1]]`
/// with `i-1` clamped to `0` and `i+1` clamped to `n - 1`. A single isolated
/// outlier frame is removed; a sustained level is preserved; a step edge is
/// sharpened, not smeared.
///
/// Note: because the RMS window is wider than the hop, a single-sample click
/// in the audio shows up in **two** envelope frames, not one. The median
/// filter is designed to remove a one-frame envelope spike; two-frame spikes
/// survive by design and are handled downstream by the detector's
/// min-silence rule.
fn median_filter_3(input: &[f32]) -> Vec<f32> {
    let n = input.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return input.to_vec();
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let a = input[i.saturating_sub(1)];
        let b = input[i];
        let c = input[(i + 1).min(n - 1)];
        let mut three = [a, b, c];
        three.sort_by(f32::total_cmp);
        out.push(three[1]);
    }
    out
}

/// Computes the dBFS envelope of `audio` using `params`.
///
/// Returns an empty envelope when `audio` is shorter than one window, when
/// the sample rate is zero, or when either `window_ms` or `hop_ms` is zero.
#[must_use]
pub fn compute_envelope(audio: &MonoAudio, params: EnvelopeParams) -> Envelope {
    let sample_rate = audio.sample_rate;
    if sample_rate == 0 || params.window_ms == 0 || params.hop_ms == 0 {
        return Envelope::empty(sample_rate);
    }

    let window_samples = ms_to_samples(params.window_ms, sample_rate);
    let hop_samples = ms_to_samples(params.hop_ms, sample_rate);
    if window_samples == 0 || hop_samples == 0 || audio.samples.len() < window_samples {
        return Envelope::empty(sample_rate);
    }

    let frame_count = (audio.samples.len() - window_samples) / hop_samples + 1;
    let mut levels = Vec::with_capacity(frame_count);
    for i in 0..frame_count {
        let start = i * hop_samples;
        let end = start + window_samples;
        levels.push(rms_dbfs(&audio.samples[start..end]));
    }

    if params.median_frames >= 3 {
        levels = median_filter_3(&levels);
    }

    Envelope {
        levels_db: levels,
        sample_rate,
        hop_samples,
        window_samples,
    }
}

/// Reduces `envelope` to at most `target_points` values for a preview chart.
///
/// Each output point is the **maximum** of its bucket, so a single loud frame
/// survives the reduction. When the envelope is already at or below
/// `target_points`, a copy is returned unchanged. `target_points == 0` and an
/// empty envelope both yield an empty vector.
#[must_use]
pub fn downsample_for_preview(envelope: &Envelope, target_points: usize) -> Vec<f32> {
    let n = envelope.levels_db.len();
    if target_points == 0 || n == 0 {
        return Vec::new();
    }
    if n <= target_points {
        return envelope.levels_db.clone();
    }

    let bucket = n as f64 / target_points as f64;
    let mut out = Vec::with_capacity(target_points);
    for b in 0..target_points {
        let start = ((b as f64) * bucket).floor() as usize;
        let end = ((((b + 1) as f64) * bucket).floor() as usize).min(n);
        let end = end.max(start + 1);
        let slice = &envelope.levels_db[start..end];
        let peak = slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        out.push(peak);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::ANALYSIS_SAMPLE_RATE;

    const RATE: u32 = ANALYSIS_SAMPLE_RATE;

    fn sine_samples(freq_hz: f64, seconds: f64, amplitude: f32) -> Vec<f32> {
        let n = (seconds * f64::from(RATE)).round() as usize;
        (0..n)
            .map(|i| {
                let t = i as f64 / f64::from(RATE);
                (f64::from(amplitude) * (2.0 * std::f64::consts::PI * freq_hz * t).sin()) as f32
            })
            .collect()
    }

    fn silence_samples(seconds: f64) -> Vec<f32> {
        let n = (seconds * f64::from(RATE)).round() as usize;
        vec![0.0; n]
    }

    fn square_samples(seconds: f64, amplitude: f32) -> Vec<f32> {
        let n = (seconds * f64::from(RATE)).round() as usize;
        (0..n)
            .map(|i| if i % 2 == 0 { amplitude } else { -amplitude })
            .collect()
    }

    fn envelope_of(samples: Vec<f32>) -> Envelope {
        compute_envelope(&MonoAudio::new(samples, RATE), EnvelopeParams::default())
    }

    #[test]
    fn empty_audio_produces_empty_envelope() {
        let env = envelope_of(Vec::new());
        assert!(env.is_empty());
        assert_eq!(env.sample_rate, RATE);
    }

    #[test]
    fn audio_shorter_than_window_produces_empty_envelope() {
        // 319 samples is one short of the 320-sample window.
        let env = envelope_of(vec![0.5; 319]);
        assert!(env.is_empty());
    }

    #[test]
    fn dc_silence_has_floor_levels() {
        let env = envelope_of(silence_samples(0.5));
        assert!(!env.is_empty());
        assert!(env.levels_db.iter().all(|&v| v == DBFS_FLOOR));
    }

    #[test]
    fn full_scale_sine_is_about_minus_three_dbfs() {
        // RMS of a full-scale sine is 1/sqrt(2) -> -3.01 dBFS.
        let env = envelope_of(sine_samples(1000.0, 0.5, 1.0));
        let middle = env.levels_db[env.len() / 2];
        assert!(
            (middle - (-3.01)).abs() < 0.15,
            "expected ~-3.01 dBFS, got {middle}"
        );
    }

    #[test]
    fn half_scale_sine_is_about_minus_nine_dbfs() {
        let env = envelope_of(sine_samples(1000.0, 0.5, 0.5));
        let middle = env.levels_db[env.len() / 2];
        assert!(
            (middle - (-9.03)).abs() < 0.15,
            "expected ~-9.03 dBFS, got {middle}"
        );
    }

    #[test]
    fn square_wave_is_zero_dbfs() {
        let env = envelope_of(square_samples(0.5, 1.0));
        let middle = env.levels_db[env.len() / 2];
        assert!((middle - 0.0).abs() < 0.1, "expected ~0 dBFS, got {middle}");
    }

    #[test]
    fn step_from_silence_to_tone_is_sharp() {
        // 0.5 s silence, then 0.5 s full-scale tone.
        let mut samples = silence_samples(0.5);
        samples.extend(sine_samples(1000.0, 0.5, 1.0));
        let env = envelope_of(samples);

        // Frames 0..40 sit entirely in the silence region.
        assert!(env.levels_db[..40].iter().all(|&v| v == DBFS_FLOOR));
        // Frames 60..end sit entirely in the tone region.
        assert!(env.levels_db[60..]
            .iter()
            .all(|&v| (v - (-3.01)).abs() < 0.15));
        // The transition is confined to a few frames near the 0.5 s mark,
        // not smeared across the whole signal.
        let transition: Vec<usize> = (0..env.len())
            .filter(|&i| env.levels_db[i] > DBFS_FLOOR + 1.0 && env.levels_db[i] < -4.0)
            .collect();
        assert!(transition.len() <= 3, "transition too wide: {transition:?}");
    }

    #[test]
    fn median_filter_removes_an_isolated_envelope_frame() {
        // Direct unit test of the median filter: a single loud frame between
        // quiet neighbours is replaced by the median of the three, which is
        // quiet.
        let levels = vec![-100.0f32, -100.0, -20.0, -100.0, -100.0];
        let filtered = median_filter_3(&levels);
        assert_eq!(filtered[2], -100.0, "spike frame was not removed");
        // The neighbours are unchanged.
        assert_eq!(filtered[0], -100.0);
        assert_eq!(filtered[1], -100.0);
        assert_eq!(filtered[3], -100.0);
        assert_eq!(filtered[4], -100.0);
    }

    #[test]
    fn median_filter_preserves_a_step_edge() {
        // A sustained level change is kept as-is.
        let levels = vec![-100.0f32, -100.0, -20.0, -20.0, -20.0];
        let filtered = median_filter_3(&levels);
        assert_eq!(filtered, vec![-100.0, -100.0, -20.0, -20.0, -20.0]);
    }

    #[test]
    fn median_filter_handles_short_inputs() {
        assert!(median_filter_3(&[]).is_empty());
        assert_eq!(median_filter_3(&[0.5]), vec![0.5]);
    }

    #[test]
    fn median_smoothing_preserves_sustained_signal() {
        // A constant tone produces constant levels; the median of three
        // equal values is that value.
        let env = envelope_of(sine_samples(1000.0, 0.5, 1.0));
        let first = env.levels_db[5];
        for &level in &env.levels_db[5..env.len() - 5] {
            assert!(
                (level - first).abs() < 0.2,
                "level {level} deviates from {first}"
            );
        }
    }

    #[test]
    fn frame_count_matches_window_and_hop() {
        // 1 s of audio at 16 kHz: 16000 samples, window 320, hop 160.
        // frame_count = 1 + (16000 - 320) / 160 = 1 + 98 = 99.
        let env = envelope_of(sine_samples(1000.0, 1.0, 1.0));
        assert_eq!(env.len(), 99);
        assert_eq!(env.window_samples, 320);
        assert_eq!(env.hop_samples, 160);
    }

    #[test]
    fn seconds_per_frame_is_hop_over_rate() {
        let env = envelope_of(sine_samples(1000.0, 0.5, 1.0));
        assert!((env.seconds_per_frame() - 0.010).abs() < 1e-12);
    }

    #[test]
    fn seconds_per_frame_with_zero_rate_is_zero() {
        let env = Envelope::empty(0);
        assert_eq!(env.seconds_per_frame(), 0.0);
    }

    #[test]
    fn frame_to_sample_maps_hop_indices() {
        let env = envelope_of(sine_samples(1000.0, 0.5, 1.0));
        assert_eq!(env.frame_to_sample(0), 0);
        assert_eq!(env.frame_to_sample(1), 160);
        assert_eq!(env.frame_to_sample(10), 1600);
    }

    #[test]
    fn level_at_seconds_returns_expected() {
        let env = envelope_of(sine_samples(1000.0, 0.5, 1.0));
        let at_mid = env.level_at_seconds(0.25).unwrap();
        assert!((at_mid - (-3.01)).abs() < 0.15);
    }

    #[test]
    fn level_at_seconds_out_of_range_is_none() {
        let env = envelope_of(sine_samples(1000.0, 0.5, 1.0));
        assert!(env.level_at_seconds(-0.1).is_none());
        assert!(env.level_at_seconds(10.0).is_none());
    }

    #[test]
    fn level_at_seconds_on_empty_envelope_is_none() {
        let env = Envelope::empty(RATE);
        assert!(env.level_at_seconds(0.0).is_none());
    }

    #[test]
    fn min_and_max_levels() {
        let mut samples = silence_samples(0.5);
        samples.extend(sine_samples(1000.0, 0.5, 1.0));
        let env = envelope_of(samples);

        assert_eq!(env.min_level(), Some(DBFS_FLOOR));
        let max = env.max_level().unwrap();
        assert!(
            (max - (-3.01)).abs() < 0.15,
            "expected max ~-3.01 dBFS, got {max}"
        );
    }

    #[test]
    fn min_and_max_of_empty_envelope_are_none() {
        let env = Envelope::empty(RATE);
        assert_eq!(env.min_level(), None);
        assert_eq!(env.max_level(), None);
    }

    #[test]
    fn downsample_identity_when_target_at_least_length() {
        let env = envelope_of(sine_samples(1000.0, 0.5, 1.0));
        let preview = downsample_for_preview(&env, env.len());
        assert_eq!(preview, env.levels_db);

        let preview_larger = downsample_for_preview(&env, env.len() + 100);
        assert_eq!(preview_larger, env.levels_db);
    }

    #[test]
    fn downsample_preserves_peaks() {
        // 0.2 s silence with one loud frame in the middle. Downsampled to 5
        // points, exactly one bucket must contain the peak.
        let mut samples = silence_samples(0.2);
        for sample in samples.iter_mut().skip(1600).take(320) {
            *sample = 1.0;
        }
        let env = envelope_of(samples);
        let preview = downsample_for_preview(&env, 5);

        assert_eq!(preview.len(), 5);
        assert!(preview.iter().any(|&v| v > -10.0));
        assert!(preview.iter().filter(|&&v| v == DBFS_FLOOR).count() >= 3);
    }

    #[test]
    fn downsample_zero_target_is_empty() {
        let env = envelope_of(sine_samples(1000.0, 0.5, 1.0));
        assert!(downsample_for_preview(&env, 0).is_empty());
    }

    #[test]
    fn downsample_empty_envelope_is_empty() {
        let env = Envelope::empty(RATE);
        assert!(downsample_for_preview(&env, 16).is_empty());
    }

    #[test]
    fn params_zero_window_is_empty() {
        let audio = MonoAudio::new(sine_samples(1000.0, 0.1, 1.0), RATE);
        let params = EnvelopeParams {
            window_ms: 0,
            hop_ms: 10,
            median_frames: 3,
        };
        assert!(compute_envelope(&audio, params).is_empty());
    }

    #[test]
    fn params_zero_hop_is_empty() {
        let audio = MonoAudio::new(sine_samples(1000.0, 0.1, 1.0), RATE);
        let params = EnvelopeParams {
            window_ms: 20,
            hop_ms: 0,
            median_frames: 3,
        };
        assert!(compute_envelope(&audio, params).is_empty());
    }

    #[test]
    fn params_zero_rate_is_empty() {
        let audio = MonoAudio::new(sine_samples(1000.0, 0.1, 1.0), 0);
        let env = compute_envelope(&audio, EnvelopeParams::default());
        assert!(env.is_empty());
        assert_eq!(env.sample_rate, 0);
    }
}
