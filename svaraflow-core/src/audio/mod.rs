//! Audio primitives for the analysis pipeline.
//!
//! The analysis pipeline reduces every source clip to a single mono `f32`
//! signal at [`ANALYSIS_SAMPLE_RATE`]. Everything downstream — the dBFS
//! envelope, threshold detection, VAD and the cut engine — operates on that
//! one signal and never touches a media file again.

pub mod decode;
pub mod envelope;
pub mod mixdown;

use crate::error::CoreError;

pub use decode::{apply_speed, decode_native_mono, decode_via_ffmpeg, FileDecoder};
pub use envelope::{
    compute_envelope, downsample_for_preview, Envelope, EnvelopeParams, DBFS_FLOOR,
};
pub use mixdown::{frames_to_samples, render_analysis_signal, select_tracks, ClipDecoder};

/// Sample rate of the analysis signal, in Hz.
pub const ANALYSIS_SAMPLE_RATE: u32 = 16_000;

/// A mono, `f32`, single-rate audio signal.
#[derive(Debug, Clone, PartialEq)]
pub struct MonoAudio {
    /// The samples, in playback order.
    pub samples: Vec<f32>,
    /// Samples per second.
    pub sample_rate: u32,
}

impl MonoAudio {
    /// Wraps a sample buffer and its rate.
    #[must_use]
    pub fn new(samples: Vec<f32>, sample_rate: u32) -> Self {
        Self {
            samples,
            sample_rate,
        }
    }

    /// An empty signal at the given rate.
    #[must_use]
    pub fn empty(sample_rate: u32) -> Self {
        Self::new(Vec::new(), sample_rate)
    }

    /// Number of samples.
    #[must_use]
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// True when there are no samples.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Duration in seconds. Returns `0.0` for a zero sample rate.
    #[must_use]
    pub fn duration_seconds(&self) -> f64 {
        if self.sample_rate == 0 {
            0.0
        } else {
            self.samples.len() as f64 / f64::from(self.sample_rate)
        }
    }
}

/// Resamples `input` to `target_rate` with an FFT resampler.
///
/// The ratio is free: `48_000 -> 16_000`, `44_100 -> 16_000` and
/// `16_000 -> 16_000` are all valid. The output length is
/// `round(input.len() * target_rate / input.sample_rate)`.
///
/// The input is zero-padded by `input.sample_rate` samples before being fed
/// to the resampler, because `FftFixedIn` produces output in discrete
/// internal chunks and can return fewer samples than the exact expected
/// count at some ratios (notably 48 kHz -> 16 kHz). The padded tail is
/// discarded by the truncation below, so the visible output length is
/// exactly `expected_out`.
///
/// # Errors
///
/// Returns [`CoreError::InvalidSampleRate`] when either rate is zero, and
/// [`CoreError::Resample`] when the resampler cannot be constructed or
/// fails.
pub fn resample_to(input: &MonoAudio, target_rate: u32) -> Result<MonoAudio, CoreError> {
    if target_rate == 0 || input.sample_rate == 0 {
        return Err(CoreError::InvalidSampleRate(if target_rate == 0 {
            target_rate
        } else {
            input.sample_rate
        }));
    }
    if input.sample_rate == target_rate {
        return Ok(input.clone());
    }
    if input.samples.is_empty() {
        return Ok(MonoAudio::empty(target_rate));
    }

    let fs_in = input.sample_rate as usize;
    let fs_out = target_rate as usize;

    // Zero-pad the input so the resampler always has enough frames to
    // produce the exact expected output length.
    let padded: Vec<f32> = {
        let mut v = Vec::with_capacity(input.samples.len() + fs_in);
        v.extend_from_slice(&input.samples);
        v.resize(input.samples.len() + fs_in, 0.0);
        v
    };
    let total = padded.len();

    const CHUNK_OUT: usize = 1024;
    const SUB_CHUNKS: usize = 2;
    const CHANNELS: usize = 1;

    let mut resampler =
        rubato::FftFixedIn::<f32>::new(fs_in, fs_out, CHUNK_OUT, SUB_CHUNKS, CHANNELS).map_err(
            |e| CoreError::Resample {
                from_hz: input.sample_rate,
                to_hz: target_rate,
                reason: e.to_string(),
            },
        )?;

    let expected_out =
        ((input.samples.len() as f64) * (fs_out as f64) / (fs_in as f64)).round() as usize;
    let mut out: Vec<f32> = Vec::with_capacity(expected_out + CHUNK_OUT);

    let mut pos = 0usize;
    let mut waves_in: Vec<Vec<f32>> = vec![Vec::new()];

    while pos < total {
        let need = rubato::Resampler::input_frames_next(&resampler);
        if waves_in[0].len() != need {
            waves_in[0] = vec![0.0f32; need];
        }
        let take = (total - pos).min(need);
        waves_in[0][..take].copy_from_slice(&padded[pos..pos + take]);
        for sample in &mut waves_in[0][take..] {
            *sample = 0.0;
        }
        let waves_out = rubato::Resampler::process(&mut resampler, &waves_in, None).map_err(|e| {
            CoreError::Resample {
                from_hz: input.sample_rate,
                to_hz: target_rate,
                reason: e.to_string(),
            }
        })?;
        out.extend_from_slice(&waves_out[0]);
        pos += take;
    }

    out.truncate(expected_out);
    Ok(MonoAudio::new(out, target_rate))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(freq_hz: f64, seconds: f64, rate: u32) -> MonoAudio {
        let n = (seconds * f64::from(rate)).round() as usize;
        let samples = (0..n)
            .map(|i| {
                let t = i as f64 / f64::from(rate);
                (2.0 * std::f64::consts::PI * freq_hz * t).sin() as f32
            })
            .collect();
        MonoAudio::new(samples, rate)
    }

    #[test]
    fn mono_audio_duration() {
        let a = MonoAudio::new(vec![0.0; 16_000], 16_000);
        assert_eq!(a.len(), 16_000);
        assert!(!a.is_empty());
        assert!((a.duration_seconds() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn mono_audio_duration_with_zero_rate_is_zero() {
        let a = MonoAudio::new(vec![1.0, 2.0], 0);
        assert_eq!(a.duration_seconds(), 0.0);
    }

    #[test]
    fn resample_identity_returns_clone() {
        let a = sine(440.0, 0.1, 16_000);
        let b = resample_to(&a, 16_000).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn resample_48k_to_16k_thirds_the_length() {
        let a = sine(440.0, 1.0, 48_000);
        let b = resample_to(&a, 16_000).unwrap();
        assert_eq!(b.sample_rate, 16_000);
        assert_eq!(b.len(), 16_000);
    }

    #[test]
    fn resample_44k1_to_16k_length_is_rounded_not_truncated() {
        let a = sine(440.0, 1.0, 44_100);
        let b = resample_to(&a, 16_000).unwrap();
        assert_eq!(b.len(), 16_000);
    }

    #[test]
    fn resample_preserves_a_tone() {
        let a = sine(1000.0, 0.25, 48_000);
        let b = resample_to(&a, 16_000).unwrap();
        // Compare peak amplitudes; the resampler must not attenuate the tone.
        let peak_a = a.samples.iter().fold(0.0f32, |m, s| m.max(s.abs()));
        let peak_b = b.samples.iter().fold(0.0f32, |m, s| m.max(s.abs()));
        assert!(peak_a > 0.9, "source peak {peak_a}");
        assert!((peak_a - peak_b).abs() < 0.1, "peak {peak_a} -> {peak_b}");
    }

    #[test]
    fn resample_empty_input_is_empty_output() {
        let a = MonoAudio::empty(48_000);
        let b = resample_to(&a, 16_000).unwrap();
        assert!(b.is_empty());
        assert_eq!(b.sample_rate, 16_000);
    }

    #[test]
    fn resample_rejects_zero_target_rate() {
        let a = sine(440.0, 0.01, 48_000);
        assert!(matches!(
            resample_to(&a, 0),
            Err(CoreError::InvalidSampleRate(0))
        ));
    }

    #[test]
    fn resample_rejects_zero_source_rate() {
        let a = MonoAudio::new(vec![0.0; 16], 0);
        assert!(matches!(
            resample_to(&a, 16_000),
            Err(CoreError::InvalidSampleRate(0))
        ));
    }
}