//! Media decoding for the analysis pipeline.
//!
//! Two backends are available:
//!
//! 1. **symphonia** — pure Rust, no external process, handles WAV, FLAC, OGG
//!    Vorbis, AAC, ALAC and MP3 depending on the enabled features.
//! 2. **ffmpeg** — a system `ffmpeg` on `PATH`, used only when symphonia
//!    cannot open or decode the file. Requires the user to have ffmpeg
//!    installed; if it is missing, the error says so.
//!
//! [`FileDecoder`] tries them in that order and reports the clip id when both
//! fail, so the CLI can tell the user exactly which source is unreadable.

use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::process::{Command, Stdio};

use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

use crate::error::CoreError;
use crate::timeline::Clip;

use super::mixdown::ClipDecoder;
use super::{resample_to, MonoAudio, ANALYSIS_SAMPLE_RATE};

/// Decodes `path` with symphonia into mono `f32` samples at the file's native
/// sample rate.
///
/// # Errors
///
/// [`CoreError::AudioDecode`] with the file path as `id`. Callers that know
/// the clip id should re-wrap the error (see [`FileDecoder::decode_clip`]).
pub fn decode_native_mono(path: &Path) -> Result<MonoAudio, CoreError> {
    let label = path.display().to_string();
    let decode_err = |reason: String| CoreError::AudioDecode {
        id: label.clone(),
        reason,
    };

    let file = File::open(path).map_err(|e| decode_err(format!("open failed: {e}")))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|e| decode_err(format!("probe failed: {e}")))?;
    let mut format = probed.format;

    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| decode_err("no decodable audio track".to_string()))?;
    let track_id = track.id;
    let codec_params = track.codec_params.clone();

    let mut decoder = symphonia::default::get_codecs()
        .make(&codec_params, &DecoderOptions::default())
        .map_err(|e| decode_err(format!("no decoder for this codec: {e}")))?;

    let mut interleaved: Vec<f32> = Vec::new();
    let mut sample_rate: u32 = 0;
    let mut channels: usize = 0;

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            // End of stream: symphonia reports it as an unexpected EOF.
            Err(SymphoniaError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                break;
            }
            // A stream reset means the format changed mid-file; stop cleanly
            // and keep what was decoded.
            Err(SymphoniaError::ResetRequired) => break,
            Err(e) => return Err(decode_err(format!("read failed: {e}"))),
        };

        if packet.track_id() != track_id {
            continue;
        }

        match decoder.decode(&packet) {
            Ok(decoded) => {
                let spec = *decoded.spec();
                sample_rate = spec.rate;
                channels = spec.channels.count();
                let mut buffer = SampleBuffer::<f32>::new(decoded.capacity() as u64, spec);
                buffer.copy_interleaved_ref(decoded);
                interleaved.extend_from_slice(buffer.samples());
            }
            // A single corrupt packet is not fatal: skip it and keep going.
            Err(SymphoniaError::DecodeError(_)) => continue,
            Err(e) => return Err(decode_err(format!("decode failed: {e}"))),
        }
    }

    if sample_rate == 0 || channels == 0 {
        return Err(decode_err("no audio frames were decoded".to_string()));
    }

    Ok(MonoAudio::new(
        downmix_to_mono(&interleaved, channels),
        sample_rate,
    ))
}

/// Decodes `path` by piping it through a system `ffmpeg` on `PATH`.
///
/// ffmpeg is asked for raw `f32le` mono PCM at `target_rate`, so no
/// resampling happens on the Rust side.
///
/// # Memory
///
/// One hour of 16 kHz mono `f32` is about 230 MB, and the whole of ffmpeg's
/// stdout is buffered before conversion. Streaming is a later optimisation.
///
/// # Errors
///
/// [`CoreError::AudioDecode`] when ffmpeg is absent, exits non-zero, or its
/// output cannot be read.
pub fn decode_via_ffmpeg(path: &Path, target_rate: u32) -> Result<MonoAudio, CoreError> {
    let label = path.display().to_string();
    let decode_err = |reason: String| CoreError::AudioDecode {
        id: label.clone(),
        reason,
    };

    let mut child = Command::new("ffmpeg")
        .args(["-nostdin", "-loglevel", "error", "-i"])
        .arg(path)
        .args(["-vn", "-f", "f32le", "-ac", "1", "-ar"])
        .arg(target_rate.to_string())
        .arg("-")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| {
            decode_err(format!(
                "ffmpeg could not be started (is it on PATH?): {e}"
            ))
        })?;

    // Drain stderr on a thread. Reading only stdout would deadlock if ffmpeg
    // filled the stderr pipe buffer while we blocked on stdout.
    let mut stderr_pipe = child
        .stderr
        .take()
        .ok_or_else(|| decode_err("ffmpeg stderr was not captured".to_string()))?;
    let stderr_thread = std::thread::spawn(move || {
        let mut text = String::new();
        let _ = stderr_pipe.read_to_string(&mut text);
        text
    });

    let mut stdout = child
        .stdout
        .take()
        .ok_or_else(|| decode_err("ffmpeg stdout was not captured".to_string()))?;
    let mut raw: Vec<u8> = Vec::new();
    stdout
        .read_to_end(&mut raw)
        .map_err(|e| decode_err(format!("reading ffmpeg stdout failed: {e}")))?;

    let status = child
        .wait()
        .map_err(|e| decode_err(format!("waiting for ffmpeg failed: {e}")))?;
    let stderr = stderr_thread.join().unwrap_or_default();

    if !status.success() {
        return Err(decode_err(format!(
            "ffmpeg exited with {status}: {}",
            stderr.trim()
        )));
    }

    Ok(MonoAudio::new(
        pcm_f32le_to_samples(&raw),
        target_rate,
    ))
}

/// Time-compresses `input` by `speed` using linear interpolation.
///
/// `speed == 1.0` returns a copy. `speed == 2.0` returns roughly half as many
/// samples, i.e. the clip plays twice as fast on the timeline.
///
/// Linear interpolation is deliberate: the result feeds RMS and VAD, never a
/// speaker, and it avoids forcing `rubato` to accept a non-integer rate such
/// as `16_000 / 1.5`.
///
/// # Errors
///
/// [`CoreError::InvalidClipSpeed`] when `speed` is not finite or not positive.
/// `Clip::validate` already rejects such values, so this is a second line of
/// defence.
pub fn apply_speed(input: &MonoAudio, speed: f64) -> Result<MonoAudio, CoreError> {
    if !speed.is_finite() || speed <= 0.0 {
        return Err(CoreError::InvalidClipSpeed {
            id: "<speed>".to_string(),
            speed,
        });
    }
    if (speed - 1.0).abs() < 1e-9 {
        return Ok(input.clone());
    }

    let out_len = ((input.samples.len() as f64) / speed).round() as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let position = i as f64 * speed;
        let index = position.floor() as usize;
        let fraction = (position - index as f64) as f32;
        let a = input.samples.get(index).copied().unwrap_or(0.0);
        let b = input.samples.get(index + 1).copied().unwrap_or(a);
        out.push(a + (b - a) * fraction);
    }

    Ok(MonoAudio::new(out, input.sample_rate))
}

/// Averaged downmix of interleaved samples.
///
/// A trailing partial frame is dropped.
fn downmix_to_mono(interleaved: &[f32], channels: usize) -> Vec<f32> {
    if channels <= 1 {
        return interleaved.to_vec();
    }
    let scale = 1.0f32 / channels as f32;
    interleaved
        .chunks_exact(channels)
        .map(|frame| frame.iter().sum::<f32>() * scale)
        .collect()
}

/// Converts raw little-endian `f32` bytes into samples, dropping any trailing
/// bytes that do not form a complete sample.
fn pcm_f32le_to_samples(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Decodes real media files into the analysis domain.
///
/// Produces 16 kHz mono audio, already time-compressed by the clip's `speed`,
/// which is exactly what [`ClipDecoder`] promises.
#[derive(Debug, Clone, Copy, Default)]
pub struct FileDecoder {
    /// When `true`, a system `ffmpeg` on `PATH` is tried if symphonia cannot
    /// open or decode the file.
    pub allow_ffmpeg_fallback: bool,
}

impl FileDecoder {
    /// A decoder that uses symphonia only.
    #[must_use]
    pub fn symphonia_only() -> Self {
        Self {
            allow_ffmpeg_fallback: false,
        }
    }

    /// A decoder that falls back to a system `ffmpeg`.
    #[must_use]
    pub fn with_ffmpeg_fallback() -> Self {
        Self {
            allow_ffmpeg_fallback: true,
        }
    }
}

impl ClipDecoder for FileDecoder {
    fn decode_clip(&mut self, clip: &Clip) -> Result<MonoAudio, CoreError> {
        let path = Path::new(&clip.source);

        let native = match decode_native_mono(path) {
            Ok(audio) => audio,
            Err(symphonia_error) => {
                if !self.allow_ffmpeg_fallback {
                    return Err(CoreError::AudioDecode {
                        id: clip.id.clone(),
                        reason: symphonia_error.to_string(),
                    });
                }
                match decode_via_ffmpeg(path, ANALYSIS_SAMPLE_RATE) {
                    Ok(audio) => {
                        return apply_speed(&audio, clip.speed).map_err(|e| {
                            CoreError::AudioDecode {
                                id: clip.id.clone(),
                                reason: e.to_string(),
                            }
                        });
                    }
                    Err(ffmpeg_error) => {
                        return Err(CoreError::AudioDecode {
                            id: clip.id.clone(),
                            reason: format!(
                                "symphonia failed ({symphonia_error}); \
                                 ffmpeg fallback failed ({ffmpeg_error})"
                            ),
                        });
                    }
                }
            }
        };

        let at_analysis_rate =
            resample_to(&native, ANALYSIS_SAMPLE_RATE).map_err(|e| CoreError::AudioDecode {
                id: clip.id.clone(),
                reason: e.to_string(),
            })?;

        apply_speed(&at_analysis_rate, clip.speed).map_err(|e| CoreError::AudioDecode {
            id: clip.id.clone(),
            reason: e.to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn downmix_averages_channels() {
        // Two frames, two channels: (1.0, 0.0) and (0.0, 1.0).
        let interleaved = [1.0f32, 0.0, 0.0, 1.0];
        let mono = downmix_to_mono(&interleaved, 2);
        assert_eq!(mono, vec![0.5, 0.5]);
    }

    #[test]
    fn downmix_passes_mono_through() {
        let interleaved = [0.25f32, -0.5];
        let mono = downmix_to_mono(&interleaved, 1);
        assert_eq!(mono, vec![0.25, -0.5]);
    }

    #[test]
    fn downmix_drops_trailing_partial_frame() {
        // Five samples at three channels is one frame plus two stray samples.
        let interleaved = [3.0f32, 3.0, 3.0, 9.0, 9.0];
        let mono = downmix_to_mono(&interleaved, 3);
        assert_eq!(mono, vec![3.0]);
    }

    #[test]
    fn pcm_bytes_round_trip() {
        let original = [0.0f32, 0.5, -0.5, 1.0];
        let mut bytes = Vec::new();
        for sample in original {
            bytes.extend_from_slice(&sample.to_le_bytes());
        }
        assert_eq!(pcm_f32le_to_samples(&bytes), original.to_vec());
    }

    #[test]
    fn pcm_bytes_drop_trailing_garbage() {
        let mut bytes = 0.25f32.to_le_bytes().to_vec();
        bytes.extend_from_slice(&[0xff, 0xff]); // half a sample
        assert_eq!(pcm_f32le_to_samples(&bytes), vec![0.25]);
    }

    #[test]
    fn apply_speed_one_is_a_copy() {
        let input = MonoAudio::new(vec![1.0, 2.0, 3.0], 16_000);
        let out = apply_speed(&input, 1.0).unwrap();
        assert_eq!(out, input);
    }

    #[test]
    fn apply_speed_two_halves_the_length() {
        let input = MonoAudio::new(vec![0.0, 1.0, 0.0, -1.0], 16_000);
        let out = apply_speed(&input, 2.0).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out.sample_rate, 16_000);
    }

    #[test]
    fn apply_speed_half_doubles_the_length() {
        let input = MonoAudio::new(vec![0.0, 1.0, 0.0, -1.0], 16_000);
        let out = apply_speed(&input, 0.5).unwrap();
        assert_eq!(out.len(), 8);
    }

    #[test]
    fn apply_speed_rejects_non_positive() {
        let input = MonoAudio::new(vec![0.0; 8], 16_000);
        assert!(matches!(
            apply_speed(&input, 0.0),
            Err(CoreError::InvalidClipSpeed { .. })
        ));
        assert!(matches!(
            apply_speed(&input, -1.0),
            Err(CoreError::InvalidClipSpeed { .. })
        ));
    }

    #[test]
    fn apply_speed_rejects_nan() {
        let input = MonoAudio::new(vec![0.0; 8], 16_000);
        assert!(matches!(
            apply_speed(&input, f64::NAN),
            Err(CoreError::InvalidClipSpeed { .. })
        ));
    }
}