//! Shared helpers for `svaraflow-cli` integration tests.
//!
//! Only `pipeline.rs` currently uses this module. `#![allow(dead_code)]`
//! silences unused-function warnings that would otherwise appear when a
//! future test file imports a subset.

#![allow(dead_code)]

use std::io::Write;
use std::path::{Path, PathBuf};

/// Returns a path under Cargo's per-test-binary temp directory.
///
/// Cargo creates `target/tmp/` fresh for each test binary, so there is no
/// cross-test contamination. Callers do not need to clean up.
pub fn tmp_path(name: &str) -> PathBuf {
    let mut dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    std::fs::create_dir_all(&dir).expect("create CARGO_TARGET_TMPDIR");
    dir.push(name);
    dir
}

/// Writes a mono 16-bit PCM WAV file at `path`.
///
/// Standard 44-byte RIFF header followed by interleaved `i16` samples in
/// little-endian order. Input `f32` values outside `[-1, 1]` are clipped
/// before conversion so the writer never wraps.
pub fn write_wav(path: &Path, samples: &[f32], sample_rate: u32) -> std::io::Result<()> {
    const BITS_PER_SAMPLE: u16 = 16;
    const CHANNELS: u16 = 1;

    let byte_rate = sample_rate * u32::from(CHANNELS) * u32::from(BITS_PER_SAMPLE) / 8;
    let block_align = CHANNELS * BITS_PER_SAMPLE / 8;
    let data_size =
        (samples.len() as u32) * u32::from(CHANNELS) * u32::from(BITS_PER_SAMPLE) / 8;
    let riff_size = 36 + data_size;

    let mut file = std::fs::File::create(path)?;
    file.write_all(b"RIFF")?;
    file.write_all(&riff_size.to_le_bytes())?;
    file.write_all(b"WAVE")?;
    file.write_all(b"fmt ")?;
    file.write_all(&16u32.to_le_bytes())?;
    file.write_all(&1u16.to_le_bytes())?;
    file.write_all(&CHANNELS.to_le_bytes())?;
    file.write_all(&sample_rate.to_le_bytes())?;
    file.write_all(&byte_rate.to_le_bytes())?;
    file.write_all(&block_align.to_le_bytes())?;
    file.write_all(&BITS_PER_SAMPLE.to_le_bytes())?;
    file.write_all(b"data")?;
    file.write_all(&data_size.to_le_bytes())?;

    for s in samples {
        let clipped = s.clamp(-1.0, 1.0);
        let value = (f64::from(clipped) * 32767.0).round() as i16;
        file.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}

/// Appends `duration_s` seconds of silence to `out` at `sample_rate`.
pub fn append_silence(out: &mut Vec<f32>, duration_s: f64, sample_rate: u32) {
    let count = (duration_s * f64::from(sample_rate)).round() as usize;
    out.resize(out.len() + count, 0.0f32);
}

/// Appends `duration_s` seconds of a 1 kHz sine at `amplitude` to `out`.
pub fn append_sine(out: &mut Vec<f32>, duration_s: f64, amplitude: f32, sample_rate: u32) {
    let count = (duration_s * f64::from(sample_rate)).round() as usize;
    let two_pi = 2.0 * std::f64::consts::PI;
    for i in 0..count {
        let t = i as f64 / f64::from(sample_rate);
        let sample = f64::from(amplitude) * (two_pi * 1000.0 * t).sin();
        out.push(sample as f32);
    }
}

/// Escapes a filesystem path for embedding in a JSON string literal.
///
/// Windows backslashes are doubled; forward slashes and other bytes pass
/// through unchanged. The result is the *contents* of the JSON string, not
/// including the surrounding quotes.
pub fn escape_for_json(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "\\\\")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_wav_produces_a_readable_file() {
        let path = tmp_path("common_write_wav_probe.wav");
        let samples: Vec<f32> = (0..1600).map(|i| (i as f32 / 1600.0) * 2.0 - 1.0).collect();
        write_wav(&path, &samples, 16_000).unwrap();
        let metadata = std::fs::metadata(&path).unwrap();
        assert_eq!(metadata.len(), 44 + 3200);
    }

    #[test]
    fn append_silence_matches_duration() {
        let mut out = Vec::new();
        append_silence(&mut out, 0.5, 16_000);
        assert_eq!(out.len(), 8_000);
        assert!(out.iter().all(|s| *s == 0.0));
    }

    #[test]
    fn append_silence_appends_to_existing_content() {
        let mut out = vec![0.5f32; 100];
        append_silence(&mut out, 0.5, 16_000);
        assert_eq!(out.len(), 8_100);
        assert!(out[..100].iter().all(|s| (*s - 0.5).abs() < 1e-6));
        assert!(out[100..].iter().all(|s| *s == 0.0));
    }

    #[test]
    fn append_sine_matches_duration() {
        let mut out = Vec::new();
        append_sine(&mut out, 0.25, 0.9, 16_000);
        assert_eq!(out.len(), 4_000);
        let peak = out.iter().fold(0.0f32, |m, s| m.max(s.abs()));
        assert!((peak - 0.9).abs() < 0.01, "peak was {peak}");
    }

    #[test]
    fn escape_for_json_doubles_backslashes() {
        let escaped = escape_for_json(Path::new("C:\\media\\a.wav"));
        assert_eq!(escaped, "C:\\\\media\\\\a.wav");
    }

    #[test]
    fn escape_for_json_leaves_forward_slashes() {
        let escaped = escape_for_json(Path::new("/tmp/a.wav"));
        assert_eq!(escaped, "/tmp/a.wav");
    }
}