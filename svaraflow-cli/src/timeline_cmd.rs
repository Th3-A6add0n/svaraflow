//! `svaraflow-cli timeline` subcommands.

use std::path::Path;
use std::process::Command;

use anyhow::{anyhow, Context, Result};
use serde::Deserialize;
use svaraflow_core::timeline::{Clip, Fps, Timeline, Track, TrackKind};

use crate::cli::{TimelineCli, TimelineCommand};

/// Runs `timeline <subcommand>`.
pub fn run(args: TimelineCli) -> Result<()> {
    match args.command {
        TimelineCommand::FromMedia(a) => from_media(&a.file, &a.out),
    }
}

/// `timeline from-media`: probe `file` with ffprobe and write a
/// single-clip timeline to `out`.
fn from_media(file: &Path, out: &Path) -> Result<()> {
    if !file.exists() {
        return Err(anyhow!("media file does not exist: {}", file.display()));
    }

    let probe = probe_with_ffprobe(file)?;

    let fps = probe
        .fps()
        .ok_or_else(|| anyhow!("ffprobe did not report a frame rate for {}", file.display()))?;
    let (width, height) = probe.dimensions().ok_or_else(|| {
        anyhow!(
            "ffprobe did not report video dimensions for {}",
            file.display()
        )
    })?;
    let duration_s = probe
        .duration_seconds()
        .ok_or_else(|| anyhow!("ffprobe did not report a duration for {}", file.display()))?;

    let duration_frames = (duration_s * fps.as_f64()).round() as i64;
    if duration_frames <= 0 {
        return Err(anyhow!(
            "ffprobe reported a non-positive duration ({duration_s:.3} s) for {}",
            file.display()
        ));
    }

    let source = file
        .canonicalize()
        .with_context(|| format!("resolving {}", file.display()))?
        .to_string_lossy()
        .into_owned();

    let has_audio = probe.has_audio();

    let mut tracks = Vec::with_capacity(2);
    tracks.push(Track {
        kind: TrackKind::Video,
        index: 1,
        clips: vec![Clip {
            id: "video-1".to_string(),
            source: source.clone(),
            source_in: 0,
            source_out: duration_frames,
            start: 0,
            speed: 1.0,
            has_audio,
            muted: false,
        }],
    });
    if has_audio {
        tracks.push(Track {
            kind: TrackKind::Audio,
            index: 1,
            clips: vec![Clip {
                id: "audio-1".to_string(),
                source,
                source_in: 0,
                source_out: duration_frames,
                start: 0,
                speed: 1.0,
                has_audio: true,
                muted: false,
            }],
        });
    }

    let timeline = Timeline {
        fps,
        width,
        height,
        tracks,
    };
    timeline
        .validate()
        .map_err(|e| anyhow!("generated timeline failed validation: {e}"))?;

    let json = timeline
        .to_json_string_pretty()
        .map_err(|e| anyhow!("serialising timeline: {e}"))?;
    std::fs::write(out, json).with_context(|| format!("writing {}", out.display()))?;

    println!(
        "Wrote {} ({} frames at {} fps, {}x{}, audio: {})",
        out.display(),
        duration_frames,
        format_fps(fps),
        width,
        height,
        if has_audio { "yes" } else { "no" }
    );
    Ok(())
}

fn format_fps(fps: Fps) -> String {
    if fps.den == 1 {
        fps.num.to_string()
    } else {
        format!("{}/{}", fps.num, fps.den)
    }
}

/// A minimal view of ffprobe's JSON output.
#[derive(Debug, Deserialize)]
struct Probe {
    streams: Vec<Stream>,
    #[serde(default)]
    format: Format,
}

#[derive(Debug, Deserialize)]
struct Stream {
    codec_type: String,
    #[serde(default)]
    width: Option<u32>,
    #[serde(default)]
    height: Option<u32>,
    #[serde(default)]
    r_frame_rate: Option<String>,
    #[serde(default)]
    avg_frame_rate: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct Format {
    #[serde(default)]
    duration: Option<String>,
}

impl Probe {
    fn has_audio(&self) -> bool {
        self.streams.iter().any(|s| s.codec_type == "audio")
    }

    fn dimensions(&self) -> Option<(u32, u32)> {
        self.streams
            .iter()
            .find(|s| s.codec_type == "video")
            .and_then(|s| s.width.zip(s.height))
            .filter(|(w, h)| *w > 0 && *h > 0)
    }

    fn fps(&self) -> Option<Fps> {
        let stream = self.streams.iter().find(|s| s.codec_type == "video")?;
        let candidate = stream
            .r_frame_rate
            .as_deref()
            .filter(|s| !s.is_empty() && *s != "0/0")
            .or_else(|| stream.avg_frame_rate.as_deref());
        let rational = candidate?;
        parse_rational(rational)
    }

    fn duration_seconds(&self) -> Option<f64> {
        self.format
            .duration
            .as_deref()?
            .trim()
            .parse::<f64>()
            .ok()
            .filter(|d| d.is_finite() && *d > 0.0)
    }
}

/// Parses an `"N/D"` frame-rate string into a validated [`Fps`].
fn parse_rational(s: &str) -> Option<Fps> {
    let (num_s, den_s) = s.split_once('/')?;
    let num: u32 = num_s.trim().parse().ok()?;
    let den: u32 = den_s.trim().parse().ok()?;
    if num == 0 || den == 0 {
        return None;
    }
    Fps::new(num, den).ok()
}

/// Runs `ffprobe` and parses its JSON output.
fn probe_with_ffprobe(file: &Path) -> Result<Probe> {
    let output = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
        ])
        .arg(file)
        .output()
        .map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                anyhow!(
                    "ffprobe was not found on PATH. Install ffmpeg (which includes \
                     ffprobe) and ensure it is on PATH, or write timeline.json by hand."
                )
            } else {
                anyhow!("failed to run ffprobe: {e}")
            }
        })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(anyhow!(
            "ffprobe exited with {}: {}",
            output.status,
            stderr.trim()
        ));
    }

    let probe: Probe =
        serde_json::from_slice(&output.stdout).context("ffprobe returned invalid JSON")?;
    Ok(probe)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_rational_accepts_common_rates() {
        assert_eq!(parse_rational("30/1"), Some(Fps { num: 30, den: 1 }));
        assert_eq!(
            parse_rational("30000/1001"),
            Some(Fps {
                num: 30_000,
                den: 1001
            })
        );
        assert_eq!(parse_rational("25/1"), Some(Fps { num: 25, den: 1 }));
    }

    #[test]
    fn parse_rational_rejects_garbage() {
        assert_eq!(parse_rational(""), None);
        assert_eq!(parse_rational("30"), None);
        assert_eq!(parse_rational("0/0"), None);
        assert_eq!(parse_rational("30/0"), None);
        assert_eq!(parse_rational("abc/def"), None);
    }

    #[test]
    fn probe_detects_video_stream() {
        let json = r#"{
            "streams": [
                {"codec_type": "video", "width": 1920, "height": 1080, "r_frame_rate": "30000/1001"},
                {"codec_type": "audio"}
            ],
            "format": {"duration": "10.5"}
        }"#;
        let probe: Probe = serde_json::from_str(json).unwrap();
        assert!(probe.has_audio());
        assert_eq!(probe.dimensions(), Some((1920, 1080)));
        assert_eq!(
            probe.fps(),
            Some(Fps {
                num: 30_000,
                den: 1001
            })
        );
        assert!((probe.duration_seconds().unwrap() - 10.5).abs() < 1e-9);
    }

    #[test]
    fn probe_handles_audio_only() {
        let json = r#"{
            "streams": [{"codec_type": "audio"}],
            "format": {"duration": "3.0"}
        }"#;
        let probe: Probe = serde_json::from_str(json).unwrap();
        assert!(probe.has_audio());
        assert_eq!(probe.dimensions(), None);
        assert_eq!(probe.fps(), None);
    }

    #[test]
    fn probe_falls_back_to_avg_frame_rate() {
        let json = r#"{
            "streams": [
                {"codec_type": "video", "width": 640, "height": 480, "avg_frame_rate": "25/1"}
            ],
            "format": {"duration": "2.0"}
        }"#;
        let probe: Probe = serde_json::from_str(json).unwrap();
        assert_eq!(probe.fps(), Some(Fps { num: 25, den: 1 }));
    }

    #[test]
    fn probe_treats_zero_over_zero_as_missing() {
        let json = r#"{
            "streams": [
                {"codec_type": "video", "width": 640, "height": 480, "r_frame_rate": "0/0"}
            ],
            "format": {"duration": "2.0"}
        }"#;
        let probe: Probe = serde_json::from_str(json).unwrap();
        assert_eq!(probe.fps(), None);
    }
}
