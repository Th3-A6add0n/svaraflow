//! `svaraflow-cli timeline` subcommands.
//!
//! `from-media` probes a media file with `ffprobe` and writes a
//! single-track timeline. Attached cover art
//! (`disposition.attached_pic == 1`) is not treated as a video stream.

use std::path::Path;
use std::process::Command;

use anyhow::{anyhow, Context, Result};
use serde::Deserialize;
use svaraflow_core::timeline::{Clip, Fps, Timeline, Track, TrackKind};

use crate::cli::{FromMediaArgs, TimelineCli, TimelineCommand};

/// Audio properties the XML writer needs but that `timeline.json` does not
/// carry. Obtained by [`probe_audio_properties`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AudioProperties {
    /// Channel count: `1` mono, `2` stereo. Falls back to `1`.
    pub channels: u32,
    /// Sample rate in Hz. Falls back to `48_000`.
    pub sample_rate: u32,
    /// Bit depth. Falls back to `16` when the probe reports `0` or
    /// nothing, which is the case for every lossy codec.
    pub bit_depth: u32,
}

impl Default for AudioProperties {
    fn default() -> Self {
        Self {
            channels: 1,
            sample_rate: 48_000,
            bit_depth: 16,
        }
    }
}

/// Runs `ffprobe` on the first audio stream of `path` and returns its
/// channel count, sample rate and bit depth.
///
/// Never fails on missing data: if the file has no audio stream, or the
/// probe cannot be run, the defaults are returned.
pub fn probe_audio_properties(path: &Path) -> AudioProperties {
    let output = match Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=channels,sample_rate,bits_per_sample",
            "-of",
            "json",
        ])
        .arg(path)
        .output()
    {
        Ok(o) => o,
        Err(e) => {
            eprintln!(
                "warning: ffprobe unavailable for {} ({e}); assuming mono 48 kHz",
                path.display()
            );
            return AudioProperties::default();
        }
    };

    if !output.status.success() {
        eprintln!(
            "warning: ffprobe failed on {}; assuming mono 48 kHz",
            path.display()
        );
        return AudioProperties::default();
    }

    let probe: AudioProbe = match serde_json::from_slice(&output.stdout) {
        Ok(p) => p,
        Err(_) => {
            eprintln!(
                "warning: ffprobe returned invalid JSON for {}; assuming mono 48 kHz",
                path.display()
            );
            return AudioProperties::default();
        }
    };

    let Some(stream) = probe.streams.into_iter().next() else {
        return AudioProperties::default();
    };

    let channels = stream.channels.unwrap_or(1).max(1);
    let sample_rate = stream
        .sample_rate
        .as_deref()
        .and_then(|s| s.trim().parse::<u32>().ok())
        .filter(|r| *r > 0)
        .unwrap_or(48_000);
    let bit_depth = stream.bits_per_sample.filter(|b| *b > 0).unwrap_or(16);

    AudioProperties {
        channels,
        sample_rate,
        bit_depth,
    }
}

/// Minimal ffprobe audio-stream JSON.
#[derive(Debug, Deserialize)]
struct AudioProbe {
    #[serde(default)]
    streams: Vec<AudioStream>,
}

#[derive(Debug, Deserialize)]
struct AudioStream {
    #[serde(default)]
    channels: Option<u32>,
    #[serde(default)]
    sample_rate: Option<String>,
    #[serde(default)]
    bits_per_sample: Option<u32>,
}

/// Runs `timeline <subcommand>`.
pub fn run(args: TimelineCli) -> Result<()> {
    match args.command {
        TimelineCommand::FromMedia(a) => from_media(&a),
    }
}

fn from_media(args: &FromMediaArgs) -> Result<()> {
    let file = &args.file;
    if !file.exists() {
        return Err(anyhow!("media file does not exist: {}", file.display()));
    }

    let fallback_fps = parse_fps_arg(&args.fps).map_err(|e| anyhow!("{e}"))?;

    // Canonicalize once, here, where the file is known to exist.
    // `dunce::canonicalize` strips the Windows `\\?\` verbatim prefix, so
    // the path we embed in `timeline.json` is the plain absolute path.
    let source = dunce::canonicalize(file)
        .with_context(|| format!("resolving {}", file.display()))?
        .to_string_lossy()
        .into_owned();

    let probe = probe_with_ffprobe(file)?;
    let built = build_from_probe(&probe, &source, args, fallback_fps)?;

    built
        .timeline
        .validate()
        .map_err(|e| anyhow!("generated timeline failed validation: {e}"))?;

    let json = built
        .timeline
        .to_json_string_pretty()
        .map_err(|e| anyhow!("serialising timeline: {e}"))?;
    std::fs::write(&args.out, json).with_context(|| format!("writing {}", args.out.display()))?;

    if probe.has_audio() {
        let props = probe_audio_properties(file);
        println!(
            "Audio source: {} channel(s), {} Hz, {}-bit",
            props.channels, props.sample_rate, props.bit_depth
        );
    }

    if built.audio_only {
        println!(
            "Note: {} is audio-only. Built an audio-only timeline at {} fps; \
             raster size ({}x{}) is a placeholder.",
            file.display(),
            format_fps_pretty(built.timeline.fps),
            built.timeline.width,
            built.timeline.height,
        );
    } else {
        let has_audio = built
            .timeline
            .tracks
            .iter()
            .any(|t| t.kind == TrackKind::Audio);
        println!(
            "Wrote {} ({} frames at {} fps, {}x{}, audio: {})",
            args.out.display(),
            built.duration_frames,
            format_fps_pretty(built.timeline.fps),
            built.timeline.width,
            built.timeline.height,
            if has_audio { "yes" } else { "no" }
        );
    }
    Ok(())
}

#[derive(Debug)]
struct Built {
    timeline: Timeline,
    audio_only: bool,
    duration_frames: i64,
}

/// Builds a timeline from a probe and a resolved source path.
///
/// `source` is the canonical, verbatim-prefix-free absolute path that will
/// be embedded in the timeline. It is produced by the caller (see
/// [`from_media`]); this function does no I/O and therefore is testable
/// with synthetic probes and synthetic paths.
fn build_from_probe(
    probe: &Probe,
    source: &str,
    args: &FromMediaArgs,
    fallback_fps: Fps,
) -> Result<Built> {
    let duration_s = probe.duration_seconds().ok_or_else(|| {
        anyhow!(
            "ffprobe did not report a duration for {source}\nStreams seen:\n{}",
            probe.describe()
        )
    })?;

    let has_audio = probe.has_audio();

    match probe.first_video() {
        Some(video) => {
            let fps = video.frame_rate().ok_or_else(|| {
                anyhow!(
                    "video stream {} has no usable frame rate \
                     (avg_frame_rate = {:?}, r_frame_rate = {:?}).\n\
                     Pass --fps <rate> to override, e.g. --fps 30000/1001.\n\
                     Streams seen:\n{}",
                    video.label(),
                    video.avg_frame_rate.as_deref().unwrap_or("missing"),
                    video.r_frame_rate.as_deref().unwrap_or("missing"),
                    probe.describe()
                )
            })?;

            let (width, height) = video
                .width
                .zip(video.height)
                .filter(|(w, h)| *w > 0 && *h > 0)
                .ok_or_else(|| {
                    anyhow!(
                        "video stream {} has no usable dimensions \
                         (width = {:?}, height = {:?}).\n\
                         Pass --width and --height to override.\n\
                         Streams seen:\n{}",
                        video.label(),
                        video.width,
                        video.height,
                        probe.describe()
                    )
                })?;

            let duration_frames = frames_for(duration_s, fps, source)?;

            let mut tracks = Vec::with_capacity(2);
            tracks.push(Track {
                kind: TrackKind::Video,
                index: 1,
                clips: vec![Clip {
                    id: "video-1".to_string(),
                    source: source.to_string(),
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
                        source: source.to_string(),
                        source_in: 0,
                        source_out: duration_frames,
                        start: 0,
                        speed: 1.0,
                        has_audio: true,
                        muted: false,
                    }],
                });
            }

            Ok(Built {
                timeline: Timeline {
                    fps,
                    width,
                    height,
                    tracks,
                },
                audio_only: false,
                duration_frames,
            })
        }
        None => {
            if !has_audio {
                return Err(anyhow!(
                    "{source} has no video stream and no audio stream; nothing to \
                     build a timeline from.\nStreams seen:\n{}",
                    probe.describe()
                ));
            }
            let fps = fallback_fps;
            let duration_frames = frames_for(duration_s, fps, source)?;

            Ok(Built {
                timeline: Timeline {
                    fps,
                    width: args.width,
                    height: args.height,
                    tracks: vec![Track {
                        kind: TrackKind::Audio,
                        index: 1,
                        clips: vec![Clip {
                            id: "audio-1".to_string(),
                            source: source.to_string(),
                            source_in: 0,
                            source_out: duration_frames,
                            start: 0,
                            speed: 1.0,
                            has_audio: true,
                            muted: false,
                        }],
                    }],
                },
                audio_only: true,
                duration_frames,
            })
        }
    }
}

fn frames_for(duration_s: f64, fps: Fps, label: &str) -> Result<i64> {
    let frames = (duration_s * fps.as_f64()).round() as i64;
    if frames <= 0 {
        return Err(anyhow!(
            "computed timeline duration for {label} is {frames} frames, which \
             is not positive; the probe reported {duration_s:.3} s at {} fps",
            format_fps_pretty(fps)
        ));
    }
    Ok(frames)
}

fn format_fps_pretty(fps: Fps) -> String {
    if fps.den == 1 {
        fps.num.to_string()
    } else {
        format!("{:.3} ({}/{})", fps.as_f64(), fps.num, fps.den)
    }
}

/// Parses the `--fps` flag.
pub fn parse_fps_arg(s: &str) -> Result<Fps, String> {
    let s = s.trim();
    if s.is_empty() {
        return Err("--fps must not be empty".to_string());
    }

    if let Some((num_s, den_s)) = s.split_once('/') {
        let num = num_s.trim().parse::<u32>().map_err(|_| {
            format!(
                "--fps fraction numerator `{}` is not a positive integer",
                num_s.trim()
            )
        })?;
        let den = den_s.trim().parse::<u32>().map_err(|_| {
            format!(
                "--fps fraction denominator `{}` is not a positive integer",
                den_s.trim()
            )
        })?;
        return Fps::new(num, den)
            .map_err(|_| format!("--fps fraction {num}/{den} is invalid (both parts must be > 0)"));
    }

    match s {
        "23.976" | "23.98" => return Fps::new(24000, 1001).map_err(|e| e.to_string()),
        "29.97" => return Fps::new(30000, 1001).map_err(|e| e.to_string()),
        "59.94" => return Fps::new(60000, 1001).map_err(|e| e.to_string()),
        _ => {}
    }

    if let Ok(n) = s.parse::<u32>() {
        if n == 0 {
            return Err("--fps must be > 0".to_string());
        }
        return Fps::new(n, 1).map_err(|e| e.to_string());
    }

    Err(format!(
        "--fps `{s}` is not a recognised frame rate. \
         Try an integer (24, 25, 30, 50, 60), an NTSC value \
         (23.976, 29.97, 59.94), or a fraction such as 30000/1001"
    ))
}

#[derive(Debug, Deserialize)]
struct Probe {
    #[serde(default)]
    streams: Vec<Stream>,
    #[serde(default)]
    format: Format,
}

#[derive(Debug, Deserialize)]
struct Stream {
    #[serde(default)]
    index: Option<u32>,
    #[serde(default)]
    codec_type: Option<String>,
    #[serde(default)]
    codec_name: Option<String>,
    #[serde(default)]
    width: Option<u32>,
    #[serde(default)]
    height: Option<u32>,
    #[serde(default)]
    r_frame_rate: Option<String>,
    #[serde(default)]
    avg_frame_rate: Option<String>,
    #[serde(default)]
    duration: Option<String>,
    #[serde(default)]
    disposition: Disposition,
}

#[derive(Debug, Default, Deserialize)]
struct Disposition {
    #[serde(default)]
    attached_pic: u8,
}

#[derive(Debug, Default, Deserialize)]
struct Format {
    #[serde(default)]
    duration: Option<String>,
}

impl Probe {
    fn has_audio(&self) -> bool {
        self.streams
            .iter()
            .any(|s| s.codec_type.as_deref() == Some("audio"))
    }

    fn first_video(&self) -> Option<&Stream> {
        self.streams.iter().find(|s| s.is_real_video())
    }

    fn duration_seconds(&self) -> Option<f64> {
        if let Some(d) = self.format.duration.as_deref().and_then(parse_duration) {
            return Some(d);
        }
        self.streams
            .iter()
            .filter_map(|s| s.duration.as_deref().and_then(parse_duration))
            .reduce(f64::max)
    }

    fn describe(&self) -> String {
        if self.streams.is_empty() {
            return "  (ffprobe reported no streams)".to_string();
        }
        self.streams
            .iter()
            .map(Stream::describe)
            .collect::<Vec<_>>()
            .join("\n")
    }
}

impl Stream {
    fn is_real_video(&self) -> bool {
        self.codec_type.as_deref() == Some("video") && self.disposition.attached_pic == 0
    }

    fn frame_rate(&self) -> Option<Fps> {
        parse_rational_rate(self.avg_frame_rate.as_deref())
            .or_else(|| parse_rational_rate(self.r_frame_rate.as_deref()))
    }

    fn label(&self) -> String {
        match self.index {
            Some(i) => format!("#{i}"),
            None => "#?".to_string(),
        }
    }

    fn describe(&self) -> String {
        let kind = self.codec_type.as_deref().unwrap_or("unknown");
        let codec = self.codec_name.as_deref().unwrap_or("?");
        let mut line = format!("  {}: {kind}/{codec}", self.label());
        if self.is_real_video() {
            let w = self.width.unwrap_or(0);
            let h = self.height.unwrap_or(0);
            let fps = self
                .frame_rate()
                .map(format_fps_pretty)
                .unwrap_or_else(|| "fps=?".to_string());
            line.push_str(&format!(" ({w}x{h}, {fps})"));
        } else if self.disposition.attached_pic != 0 {
            line.push_str(" (attached picture)");
        }
        line
    }
}

fn parse_rational_rate(s: Option<&str>) -> Option<Fps> {
    let s = s?.trim();
    if s.is_empty() {
        return None;
    }
    let (num_s, den_s) = s.split_once('/')?;
    let num: u32 = num_s.trim().parse().ok()?;
    let den: u32 = den_s.trim().parse().ok()?;
    if num == 0 || den == 0 {
        return None;
    }
    Fps::new(num, den).ok()
}

fn parse_duration(s: &str) -> Option<f64> {
    s.trim()
        .parse::<f64>()
        .ok()
        .filter(|d| d.is_finite() && *d > 0.0)
}

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
                    "ffprobe was not found on PATH. Install ffmpeg (which \
                     includes ffprobe) and ensure it is on PATH, or write \
                     timeline.json by hand."
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

    /// A stable absolute path used in tests that exercise
    /// `build_from_probe` without touching the filesystem. It must look
    /// like the output of `dunce::canonicalize`: no `\\?\` prefix.
    const TEST_SOURCE: &str = "C:\\media\\a.mp4";

    fn args_with(fps: &str, width: u32, height: u32) -> FromMediaArgs {
        FromMediaArgs {
            file: "media.mp4".into(),
            out: "tl.json".into(),
            fps: fps.to_string(),
            width,
            height,
        }
    }

    #[test]
    fn parse_fps_arg_integer() {
        assert_eq!(parse_fps_arg("30").unwrap(), Fps { num: 30, den: 1 });
    }

    #[test]
    fn parse_fps_arg_ntsc_decimals() {
        assert_eq!(
            parse_fps_arg("29.97").unwrap(),
            Fps {
                num: 30_000,
                den: 1001
            }
        );
    }

    #[test]
    fn parse_fps_arg_fraction() {
        assert_eq!(
            parse_fps_arg("30000/1001").unwrap(),
            Fps {
                num: 30_000,
                den: 1001
            }
        );
    }

    #[test]
    fn parse_fps_arg_rejects_garbage() {
        assert!(parse_fps_arg("").is_err());
        assert!(parse_fps_arg("0").is_err());
        assert!(parse_fps_arg("abc").is_err());
    }

    fn json_audio_only() -> &'static str {
        r#"{
            "streams": [
                {"index": 0, "codec_type": "audio", "codec_name": "aac"}
            ],
            "format": {"duration": "12.5"}
        }"#
    }

    fn json_video_plus_audio() -> &'static str {
        r#"{
            "streams": [
                {"index": 0, "codec_type": "video", "codec_name": "h264",
                 "width": 1920, "height": 1080,
                 "avg_frame_rate": "30000/1001", "r_frame_rate": "30000/1001",
                 "disposition": {"attached_pic": 0}},
                {"index": 1, "codec_type": "audio", "codec_name": "aac"}
            ],
            "format": {"duration": "10.5"}
        }"#
    }

    fn json_cover_art() -> &'static str {
        r#"{
            "streams": [
                {"index": 0, "codec_type": "audio", "codec_name": "mp3"},
                {"index": 1, "codec_type": "video", "codec_name": "mjpeg",
                 "width": 600, "height": 600,
                 "avg_frame_rate": "90000/1", "r_frame_rate": "90000/1",
                 "disposition": {"attached_pic": 1}}
            ],
            "format": {"duration": "180.0"}
        }"#
    }

    fn parse(json: &str) -> Probe {
        serde_json::from_str(json).expect("parse probe json")
    }

    #[test]
    fn probe_audio_only() {
        let p = parse(json_audio_only());
        assert!(p.has_audio());
        assert!(p.first_video().is_none());
    }

    #[test]
    fn probe_ignores_attached_picture() {
        let p = parse(json_cover_art());
        assert!(p.first_video().is_none());
    }

    #[test]
    fn build_audio_only_uses_fallback_fps() {
        let probe = parse(json_audio_only());
        let args = args_with("30000/1001", 1920, 1080);
        let fps = parse_fps_arg(&args.fps).unwrap();
        let built = build_from_probe(&probe, TEST_SOURCE, &args, fps).unwrap();
        assert!(built.audio_only);
        assert_eq!(built.timeline.tracks.len(), 1);
        assert_eq!(built.duration_frames, 375);
        assert_eq!(built.timeline.tracks[0].kind, TrackKind::Audio);
        assert_eq!(built.timeline.tracks[0].clips[0].source, TEST_SOURCE);
    }

    #[test]
    fn build_video_plus_audio() {
        let probe = parse(json_video_plus_audio());
        let args = args_with("30000/1001", 1920, 1080);
        let fps = parse_fps_arg(&args.fps).unwrap();
        let built = build_from_probe(&probe, TEST_SOURCE, &args, fps).unwrap();
        assert!(!built.audio_only);
        assert_eq!(built.timeline.tracks.len(), 2);
        assert_eq!(built.timeline.tracks[0].kind, TrackKind::Video);
        assert_eq!(built.timeline.tracks[1].kind, TrackKind::Audio);
        assert_eq!(built.timeline.tracks[0].clips[0].source, TEST_SOURCE);
    }

    #[test]
    fn build_cover_art_is_audio_only() {
        let probe = parse(json_cover_art());
        let args = args_with("30000/1001", 1920, 1080);
        let fps = parse_fps_arg(&args.fps).unwrap();
        let built = build_from_probe(&probe, TEST_SOURCE, &args, fps).unwrap();
        assert!(built.audio_only);
        assert_eq!(built.timeline.tracks.len(), 1);
    }

    #[test]
    fn build_no_av_streams_errors() {
        let json = r#"{
            "streams": [
                {"index": 0, "codec_type": "data", "codec_name": "bin_data"}
            ],
            "format": {"duration": "1.0"}
        }"#;
        let probe = parse(json);
        let args = args_with("30000/1001", 1920, 1080);
        let fps = parse_fps_arg(&args.fps).unwrap();
        let err = build_from_probe(&probe, TEST_SOURCE, &args, fps).unwrap_err();
        assert!(err.to_string().contains("no video stream and no audio stream"));
    }

    #[test]
    fn audio_properties_default_is_mono_48k_16() {
        let d = AudioProperties::default();
        assert_eq!(d.channels, 1);
        assert_eq!(d.sample_rate, 48_000);
        assert_eq!(d.bit_depth, 16);
    }

    #[test]
    fn audio_probe_parses_stereo_aac() {
        let json = r#"{
            "streams": [
                {"channels": 2, "sample_rate": "22050", "bits_per_sample": 0}
            ]
        }"#;
        let probe: AudioProbe = serde_json::from_str(json).unwrap();
        let stream = probe.streams.into_iter().next().unwrap();
        assert_eq!(stream.channels, Some(2));
        assert_eq!(stream.sample_rate.as_deref(), Some("22050"));
        assert_eq!(stream.bits_per_sample, Some(0));
    }
}