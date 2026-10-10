//! Command-line interface definitions for `svaraflow-cli`.

use std::path::PathBuf;

use clap::{Args, Parser, Subcommand, ValueEnum};

/// SvaraFlow command-line interface.
#[derive(Debug, Parser)]
#[command(
    name = "svaraflow-cli",
    version,
    about = "SvaraFlow CLI — silence removal for Premiere Pro and DaVinci Resolve",
    long_about = None,
)]
pub struct Cli {
    /// Subcommand to run.
    #[command(subcommand)]
    pub command: Command,
}

/// Top-level subcommands.
#[derive(Debug, Subcommand)]
pub enum Command {
    /// Timeline utilities.
    Timeline(TimelineCli),
    /// Silence detection and cutting.
    Silence(SilenceCli),
}

/// `timeline` subcommand tree.
#[derive(Debug, Args)]
pub struct TimelineCli {
    /// Action to perform.
    #[command(subcommand)]
    pub command: TimelineCommand,
}

/// Actions under `timeline`.
#[derive(Debug, Subcommand)]
pub enum TimelineCommand {
    /// Build a single-clip timeline.json from a media file.
    ///
    /// Uses `ffprobe` (from a system ffmpeg install) to read the media's
    /// duration, dimensions and frame rate.
    FromMedia(FromMediaArgs),
}

/// Arguments for `timeline from-media`.
#[derive(Debug, Args)]
pub struct FromMediaArgs {
    /// Path to the media file.
    pub file: PathBuf,
    /// Where to write `timeline.json`.
    #[arg(long)]
    pub out: PathBuf,
}

/// `silence` subcommand tree.
#[derive(Debug, Args)]
pub struct SilenceCli {
    /// Action to perform.
    #[command(subcommand)]
    pub command: SilenceCommand,
}

/// Actions under `silence`.
#[derive(Debug, Subcommand)]
pub enum SilenceCommand {
    /// Analyze the timeline and report keep ranges and stats.
    Analyze(AnalyzeArgs),
    /// Analyze the timeline and write the cut result as XML and/or EDL.
    Apply(ApplyArgs),
}

/// Options common to `silence analyze` and `silence apply`.
#[derive(Debug, Args)]
pub struct DetectionOptions {
    /// Path to `timeline.json`.
    pub timeline: PathBuf,
    /// Noise floor: `auto` (default) or a dBFS value such as `-45`.
    ///
    /// Values are negative dBFS numbers, so the leading hyphen must be
    /// accepted as part of the value rather than parsed as a flag.
    #[arg(long, default_value = "auto", allow_hyphen_values = true)]
    pub noise: String,
    /// Minimum silence to remove, in seconds. Overrides the preset.
    #[arg(long)]
    pub min_silence: Option<f64>,
    /// Minimum talk to keep, in seconds. Overrides the preset.
    #[arg(long)]
    pub min_talk: Option<f64>,
    /// Pad before each sound run, in seconds. Overrides the preset.
    #[arg(long)]
    pub pad_before: Option<f64>,
    /// Pad after each sound run, in seconds. Overrides the preset.
    #[arg(long)]
    pub pad_after: Option<f64>,
    /// Detection method.
    #[arg(long, value_enum, default_value_t = MethodArg::Threshold)]
    pub method: MethodArg,
    /// Cut mode. Ignored by `silence analyze`.
    #[arg(long, value_enum, default_value_t = ModeArg::Remove)]
    pub mode: ModeArg,
    /// Tuning preset. Provides defaults for the four numeric options above;
    /// explicit values on the command line win.
    #[arg(long, value_enum)]
    pub preset: Option<PresetArg>,
    /// Path to the Silero VAD ONNX model.
    ///
    /// Required for `--method vad` and `--method hybrid`. If omitted, the
    /// CLI looks for `models/silero_vad_v5.onnx` relative to the current
    /// directory.
    #[arg(long)]
    pub vad_model: Option<PathBuf>,
    /// Analyze a pre-made mixdown WAV instead of the timeline's clips.
    #[arg(long)]
    pub audio: Option<PathBuf>,
    /// Audio track indices to analyze and cut, comma-separated.
    #[arg(long, value_delimiter = ',', num_args = 0..)]
    pub audio_tracks: Option<Vec<u32>>,
    /// Disable the ffmpeg fallback when symphonia cannot decode a clip.
    #[arg(long)]
    pub no_ffmpeg: bool,
}

/// Arguments for `silence analyze`.
#[derive(Debug, Args)]
pub struct AnalyzeArgs {
    /// Detection and analysis options.
    #[command(flatten)]
    pub opts: DetectionOptions,
    /// Print keep ranges and stats as JSON instead of a human-readable
    /// table.
    #[arg(long)]
    pub json: bool,
}

/// Arguments for `silence apply`.
#[derive(Debug, Args)]
pub struct ApplyArgs {
    /// Detection and analysis options.
    #[command(flatten)]
    pub opts: DetectionOptions,
    /// Write the cut timeline as FCP7 XML.
    #[arg(long)]
    pub xml: Option<PathBuf>,
    /// Write the cut timeline as CMX3600 EDL.
    #[arg(long)]
    pub edl: Option<PathBuf>,
    /// Sequence name to use in the XML and EDL output. Defaults to the
    /// timeline file's stem.
    #[arg(long)]
    pub sequence_name: Option<String>,
}

/// Detection method, as accepted on the command line.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum MethodArg {
    /// Threshold on the dBFS envelope.
    Threshold,
    /// Silero VAD only.
    Vad,
    /// Silero VAD for decisions, envelope for boundary snapping.
    Hybrid,
}

/// Cut mode, as accepted on the command line.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum ModeArg {
    /// Keep ranges concatenated from frame 0.
    Remove,
    /// Every range becomes its own clip; timeline length unchanged.
    Keep,
    /// As Keep, but silence clips on audio tracks are muted.
    Mute,
    /// Keep ranges stay at their original positions; silences are empty.
    Gaps,
}

/// Preset name, as accepted on the command line.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum PresetArg {
    /// remove 0.80 s, keep 0.30 s.
    Calm,
    /// remove 0.50 s, keep 0.25 s.
    Measured,
    /// remove 0.35 s, keep 0.20 s.
    Paced,
    /// remove 0.20 s, keep 0.15 s, padding 0.08 s each side.
    Fast,
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn version_flag_is_intercepted_by_clap() {
        assert!(Cli::try_parse_from(["svaraflow-cli", "--version"]).is_err());
    }

    #[test]
    fn help_flag_is_intercepted_by_clap() {
        assert!(Cli::try_parse_from(["svaraflow-cli", "--help"]).is_err());
    }

    #[test]
    fn parses_silence_analyze_defaults() {
        let cli =
            Cli::try_parse_from(["svaraflow-cli", "silence", "analyze", "tl.json"]).expect("parse");
        let Command::Silence(s) = cli.command else {
            panic!();
        };
        let SilenceCommand::Analyze(a) = s.command else {
            panic!();
        };
        assert_eq!(a.opts.timeline, PathBuf::from("tl.json"));
        assert_eq!(a.opts.method, MethodArg::Threshold);
        assert!(a.opts.vad_model.is_none());
    }

    #[test]
    fn parses_vad_model_flag() {
        let cli = Cli::try_parse_from([
            "svaraflow-cli",
            "silence",
            "analyze",
            "tl.json",
            "--method",
            "vad",
            "--vad-model",
            "C:\\models\\silero.onnx",
        ])
        .expect("parse");
        let Command::Silence(s) = cli.command else {
            panic!();
        };
        let SilenceCommand::Analyze(a) = s.command else {
            panic!();
        };
        assert_eq!(a.opts.method, MethodArg::Vad);
        assert_eq!(
            a.opts.vad_model,
            Some(PathBuf::from("C:\\models\\silero.onnx"))
        );
    }

    #[test]
    fn parses_silence_analyze_overrides() {
        let cli = Cli::try_parse_from([
            "svaraflow-cli",
            "silence",
            "analyze",
            "tl.json",
            "--noise",
            "-42.5",
            "--min-silence",
            "0.8",
            "--min-talk",
            "0.3",
            "--pad-before",
            "0.15",
            "--pad-after",
            "0.15",
            "--method",
            "hybrid",
            "--mode",
            "mute",
            "--json",
        ])
        .expect("parse");
        let Command::Silence(s) = cli.command else {
            panic!();
        };
        let SilenceCommand::Analyze(a) = s.command else {
            panic!();
        };
        assert_eq!(a.opts.noise, "-42.5");
        assert_eq!(a.opts.min_silence, Some(0.8));
        assert_eq!(a.opts.method, MethodArg::Hybrid);
        assert_eq!(a.opts.mode, ModeArg::Mute);
        assert!(a.json);
    }

    #[test]
    fn parses_negative_noise_with_equals_syntax() {
        let cli = Cli::try_parse_from([
            "svaraflow-cli",
            "silence",
            "analyze",
            "tl.json",
            "--noise=-45",
        ])
        .expect("parse");
        let Command::Silence(s) = cli.command else {
            panic!();
        };
        let SilenceCommand::Analyze(a) = s.command else {
            panic!();
        };
        assert_eq!(a.opts.noise, "-45");
    }

    #[test]
    fn parses_apply_with_xml_and_edl() {
        let cli = Cli::try_parse_from([
            "svaraflow-cli",
            "silence",
            "apply",
            "tl.json",
            "--xml",
            "out.xml",
            "--edl",
            "out.edl",
            "--sequence-name",
            "Interview",
        ])
        .expect("parse");
        let Command::Silence(s) = cli.command else {
            panic!();
        };
        let SilenceCommand::Apply(a) = s.command else {
            panic!();
        };
        assert_eq!(a.xml, Some(PathBuf::from("out.xml")));
        assert_eq!(a.edl, Some(PathBuf::from("out.edl")));
        assert_eq!(a.sequence_name.as_deref(), Some("Interview"));
    }

    #[test]
    fn parses_audio_tracks_list() {
        let cli = Cli::try_parse_from([
            "svaraflow-cli",
            "silence",
            "analyze",
            "tl.json",
            "--audio-tracks",
            "1,2,5",
        ])
        .expect("parse");
        let Command::Silence(s) = cli.command else {
            panic!();
        };
        let SilenceCommand::Analyze(a) = s.command else {
            panic!();
        };
        assert_eq!(a.opts.audio_tracks, Some(vec![1, 2, 5]));
    }

    #[test]
    fn parses_preset_flag() {
        let cli = Cli::try_parse_from([
            "svaraflow-cli",
            "silence",
            "analyze",
            "tl.json",
            "--preset",
            "fast",
        ])
        .expect("parse");
        let Command::Silence(s) = cli.command else {
            panic!();
        };
        let SilenceCommand::Analyze(a) = s.command else {
            panic!();
        };
        assert_eq!(a.opts.preset, Some(PresetArg::Fast));
    }

    #[test]
    fn unknown_subcommand_is_rejected() {
        assert!(Cli::try_parse_from(["svaraflow-cli", "nonsense"]).is_err());
    }

    #[test]
    fn timeline_from_media_requires_out() {
        assert!(Cli::try_parse_from(["svaraflow-cli", "timeline", "from-media", "a.mov"]).is_err());
    }
}