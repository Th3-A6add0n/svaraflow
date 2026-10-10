//! Command-line interface definitions for `svaraflow-cli`.

use std::path::PathBuf;

use clap::{Args, Parser, Subcommand, ValueEnum};

/// SvaraFlow command-line interface.
#[derive(Debug, Parser)]
#[command(
    name = "svaraflow-cli",
    version,
    about = "SvaraFlow CLI — silence removal and captions for Premiere Pro and DaVinci Resolve",
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
    /// Transcribe speech to Roman Hinglish word timings.
    Transcribe(TranscribeArgs),
    /// Manage speech-to-text model files.
    Models(ModelsCli),
}

/// `models` subcommand tree.
#[derive(Debug, Args)]
pub struct ModelsCli {
    /// Action to perform.
    #[command(subcommand)]
    pub command: ModelsCommand,
}

/// Actions under `models`.
#[derive(Debug, Subcommand)]
pub enum ModelsCommand {
    /// List the models in the manifest and whether they are on disk.
    List,
    /// Download a model by name or tier.
    Download(ModelsDownloadArgs),
    /// Print the SHA-256 of a file.
    Hash(ModelsHashArgs),
    /// Delete a downloaded model.
    Remove(ModelsRemoveArgs),
}

/// Arguments for `models download`.
#[derive(Debug, Args)]
pub struct ModelsDownloadArgs {
    /// Model name (from `models list`) or tier (`light`, `standard`).
    pub name: String,
}

/// Arguments for `models hash`.
#[derive(Debug, Args)]
pub struct ModelsHashArgs {
    /// File to hash.
    pub file: PathBuf,
}

/// Arguments for `models remove`.
#[derive(Debug, Args)]
pub struct ModelsRemoveArgs {
    /// Model name or tier.
    pub name: String,
}

/// Arguments for `transcribe`.
#[derive(Debug, Args)]
pub struct TranscribeArgs {
    /// Audio or video file to transcribe.
    pub input: PathBuf,
    /// Which model tier to use.
    #[arg(long, value_enum, default_value_t = TierArg::Auto)]
    pub tier: TierArg,
    /// Print the transcript JSON to stdout.
    #[arg(long)]
    pub json: bool,
    /// Write the transcript JSON to this file.
    #[arg(long)]
    pub out_json: Option<PathBuf>,
    /// Write an SRT subtitle file.
    #[arg(long)]
    pub srt: Option<PathBuf>,
    /// Write a WebVTT subtitle file.
    #[arg(long)]
    pub vtt: Option<PathBuf>,
    /// Write an ASS subtitle file with karaoke word highlighting.
    #[arg(long)]
    pub ass: Option<PathBuf>,
    /// Path to the Silero VAD ONNX model. Defaults to
    /// `models/silero_vad_v5.onnx` relative to the CWD.
    #[arg(long)]
    pub vad_model: Option<PathBuf>,
    /// Do not request token-level DTW timestamps. Word timing will be
    /// proportional and flagged as estimated.
    #[arg(long)]
    pub no_dtw: bool,
    /// Beam size for beam search. Greedy is used when omitted or `1`.
    #[arg(long)]
    pub beam: Option<u32>,
    /// Domain vocabulary to bias the decoder. Comma-separated.
    #[arg(long, value_delimiter = ',')]
    pub vocabulary: Vec<String>,
    /// Language hint: `hi` (default) or `auto`.
    #[arg(long, default_value = "hi")]
    pub language: String,
    /// Do not read from or write to the transcript cache.
    #[arg(long)]
    pub no_cache: bool,
}

/// Model tier, as accepted on the command line.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum TierArg {
    /// Pick the tier automatically from a benchmark or hardware probe.
    Auto,
    /// Whisper-base-sized model. Fast.
    Light,
    /// Large-v3-turbo-sized model, Q5_0. Accurate.
    Standard,
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
    /// Build a single-track timeline.json from a media file.
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
    /// Frame rate used when the probe cannot determine one, or when the
    /// file has no video stream at all.
    #[arg(long, default_value = "30000/1001")]
    pub fps: String,
    /// Raster width used when there is no video stream.
    #[arg(long, default_value_t = 1920)]
    pub width: u32,
    /// Raster height used when there is no video stream.
    #[arg(long, default_value_t = 1080)]
    pub height: u32,
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
    /// Print the table of built-in and user presets.
    Presets,
    /// Manage user presets.
    Preset(PresetCli),
}

/// `silence preset` subcommand tree.
#[derive(Debug, Args)]
pub struct PresetCli {
    /// Action to perform.
    #[command(subcommand)]
    pub command: PresetCommand,
}

/// Actions under `silence preset`.
#[derive(Debug, Subcommand)]
pub enum PresetCommand {
    /// Save a user preset. Built-in names are rejected.
    Save(PresetSaveArgs),
    /// Rename a user preset.
    Rename(PresetRenameArgs),
    /// Delete a user preset.
    Delete(PresetDeleteArgs),
    /// List user presets.
    List,
    /// Import presets from a JSON file. Merges into existing.
    Import(PresetFileArg),
    /// Export user presets to a JSON file.
    Export(PresetFileArg),
}

/// Arguments for `silence preset save`.
#[derive(Debug, Args)]
pub struct PresetSaveArgs {
    /// Name of the preset. Must not collide with a built-in.
    pub name: String,
    /// Silence removal threshold, milliseconds.
    #[arg(long)]
    pub remove_silences_ms: Option<u32>,
    /// Minimum kept speech, milliseconds.
    #[arg(long)]
    pub keep_talks_ms: Option<u32>,
    /// Padding before each kept run, milliseconds.
    #[arg(long)]
    pub margin_before_ms: Option<u32>,
    /// Padding after each kept run, milliseconds.
    #[arg(long)]
    pub margin_after_ms: Option<u32>,
    /// Minimum gap between cuts, milliseconds. `0` disables the rule.
    #[arg(long)]
    pub min_between_cuts_ms: Option<u32>,
}

/// Arguments for `silence preset rename`.
#[derive(Debug, Args)]
pub struct PresetRenameArgs {
    /// Existing user preset name.
    pub old: String,
    /// New user preset name.
    pub new: String,
}

/// Arguments for `silence preset delete`.
#[derive(Debug, Args)]
pub struct PresetDeleteArgs {
    /// User preset to delete.
    pub name: String,
}

/// Arguments for `silence preset import` / `export`.
#[derive(Debug, Args)]
pub struct PresetFileArg {
    /// Path to the JSON file.
    pub path: PathBuf,
}

/// Options common to `silence analyze` and `silence apply`.
#[derive(Debug, Args)]
pub struct DetectionOptions {
    /// Path to `timeline.json`.
    pub timeline: PathBuf,
    /// Noise floor: `auto` (default) or a dBFS value such as `-45`.
    #[arg(long, default_value = "auto", allow_hyphen_values = true)]
    pub noise: String,
    /// Pacing preset. Provides defaults for the ms flags below.
    #[arg(long, value_enum, default_value_t = PresetArg::Natural)]
    pub preset: PresetArg,
    /// Silence removal threshold, milliseconds. Overrides the preset.
    #[arg(long)]
    pub remove_silences_ms: Option<u32>,
    /// Minimum kept speech, milliseconds. Overrides the preset.
    #[arg(long)]
    pub keep_talks_ms: Option<u32>,
    /// Padding before each kept run, milliseconds. Overrides the preset.
    #[arg(long)]
    pub margin_before_ms: Option<u32>,
    /// Padding after each kept run, milliseconds. Overrides the preset.
    #[arg(long)]
    pub margin_after_ms: Option<u32>,
    /// Minimum gap between cuts, milliseconds. `0` disables the rule.
    #[arg(long)]
    pub min_between_cuts_ms: Option<u32>,
    /// Boundary snap mode.
    #[arg(long, value_enum)]
    pub snap: Option<SnapArg>,
    /// Audio channel layout used when writing XML.
    #[arg(long, value_enum, default_value_t = AudioChannelsArg::Auto)]
    pub audio_channels: AudioChannelsArg,
    /// Deprecated. Use `--remove-silences-ms` instead.
    #[arg(long, hide = true, allow_hyphen_values = true)]
    pub min_silence: Option<f64>,
    /// Deprecated. Use `--keep-talks-ms` instead.
    #[arg(long, hide = true, allow_hyphen_values = true)]
    pub min_talk: Option<f64>,
    /// Deprecated. Use `--margin-before-ms` instead.
    #[arg(long, hide = true, allow_hyphen_values = true)]
    pub pad_before: Option<f64>,
    /// Deprecated. Use `--margin-after-ms` instead.
    #[arg(long, hide = true, allow_hyphen_values = true)]
    pub pad_after: Option<f64>,
    /// Detection method.
    #[arg(long, value_enum, default_value_t = MethodArg::Threshold)]
    pub method: MethodArg,
    /// Cut mode. Ignored by `silence analyze`.
    #[arg(long, value_enum, default_value_t = ModeArg::Remove)]
    pub mode: ModeArg,
    /// Path to the Silero VAD ONNX model.
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
    /// Write a full report JSON to this path.
    #[arg(long)]
    pub report: Option<PathBuf>,
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
    /// Which FCP7 XML profile to write.
    #[arg(long, value_enum, default_value_t = XmlProfileArg::Full)]
    pub xml_profile: XmlProfileArg,
    /// Write the cut timeline as CMX3600 EDL.
    #[arg(long)]
    pub edl: Option<PathBuf>,
    /// Sequence name to use in the XML and EDL output.
    #[arg(long)]
    pub sequence_name: Option<String>,
}

/// FCP7 XML profile, as accepted on the command line.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum XmlProfileArg {
    /// One audio clipitem covering the first keep range.
    Minimal,
    /// Every cut clipitem, no sequence timecode or video block.
    Cuts,
    /// Full FCP7 structure. The default.
    Full,
}

/// Pacing preset, as accepted on the command line.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum PresetArg {
    /// Long, gentle cuts.
    Relaxed,
    /// Balanced. The default.
    Natural,
    /// Tighter cuts.
    Brisk,
    /// Tight cuts for dense speech.
    Tight,
    /// Maximum removal; uses `nearest` snap by default.
    Max,
}

/// Snap mode, as accepted on the command line.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum SnapArg {
    /// Shrink inward. Never clips a word. The default.
    Inward,
    /// Round to the nearest frame.
    Nearest,
}

/// Audio channel layout for XML export.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum AudioChannelsArg {
    /// Probe each source with `ffprobe`.
    Auto,
    /// Force one mono clipitem per source.
    Mono,
    /// Force the linked left/right clipitem pair.
    Stereo,
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

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn version_flag_is_intercepted_by_clap() {
        assert!(Cli::try_parse_from(["svaraflow-cli", "--version"]).is_err());
    }

    #[test]
    fn parses_transcribe_defaults() {
        let cli = Cli::try_parse_from(["svaraflow-cli", "transcribe", "a.mp4"]).expect("parse");
        let Command::Transcribe(t) = cli.command else {
            panic!();
        };
        assert_eq!(t.input, PathBuf::from("a.mp4"));
        assert_eq!(t.tier, TierArg::Auto);
        assert_eq!(t.language, "hi");
        assert!(!t.json);
        assert!(t.srt.is_none());
        assert!(t.vtt.is_none());
        assert!(t.ass.is_none());
        assert!(!t.no_dtw);
        assert!(!t.no_cache);
    }

    #[test]
    fn parses_transcribe_all_exports() {
        let cli = Cli::try_parse_from([
            "svaraflow-cli",
            "transcribe",
            "a.mp4",
            "--tier",
            "standard",
            "--json",
            "--srt",
            "a.srt",
            "--vtt",
            "a.vtt",
            "--ass",
            "a.ass",
        ])
        .expect("parse");
        let Command::Transcribe(t) = cli.command else {
            panic!();
        };
        assert_eq!(t.tier, TierArg::Standard);
        assert!(t.json);
        assert_eq!(t.srt, Some(PathBuf::from("a.srt")));
        assert_eq!(t.vtt, Some(PathBuf::from("a.vtt")));
        assert_eq!(t.ass, Some(PathBuf::from("a.ass")));
    }

    #[test]
    fn parses_transcribe_vocabulary_and_beam() {
        let cli = Cli::try_parse_from([
            "svaraflow-cli",
            "transcribe",
            "a.mp4",
            "--vocabulary",
            "SvaraFlow,Th3",
            "--beam",
            "5",
        ])
        .expect("parse");
        let Command::Transcribe(t) = cli.command else {
            panic!();
        };
        assert_eq!(t.vocabulary, vec!["SvaraFlow", "Th3"]);
        assert_eq!(t.beam, Some(5));
    }

    #[test]
    fn parses_models_list() {
        let cli = Cli::try_parse_from(["svaraflow-cli", "models", "list"]).expect("parse");
        let Command::Models(m) = cli.command else {
            panic!();
        };
        assert!(matches!(m.command, ModelsCommand::List));
    }

    #[test]
    fn parses_models_download() {
        let cli = Cli::try_parse_from([
            "svaraflow-cli",
            "models",
            "download",
            "Whisper-Hindi2Hinglish-Apex",
        ])
        .expect("parse");
        let Command::Models(m) = cli.command else {
            panic!();
        };
        let ModelsCommand::Download(a) = m.command else {
            panic!();
        };
        assert_eq!(a.name, "Whisper-Hindi2Hinglish-Apex");
    }

    #[test]
    fn parses_models_hash() {
        let cli =
            Cli::try_parse_from(["svaraflow-cli", "models", "hash", "m.bin"]).expect("parse");
        let Command::Models(m) = cli.command else {
            panic!();
        };
        let ModelsCommand::Hash(a) = m.command else {
            panic!();
        };
        assert_eq!(a.file, PathBuf::from("m.bin"));
    }

    #[test]
    fn parses_models_remove() {
        let cli = Cli::try_parse_from(["svaraflow-cli", "models", "remove", "light"])
            .expect("parse");
        let Command::Models(m) = cli.command else {
            panic!();
        };
        let ModelsCommand::Remove(a) = m.command else {
            panic!();
        };
        assert_eq!(a.name, "light");
    }

    #[test]
    fn transcribe_requires_input() {
        assert!(Cli::try_parse_from(["svaraflow-cli", "transcribe"]).is_err());
    }
}