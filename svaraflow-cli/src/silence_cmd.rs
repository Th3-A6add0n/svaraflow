//! `svaraflow-cli silence` subcommands: `analyze` and `apply`.

use std::path::Path;

use anyhow::{anyhow, Context, Result};
use serde::Serialize;
use svaraflow_core::audio::{
    compute_envelope, decode_native_mono, render_analysis_signal, resample_to, select_tracks,
    Envelope, EnvelopeParams, FileDecoder, MonoAudio, ANALYSIS_SAMPLE_RATE,
};
use svaraflow_core::cut::{apply_cuts, CutMode, CutResult};
use svaraflow_core::edl::write_edl;
use svaraflow_core::silence::{
    detect_silence, detect_silence_with_vad, FrameRange, Method, NoiseSetting, Preset,
    SilenceParams, SilenceRanges,
};
use svaraflow_core::timeline::{Fps, Timeline};
use svaraflow_core::xmeml::write_xmeml;

use crate::cli::{
    ApplyArgs, DetectionOptions, MethodArg, ModeArg, PresetArg, SilenceCli, SilenceCommand,
};

/// Default VAD model path, relative to the current directory.
const DEFAULT_VAD_MODEL: &str = "models/silero_vad_v5.onnx";

/// Runs `silence <subcommand>`.
pub fn run(args: SilenceCli) -> Result<()> {
    match args.command {
        SilenceCommand::Analyze(a) => analyze(&a.opts, a.json),
        SilenceCommand::Apply(a) => apply(&a),
    }
}

// ---------------------------------------------------------------------------
// Shared pipeline
// ---------------------------------------------------------------------------

/// The result of loading a timeline, building the analysis signal, computing
/// the envelope, and detecting silence.
struct Analysis {
    timeline: Timeline,
    envelope: Envelope,
    ranges: SilenceRanges,
    params: SilenceParams,
}

/// Runs the analysis pipeline: load → envelope → detect.
fn run_analysis(opts: &DetectionOptions) -> Result<Analysis> {
    let timeline = load_timeline(&opts.timeline)?;
    let params = resolve_params(opts)?;

    let (audio, envelope) = match &opts.audio {
        Some(wav) => {
            let audio = wav_to_analysis_audio(wav)?;
            let envelope = compute_envelope(&audio, EnvelopeParams::default());
            (audio, envelope)
        }
        None => {
            let audio = build_timeline_audio(&timeline, opts)?;
            let envelope = compute_envelope(&audio, EnvelopeParams::default());
            (audio, envelope)
        }
    };

    let ranges = match params.method {
        Method::Threshold => detect_silence(&envelope, &params, timeline.fps)?,
        Method::Vad | Method::Hybrid => {
            #[cfg(feature = "vad")]
            {
                let mut vad = load_silero_vad(opts)?;
                detect_silence_with_vad(&envelope, &params, timeline.fps, &mut vad, &audio.samples)?
            }
            #[cfg(not(feature = "vad"))]
            {
                return Err(anyhow!(
                    "--method {:?} requires the `vad` feature on `svaraflow-cli`, \
                     which was not enabled at build time",
                    params.method
                ));
            }
        }
    };

    Ok(Analysis {
        timeline,
        envelope,
        ranges,
        params,
    })
}

/// Reads `timeline.json` and validates it.
fn load_timeline(path: &Path) -> Result<Timeline> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    Timeline::from_json_str(&text).map_err(|e| anyhow!(e))
}

/// Builds the analysis signal from the timeline's own clips.
fn build_timeline_audio(timeline: &Timeline, opts: &DetectionOptions) -> Result<MonoAudio> {
    let selected = select_tracks(timeline, opts.audio_tracks.as_deref());
    if selected.is_empty() {
        return Err(anyhow!(
            "no audio tracks selected; the timeline has {} tracks and --audio-tracks \
             filtered to none",
            timeline.tracks.len()
        ));
    }

    let mut decoder = if opts.no_ffmpeg {
        FileDecoder::symphonia_only()
    } else {
        FileDecoder::with_ffmpeg_fallback()
    };
    render_analysis_signal(timeline, &selected, &mut decoder)
        .context("building the analysis signal")
}

/// Loads a mixdown WAV at 16 kHz mono.
fn wav_to_analysis_audio(path: &Path) -> Result<MonoAudio> {
    let native =
        decode_native_mono(path).with_context(|| format!("decoding {}", path.display()))?;
    resample_to(&native, ANALYSIS_SAMPLE_RATE)
        .with_context(|| format!("resampling {} to 16 kHz", path.display()))
}

/// Loads the Silero VAD model from `--vad-model` or the default path.
#[cfg(feature = "vad")]
fn load_silero_vad(opts: &DetectionOptions) -> Result<svaraflow_asr::SileroVad> {
    let path = opts
        .vad_model
        .clone()
        .unwrap_or_else(|| std::path::PathBuf::from(DEFAULT_VAD_MODEL));
    svaraflow_asr::SileroVad::from_file(&path).map_err(|e| {
        anyhow!(
            "loading VAD model from {}: {e}\n\
             hint: pass --vad-model <path>, or place the Silero VAD v5 ONNX export \
             at {DEFAULT_VAD_MODEL}, and make sure ONNX Runtime is on ORT_DYLIB_PATH \
             or PATH",
            path.display()
        )
    })
}

/// Merges the preset, the explicit numeric overrides, and the method/noise
/// settings into a single [`SilenceParams`].
fn resolve_params(opts: &DetectionOptions) -> Result<SilenceParams> {
    let mut params = match opts.preset {
        Some(p) => SilenceParams::preset(match p {
            PresetArg::Calm => Preset::Calm,
            PresetArg::Measured => Preset::Measured,
            PresetArg::Paced => Preset::Paced,
            PresetArg::Fast => Preset::Fast,
        }),
        None => SilenceParams::default(),
    };

    if let Some(v) = opts.min_silence {
        params.remove_silences_s = v;
    }
    if let Some(v) = opts.min_talk {
        params.keep_talks_s = v;
    }
    if let Some(v) = opts.pad_before {
        params.pad_before_s = v;
    }
    if let Some(v) = opts.pad_after {
        params.pad_after_s = v;
    }

    params.method = match opts.method {
        MethodArg::Threshold => Method::Threshold,
        MethodArg::Vad => Method::Vad,
        MethodArg::Hybrid => Method::Hybrid,
    };

    params.noise = parse_noise(&opts.noise)?;

    Ok(params)
}

/// Parses `--noise`: the literal `auto`, or a finite dBFS number.
fn parse_noise(s: &str) -> Result<NoiseSetting> {
    let trimmed = s.trim();
    if trimmed.eq_ignore_ascii_case("auto") {
        return Ok(NoiseSetting::Auto);
    }
    let value: f32 = trimmed
        .parse()
        .with_context(|| format!("--noise must be `auto` or a dBFS number, got `{s}`"))?;
    if !value.is_finite() {
        return Err(anyhow!("--noise must be a finite dBFS value"));
    }
    Ok(NoiseSetting::Manual(value))
}

fn method_name(m: Method) -> &'static str {
    match m {
        Method::Threshold => "threshold",
        Method::Vad => "vad",
        Method::Hybrid => "hybrid",
    }
}

fn mode_from(p: ModeArg) -> CutMode {
    match p {
        ModeArg::Remove => CutMode::Remove,
        ModeArg::Keep => CutMode::Keep,
        ModeArg::Mute => CutMode::Mute,
        ModeArg::Gaps => CutMode::Gaps,
    }
}

fn mode_name(m: CutMode) -> &'static str {
    match m {
        CutMode::Remove => "remove",
        CutMode::Keep => "keep",
        CutMode::Mute => "mute",
        CutMode::Gaps => "gaps",
    }
}

// ---------------------------------------------------------------------------
// analyze
// ---------------------------------------------------------------------------

fn analyze(opts: &DetectionOptions, json: bool) -> Result<()> {
    let a = run_analysis(opts)?;
    if json {
        let payload = JsonOutput::from(&a);
        println!("{}", serde_json::to_string_pretty(&payload)?);
    } else {
        print_analysis_human(&a);
    }
    Ok(())
}

/// JSON shape emitted by `silence analyze --json`.
#[derive(Debug, Serialize)]
struct JsonOutput {
    noise_db: f32,
    method: String,
    keep: Vec<JsonRange>,
    silences: Vec<JsonRange>,
    stats: JsonStats,
    warnings: Vec<String>,
}

#[derive(Debug, Serialize)]
struct JsonRange {
    start: i64,
    end: i64,
}

#[derive(Debug, Serialize)]
struct JsonStats {
    total_frames: i64,
    kept_frames: i64,
    removed_frames: i64,
    cut_count: usize,
    percent_shorter: f64,
}

impl From<&Analysis> for JsonOutput {
    fn from(a: &Analysis) -> Self {
        Self {
            noise_db: a.ranges.noise_db,
            method: method_name(a.params.method).to_string(),
            keep: a.ranges.keep.iter().map(JsonRange::from).collect(),
            silences: a.ranges.silences.iter().map(JsonRange::from).collect(),
            stats: JsonStats {
                total_frames: a.ranges.stats.total_frames,
                kept_frames: a.ranges.stats.kept_frames,
                removed_frames: a.ranges.stats.removed_frames,
                cut_count: a.ranges.stats.cut_count,
                percent_shorter: a.ranges.stats.percent_shorter,
            },
            warnings: a.ranges.warnings.iter().map(|w| w.to_string()).collect(),
        }
    }
}

impl From<&FrameRange> for JsonRange {
    fn from(r: &FrameRange) -> Self {
        Self {
            start: r.start,
            end: r.end,
        }
    }
}

fn print_analysis_human(a: &Analysis) {
    let fps = a.timeline.fps;

    println!("Timeline:");
    println!("  fps:   {} ({:.3})", format_fps(fps), fps.as_f64());
    println!(
        "  total: {} frames ({:.2} s)",
        a.ranges.stats.total_frames,
        fps.frames_to_seconds(a.ranges.stats.total_frames)
    );
    println!(
        "  audio: {} envelope frames ({:.2} s at 100 Hz)",
        a.envelope.len(),
        a.envelope.len() as f64 * a.envelope.seconds_per_frame()
    );

    println!();
    println!("Analysis:");
    println!("  method:      {}", method_name(a.params.method));
    println!("  noise:       {:.1} dBFS", a.ranges.noise_db);
    println!("  min-silence: {:.2} s", a.params.remove_silences_s);
    println!("  min-talk:    {:.2} s", a.params.keep_talks_s);
    println!(
        "  padding:     {:.2} s before / {:.2} s after",
        a.params.pad_before_s, a.params.pad_after_s
    );

    println!();
    println!("Result:");
    println!(
        "  kept:    {:>10} frames  {:>8.2} s  {:>6.2}%",
        a.ranges.stats.kept_frames,
        fps.frames_to_seconds(a.ranges.stats.kept_frames),
        100.0 - a.ranges.stats.percent_shorter
    );
    println!(
        "  removed: {:>10} frames  {:>8.2} s  {:>6.2}%",
        a.ranges.stats.removed_frames,
        fps.frames_to_seconds(a.ranges.stats.removed_frames),
        a.ranges.stats.percent_shorter
    );
    println!("  cuts:    {:>10}", a.ranges.stats.cut_count);

    println!();
    println!("Keep ranges ({}):", a.ranges.keep.len());
    for r in &a.ranges.keep {
        println!(
            "  [{}, {})  {} frames  {:.3} s – {:.3} s",
            r.start,
            r.end,
            r.len(),
            fps.frames_to_seconds(r.start),
            fps.frames_to_seconds(r.end),
        );
    }

    if !a.ranges.warnings.is_empty() {
        println!();
        println!("Warnings:");
        for w in &a.ranges.warnings {
            println!("  - {w}");
        }
    }
}

// ---------------------------------------------------------------------------
// apply
// ---------------------------------------------------------------------------

fn apply(args: &ApplyArgs) -> Result<()> {
    if args.xml.is_none() && args.edl.is_none() {
        return Err(anyhow!(
            "silence apply needs at least one of --xml or --edl to write output; \
             use `silence analyze` to inspect only"
        ));
    }

    let a = run_analysis(&args.opts)?;
    let mode = mode_from(args.opts.mode);

    if args.edl.is_some() && mode != CutMode::Remove {
        eprintln!(
            "warning: --edl is written for Remove-mode semantics; the destination \
             timecodes assume the timeline is concatenated. Use --xml for {:?}.",
            mode
        );
    }

    let cut: CutResult = apply_cuts(
        &a.timeline,
        &a.ranges,
        mode,
        args.opts.audio_tracks.as_deref(),
    );

    print_analysis_human(&a);

    println!();
    println!("Cut:");
    println!("  mode:          {}", mode_name(mode));
    println!("  frames before: {}", cut.stats.frames_before);
    println!("  frames after:  {}", cut.stats.frames_after);
    println!("  clips created: {}", cut.stats.clips_created);
    println!("  clips removed: {}", cut.stats.clips_removed);

    let sequence_name = args
        .sequence_name
        .clone()
        .or_else(|| {
            args.opts
                .timeline
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
        })
        .unwrap_or_else(|| "SvaraFlow".to_string());

    if let Some(path) = &args.xml {
        let xml = write_xmeml(&cut.timeline, &sequence_name)?;
        std::fs::write(path, xml).with_context(|| format!("writing {}", path.display()))?;
        println!("Wrote XML: {}", path.display());
    }

    if let Some(path) = &args.edl {
        let edl = write_edl(&a.timeline, &a.ranges, &sequence_name)?;
        std::fs::write(path, edl).with_context(|| format!("writing {}", path.display()))?;
        println!("Wrote EDL: {}", path.display());
    }

    Ok(())
}

fn format_fps(fps: Fps) -> String {
    if fps.den == 1 {
        fps.num.to_string()
    } else {
        format!("{}/{}", fps.num, fps.den)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_noise_accepts_auto() {
        assert_eq!(parse_noise("auto").unwrap(), NoiseSetting::Auto);
        assert_eq!(parse_noise("AUTO").unwrap(), NoiseSetting::Auto);
        assert_eq!(parse_noise("  Auto  ").unwrap(), NoiseSetting::Auto);
    }

    #[test]
    fn parse_noise_accepts_number() {
        assert_eq!(parse_noise("-42.5").unwrap(), NoiseSetting::Manual(-42.5));
        assert_eq!(parse_noise("-60").unwrap(), NoiseSetting::Manual(-60.0));
    }

    #[test]
    fn parse_noise_rejects_garbage() {
        assert!(parse_noise("").is_err());
        assert!(parse_noise("silent").is_err());
        assert!(parse_noise("nan").is_err());
    }

    #[test]
    fn resolve_params_without_preset_uses_defaults() {
        let opts = DetectionOptions {
            timeline: "x.json".into(),
            noise: "auto".to_string(),
            min_silence: None,
            min_talk: None,
            pad_before: None,
            pad_after: None,
            method: MethodArg::Threshold,
            mode: ModeArg::Remove,
            preset: None,
            vad_model: None,
            audio: None,
            audio_tracks: None,
            no_ffmpeg: false,
        };
        let p = resolve_params(&opts).unwrap();
        assert!((p.remove_silences_s - 0.50).abs() < 1e-12);
        assert!((p.keep_talks_s - 0.25).abs() < 1e-12);
        assert!((p.pad_before_s - 0.10).abs() < 1e-12);
        assert!((p.pad_after_s - 0.10).abs() < 1e-12);
    }

    #[test]
    fn resolve_params_preset_then_explicit_override() {
        let opts = DetectionOptions {
            timeline: "x.json".into(),
            noise: "auto".to_string(),
            min_silence: Some(0.99),
            min_talk: None,
            pad_before: None,
            pad_after: Some(0.05),
            method: MethodArg::Threshold,
            mode: ModeArg::Remove,
            preset: Some(PresetArg::Fast),
            vad_model: None,
            audio: None,
            audio_tracks: None,
            no_ffmpeg: false,
        };
        let p = resolve_params(&opts).unwrap();
        assert!((p.remove_silences_s - 0.99).abs() < 1e-12);
        assert!((p.keep_talks_s - 0.15).abs() < 1e-12);
        assert!((p.pad_before_s - 0.08).abs() < 1e-12);
        assert!((p.pad_after_s - 0.05).abs() < 1e-12);
    }

    #[test]
    fn resolve_params_maps_methods_and_modes() {
        let base = DetectionOptions {
            timeline: "x.json".into(),
            noise: "auto".to_string(),
            min_silence: None,
            min_talk: None,
            pad_before: None,
            pad_after: None,
            method: MethodArg::Hybrid,
            mode: ModeArg::Mute,
            preset: None,
            vad_model: None,
            audio: None,
            audio_tracks: None,
            no_ffmpeg: false,
        };
        let p = resolve_params(&base).unwrap();
        assert_eq!(p.method, Method::Hybrid);

        assert_eq!(mode_from(ModeArg::Remove), CutMode::Remove);
        assert_eq!(mode_from(ModeArg::Keep), CutMode::Keep);
        assert_eq!(mode_from(ModeArg::Mute), CutMode::Mute);
        assert_eq!(mode_from(ModeArg::Gaps), CutMode::Gaps);
    }
}