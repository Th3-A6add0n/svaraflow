//! `svaraflow-cli silence` subcommands: `analyze`, `apply`, `presets` and
//! `preset`.

use std::collections::{HashMap, HashSet};
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
    detect_silence, detect_silence_with_vad, AnalysisReport, BuiltinPreset, FrameRange, Method,
    NoiseSetting, NoiseSource, SilenceParams, SilenceRanges, SnapMode,
};
use svaraflow_core::timeline::{Fps, Timeline, TrackKind};
use svaraflow_core::xmeml::{write_xmeml, SourceInfo, XmlProfile};

use crate::cli::{
    ApplyArgs, AudioChannelsArg, DetectionOptions, MethodArg, ModeArg, PresetArg, PresetCli,
    PresetCommand, SilenceCli, SilenceCommand, SnapArg, XmlProfileArg,
};
use crate::presets;
use crate::timeline_cmd::{probe_audio_properties, AudioProperties};

/// Default VAD model path, relative to the current directory.
const DEFAULT_VAD_MODEL: &str = "models/silero_vad_v5.onnx";

/// Runs `silence <subcommand>`.
pub fn run(args: SilenceCli) -> Result<()> {
    match args.command {
        SilenceCommand::Analyze(a) => analyze(&a.opts, a.json),
        SilenceCommand::Apply(a) => apply(&a),
        SilenceCommand::Presets => print_presets(),
        SilenceCommand::Preset(p) => run_preset(p),
    }
}

// ---------------------------------------------------------------------------
// Presets table and management
// ---------------------------------------------------------------------------

fn print_presets() -> Result<()> {
    let user = presets::list_user()?;
    println!(
        "{:<16} {:>8} {:>6} {:>8} {:>8} {:>12} {:<8} {}",
        "NAME", "REMOVE", "KEEP", "MARG-B", "MARG-A", "MIN-BETWEEN", "SNAP", "SOURCE"
    );
    for p in BuiltinPreset::ALL {
        let v = p.values();
        println!(
            "{:<16} {:>8} {:>6} {:>8} {:>8} {:>12} {:<8} builtin",
            p.name(),
            v.remove_silences_ms,
            v.keep_talks_ms,
            v.margin_before_ms,
            v.margin_after_ms,
            v.min_between_cuts_ms,
            snap_name(p.default_snap())
        );
    }
    for (name, v) in user {
        println!(
            "{:<16} {:>8} {:>6} {:>8} {:>8} {:>12} {:<8} user",
            name,
            v.remove_silences_ms,
            v.keep_talks_ms,
            v.margin_before_ms,
            v.margin_after_ms,
            v.min_between_cuts_ms,
            "inward"
        );
    }
    Ok(())
}

fn snap_name(m: SnapMode) -> &'static str {
    match m {
        SnapMode::Inward => "inward",
        SnapMode::Nearest => "nearest",
    }
}

fn run_preset(cli: PresetCli) -> Result<()> {
    match cli.command {
        PresetCommand::Save(a) => {
            let mut v = BuiltinPreset::Natural.values();
            if let Some(x) = a.remove_silences_ms {
                v.remove_silences_ms = x;
            }
            if let Some(x) = a.keep_talks_ms {
                v.keep_talks_ms = x;
            }
            if let Some(x) = a.margin_before_ms {
                v.margin_before_ms = x;
            }
            if let Some(x) = a.margin_after_ms {
                v.margin_after_ms = x;
            }
            if let Some(x) = a.min_between_cuts_ms {
                v.min_between_cuts_ms = x;
            }
            presets::save_user(&a.name, v)?;
            println!("Saved user preset `{}`.", a.name);
            Ok(())
        }
        PresetCommand::Rename(a) => {
            presets::rename_user(&a.old, &a.new)?;
            println!("Renamed `{}` to `{}`.", a.old, a.new);
            Ok(())
        }
        PresetCommand::Delete(a) => {
            presets::delete_user(&a.name)?;
            println!("Deleted user preset `{}`.", a.name);
            Ok(())
        }
        PresetCommand::List => {
            let user = presets::list_user()?;
            if user.is_empty() {
                println!("No user presets.");
            } else {
                for (name, v) in user {
                    println!(
                        "{name}: remove={} keep={} marg_b={} marg_a={} min_between={}",
                        v.remove_silences_ms,
                        v.keep_talks_ms,
                        v.margin_before_ms,
                        v.margin_after_ms,
                        v.min_between_cuts_ms
                    );
                }
            }
            Ok(())
        }
        PresetCommand::Import(a) => {
            let n = presets::import(&a.path)?;
            println!("Imported {n} preset(s).");
            Ok(())
        }
        PresetCommand::Export(a) => {
            let n = presets::export(&a.path)?;
            println!("Exported {n} preset(s) to {}.", a.path.display());
            Ok(())
        }
    }
}

// ---------------------------------------------------------------------------
// Shared pipeline
// ---------------------------------------------------------------------------

struct Analysis {
    timeline: Timeline,
    envelope: Envelope,
    ranges: SilenceRanges,
    params: SilenceParams,
}

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
                    "--method {:?} requires the `vad` feature on `svaraflow-cli`",
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

fn load_timeline(path: &Path) -> Result<Timeline> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    Timeline::from_json_str(&text).map_err(|e| anyhow!(e))
}

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

fn wav_to_analysis_audio(path: &Path) -> Result<MonoAudio> {
    let native =
        decode_native_mono(path).with_context(|| format!("decoding {}", path.display()))?;
    resample_to(&native, ANALYSIS_SAMPLE_RATE)
        .with_context(|| format!("resampling {} to 16 kHz", path.display()))
}

#[cfg(feature = "vad")]
fn load_silero_vad(opts: &DetectionOptions) -> Result<svaraflow_asr::SileroVad> {
    let path = opts
        .vad_model
        .clone()
        .unwrap_or_else(|| std::path::PathBuf::from(DEFAULT_VAD_MODEL));
    svaraflow_asr::SileroVad::from_file(&path).map_err(|e| {
        anyhow!(
            "loading VAD model from {}: {e}\n\
             hint: pass --vad-model <path>, and make sure ONNX Runtime is on \
             ORT_DYLIB_PATH or PATH",
            path.display()
        )
    })
}

/// Merges the preset, explicit ms flags, deprecated second flags and
/// method/noise settings into a single [`SilenceParams`].
fn resolve_params(opts: &DetectionOptions) -> Result<SilenceParams> {
    let builtin = preset_from_arg(opts.preset);
    let mut params = SilenceParams::from_preset(builtin);

    // Explicit ms flags override the preset.
    if let Some(x) = opts.remove_silences_ms {
        params.remove_silences_ms = x;
    }
    if let Some(x) = opts.keep_talks_ms {
        params.keep_talks_ms = x;
    }
    if let Some(x) = opts.margin_before_ms {
        params.margin_before_ms = x;
    }
    if let Some(x) = opts.margin_after_ms {
        params.margin_after_ms = x;
    }
    if let Some(x) = opts.min_between_cuts_ms {
        params.min_between_cuts_ms = x;
    }

    // Deprecated second-based flags override the preset too, but the ms
    // flags win when both are supplied.
    if let Some(s) = opts.min_silence {
        if opts.remove_silences_ms.is_none() {
            eprintln!("warning: --min-silence is deprecated; use --remove-silences-ms");
            params.remove_silences_ms = (s * 1000.0).round() as u32;
        }
    }
    if let Some(s) = opts.min_talk {
        if opts.keep_talks_ms.is_none() {
            eprintln!("warning: --min-talk is deprecated; use --keep-talks-ms");
            params.keep_talks_ms = (s * 1000.0).round() as u32;
        }
    }
    if let Some(s) = opts.pad_before {
        if opts.margin_before_ms.is_none() {
            eprintln!("warning: --pad-before is deprecated; use --margin-before-ms");
            params.margin_before_ms = (s * 1000.0).round() as u32;
        }
    }
    if let Some(s) = opts.pad_after {
        if opts.margin_after_ms.is_none() {
            eprintln!("warning: --pad-after is deprecated; use --margin-after-ms");
            params.margin_after_ms = (s * 1000.0).round() as u32;
        }
    }

    if let Some(s) = opts.snap {
        params.snap = match s {
            SnapArg::Inward => SnapMode::Inward,
            SnapArg::Nearest => SnapMode::Nearest,
        };
    }

    params.method = match opts.method {
        MethodArg::Threshold => Method::Threshold,
        MethodArg::Vad => Method::Vad,
        MethodArg::Hybrid => Method::Hybrid,
    };

    params.noise = parse_noise(&opts.noise)?;

    Ok(params)
}

fn preset_from_arg(p: PresetArg) -> BuiltinPreset {
    match p {
        PresetArg::Relaxed => BuiltinPreset::Relaxed,
        PresetArg::Natural => BuiltinPreset::Natural,
        PresetArg::Brisk => BuiltinPreset::Brisk,
        PresetArg::Tight => BuiltinPreset::Tight,
        PresetArg::Max => BuiltinPreset::Max,
    }
}

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

fn profile_from(p: XmlProfileArg) -> XmlProfile {
    match p {
        XmlProfileArg::Minimal => XmlProfile::Minimal,
        XmlProfileArg::Cuts => XmlProfile::Cuts,
        XmlProfileArg::Full => XmlProfile::Full,
    }
}

fn profile_name(p: XmlProfile) -> &'static str {
    match p {
        XmlProfile::Minimal => "minimal",
        XmlProfile::Cuts => "cuts",
        XmlProfile::Full => "full",
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
    if let Some(path) = &opts.report {
        write_report(path, &a)?;
    }
    Ok(())
}

#[derive(Debug, Serialize)]
struct JsonOutput {
    noise_db: f32,
    method: String,
    keep: Vec<JsonRange>,
    silences: Vec<JsonRange>,
    stats: JsonStats,
    warnings: Vec<String>,
    report: JsonReport,
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

#[derive(Debug, Serialize)]
struct JsonReport {
    noise_db: f32,
    noise_source: &'static str,
    frames_below_noise_pct: f64,
    silences_found: usize,
    silences_cut: usize,
    cuts_prevented: usize,
    average_cut_frames: f64,
    max_cut_frames: i64,
    shortest_cut_frames: i64,
    longest_cuts: Vec<JsonRange>,
    per_cut: Vec<JsonRange>,
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
            report: JsonReport::from(&a.ranges.report),
        }
    }
}

impl From<&AnalysisReport> for JsonReport {
    fn from(r: &AnalysisReport) -> Self {
        Self {
            noise_db: r.noise_db,
            noise_source: match r.noise_source {
                NoiseSource::Auto => "auto",
                NoiseSource::Manual => "manual",
            },
            frames_below_noise_pct: r.frames_below_noise_pct,
            silences_found: r.silences_found,
            silences_cut: r.silences_cut,
            cuts_prevented: r.cuts_prevented,
            average_cut_frames: r.average_cut_frames,
            max_cut_frames: r.max_cut_frames,
            shortest_cut_frames: r.shortest_cut_frames,
            longest_cuts: r.longest_cuts.iter().map(JsonRange::from).collect(),
            per_cut: r
                .per_cut
                .iter()
                .map(|c| JsonRange {
                    start: c.start,
                    end: c.end,
                })
                .collect(),
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

    let p = &a.params;
    println!();
    println!("Parameters:");
    println!("  preset:      {}", preset_label(p));
    println!("  method:      {}", method_name(p.method));
    println!("  remove:      {} ms", p.remove_silences_ms);
    println!("  keep:        {} ms", p.keep_talks_ms);
    println!(
        "  margins:     {} ms before / {} ms after",
        p.margin_before_ms, p.margin_after_ms
    );
    if p.min_between_cuts_ms > 0 {
        println!("  min-between: {} ms", p.min_between_cuts_ms);
    } else {
        println!("  min-between: off");
    }
    println!("  snap:        {}", snap_name(p.snap));

    println!();
    println!("Analysis:");
    let r = &a.ranges.report;
    let source = match r.noise_source {
        NoiseSource::Auto => "auto",
        NoiseSource::Manual => "manual",
    };
    println!("  noise:       {:.1} dBFS ({})", r.noise_db, source);
    println!("  below noise: {:.1}%", r.frames_below_noise_pct);
    println!(
        "  silences:    {} found, {} cut",
        r.silences_found, r.silences_cut
    );
    println!("  cuts prevented: {}", r.cuts_prevented);
    if r.silences_cut > 0 {
        println!(
            "  cut length:  avg {:.1} frames ({:.2} s), max {} frames ({:.2} s), \
             shortest {} frames ({:.2} s)",
            r.average_cut_frames,
            fps.frames_to_seconds(r.average_cut_frames.round() as i64),
            r.max_cut_frames,
            fps.frames_to_seconds(r.max_cut_frames),
            r.shortest_cut_frames,
            fps.frames_to_seconds(r.shortest_cut_frames),
        );
    }

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

    if !r.longest_cuts.is_empty() {
        println!();
        println!("Longest remaining cuts ({} shown):", r.longest_cuts.len().min(5));
        for c in r.longest_cuts.iter().take(5) {
            println!(
                "  [{}, {})  {} frames  {:.3} s",
                c.start,
                c.end,
                c.len(),
                fps.frames_to_seconds(c.len()),
            );
        }
    }

    println!();
    println!("Keep ranges ({}):", a.ranges.keep.len());
    for k in &a.ranges.keep {
        println!(
            "  [{}, {})  {} frames  {:.3} s – {:.3} s",
            k.start,
            k.end,
            k.len(),
            fps.frames_to_seconds(k.start),
            fps.frames_to_seconds(k.end),
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

fn preset_label(p: &SilenceParams) -> String {
    let v = p.values();
    for b in BuiltinPreset::ALL {
        if b.values() == v && b.default_snap() == p.snap {
            return b.name().to_string();
        }
    }
    for b in BuiltinPreset::ALL {
        if b.values() == v {
            return format!("{} (values; snap={})", b.name(), snap_name(p.snap));
        }
    }
    "custom".to_string()
}

fn write_report(path: &Path, a: &Analysis) -> Result<()> {
    let payload = JsonOutput::from(a);
    let text = serde_json::to_string_pretty(&payload)?;
    std::fs::write(path, text).with_context(|| format!("writing {}", path.display()))?;
    println!("Wrote report: {}", path.display());
    Ok(())
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

    if a.ranges.keep.is_empty() {
        return Err(anyhow!(
            "no keep ranges: every frame was classified as silence. \
             Try a lower noise level (--noise auto or --noise -50), a smaller \
             --keep-talks-ms, or a larger --margin-before-ms/--margin-after-ms."
        ));
    }

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
        let profile = profile_from(args.xml_profile);
        let sources = build_source_info(&a.timeline, &cut.timeline, args.opts.audio_channels);
        let xml = write_xmeml(&cut.timeline, &sequence_name, profile, &sources)?;
        std::fs::write(path, xml).with_context(|| format!("writing {}", path.display()))?;
        println!(
            "Wrote XML ({} profile): {}",
            profile_name(profile),
            path.display()
        );
    }

    if let Some(path) = &args.edl {
        let edl = write_edl(&a.timeline, &a.ranges, &sequence_name)?;
        std::fs::write(path, edl).with_context(|| format!("writing {}", path.display()))?;
        println!("Wrote EDL: {}", path.display());
    }

    if let Some(path) = &args.opts.report {
        write_report(path, &a)?;
    }

    Ok(())
}

/// Builds per-source metadata for the XML writer.
///
/// `total_frames` for each source is the maximum `source_out` across the
/// **original** timeline, so the file duration reflects the true source
/// length even after the cut dropped the tail. Sources that appear only in
/// the cut timeline fall back to their own `source_out`.
///
/// Channel count, sample rate and bit depth come from `ffprobe` on each
/// source that appears on an audio track. When `channels_arg` is
/// [`AudioChannelsArg::Mono`] or [`AudioChannelsArg::Stereo`], the probe is
/// still consulted for the sample rate and bit depth, but the channel count
/// is forced to the requested value. When it is [`AudioChannelsArg::Auto`],
/// the probe's channel count is used and a failure falls back to mono
/// 48 kHz with a warning on stderr.
fn build_source_info(
    original: &Timeline,
    cut: &Timeline,
    channels_arg: AudioChannelsArg,
) -> HashMap<String, SourceInfo> {
    let mut map: HashMap<String, SourceInfo> = HashMap::new();
    let mut audio_sources: HashSet<String> = HashSet::new();

    for track in &original.tracks {
        for clip in &track.clips {
            let entry = map
                .entry(clip.source.clone())
                .or_insert_with(|| source_info_stub(track.kind, original));
            entry.total_frames = entry.total_frames.max(clip.source_out);
            if track.kind == TrackKind::Audio {
                audio_sources.insert(clip.source.clone());
            }
        }
    }
    for track in &cut.tracks {
        for clip in &track.clips {
            map.entry(clip.source.clone())
                .or_insert_with(|| source_info_stub(track.kind, original));
            if track.kind == TrackKind::Audio {
                audio_sources.insert(clip.source.clone());
            }
        }
    }

    for source in &audio_sources {
        if let Some(info) = map.get_mut(source) {
            let probed = match channels_arg {
                AudioChannelsArg::Mono => AudioProperties {
                    channels: 1,
                    ..probe_audio_properties(Path::new(source))
                },
                AudioChannelsArg::Stereo => AudioProperties {
                    channels: 2,
                    ..probe_audio_properties(Path::new(source))
                },
                AudioChannelsArg::Auto => probe_audio_properties(Path::new(source)),
            };
            info.audio_channels = probed.channels;
            info.audio_sample_rate = probed.sample_rate;
            info.audio_bit_depth = probed.bit_depth;
        }
    }

    map
}

/// Builds a [`SourceInfo`] with `total_frames = 0` and default audio
/// properties. Audio and video presence are set from the track kind; the
/// audio properties are overwritten by the probe pass in
/// [`build_source_info`].
fn source_info_stub(kind: TrackKind, timeline: &Timeline) -> SourceInfo {
    let mut info = SourceInfo::default();
    info.has_video = kind == TrackKind::Video;
    info.video_width = timeline.width;
    info.video_height = timeline.height;
    info
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
    use svaraflow_core::timeline::{Clip, Track};

    fn base_opts() -> DetectionOptions {
        DetectionOptions {
            timeline: "x.json".into(),
            noise: "auto".to_string(),
            preset: PresetArg::Natural,
            remove_silences_ms: None,
            keep_talks_ms: None,
            margin_before_ms: None,
            margin_after_ms: None,
            min_between_cuts_ms: None,
            snap: None,
            audio_channels: AudioChannelsArg::Auto,
            min_silence: None,
            min_talk: None,
            pad_before: None,
            pad_after: None,
            method: MethodArg::Threshold,
            mode: ModeArg::Remove,
            vad_model: None,
            audio: None,
            audio_tracks: None,
            no_ffmpeg: false,
            report: None,
        }
    }

    fn track(kind: TrackKind, index: u32, clips: Vec<Clip>) -> Track {
        Track { kind, index, clips }
    }

    fn clip(id: &str, source: &str, source_out: i64) -> Clip {
        Clip {
            id: id.to_string(),
            source: source.to_string(),
            source_in: 0,
            source_out,
            start: 0,
            speed: 1.0,
            has_audio: true,
            muted: false,
        }
    }

    #[test]
    fn parse_noise_accepts_auto() {
        assert_eq!(parse_noise("auto").unwrap(), NoiseSetting::Auto);
        assert_eq!(parse_noise("AUTO").unwrap(), NoiseSetting::Auto);
    }

    #[test]
    fn parse_noise_accepts_number() {
        assert_eq!(parse_noise("-42.5").unwrap(), NoiseSetting::Manual(-42.5));
    }

    #[test]
    fn parse_noise_rejects_garbage() {
        assert!(parse_noise("").is_err());
        assert!(parse_noise("silent").is_err());
        assert!(parse_noise("nan").is_err());
    }

    #[test]
    fn natural_preset_is_used_by_default() {
        let p = resolve_params(&base_opts()).unwrap();
        assert_eq!(p.remove_silences_ms, 300);
        assert_eq!(p.keep_talks_ms, 150);
        assert_eq!(p.margin_before_ms, 100);
        assert_eq!(p.margin_after_ms, 100);
    }

    #[test]
    fn explicit_ms_flags_override_preset() {
        let mut opts = base_opts();
        opts.remove_silences_ms = Some(220);
        opts.min_between_cuts_ms = Some(180);
        let p = resolve_params(&opts).unwrap();
        assert_eq!(p.remove_silences_ms, 220);
        assert_eq!(p.min_between_cuts_ms, 180);
        assert_eq!(p.keep_talks_ms, 150);
    }

    #[test]
    fn deprecated_seconds_flag_is_converted() {
        let mut opts = base_opts();
        opts.min_silence = Some(0.5);
        let p = resolve_params(&opts).unwrap();
        assert_eq!(p.remove_silences_ms, 500);
    }

    #[test]
    fn ms_flag_wins_over_deprecated_seconds_flag() {
        let mut opts = base_opts();
        opts.min_silence = Some(0.5);
        opts.remove_silences_ms = Some(200);
        let p = resolve_params(&opts).unwrap();
        assert_eq!(p.remove_silences_ms, 200);
    }

    #[test]
    fn snap_flag_overrides_preset_default() {
        let mut opts = base_opts();
        opts.preset = PresetArg::Max;
        let p = resolve_params(&opts).unwrap();
        assert_eq!(p.snap, SnapMode::Nearest);
        opts.snap = Some(SnapArg::Inward);
        let p = resolve_params(&opts).unwrap();
        assert_eq!(p.snap, SnapMode::Inward);
    }

    #[test]
    fn all_presets_resolve_to_their_documented_values() {
        for (arg, preset) in [
            (PresetArg::Relaxed, BuiltinPreset::Relaxed),
            (PresetArg::Natural, BuiltinPreset::Natural),
            (PresetArg::Brisk, BuiltinPreset::Brisk),
            (PresetArg::Tight, BuiltinPreset::Tight),
            (PresetArg::Max, BuiltinPreset::Max),
        ] {
            let mut opts = base_opts();
            opts.preset = arg;
            let p = resolve_params(&opts).unwrap();
            let v = preset.values();
            assert_eq!(p.remove_silences_ms, v.remove_silences_ms);
            assert_eq!(p.keep_talks_ms, v.keep_talks_ms);
            assert_eq!(p.margin_before_ms, v.margin_before_ms);
            assert_eq!(p.margin_after_ms, v.margin_after_ms);
            assert_eq!(p.min_between_cuts_ms, v.min_between_cuts_ms);
            assert_eq!(p.snap, preset.default_snap());
        }
    }

    #[test]
    fn build_source_info_uses_original_total() {
        let original = Timeline {
            fps: Fps { num: 30, den: 1 },
            width: 1920,
            height: 1080,
            tracks: vec![track(
                TrackKind::Audio,
                1,
                vec![clip("a1", "speech1.mp4", 500)],
            )],
        };
        let cut = Timeline {
            fps: Fps { num: 30, den: 1 },
            width: 1920,
            height: 1080,
            tracks: vec![track(
                TrackKind::Audio,
                1,
                vec![clip("a1__0", "speech1.mp4", 300)],
            )],
        };
        let map = build_source_info(&original, &cut, AudioChannelsArg::Mono);
        assert_eq!(map["speech1.mp4"].total_frames, 500);
        assert_eq!(map["speech1.mp4"].audio_channels, 1);
    }

    #[test]
    fn forced_stereo_sets_two_channels() {
        let original = Timeline {
            fps: Fps { num: 30, den: 1 },
            width: 1920,
            height: 1080,
            tracks: vec![track(
                TrackKind::Audio,
                1,
                vec![clip("a1", "speech1.mp4", 500)],
            )],
        };
        let cut = original.clone();
        let map = build_source_info(&original, &cut, AudioChannelsArg::Stereo);
        assert_eq!(map["speech1.mp4"].audio_channels, 2);
    }

    #[test]
    fn build_source_info_includes_cut_only_sources() {
        let original = Timeline {
            fps: Fps { num: 30, den: 1 },
            width: 1920,
            height: 1080,
            tracks: vec![track(
                TrackKind::Audio,
                1,
                vec![clip("a1", "speech1.mp4", 500)],
            )],
        };
        let cut = Timeline {
            fps: Fps { num: 30, den: 1 },
            width: 1920,
            height: 1080,
            tracks: vec![track(
                TrackKind::Audio,
                1,
                vec![
                    clip("a1__0", "speech1.mp4", 300),
                    clip("a1__1", "music.mp3", 200),
                ],
            )],
        };
        let map = build_source_info(&original, &cut, AudioChannelsArg::Mono);
        assert!(map.contains_key("music.mp3"));
    }

    #[test]
    fn profile_from_maps_all_variants() {
        assert_eq!(profile_from(XmlProfileArg::Minimal), XmlProfile::Minimal);
        assert_eq!(profile_from(XmlProfileArg::Cuts), XmlProfile::Cuts);
        assert_eq!(profile_from(XmlProfileArg::Full), XmlProfile::Full);
    }
}