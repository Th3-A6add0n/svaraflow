// svaraflow-core/src/xmeml.rs
//! FCP7 XML (`xmeml` version 4) export.
//!
//! Writes a new sequence named `"{original} - SvaraFlow cut"` that references
//! the original media by file URL. The original timeline is never touched.
//!
//! # Structure
//!
//! The output follows the FCP7 DTD (Apple, *Final Cut Pro 7 XML Interchange
//! Format*). The root is `<xmeml version="4">` with a bare
//! `<!DOCTYPE xmeml>`. The sequence carries a `<uuid>`, `<name>`,
//! `<duration>`, `<rate>`, `<timecode>` and `<media>`. `<media>` always
//! contains a `<video>` block — with an empty `<track/>` when there are no
//! video clips — and an `<audio>` block with `<numOutputChannels>`, a
//! `<format><samplecharacteristics>` and one `<track>` per audio channel.
//!
//! # File definitions and references
//!
//! Each distinct source path gets **one** `<file>` element. The first
//! clipitem that uses a source carries the full definition (name, pathurl,
//! rate, total source duration, timecode, media). Every later clipitem that
//! uses the same source carries `<file id="file-N"/>`, a reference. This
//! keeps `<file><duration>` equal to the *total* source duration rather
//! than a segment length, so `out <= duration` holds for every clipitem.
//!
//! # Profiles
//!
//! [`XmlProfile`] controls how much is written:
//!
//! | Profile | Clips | Sequence metadata | Video block | Links |
//! | --- | --- | --- | --- | --- |
//! | `Minimal` | first only | name, duration, rate | omitted | no |
//! | `Cuts` | all | name, duration, rate | omitted | yes |
//! | `Full` | all | uuid, timecode, name, duration, rate | yes (possibly empty) | yes |
//!
//! `Full` is the default and the one that imports into both Resolve and
//! Premiere. `Minimal` and `Cuts` exist for debugging and for hosts that
//! choke on the empty video block.
//!
//! # Paths
//!
//! Paths in this module are treated as **strings**, never as
//! [`std::path::Path`]. A source string is interpreted the same way on
//! every host: Windows drive letters (`C:\media\x.mov`), the Windows
//! verbatim prefix (`\\?\C:\media\x.mov`), and POSIX paths
//! (`/home/user/media/x.mov`) all parse the same whether the code is
//! compiled for Windows, Linux or macOS.
//!
//! [`encode_path_url`] strips the Windows `\\?\` verbatim prefix (but not
//! the `\\?\UNC\` prefix — removing it would change the path's meaning)
//! and percent-encodes every byte outside the unreserved set plus `/`,
//! with **lowercase** hex. A POSIX-leading `/` is absorbed so the result is
//! always `file://localhost/…`, never `file://localhost//…`.
//!
//! [`source_basename`] returns the final path component of a source
//! string, splitting on both `/` and `\` regardless of host. This is what
//! goes into `<file><name>` so the element holds `interview.mov`, not
//! `C:\media\interview.mov`, on every OS.
//!
//! Neither function can emit a `\\?\` prefix: `source_basename` uses the
//! stripped form, and `encode_path_url` percent-encodes `?` and `\` in the
//! body if any survive.

use std::collections::HashMap;

use quick_xml::events::{BytesDecl, BytesEnd, BytesStart, BytesText, Event};
use quick_xml::Writer;

use crate::error::CoreError;
use crate::timeline::{Fps, Timeline, Track, TrackKind};

/// Version attribute written into the root `<xmeml>` element.
pub const XMEML_VERSION: u32 = 4;

/// Name suffix appended to the original sequence name.
pub const SEQUENCE_SUFFIX: &str = " - SvaraFlow cut";

/// How much of the timeline to write.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum XmlProfile {
    /// One audio clipitem covering the first clip. Smallest importable file.
    Minimal,
    /// Every cut clipitem, with links, but no sequence timecode or video
    /// block.
    Cuts,
    /// Full FCP7 structure: uuid, timecode, empty video block, audio
    /// format, per-track outputchannelindex, links. The default, and the
    /// one that imports into Resolve and Premiere.
    #[default]
    Full,
}

/// Per-source metadata the writer needs but the timeline does not carry.
#[derive(Debug, Clone, PartialEq)]
pub struct SourceInfo {
    /// Total source duration in frames at the timeline's fps.
    pub total_frames: i64,
    /// Audio channel count. `1` for mono, `2` for stereo.
    pub audio_channels: u32,
    /// Audio sample rate in Hz.
    pub audio_sample_rate: u32,
    /// Audio bit depth.
    pub audio_bit_depth: u32,
    /// True when the source carries a video stream.
    pub has_video: bool,
    /// Video frame width in pixels.
    pub video_width: u32,
    /// Video frame height in pixels.
    pub video_height: u32,
}

impl Default for SourceInfo {
    fn default() -> Self {
        Self {
            total_frames: 0,
            audio_channels: 1,
            audio_sample_rate: 48_000,
            audio_bit_depth: 16,
            has_video: false,
            video_width: 1920,
            video_height: 1080,
        }
    }
}

impl SourceInfo {
    /// Builds a best-effort map from a timeline.
    #[must_use]
    pub fn from_timeline(timeline: &Timeline) -> HashMap<String, SourceInfo> {
        let mut map: HashMap<String, SourceInfo> = HashMap::new();
        for track in &timeline.tracks {
            for clip in &track.clips {
                let entry = map
                    .entry(clip.source.clone())
                    .or_insert_with(|| SourceInfo {
                        total_frames: 0,
                        audio_channels: if track.kind == TrackKind::Audio { 1 } else { 0 },
                        audio_sample_rate: 48_000,
                        audio_bit_depth: 16,
                        has_video: track.kind == TrackKind::Video,
                        video_width: timeline.width,
                        video_height: timeline.height,
                    });
                entry.total_frames = entry.total_frames.max(clip.source_out);
                if track.kind == TrackKind::Video {
                    entry.has_video = true;
                }
            }
        }
        map
    }
}

/// Writes `timeline` as FCP7 XML.
///
/// `original_name` is the sequence name the user sees in their host.
/// `sources` maps each `clip.source` string to its [`SourceInfo`]. Missing
/// entries fall back to [`SourceInfo::default`], with `total_frames`
/// clamped up to the clip's `source_out` so `out <= duration` still holds.
///
/// # Errors
///
/// - [`CoreError::InvalidTimeline`] when the timeline has no clips.
/// - [`CoreError::Io`] for any write failure.
pub fn write_xmeml(
    timeline: &Timeline,
    original_name: &str,
    profile: XmlProfile,
    sources: &HashMap<String, SourceInfo>,
) -> Result<String, CoreError> {
    let clip_count: usize = timeline.tracks.iter().map(|t| t.clips.len()).sum();
    if clip_count == 0 {
        return Err(CoreError::InvalidTimeline(
            "cannot export a timeline with no clips: every frame was \
             classified as silence. Try a lower noise level (--noise auto \
             or --noise -50) or a smaller --keep-talks-ms."
                .into(),
        ));
    }

    let mut writer = Writer::new_with_indent(Vec::new(), b' ', 2);

    writer
        .write_event(Event::Decl(BytesDecl::new("1.0", Some("UTF-8"), None)))
        .map_err(io_err)?;
    // Bare DOCTYPE. No SYSTEM identifier: the DTD is not on disk and hosts
    // do not fetch it. A SYSTEM reference to a missing file is what caused
    // Premiere's "EMPTY" import failure.
    writer
        .write_event(Event::DocType(BytesText::from_escaped("xmeml")))
        .map_err(io_err)?;

    let mut root = BytesStart::new("xmeml");
    root.push_attribute(("version", XMEML_VERSION.to_string().as_str()));
    writer.write_event(Event::Start(root)).map_err(io_err)?;

    match profile {
        XmlProfile::Minimal => {
            write_sequence_minimal(&mut writer, timeline, original_name, sources)?
        }
        XmlProfile::Cuts => write_sequence_cuts(&mut writer, timeline, original_name, sources)?,
        XmlProfile::Full => write_sequence_full(&mut writer, timeline, original_name, sources)?,
    }

    writer
        .write_event(Event::End(BytesEnd::new("xmeml")))
        .map_err(io_err)?;

    String::from_utf8(writer.into_inner()).map_err(|e| {
        CoreError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("xmeml writer produced non-UTF-8 bytes: {e}"),
        ))
    })
}

// ---------------------------------------------------------------------------
// Sequences
// ---------------------------------------------------------------------------

fn write_sequence_full(
    writer: &mut Writer<Vec<u8>>,
    timeline: &Timeline,
    original_name: &str,
    sources: &HashMap<String, SourceInfo>,
) -> Result<(), CoreError> {
    let mut seq = BytesStart::new("sequence");
    seq.push_attribute(("id", "sequence-1"));
    writer.write_event(Event::Start(seq)).map_err(io_err)?;

    write_element(writer, "uuid", &make_uuid(original_name))?;
    write_element(writer, "name", &format!("{original_name}{SEQUENCE_SUFFIX}"))?;
    write_element(writer, "duration", &timeline.duration_frames().to_string())?;
    write_rate(writer, timeline.fps)?;
    write_timecode(writer, timeline.fps)?;

    let clips = build_xml_clips(timeline, sources);
    let file_ids = assign_file_ids(&clips);
    write_media_full(writer, timeline, &clips, &file_ids, sources)?;

    writer
        .write_event(Event::End(BytesEnd::new("sequence")))
        .map_err(io_err)?;
    Ok(())
}

fn write_sequence_cuts(
    writer: &mut Writer<Vec<u8>>,
    timeline: &Timeline,
    original_name: &str,
    sources: &HashMap<String, SourceInfo>,
) -> Result<(), CoreError> {
    writer
        .write_event(Event::Start(BytesStart::new("sequence")))
        .map_err(io_err)?;

    write_element(writer, "name", &format!("{original_name}{SEQUENCE_SUFFIX}"))?;
    write_element(writer, "duration", &timeline.duration_frames().to_string())?;
    write_rate(writer, timeline.fps)?;

    let clips = build_xml_clips(timeline, sources);
    let file_ids = assign_file_ids(&clips);
    write_media_cuts(writer, timeline, &clips, &file_ids, sources)?;

    writer
        .write_event(Event::End(BytesEnd::new("sequence")))
        .map_err(io_err)?;
    Ok(())
}

fn write_sequence_minimal(
    writer: &mut Writer<Vec<u8>>,
    timeline: &Timeline,
    original_name: &str,
    sources: &HashMap<String, SourceInfo>,
) -> Result<(), CoreError> {
    writer
        .write_event(Event::Start(BytesStart::new("sequence")))
        .map_err(io_err)?;

    write_element(writer, "name", &format!("{original_name}{SEQUENCE_SUFFIX}"))?;
    write_element(writer, "duration", &timeline.duration_frames().to_string())?;
    write_rate(writer, timeline.fps)?;

    let all = build_xml_clips(timeline, sources);
    let first = all.into_iter().next();
    let clips: Vec<XmlClip> = first.into_iter().collect();
    let file_ids = assign_file_ids(&clips);

    writer
        .write_event(Event::Start(BytesStart::new("media")))
        .map_err(io_err)?;
    writer
        .write_event(Event::Start(BytesStart::new("audio")))
        .map_err(io_err)?;
    writer
        .write_event(Event::Start(BytesStart::new("track")))
        .map_err(io_err)?;
    for (i, clip) in clips.iter().enumerate() {
        write_clipitem(writer, clip, &file_ids, sources, timeline.fps, i == 0)?;
    }
    writer
        .write_event(Event::End(BytesEnd::new("track")))
        .map_err(io_err)?;
    writer
        .write_event(Event::End(BytesEnd::new("audio")))
        .map_err(io_err)?;
    writer
        .write_event(Event::End(BytesEnd::new("media")))
        .map_err(io_err)?;

    writer
        .write_event(Event::End(BytesEnd::new("sequence")))
        .map_err(io_err)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Media blocks
// ---------------------------------------------------------------------------

fn write_media_full(
    writer: &mut Writer<Vec<u8>>,
    timeline: &Timeline,
    clips: &[XmlClip],
    file_ids: &HashMap<String, String>,
    sources: &HashMap<String, SourceInfo>,
) -> Result<(), CoreError> {
    writer
        .write_event(Event::Start(BytesStart::new("media")))
        .map_err(io_err)?;

    writer
        .write_event(Event::Start(BytesStart::new("video")))
        .map_err(io_err)?;
    writer
        .write_event(Event::Start(BytesStart::new("format")))
        .map_err(io_err)?;
    writer
        .write_event(Event::Start(BytesStart::new("samplecharacteristics")))
        .map_err(io_err)?;
    write_element(writer, "width", &timeline.width.to_string())?;
    write_element(writer, "height", &timeline.height.to_string())?;
    write_element(writer, "pixelaspectratio", "square")?;
    write_rate(writer, timeline.fps)?;
    write_element(writer, "fielddominance", "none")?;
    writer
        .write_event(Event::End(BytesEnd::new("samplecharacteristics")))
        .map_err(io_err)?;
    writer
        .write_event(Event::End(BytesEnd::new("format")))
        .map_err(io_err)?;

    let video_tracks: Vec<&Track> = timeline
        .tracks
        .iter()
        .filter(|t| t.kind == TrackKind::Video)
        .collect();
    if video_tracks.is_empty() {
        write_empty(writer, "track")?;
    } else {
        for track in video_tracks {
            writer
                .write_event(Event::Start(BytesStart::new("track")))
                .map_err(io_err)?;
            for (i, clip) in track.clips.iter().enumerate() {
                let xml_clip = XmlClip {
                    id: format!("vclip-{}", i),
                    name: clip.id.clone(),
                    source: clip.source.clone(),
                    source_in: clip.source_in,
                    source_out: clip.source_out,
                    start: clip.start,
                    end: clip.end(),
                    track_index: track.index,
                    link_partner: None,
                    muted: clip.muted,
                    mediatype: TrackKind::Video,
                };
                write_clipitem(writer, &xml_clip, file_ids, sources, timeline.fps, i == 0)?;
            }
            writer
                .write_event(Event::End(BytesEnd::new("track")))
                .map_err(io_err)?;
        }
    }
    writer
        .write_event(Event::End(BytesEnd::new("video")))
        .map_err(io_err)?;

    write_audio_block(writer, timeline, clips, file_ids, sources, true)?;

    writer
        .write_event(Event::End(BytesEnd::new("media")))
        .map_err(io_err)?;
    Ok(())
}

fn write_media_cuts(
    writer: &mut Writer<Vec<u8>>,
    timeline: &Timeline,
    clips: &[XmlClip],
    file_ids: &HashMap<String, String>,
    sources: &HashMap<String, SourceInfo>,
) -> Result<(), CoreError> {
    writer
        .write_event(Event::Start(BytesStart::new("media")))
        .map_err(io_err)?;
    write_audio_block(writer, timeline, clips, file_ids, sources, false)?;
    writer
        .write_event(Event::End(BytesEnd::new("media")))
        .map_err(io_err)?;
    Ok(())
}

fn write_audio_block(
    writer: &mut Writer<Vec<u8>>,
    timeline: &Timeline,
    clips: &[XmlClip],
    file_ids: &HashMap<String, String>,
    sources: &HashMap<String, SourceInfo>,
    with_format: bool,
) -> Result<(), CoreError> {
    writer
        .write_event(Event::Start(BytesStart::new("audio")))
        .map_err(io_err)?;

    let num_tracks = clips
        .iter()
        .map(|c| c.track_index)
        .max()
        .unwrap_or(1)
        .max(1);

    if with_format {
        write_element(writer, "numOutputChannels", &num_tracks.to_string())?;
        writer
            .write_event(Event::Start(BytesStart::new("format")))
            .map_err(io_err)?;
        writer
            .write_event(Event::Start(BytesStart::new("samplecharacteristics")))
            .map_err(io_err)?;
        write_element(writer, "depth", "16")?;
        write_element(writer, "samplerate", "48000")?;
        writer
            .write_event(Event::End(BytesEnd::new("samplecharacteristics")))
            .map_err(io_err)?;
        writer
            .write_event(Event::End(BytesEnd::new("format")))
            .map_err(io_err)?;
    }

    for track_index in 1..=num_tracks {
        writer
            .write_event(Event::Start(BytesStart::new("track")))
            .map_err(io_err)?;
        if with_format {
            write_element(writer, "outputchannelindex", &track_index.to_string())?;
        }
        let mut first_in_track = true;
        for clip in clips.iter().filter(|c| c.track_index == track_index) {
            write_clipitem(
                writer,
                clip,
                file_ids,
                sources,
                timeline.fps,
                first_in_track,
            )?;
            first_in_track = false;
        }
        writer
            .write_event(Event::End(BytesEnd::new("track")))
            .map_err(io_err)?;
    }

    writer
        .write_event(Event::End(BytesEnd::new("audio")))
        .map_err(io_err)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Clipitems and files
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct XmlClip {
    id: String,
    name: String,
    source: String,
    source_in: i64,
    source_out: i64,
    start: i64,
    end: i64,
    track_index: u32,
    link_partner: Option<String>,
    muted: bool,
    mediatype: TrackKind,
}

fn build_xml_clips(timeline: &Timeline, sources: &HashMap<String, SourceInfo>) -> Vec<XmlClip> {
    let mut out = Vec::new();
    let mut counter = 0usize;
    for track in &timeline.tracks {
        if track.kind != TrackKind::Audio {
            continue;
        }
        for clip in &track.clips {
            let channels = sources
                .get(&clip.source)
                .map(|s| s.audio_channels)
                .unwrap_or(1)
                .max(1);
            if channels >= 2 {
                let left_id = format!("clip-{counter}");
                let right_id = format!("clip-{}", counter + 1);
                counter += 2;
                out.push(XmlClip {
                    id: left_id.clone(),
                    name: clip.id.clone(),
                    source: clip.source.clone(),
                    source_in: clip.source_in,
                    source_out: clip.source_out,
                    start: clip.start,
                    end: clip.end(),
                    track_index: 1,
                    link_partner: Some(right_id.clone()),
                    muted: clip.muted,
                    mediatype: TrackKind::Audio,
                });
                out.push(XmlClip {
                    id: right_id,
                    name: clip.id.clone(),
                    source: clip.source.clone(),
                    source_in: clip.source_in,
                    source_out: clip.source_out,
                    start: clip.start,
                    end: clip.end(),
                    track_index: 2,
                    link_partner: Some(left_id),
                    muted: clip.muted,
                    mediatype: TrackKind::Audio,
                });
            } else {
                let id = format!("clip-{counter}");
                counter += 1;
                out.push(XmlClip {
                    id,
                    name: clip.id.clone(),
                    source: clip.source.clone(),
                    source_in: clip.source_in,
                    source_out: clip.source_out,
                    start: clip.start,
                    end: clip.end(),
                    track_index: 1,
                    link_partner: None,
                    muted: clip.muted,
                    mediatype: TrackKind::Audio,
                });
            }
        }
    }
    out
}

fn assign_file_ids(clips: &[XmlClip]) -> HashMap<String, String> {
    let mut map = HashMap::new();
    let mut next = 1usize;
    for clip in clips {
        map.entry(clip.source.clone()).or_insert_with(|| {
            let id = format!("file-{next}");
            next += 1;
            id
        });
    }
    map
}

fn write_clipitem(
    writer: &mut Writer<Vec<u8>>,
    clip: &XmlClip,
    file_ids: &HashMap<String, String>,
    sources: &HashMap<String, SourceInfo>,
    fps: Fps,
    write_full_file: bool,
) -> Result<(), CoreError> {
    let mut start = BytesStart::new("clipitem");
    start.push_attribute(("id", clip.id.as_str()));
    writer.write_event(Event::Start(start)).map_err(io_err)?;

    write_element(writer, "name", &clip.name)?;

    let source_info = sources.get(&clip.source);
    let total_frames = source_info
        .map(|s| s.total_frames)
        .unwrap_or(clip.source_out)
        .max(clip.source_out);
    write_element(writer, "duration", &total_frames.to_string())?;

    write_rate(writer, fps)?;
    write_element(writer, "enabled", if clip.muted { "FALSE" } else { "TRUE" })?;
    write_element(writer, "in", &clip.source_in.to_string())?;
    write_element(writer, "out", &clip.source_out.to_string())?;
    write_element(writer, "start", &clip.start.to_string())?;
    write_element(writer, "end", &clip.end.to_string())?;

    let file_id = file_ids
        .get(&clip.source)
        .cloned()
        .unwrap_or_else(|| "file-1".to_string());
    if write_full_file {
        write_file_full(writer, &file_id, clip, sources, fps)?;
    } else {
        let mut file = BytesStart::new("file");
        file.push_attribute(("id", file_id.as_str()));
        writer.write_event(Event::Empty(file)).map_err(io_err)?;
    }

    writer
        .write_event(Event::Start(BytesStart::new("sourcetrack")))
        .map_err(io_err)?;
    let mediatype = match clip.mediatype {
        TrackKind::Video => "video",
        TrackKind::Audio => "audio",
    };
    write_element(writer, "mediatype", mediatype)?;
    write_element(writer, "trackindex", &clip.track_index.to_string())?;
    writer
        .write_event(Event::End(BytesEnd::new("sourcetrack")))
        .map_err(io_err)?;

    if let Some(partner) = &clip.link_partner {
        let partner_track = if clip.track_index == 1 { 2 } else { 1 };
        writer
            .write_event(Event::Start(BytesStart::new("link")))
            .map_err(io_err)?;
        write_element(writer, "linkclipref", partner)?;
        write_element(writer, "mediatype", "audio")?;
        write_element(writer, "trackindex", &partner_track.to_string())?;
        write_element(writer, "clipindex", "1")?;
        write_element(writer, "groupindex", "1")?;
        writer
            .write_event(Event::End(BytesEnd::new("link")))
            .map_err(io_err)?;
    }

    writer
        .write_event(Event::End(BytesEnd::new("clipitem")))
        .map_err(io_err)?;
    Ok(())
}

fn write_file_full(
    writer: &mut Writer<Vec<u8>>,
    file_id: &str,
    clip: &XmlClip,
    sources: &HashMap<String, SourceInfo>,
    fps: Fps,
) -> Result<(), CoreError> {
    let info = sources.get(&clip.source).cloned().unwrap_or_default();
    let total_frames = info.total_frames.max(clip.source_out);

    let mut file = BytesStart::new("file");
    file.push_attribute(("id", file_id));
    writer.write_event(Event::Start(file)).map_err(io_err)?;

    // Use the pure-string basename helper, NOT `std::path::Path::file_name`.
    // The latter applies the host's separator rules, so the same source
    // string yields a different `<name>` on Windows and Linux, which is
    // what made the golden test fail on CI.
    let basename = source_basename(&clip.source);
    write_element(writer, "name", basename)?;
    write_element(writer, "pathurl", &encode_path_url(&clip.source))?;
    write_rate(writer, fps)?;
    write_element(writer, "duration", &total_frames.to_string())?;
    write_timecode(writer, fps)?;

    writer
        .write_event(Event::Start(BytesStart::new("media")))
        .map_err(io_err)?;
    if info.has_video {
        writer
            .write_event(Event::Start(BytesStart::new("video")))
            .map_err(io_err)?;
        writer
            .write_event(Event::Start(BytesStart::new("samplecharacteristics")))
            .map_err(io_err)?;
        write_element(writer, "width", &info.video_width.to_string())?;
        write_element(writer, "height", &info.video_height.to_string())?;
        writer
            .write_event(Event::End(BytesEnd::new("samplecharacteristics")))
            .map_err(io_err)?;
        writer
            .write_event(Event::End(BytesEnd::new("video")))
            .map_err(io_err)?;
    }
    writer
        .write_event(Event::Start(BytesStart::new("audio")))
        .map_err(io_err)?;
    writer
        .write_event(Event::Start(BytesStart::new("samplecharacteristics")))
        .map_err(io_err)?;
    write_element(writer, "depth", &info.audio_bit_depth.to_string())?;
    write_element(writer, "samplerate", &info.audio_sample_rate.to_string())?;
    writer
        .write_event(Event::End(BytesEnd::new("samplecharacteristics")))
        .map_err(io_err)?;
    write_element(writer, "channelcount", &info.audio_channels.to_string())?;
    writer
        .write_event(Event::End(BytesEnd::new("audio")))
        .map_err(io_err)?;
    writer
        .write_event(Event::End(BytesEnd::new("media")))
        .map_err(io_err)?;

    writer
        .write_event(Event::End(BytesEnd::new("file")))
        .map_err(io_err)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Small writers
// ---------------------------------------------------------------------------

fn write_rate(writer: &mut Writer<Vec<u8>>, fps: Fps) -> Result<(), CoreError> {
    writer
        .write_event(Event::Start(BytesStart::new("rate")))
        .map_err(io_err)?;
    write_element(writer, "ntsc", if fps.is_ntsc() { "TRUE" } else { "FALSE" })?;
    write_element(writer, "timebase", &fps.integer_rate().to_string())?;
    writer
        .write_event(Event::End(BytesEnd::new("rate")))
        .map_err(io_err)?;
    Ok(())
}

fn write_timecode(writer: &mut Writer<Vec<u8>>, fps: Fps) -> Result<(), CoreError> {
    writer
        .write_event(Event::Start(BytesStart::new("timecode")))
        .map_err(io_err)?;
    write_rate(writer, fps)?;
    write_element(writer, "string", "00:00:00:00")?;
    write_element(writer, "frame", "0")?;
    write_element(
        writer,
        "displayformat",
        if is_drop_frame(fps) { "DF" } else { "NDF" },
    )?;
    writer
        .write_event(Event::End(BytesEnd::new("timecode")))
        .map_err(io_err)?;
    Ok(())
}

fn write_element(writer: &mut Writer<Vec<u8>>, name: &str, text: &str) -> Result<(), CoreError> {
    writer
        .write_event(Event::Start(BytesStart::new(name)))
        .map_err(io_err)?;
    writer
        .write_event(Event::Text(BytesText::new(text)))
        .map_err(io_err)?;
    writer
        .write_event(Event::End(BytesEnd::new(name)))
        .map_err(io_err)?;
    Ok(())
}

fn write_empty(writer: &mut Writer<Vec<u8>>, name: &str) -> Result<(), CoreError> {
    writer
        .write_event(Event::Empty(BytesStart::new(name)))
        .map_err(io_err)?;
    Ok(())
}

fn is_drop_frame(fps: Fps) -> bool {
    fps.den == 1001 && matches!(fps.num, 30_000 | 60_000)
}

// ---------------------------------------------------------------------------
// Paths and ids
// ---------------------------------------------------------------------------

/// Returns the final path component of a source string, splitting on both
/// `/` and `\` regardless of host platform.
///
/// This is the string used for `<file><name>`. It is deliberately *not*
/// implemented with [`std::path::Path::file_name`]: that function applies
/// the host's separator rules, so `C:\media\x.mov` returns `x.mov` on
/// Windows but the whole string on Linux. Both are legal Cargo builds of
/// the same code, and both must produce the same XML.
///
/// The Windows `\\?\` prefix is stripped first (see
/// [`strip_verbatim_prefix`]). The `\\?\UNC\…` prefix is kept, but its
/// final component is still extracted correctly because the split is on
/// the separator character, not on a parsed path.
///
/// # Examples
///
/// ```
/// # use svaraflow_core::xmeml::source_basename;
/// assert_eq!(source_basename(r"C:\media\interview.mov"), "interview.mov");
/// assert_eq!(source_basename(r"\\?\C:\media\interview.mov"), "interview.mov");
/// assert_eq!(source_basename("/home/user/media/interview.mov"), "interview.mov");
/// assert_eq!(source_basename("interview.mov"), "interview.mov");
/// ```
#[must_use]
pub fn source_basename(source: &str) -> &str {
    let cleaned = strip_verbatim_prefix(source);
    cleaned
        .rsplit(['/', '\\'])
        .find(|s| !s.is_empty())
        .unwrap_or(cleaned)
}

/// Percent-encodes a media path into a `file://localhost/…` URL.
///
/// Handles both Windows and POSIX source strings, and the same input
/// produces the same output on every host. Steps:
///
/// 1. Strips the Windows `\\?\` verbatim prefix (see
///    [`strip_verbatim_prefix`]). The `\\?\UNC\…` form is kept intact
///    because dropping the prefix would change the path's meaning.
/// 2. Replaces every `\` with `/`.
/// 3. If the result begins with `/` (a POSIX absolute path), absorbs the
///    leading slash so the URL is `file://localhost/…` rather than
///    `file://localhost//…`.
/// 4. Percent-encodes every byte outside the unreserved set plus `/`,
///    using **lowercase** hex digits.
///
/// # Examples
///
/// ```
/// # use svaraflow_core::xmeml::encode_path_url;
/// assert_eq!(
///     encode_path_url(r"C:\Users\me\a b.mp4"),
///     "file://localhost/C%3a/Users/me/a%20b.mp4"
/// );
/// assert_eq!(
///     encode_path_url(r"\\?\C:\Users\me\a b.mp4"),
///     "file://localhost/C%3a/Users/me/a%20b.mp4"
/// );
/// assert_eq!(
///     encode_path_url("/home/user/a b.mp4"),
///     "file://localhost/home/user/a%20b.mp4"
/// );
/// ```
#[must_use]
pub fn encode_path_url(path: &str) -> String {
    let cleaned = strip_verbatim_prefix(path);
    let normalized = cleaned.replace('\\', "/");
    // Absorb one leading slash so POSIX absolute paths do not produce a
    // double slash after the `file://localhost/` prefix.
    let body = normalized.strip_prefix('/').unwrap_or(&normalized);
    let mut out = String::with_capacity(body.len() + 24);
    out.push_str("file://localhost/");
    for byte in body.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' | b'/' => {
                out.push(byte as char);
            }
            _ => {
                out.push('%');
                out.push_str(&format!("{byte:02x}"));
            }
        }
    }
    out
}

/// Strips the Windows `\\?\` extended-length prefix when present.
///
/// `\\?\C:\foo` becomes `C:\foo`. `\\?\UNC\server\share` is returned
/// unchanged because stripping it would change the path's meaning.
fn strip_verbatim_prefix(path: &str) -> &str {
    match path.strip_prefix(r"\\?\") {
        Some(rest) if !rest.starts_with(r"UNC\") => rest,
        _ => path,
    }
}

/// Produces a UUID-shaped string derived from `seed` and the current time.
///
/// Not cryptographically random. FCP7 uses the uuid only as a stable
/// identifier inside one document; it is not a security token. The golden
/// test normalises its content away, so the value is free to change
/// between runs.
fn make_uuid(seed: &str) -> String {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};
    use std::time::{SystemTime, UNIX_EPOCH};

    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);

    let mut h1 = RandomState::new().build_hasher();
    h1.write(seed.as_bytes());
    h1.write_u128(nanos);
    let a = h1.finish();

    let mut h2 = RandomState::new().build_hasher();
    h2.write_u128(nanos);
    h2.write(seed.as_bytes());
    let b = h2.finish();

    format!(
        "{:08X}-{:04X}-{:04X}-{:04X}-{:012X}",
        (a >> 32) as u32,
        (a >> 16) as u16,
        a as u16,
        (b >> 48) as u16,
        b & 0xffff_ffff_ffff,
    )
}

fn io_err(e: std::io::Error) -> CoreError {
    CoreError::Io(e)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::timeline::Clip;

    fn tl_audio_only() -> Timeline {
        Timeline {
            fps: Fps {
                num: 30_000,
                den: 1001,
            },
            width: 1920,
            height: 1080,
            tracks: vec![Track {
                kind: TrackKind::Audio,
                index: 1,
                clips: vec![
                    Clip {
                        id: "a1__0".into(),
                        source: "\\\\?\\C:\\media\\speech1.mp4".into(),
                        source_in: 52,
                        source_out: 118,
                        start: 0,
                        speed: 1.0,
                        has_audio: true,
                        muted: false,
                    },
                    Clip {
                        id: "a1__1".into(),
                        source: "\\\\?\\C:\\media\\speech1.mp4".into(),
                        source_in: 200,
                        source_out: 300,
                        start: 66,
                        speed: 1.0,
                        has_audio: true,
                        muted: false,
                    },
                ],
            }],
        }
    }

    fn sources_for() -> HashMap<String, SourceInfo> {
        let mut m = HashMap::new();
        m.insert(
            "\\\\?\\C:\\media\\speech1.mp4".to_string(),
            SourceInfo {
                total_frames: 375,
                audio_channels: 1,
                audio_sample_rate: 48_000,
                audio_bit_depth: 16,
                has_video: false,
                video_width: 1920,
                video_height: 1080,
            },
        );
        m
    }

    #[test]
    fn strip_verbatim_prefix_removes_drive_prefix() {
        assert_eq!(strip_verbatim_prefix(r"\\?\C:\foo"), r"C:\foo");
        assert_eq!(strip_verbatim_prefix(r"C:\foo"), r"C:\foo");
    }

    #[test]
    fn strip_verbatim_prefix_leaves_unc_alone() {
        assert_eq!(
            strip_verbatim_prefix(r"\\?\UNC\srv\share"),
            r"\\?\UNC\srv\share"
        );
    }

    #[test]
    fn source_basename_windows_path() {
        assert_eq!(source_basename(r"C:\media\interview.mov"), "interview.mov");
    }

    #[test]
    fn source_basename_windows_verbatim_path() {
        assert_eq!(
            source_basename(r"\\?\C:\media\interview.mov"),
            "interview.mov"
        );
    }

    #[test]
    fn source_basename_posix_path() {
        assert_eq!(
            source_basename("/home/user/media/interview.mov"),
            "interview.mov"
        );
    }

    #[test]
    fn source_basename_mixed_separators() {
        assert_eq!(
            source_basename(r"C:\media/subdir/interview.mov"),
            "interview.mov"
        );
    }

    #[test]
    fn source_basename_bare_filename() {
        assert_eq!(source_basename("interview.mov"), "interview.mov");
    }

    #[test]
    fn source_basename_is_identical_on_every_host() {
        // This is the property the golden test relies on: the string is
        // produced by pure parsing, not by `std::path::Path`.
        let cases = [
            (r"C:\media\interview.mov", "interview.mov"),
            (r"\\?\C:\media\interview.mov", "interview.mov"),
            ("/home/user/media/interview.mov", "interview.mov"),
            (r"C:\media/a.mp4", "a.mp4"),
        ];
        for (input, expected) in cases {
            assert_eq!(source_basename(input), expected, "input = {input:?}");
        }
    }

    #[test]
    fn encode_path_url_strips_prefix() {
        assert_eq!(
            encode_path_url(r"\\?\C:\Users\me\a b.mp4"),
            "file://localhost/C%3a/Users/me/a%20b.mp4"
        );
    }

    #[test]
    fn encode_path_url_handles_posix_absolute_path() {
        assert_eq!(
            encode_path_url("/home/user/a b.mp4"),
            "file://localhost/home/user/a%20b.mp4"
        );
    }

    #[test]
    fn encode_path_url_posix_has_no_double_slash() {
        let url = encode_path_url("/home/user/media/interview.mov");
        assert!(!url.contains("//localhost//"), "{url}");
        assert_eq!(url, "file://localhost/home/user/media/interview.mov");
    }

    #[test]
    fn encode_path_url_windows_and_posix_are_shape_identical() {
        // A Windows drive path and a POSIX path both reach `file://localhost/…`
        // with exactly one slash after the host.
        let win = encode_path_url(r"C:\media\a.mp4");
        let posix = encode_path_url("/home/user/a.mp4");
        assert!(win.starts_with("file://localhost/"), "{win}");
        assert!(posix.starts_with("file://localhost/"), "{posix}");
        assert!(!win.starts_with("file://localhost//"), "{win}");
        assert!(!posix.starts_with("file://localhost//"), "{posix}");
    }

    #[test]
    fn encode_path_url_no_percent_3f() {
        let url = encode_path_url(r"\\?\C:\media\speech1.mp4");
        assert!(!url.contains("%3f"), "{url}");
    }

    #[test]
    fn encode_path_url_never_contains_verbatim_prefix() {
        let url = encode_path_url(r"\\?\C:\media\speech1.mp4");
        assert!(!url.contains(r"\\?\"), "{url}");
    }

    #[test]
    fn doctype_is_bare() {
        let tl = tl_audio_only();
        let xml = write_xmeml(&tl, "T", XmlProfile::Full, &sources_for()).unwrap();
        assert!(xml.contains("<!DOCTYPE xmeml>"), "{xml}");
        assert!(!xml.contains("SYSTEM"), "{xml}");
    }

    #[test]
    fn full_profile_has_uuid_timecode_and_video_block() {
        let tl = tl_audio_only();
        let xml = write_xmeml(&tl, "T", XmlProfile::Full, &sources_for()).unwrap();
        assert!(xml.contains("<uuid>"), "{xml}");
        assert!(xml.contains("<timecode>"), "{xml}");
        assert!(xml.contains("<video>"), "{xml}");
        assert!(xml.contains("<track/>"), "{xml}");
        assert!(xml.contains("<numOutputChannels>"), "{xml}");
        assert!(xml.contains("<outputchannelindex>"), "{xml}");
    }

    #[test]
    fn one_file_definition_per_source() {
        let tl = tl_audio_only();
        let xml = write_xmeml(&tl, "T", XmlProfile::Full, &sources_for()).unwrap();
        let full = xml.matches("<file id=\"file-1\">").count();
        let refs = xml.matches("<file id=\"file-1\"/>").count();
        assert_eq!(full, 1, "exactly one full definition: {xml}");
        assert_eq!(refs, 1, "later clipitem references the file: {xml}");
    }

    #[test]
    fn file_name_element_is_basename_not_full_path() {
        // Regression: `Path::file_name()` produced different output on
        // Windows and Linux, which broke the golden test on CI.
        let tl = tl_audio_only();
        let xml = write_xmeml(&tl, "T", XmlProfile::Full, &sources_for()).unwrap();
        assert!(xml.contains("<name>speech1.mp4</name>"), "{xml}");
        assert!(
            !xml.contains(r"<name>C:\media\speech1.mp4</name>"),
            "full path leaked into <name>: {xml}"
        );
    }

    #[test]
    fn file_duration_is_total_source_not_segment() {
        let tl = tl_audio_only();
        let xml = write_xmeml(&tl, "T", XmlProfile::Full, &sources_for()).unwrap();
        assert!(xml.contains("<duration>375</duration>"), "{xml}");
    }

    #[test]
    fn clipitem_has_sourcetrack() {
        let tl = tl_audio_only();
        let xml = write_xmeml(&tl, "T", XmlProfile::Full, &sources_for()).unwrap();
        assert!(xml.contains("<sourcetrack>"), "{xml}");
        assert!(xml.contains("<mediatype>audio</mediatype>"), "{xml}");
    }

    #[test]
    fn minimal_profile_writes_one_clip() {
        let tl = tl_audio_only();
        let xml = write_xmeml(&tl, "T", XmlProfile::Minimal, &sources_for()).unwrap();
        assert_eq!(xml.matches("<clipitem").count(), 1, "{xml}");
        assert!(!xml.contains("<uuid>"), "{xml}");
    }

    #[test]
    fn cuts_profile_writes_all_clips_without_video_block() {
        let tl = tl_audio_only();
        let xml = write_xmeml(&tl, "T", XmlProfile::Cuts, &sources_for()).unwrap();
        assert_eq!(xml.matches("<clipitem").count(), 2, "{xml}");
        assert!(!xml.contains("<video>"), "{xml}");
    }

    #[test]
    fn empty_timeline_errors() {
        let tl = Timeline {
            fps: Fps { num: 30, den: 1 },
            width: 1920,
            height: 1080,
            tracks: vec![Track {
                kind: TrackKind::Audio,
                index: 1,
                clips: vec![],
            }],
        };
        let err = write_xmeml(&tl, "T", XmlProfile::Full, &HashMap::new()).unwrap_err();
        assert!(err.to_string().contains("no clips"), "{err}");
    }
}
