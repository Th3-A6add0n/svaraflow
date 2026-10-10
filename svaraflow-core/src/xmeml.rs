//! FCP7 XML (`xmeml` version 4) export.
//!
//! Writes a new sequence named `"{original} - SvaraFlow cut"` that references
//! the original media by file URL. The original timeline is never touched:
//! this module only reads a [`Timeline`] and writes XML.
//!
//! # Frame semantics
//!
//! Inside a `<clipitem>`:
//!
//! - `<start>` / `<end>` are **destination** frames: where the clip sits on
//!   the new sequence's timeline, half-open (`end` exclusive).
//! - `<in>` / `<out>` are the **source** frames the clipitem consumes, in
//!   the source file's timebase. Because SvaraFlow does not retime during a
//!   cut, these are the same as the original `Clip::source_in` /
//!   `Clip::source_out`.
//! - `<duration>` is the source duration, `out - in`.
//!
//! The nested `<file>` block repeats `<in>` / `<out>` for the file's own
//! range. See ADR-0015 for the DTD reading this is based on.
//!
//! # File URLs
//!
//! `<pathurl>` is written as `file://localhost/C%3a/media/file.mov`: drive
//! letter percent-encoded, backslashes replaced by `/`, spaces as `%20`.
//!
//! # Linked clips
//!
//! Every `clipitem` carries one `<link>` block per member of its link group,
//! including itself. A link group is the set of clips that share the same
//! `(source, start)` **and** includes at least one video and one audio clip;
//! clips that do not share both a source and a start frame are not linked.
//! This is what makes Premiere and Resolve arrive with the video and audio
//! still paired after import.

use std::collections::HashMap;

use quick_xml::events::{BytesDecl, BytesEnd, BytesStart, BytesText, Event};
use quick_xml::Writer;

use crate::error::CoreError;
use crate::timeline::{Clip, Timeline, Track, TrackKind};

/// Version attribute written into the root `<xmeml>` element.
pub const XMEML_VERSION: u32 = 4;

/// Name suffix appended to the original sequence name.
pub const SEQUENCE_SUFFIX: &str = " - SvaraFlow cut";

/// Writes `timeline` as FCP7 XML.
///
/// `original_name` is the sequence name the user sees in their host. The
/// output sequence is named `"{original_name}{SEQUENCE_SUFFIX}"`.
///
/// # Errors
///
/// [`CoreError::Io`] for any write failure, [`CoreError::InvalidTimeline`]
/// if `timeline` has no tracks.
pub fn write_xmeml(timeline: &Timeline, original_name: &str) -> Result<String, CoreError> {
    if timeline.tracks.is_empty() {
        return Err(CoreError::InvalidTimeline(
            "cannot export a timeline with no tracks".into(),
        ));
    }

    let link_index = LinkIndex::build(timeline);

    let mut writer = Writer::new_with_indent(Vec::new(), b' ', 2);

    writer
        .write_event(Event::Decl(BytesDecl::new("1.0", Some("UTF-8"), None)))
        .map_err(io_err)?;
    writer
        .write_event(Event::DocType(BytesText::from_escaped(
            "xmeml SYSTEM \"fcpxmlv4.dtd\"",
        )))
        .map_err(io_err)?;

    let mut root = BytesStart::new("xmeml");
    root.push_attribute(("version", XMEML_VERSION.to_string().as_str()));
    writer.write_event(Event::Start(root)).map_err(io_err)?;

    writer
        .write_event(Event::Start(BytesStart::new("sequence")))
        .map_err(io_err)?;

    write_element(&mut writer, "name", &format!("{original_name}{SEQUENCE_SUFFIX}"))?;
    write_element(
        &mut writer,
        "duration",
        &timeline.duration_frames().to_string(),
    )?;
    write_rate(&mut writer, timeline)?;
    write_element(&mut writer, "in", "-1")?;
    write_element(&mut writer, "out", "-1")?;

    writer
        .write_event(Event::Start(BytesStart::new("media")))
        .map_err(io_err)?;
    write_tracks(&mut writer, timeline, &link_index)?;
    writer
        .write_event(Event::End(BytesEnd::new("media")))
        .map_err(io_err)?;
    writer
        .write_event(Event::End(BytesEnd::new("sequence")))
        .map_err(io_err)?;
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

/// Writes `<video>` and `<audio>` blocks with their tracks.
fn write_tracks(
    writer: &mut Writer<Vec<u8>>,
    timeline: &Timeline,
    link_index: &LinkIndex,
) -> Result<(), CoreError> {
    for kind in [TrackKind::Video, TrackKind::Audio] {
        let element = match kind {
            TrackKind::Video => "video",
            TrackKind::Audio => "audio",
        };
        let matching: Vec<&Track> = timeline
            .tracks
            .iter()
            .filter(|t| t.kind == kind)
            .collect();
        if matching.is_empty() {
            continue;
        }

        writer
            .write_event(Event::Start(BytesStart::new(element)))
            .map_err(io_err)?;

        for track in matching {
            writer
                .write_event(Event::Start(BytesStart::new("track")))
                .map_err(io_err)?;
            for clip in &track.clips {
                write_clipitem(writer, clip, kind, link_index, timeline)?;
            }
            writer
                .write_event(Event::End(BytesEnd::new("track")))
                .map_err(io_err)?;
        }

        writer
            .write_event(Event::End(BytesEnd::new(element)))
            .map_err(io_err)?;
    }
    Ok(())
}

/// Writes one `<clipitem>`.
fn write_clipitem(
    writer: &mut Writer<Vec<u8>>,
    clip: &Clip,
    kind: TrackKind,
    link_index: &LinkIndex,
    timeline: &Timeline,
) -> Result<(), CoreError> {
    let mut start = BytesStart::new("clipitem");
    start.push_attribute(("id", clip.id.as_str()));
    writer.write_event(Event::Start(start)).map_err(io_err)?;

    write_element(writer, "name", &clip.id)?;
    write_element(writer, "duration", &clip.source_duration().to_string())?;
    write_rate(writer, timeline)?;

    // `<enabled>` is where a muted audio clipitem is expressed. Video clips
    // are always enabled, regardless of `muted`, because Mute mode only
    // touches audio (ADR-0014).
    let enabled = match kind {
        TrackKind::Video => true,
        TrackKind::Audio => !clip.muted,
    };
    write_element(writer, "enabled", if enabled { "TRUE" } else { "FALSE" })?;

    write_element(writer, "start", &clip.start.to_string())?;
    write_element(writer, "end", &clip.end().to_string())?;
    write_element(writer, "in", &clip.source_in.to_string())?;
    write_element(writer, "out", &clip.source_out.to_string())?;

    write_file(writer, clip, timeline)?;

    // One `<link>` block per member of the group, including self.
    if let Some(members) = link_index.group_of(&clip.id) {
        for member in members {
            write_link_block(writer, member)?;
        }
    }

    writer
        .write_event(Event::End(BytesEnd::new("clipitem")))
        .map_err(io_err)?;
    Ok(())
}

/// Writes the nested `<file>` block for a clip.
fn write_file(
    writer: &mut Writer<Vec<u8>>,
    clip: &Clip,
    timeline: &Timeline,
) -> Result<(), CoreError> {
    let mut file = BytesStart::new("file");
    file.push_attribute(("id", format!("file-{}", clip.id).as_str()));
    writer.write_event(Event::Start(file)).map_err(io_err)?;

    let basename = std::path::Path::new(&clip.source)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(&clip.source);
    write_element(writer, "name", basename)?;
    write_element(writer, "pathurl", &encode_path_url(&clip.source))?;
    write_rate(writer, timeline)?;
    write_element(writer, "duration", &clip.source_duration().to_string())?;
    write_element(writer, "in", &clip.source_in.to_string())?;
    write_element(writer, "out", &clip.source_out.to_string())?;

    writer
        .write_event(Event::End(BytesEnd::new("file")))
        .map_err(io_err)?;
    Ok(())
}

/// Writes one `<link>` block describing a link-group member.
fn write_link_block(writer: &mut Writer<Vec<u8>>, member: &LinkMember) -> Result<(), CoreError> {
    writer
        .write_event(Event::Start(BytesStart::new("link")))
        .map_err(io_err)?;

    write_element(writer, "linkclipref", &member.clip_id)?;
    let mediatype = match member.mediatype {
        TrackKind::Video => "video",
        TrackKind::Audio => "audio",
    };
    write_element(writer, "mediatype", mediatype)?;
    write_element(writer, "trackindex", &member.track_index.to_string())?;
    write_element(writer, "clipindex", "1")?;

    writer
        .write_event(Event::End(BytesEnd::new("link")))
        .map_err(io_err)?;
    Ok(())
}

/// Writes `<rate><ntsc>…</ntsc><timebase>…</timebase></rate>`.
fn write_rate(writer: &mut Writer<Vec<u8>>, timeline: &Timeline) -> Result<(), CoreError> {
    writer
        .write_event(Event::Start(BytesStart::new("rate")))
        .map_err(io_err)?;
    write_element(
        writer,
        "ntsc",
        if timeline.fps.is_ntsc() { "TRUE" } else { "FALSE" },
    )?;
    write_element(
        writer,
        "timebase",
        &timeline.fps.integer_rate().to_string(),
    )?;
    writer
        .write_event(Event::End(BytesEnd::new("rate")))
        .map_err(io_err)?;
    Ok(())
}

/// Writes `<name>text</name>`.
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

/// Percent-encodes an absolute media path into a `file://localhost/…` URL.
///
/// `C:\media\a b.mov` becomes
/// `file://localhost/C%3a/media/a%20b.mov`. Drive letters are case-preserved;
/// backslashes become forward slashes; every byte outside the unreserved set
/// plus `/` is percent-encoded as uppercase hex.
#[must_use]
pub fn encode_path_url(path: &str) -> String {
    let normalized = path.replace('\\', "/");
    let mut out = String::with_capacity(normalized.len() + 24);
    out.push_str("file://localhost/");
    for byte in normalized.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' | b'/' => {
                out.push(byte as char);
            }
            _ => {
                out.push('%');
                out.push_str(&format!("{byte:02X}"));
            }
        }
    }
    out
}

/// One member of a link group.
#[derive(Debug, Clone, PartialEq, Eq)]
struct LinkMember {
    clip_id: String,
    mediatype: TrackKind,
    /// 1-based index within the media kind.
    track_index: u32,
}

/// Key that identifies a potential link group: same source file, same start
/// frame on the timeline.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct LinkKey {
    source: String,
    start: i64,
}

/// Maps each linked clip id to all members of its group, including itself.
///
/// A group is kept only when it contains at least one video and at least one
/// audio clip. Clips whose key is unique to their kind are not linked and do
/// not appear in the index.
#[derive(Debug, Default)]
struct LinkIndex {
    groups: HashMap<String, Vec<LinkMember>>,
}

impl LinkIndex {
    /// Builds the index by scanning every track.
    fn build(timeline: &Timeline) -> Self {
        let mut by_key: HashMap<LinkKey, Vec<LinkMember>> = HashMap::new();
        for track in &timeline.tracks {
            for clip in &track.clips {
                let key = LinkKey {
                    source: clip.source.clone(),
                    start: clip.start,
                };
                by_key.entry(key).or_default().push(LinkMember {
                    clip_id: clip.id.clone(),
                    mediatype: track.kind,
                    track_index: track.index,
                });
            }
        }

        let mut groups: HashMap<String, Vec<LinkMember>> = HashMap::new();
        for (_, members) in by_key {
            let has_video = members.iter().any(|m| m.mediatype == TrackKind::Video);
            let has_audio = members.iter().any(|m| m.mediatype == TrackKind::Audio);
            if !(has_video && has_audio) {
                continue;
            }
            // Sort so the output is deterministic: video first, then by
            // track index, then by clip id.
            let mut sorted = members;
            sorted.sort_by(|a, b| {
                let kind_rank = |k: TrackKind| match k {
                    TrackKind::Video => 0u8,
                    TrackKind::Audio => 1,
                };
                kind_rank(a.mediatype)
                    .cmp(&kind_rank(b.mediatype))
                    .then(a.track_index.cmp(&b.track_index))
                    .then(a.clip_id.cmp(&b.clip_id))
            });
            for member in &sorted {
                groups.insert(member.clip_id.clone(), sorted.clone());
            }
        }

        Self { groups }
    }

    /// Returns every member of `clip_id`'s group, including the clip itself,
    /// or `None` when the clip is not linked.
    fn group_of(&self, clip_id: &str) -> Option<&[LinkMember]> {
        self.groups.get(clip_id).map(Vec::as_slice)
    }
}

/// Wraps any `std::io::Error` from the XML writer.
fn io_err(e: std::io::Error) -> CoreError {
    CoreError::Io(e)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::timeline::{Clip, Fps, Timeline, Track, TrackKind};

    fn tl_30() -> Timeline {
        Timeline {
            fps: Fps { num: 30, den: 1 },
            width: 1920,
            height: 1080,
            tracks: vec![Track {
                kind: TrackKind::Video,
                index: 1,
                clips: vec![Clip {
                    id: "v1".into(),
                    source: "C:\\media\\a b.mov".into(),
                    source_in: 0,
                    source_out: 300,
                    start: 0,
                    speed: 1.0,
                    has_audio: true,
                    muted: false,
                }],
            }],
        }
    }

    fn linked_tl() -> Timeline {
        let mut tl = tl_30();
        tl.tracks.push(Track {
            kind: TrackKind::Audio,
            index: 1,
            clips: vec![Clip {
                id: "a1".into(),
                source: "C:\\media\\a b.mov".into(),
                source_in: 0,
                source_out: 300,
                start: 0,
                speed: 1.0,
                has_audio: true,
                muted: false,
            }],
        });
        tl
    }

    #[test]
    fn encode_path_url_windows() {
        assert_eq!(
            encode_path_url("C:\\media\\a b.mov"),
            "file://localhost/C%3a/media/a%20b.mov"
        );
    }

    #[test]
    fn encode_path_url_keeps_safe_bytes() {
        assert_eq!(
            encode_path_url("C:\\a-b_c.d~e.mov"),
            "file://localhost/C%3a/a-b_c.d~e.mov"
        );
    }

    #[test]
    fn encode_path_url_percent_encodes_hash_and_amp() {
        assert_eq!(
            encode_path_url("C:\\x#y&z.mov"),
            "file://localhost/C%3a/x%23y%26z.mov"
        );
    }

    #[test]
    fn xmeml_has_declaration_and_root() {
        let xml = write_xmeml(&tl_30(), "Original").unwrap();
        assert!(xml.starts_with("<?xml"));
        assert!(xml.contains("<xmeml version=\"4\">"));
        assert!(xml.contains("</xmeml>"));
    }

    #[test]
    fn xmeml_names_the_new_sequence() {
        let xml = write_xmeml(&tl_30(), "Interview").unwrap();
        assert!(xml.contains("<name>Interview - SvaraFlow cut</name>"));
    }

    #[test]
    fn xmeml_rejects_empty_tracks() {
        let mut tl = tl_30();
        tl.tracks.clear();
        assert!(matches!(
            write_xmeml(&tl, "x"),
            Err(CoreError::InvalidTimeline(_))
        ));
    }

    #[test]
    fn xmeml_writes_rate_with_ntsc_flag() {
        let xml = write_xmeml(&tl_30(), "x").unwrap();
        assert!(xml.contains("<ntsc>FALSE</ntsc>"));
        assert!(xml.contains("<timebase>30</timebase>"));
    }

    #[test]
    fn xmeml_writes_ntsc_true_for_29_97() {
        let mut tl = tl_30();
        tl.fps = Fps {
            num: 30_000,
            den: 1001,
        };
        let xml = write_xmeml(&tl, "x").unwrap();
        assert!(xml.contains("<ntsc>TRUE</ntsc>"));
        assert!(xml.contains("<timebase>30</timebase>"));
    }

    #[test]
    fn xmeml_writes_clipitem_frames() {
        let xml = write_xmeml(&tl_30(), "x").unwrap();
        assert!(xml.contains("<start>0</start>"));
        assert!(xml.contains("<end>300</end>"));
        assert!(xml.contains("<in>0</in>"));
        assert!(xml.contains("<out>300</out>"));
    }

    #[test]
    fn xmeml_writes_file_pathurl() {
        let xml = write_xmeml(&tl_30(), "x").unwrap();
        assert!(xml.contains("<pathurl>file://localhost/C%3a/media/a%20b.mov</pathurl>"));
    }

    #[test]
    fn xmeml_writes_enabled_false_for_muted_audio() {
        let tl = Timeline {
            fps: Fps { num: 30, den: 1 },
            width: 1920,
            height: 1080,
            tracks: vec![Track {
                kind: TrackKind::Audio,
                index: 1,
                clips: vec![Clip {
                    id: "a1".into(),
                    source: "C:\\media\\a.wav".into(),
                    source_in: 0,
                    source_out: 300,
                    start: 0,
                    speed: 1.0,
                    has_audio: true,
                    muted: true,
                }],
            }],
        };
        let xml = write_xmeml(&tl, "x").unwrap();
        assert!(xml.contains("<enabled>FALSE</enabled>"));
    }

    #[test]
    fn xmeml_never_mutes_video() {
        let mut tl = tl_30();
        tl.tracks[0].clips[0].muted = true;
        let xml = write_xmeml(&tl, "x").unwrap();
        assert!(xml.contains("<enabled>TRUE</enabled>"));
        assert!(!xml.contains("<enabled>FALSE</enabled>"));
    }

    // --- linking ---------------------------------------------------------

    #[test]
    fn standalone_clip_has_no_link_block() {
        let xml = write_xmeml(&tl_30(), "x").unwrap();
        assert!(
            !xml.contains("<link>"),
            "single video clip must not emit link blocks: {xml}"
        );
    }

    #[test]
    fn linked_pair_writes_two_link_blocks_per_clip() {
        let xml = write_xmeml(&linked_tl(), "x").unwrap();
        // Two clips, each with two link blocks (self and partner) = four.
        let count = xml.matches("<link>").count();
        assert_eq!(count, 4, "expected 4 link blocks, got {count} in:\n{xml}");
    }

    #[test]
    fn linked_pair_references_the_other_clip() {
        let xml = write_xmeml(&linked_tl(), "x").unwrap();
        // Each link group lists both members. Look for the cross references.
        assert!(xml.contains("<linkclipref>v1</linkclipref>"));
        assert!(xml.contains("<linkclipref>a1</linkclipref>"));
        assert!(xml.contains("<mediatype>video</mediatype>"));
        assert!(xml.contains("<mediatype>audio</mediatype>"));
    }

    #[test]
    fn clips_with_different_starts_are_not_linked() {
        let mut tl = linked_tl();
        tl.tracks[1].clips[0].start = 100;
        let xml = write_xmeml(&tl, "x").unwrap();
        assert!(
            !xml.contains("<link>"),
            "clips with different starts must not be linked: {xml}"
        );
    }

    #[test]
    fn two_video_clips_are_not_linked() {
        let mut tl = tl_30();
        tl.tracks.push(Track {
            kind: TrackKind::Video,
            index: 2,
            clips: vec![tl.tracks[0].clips[0].clone()],
        });
        // Same source, same start, both video. No audio in the group.
        let xml = write_xmeml(&tl, "x").unwrap();
        assert!(
            !xml.contains("<link>"),
            "video-only groups must not be linked: {xml}"
        );
    }
}