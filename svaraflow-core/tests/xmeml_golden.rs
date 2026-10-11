// svaraflow-core/tests/xmeml_golden.rs
//! Golden and structural tests for the FCP7 XML and CMX3600 EDL writers.

use std::collections::HashMap;
use std::path::PathBuf;

use svaraflow_core::silence::{
    AnalysisReport, FrameRange, NoiseSource, SilenceRanges, SilenceStats, SilenceWarning,
};
use svaraflow_core::timeline::{Clip, Fps, Timeline, Track, TrackKind};
use svaraflow_core::write_edl;
use svaraflow_core::xmeml::{write_xmeml, SourceInfo, XmlProfile};

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn golden(name: &str) -> String {
    let path = fixtures_dir().join(name);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read golden fixture {}: {e}", path.display()))
}

fn update(name: &str, actual: &str) {
    if std::env::var_os("UPDATE_GOLDEN").is_some() {
        let path = fixtures_dir().join(name);
        std::fs::write(&path, actual).unwrap();
    }
}

/// Replaces the content of every `<uuid>...</uuid>` with the fixed literal
/// `UUID` so golden comparison does not depend on the value `write_xmeml`
/// generates.
///
/// The sequence UUID is seeded from `RandomState` and `SystemTime` and is
/// documented as not being a security token. Its value is guaranteed to
/// differ between calls, so it can never be part of a byte-for-byte golden.
/// Its *shape* is asserted separately by `uuid_has_expected_shape` so this
/// normalisation does not lose coverage.
fn normalize_uuid(xml: &str) -> String {
    let mut out = String::with_capacity(xml.len());
    let mut split = xml.split("<uuid>");
    if let Some(head) = split.next() {
        out.push_str(head);
    }
    for part in split {
        match part.find("</uuid>") {
            Some(i) => {
                out.push_str("<uuid>UUID");
                out.push_str(&part[i..]);
            }
            None => {
                out.push_str("<uuid>");
                out.push_str(part);
            }
        }
    }
    out
}

fn golden_timeline() -> Timeline {
    let video = Clip {
        id: "v1".into(),
        source: "C:\\media\\interview.mov".into(),
        source_in: 0,
        source_out: 300,
        start: 0,
        speed: 1.0,
        has_audio: true,
        muted: false,
    };
    let audio = Clip {
        id: "a1".into(),
        source: "C:\\media\\interview.mov".into(),
        source_in: 0,
        source_out: 300,
        start: 0,
        speed: 1.0,
        has_audio: true,
        muted: false,
    };
    Timeline {
        fps: Fps { num: 30, den: 1 },
        width: 1920,
        height: 1080,
        tracks: vec![
            Track {
                kind: TrackKind::Video,
                index: 1,
                clips: vec![video],
            },
            Track {
                kind: TrackKind::Audio,
                index: 1,
                clips: vec![audio],
            },
        ],
    }
}

fn golden_ranges() -> SilenceRanges {
    let keep = vec![FrameRange::new(0, 100), FrameRange::new(200, 300)];
    let silences = vec![FrameRange::new(100, 200)];
    SilenceRanges {
        keep,
        silences: silences.clone(),
        stats: SilenceStats {
            total_frames: 300,
            kept_frames: 200,
            removed_frames: 100,
            cut_count: 1,
            percent_shorter: 100.0 / 3.0,
        },
        noise_db: -50.0,
        warnings: Vec::<SilenceWarning>::new(),
        report: AnalysisReport::empty(-50.0, NoiseSource::Manual),
    }
}

fn sources_for(tl: &Timeline) -> HashMap<String, SourceInfo> {
    SourceInfo::from_timeline(tl)
}

#[test]
fn xmeml_matches_golden() {
    let tl = golden_timeline();
    let sources = sources_for(&tl);
    let actual = write_xmeml(&tl, "Original Sequence", XmlProfile::Full, &sources).unwrap();

    // Write the fixture with the UUID normalised, so regeneration under
    // UPDATE_GOLDEN=1 produces a file this test can match on every later
    // run. Any other byte difference is a real change and still fails.
    let actual_norm = normalize_uuid(&actual);
    update("golden.xml", &actual_norm);

    assert_eq!(
        actual_norm,
        normalize_uuid(&golden("golden.xml")),
        "xmeml output changed"
    );
}

#[test]
fn edl_matches_golden() {
    let tl = golden_timeline();
    let r = golden_ranges();
    let actual = write_edl(&tl, &r, "Original Sequence").unwrap();
    update("golden.edl", &actual);
    assert_eq!(actual, golden("golden.edl"), "edl output changed");
}

#[test]
fn xmeml_is_valid_utf8_xml_shape() {
    let tl = golden_timeline();
    let sources = sources_for(&tl);
    let xml = write_xmeml(&tl, "x", XmlProfile::Full, &sources).unwrap();
    assert!(xml.starts_with("<?xml version=\"1.0\" encoding=\"UTF-8\"?>"));
    assert!(xml.contains("<!DOCTYPE xmeml>"));
    assert!(xml.contains("<xmeml version=\"4\">"));
    assert!(xml.trim_end().ends_with("</xmeml>"));
    assert!(!xml.contains("..."));
    assert!(!xml.contains("TODO"));
}

/// The sequence UUID is normalised out of the golden comparison, so its
/// shape must be checked here instead. FCP7 expects the canonical
/// 8-4-4-4-12 hexadecimal form.
#[test]
fn uuid_has_expected_shape() {
    let tl = golden_timeline();
    let sources = sources_for(&tl);
    let xml = write_xmeml(&tl, "T", XmlProfile::Full, &sources).unwrap();

    let start = xml.find("<uuid>").expect("uuid element present");
    let after = &xml[start + "<uuid>".len()..];
    let end = after.find("</uuid>").expect("uuid closing tag present");
    let uuid = &after[..end];

    let parts: Vec<&str> = uuid.split('-').collect();
    assert_eq!(parts.len(), 5, "uuid has five hyphen-separated groups: {uuid:?}");
    for (group, expected_len) in parts.iter().zip([8usize, 4, 4, 4, 12]) {
        assert_eq!(group.len(), expected_len, "group length in {uuid:?}");
        assert!(
            group.chars().all(|c| c.is_ascii_hexdigit()),
            "non-hex character in {uuid:?}"
        );
    }
}

#[test]
fn xmeml_audio_only_has_no_video_block_clips() {
    let tl = Timeline {
        fps: Fps {
            num: 30_000,
            den: 1001,
        },
        width: 1920,
        height: 1080,
        tracks: vec![Track {
            kind: TrackKind::Audio,
            index: 1,
            clips: vec![Clip {
                id: "audio-1".into(),
                source: "C:\\media\\speech.mp4".into(),
                source_in: 0,
                source_out: 375,
                start: 0,
                speed: 1.0,
                has_audio: true,
                muted: false,
            }],
        }],
    };
    let sources = sources_for(&tl);
    let xml = write_xmeml(&tl, "Audio Only", XmlProfile::Full, &sources).unwrap();

    assert!(xml.contains("<audio>"), "{xml}");
    assert!(xml.contains("</audio>"), "{xml}");
    // The Full profile still emits an empty video block for host
    // compatibility, but no clipitem inside it.
    assert!(xml.contains("<video>"), "{xml}");
    assert!(xml.contains("<track/>"), "{xml}");
    assert!(!xml.contains("clipitem id=\"vclip"), "{xml}");

    assert!(xml.contains("<ntsc>TRUE</ntsc>"), "{xml}");
    assert!(xml.contains("<timebase>30</timebase>"), "{xml}");
    assert!(xml.contains("<clipitem id=\"clip-0\">"), "{xml}");
    assert!(xml.contains("<end>375</end>"), "{xml}");
}

#[test]
fn edl_audio_only_writes_events() {
    let tl = Timeline {
        fps: Fps {
            num: 30_000,
            den: 1001,
        },
        width: 1920,
        height: 1080,
        tracks: vec![Track {
            kind: TrackKind::Audio,
            index: 1,
            clips: vec![Clip {
                id: "audio-1".into(),
                source: "C:\\media\\speech.mp4".into(),
                source_in: 0,
                source_out: 375,
                start: 0,
                speed: 1.0,
                has_audio: true,
                muted: false,
            }],
        }],
    };
    let ranges = SilenceRanges {
        keep: vec![FrameRange::new(0, 100), FrameRange::new(200, 375)],
        silences: vec![FrameRange::new(100, 200)],
        stats: SilenceStats {
            total_frames: 375,
            kept_frames: 275,
            removed_frames: 100,
            cut_count: 1,
            percent_shorter: 100.0 / 3.75,
        },
        noise_db: -50.0,
        warnings: Vec::<SilenceWarning>::new(),
        report: AnalysisReport::empty(-50.0, NoiseSource::Manual),
    };
    let edl = write_edl(&tl, &ranges, "Audio Only").unwrap();
    assert!(edl.starts_with("TITLE: Audio Only\n"));
    assert!(edl.contains("FCM: DROP FRAME"));
    assert!(edl.contains("001  AX"));
    assert!(edl.contains("002  AX"));
}