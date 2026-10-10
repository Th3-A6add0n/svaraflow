//! Golden and structural tests for the FCP7 XML and CMX3600 EDL writers.
//!
//! The golden tests compare the writers' output byte-for-byte against
//! `tests/fixtures/golden.xml` and `tests/fixtures/golden.edl`. The
//! structural tests assert shape properties that must hold regardless of
//! fixture contents (audio-only output has no `<video>` block, and so on).
//!
//! When a golden fixture changes intentionally, run with `UPDATE_GOLDEN=1`
//! and review the diff. The escape hatch exists so a deliberate format
//! change does not require hand-editing XML.

use std::path::PathBuf;

use svaraflow_core::silence::{FrameRange, SilenceRanges, SilenceStats, SilenceWarning};
use svaraflow_core::timeline::{Clip, Fps, Timeline, Track, TrackKind};
use svaraflow_core::{write_edl, write_xmeml};

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

/// A two-track, two-clip timeline at 30 fps: one video clip, one audio
/// clip, both spanning frames 0..300.
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

/// Keep [0, 100) and [200, 300); remove [100, 200).
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
    }
}

#[test]
fn xmeml_matches_golden() {
    let tl = golden_timeline();
    let actual = write_xmeml(&tl, "Original Sequence").unwrap();
    update("golden.xml", &actual);
    assert_eq!(actual, golden("golden.xml"), "xmeml output changed");
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
    let xml = write_xmeml(&tl, "x").unwrap();
    assert!(xml.starts_with("<?xml version=\"1.0\" encoding=\"UTF-8\"?>"));
    assert!(xml.contains("<!DOCTYPE xmeml SYSTEM \"fcpxmlv4.dtd\">"));
    assert!(xml.contains("<xmeml version=\"4\">"));
    assert!(xml.trim_end().ends_with("</xmeml>"));
    // No placeholder text.
    assert!(!xml.contains("..."));
    assert!(!xml.contains("TODO"));
}

/// An audio-only timeline produces an `<audio>` block and no `<video>`
/// block. This is what `svaraflow-cli timeline from-media` writes for an
/// audio-only source, and both Resolve and Premiere import it.
#[test]
fn xmeml_audio_only_has_no_video_block() {
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
    let xml = write_xmeml(&tl, "Audio Only").unwrap();

    assert!(xml.contains("<audio>"), "{xml}");
    assert!(xml.contains("</audio>"), "{xml}");
    assert!(!xml.contains("<video>"), "audio-only XML must not emit a video block: {xml}");
    assert!(!xml.contains("<video/>"), "{xml}");

    // The NTSC flag is set because the timeline is 30000/1001.
    assert!(xml.contains("<ntsc>TRUE</ntsc>"), "{xml}");
    assert!(xml.contains("<timebase>30</timebase>"), "{xml}");

    // The clipitem is present with the right frames.
    assert!(xml.contains("<clipitem id=\"audio-1\">"), "{xml}");
    assert!(xml.contains("<end>375</end>"), "{xml}");
}

/// An EDL for an audio-only timeline still writes the header and events.
/// The EDL has no notion of track kind, so the shape is the same as for a
/// video timeline.
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
    };
    let edl = write_edl(&tl, &ranges, "Audio Only").unwrap();
    assert!(edl.starts_with("TITLE: Audio Only\n"));
    // 30000/1001 is drop-frame by our `is_ntsc()` rule.
    assert!(edl.contains("FCM: DROP FRAME"));
    assert!(edl.contains("001  AX"));
    assert!(edl.contains("002  AX"));
}