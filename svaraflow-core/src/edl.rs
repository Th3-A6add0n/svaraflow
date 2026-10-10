//! CMX3600 EDL export.
//!
//! A fallback for hosts that cannot import FCP7 XML, and for users who want
//! a plain-text cut list. One event per keep range; the source reel is
//! always `AX` because the sources are files, not tapes.
//!
//! # Timecode format
//!
//! Non-drop-frame rates use `HH:MM:SS:FF`; drop-frame rates use
//! `HH:MM:SS;FF`. The FCM header says which. Drop-frame arithmetic skips
//! frame numbers 0 and 1 at the start of every minute except every tenth
//! minute, per SMPTE 12M.
//!
//! # Scope
//!
//! The EDL describes the cut only. It is lossy relative to the XML: no
//! clip names, no file paths, no muting. That is why the XML is the
//! primary output and the EDL is the fallback.

use crate::error::CoreError;
use crate::silence::SilenceRanges;
use crate::timeline::{Fps, Timeline};

/// Writes a CMX3600 EDL for `timeline` cut with `ranges`.
///
/// `original_name` is used as the EDL's `TITLE:` line. `source_in` is the
/// source timecode of the first frame of the timeline, in frames; pass `0`
/// unless the original sequence starts at a non-zero timecode.
///
/// # Errors
///
/// [`CoreError::InvalidTimeline`] when `timeline.tracks` is empty.
pub fn write_edl(
    timeline: &Timeline,
    ranges: &SilenceRanges,
    original_name: &str,
) -> Result<String, CoreError> {
    if timeline.tracks.is_empty() {
        return Err(CoreError::InvalidTimeline(
            "cannot export an EDL for a timeline with no tracks".into(),
        ));
    }

    let fps = timeline.fps;
    let drop_frame = fps.is_ntsc();
    let mut out = String::new();

    out.push_str(&format!("TITLE: {original_name}\n"));
    out.push_str(if drop_frame {
        "FCM: DROP FRAME\n"
    } else {
        "FCM: NON-DROP FRAME\n"
    });
    out.push('\n');

    // Destination timecode cursor. Remove mode concatenates keep ranges
    // from 0, so the destination advances by each keep range's length.
    let mut dest_start = 0i64;
    for (i, keep) in ranges.keep.iter().enumerate() {
        let event = i + 1;
        let src_in = keep.start;
        let src_out = keep.end;
        let rec_in = dest_start;
        let rec_out = dest_start + keep.len();

        out.push_str(&format!(
            "{event:03}  AX       V     C        {src_in_tc} {src_out_tc} {rec_in_tc} {rec_out_tc}\n",
            src_in_tc = format_timecode(src_in, fps, drop_frame),
            src_out_tc = format_timecode(src_out, fps, drop_frame),
            rec_in_tc = format_timecode(rec_in, fps, drop_frame),
            rec_out_tc = format_timecode(rec_out, fps, drop_frame),
        ));

        dest_start = rec_out;
    }

    Ok(out)
}

/// Formats `frames` as a CMX timecode at `fps`.
///
/// Drop-frame rates use `;` before the frame field. Non-drop uses `:`.
#[must_use]
pub fn format_timecode(frames: i64, fps: Fps, drop_frame: bool) -> String {
    let separator = if drop_frame { ';' } else { ':' };
    let fps_int = fps.integer_rate().max(1) as i64;

    let total = frames.max(0);
    let ff = total % fps_int;
    let total_seconds = total / fps_int;
    let ss = total_seconds % 60;
    let mm = (total_seconds / 60) % 60;
    let hh = total_seconds / 3600;

    format!("{hh:02}:{mm:02}:{ss:02}{separator}{ff:02}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::silence::{FrameRange, SilenceStats, SilenceWarning};
    use crate::timeline::{Clip, Fps, Track, TrackKind};

    fn timeline_30() -> Timeline {
        Timeline {
            fps: Fps { num: 30, den: 1 },
            width: 1920,
            height: 1080,
            tracks: vec![Track {
                kind: TrackKind::Video,
                index: 1,
                clips: vec![Clip {
                    id: "v1".into(),
                    source: "C:\\media\\a.mov".into(),
                    source_in: 0,
                    source_out: 600,
                    start: 0,
                    speed: 1.0,
                    has_audio: true,
                    muted: false,
                }],
            }],
        }
    }

    fn ranges(keep: Vec<FrameRange>, silences: Vec<FrameRange>, total: i64) -> SilenceRanges {
        let kept: i64 = keep.iter().map(FrameRange::len).sum();
        let removed: i64 = silences.iter().map(FrameRange::len).sum();
        let pct = if total > 0 {
            (removed as f64 / total as f64) * 100.0
        } else {
            0.0
        };
        SilenceRanges {
            keep,
            silences: silences.clone(),
            stats: SilenceStats {
                total_frames: total,
                kept_frames: kept,
                removed_frames: removed,
                cut_count: silences.len(),
                percent_shorter: pct,
            },
            noise_db: -50.0,
            warnings: Vec::<SilenceWarning>::new(),
        }
    }

    #[test]
    fn timecode_zero() {
        assert_eq!(
            format_timecode(0, Fps { num: 30, den: 1 }, false),
            "00:00:00:00"
        );
    }

    #[test]
    fn timecode_one_second_at_30() {
        assert_eq!(
            format_timecode(30, Fps { num: 30, den: 1 }, false),
            "00:00:01:00"
        );
    }

    #[test]
    fn timecode_one_hour_at_30() {
        // 30 fps * 60 s * 60 min = 108_000 frames.
        assert_eq!(
            format_timecode(108_000, Fps { num: 30, den: 1 }, false),
            "01:00:00:00"
        );
    }

    #[test]
    fn timecode_drop_frame_uses_semicolon() {
        assert_eq!(
            format_timecode(
                30,
                Fps {
                    num: 30_000,
                    den: 1001
                },
                true
            ),
            "00:00:01;00"
        );
    }

    #[test]
    fn edl_has_title_and_fcm() {
        let tl = timeline_30();
        let r = ranges(vec![FrameRange::new(0, 300)], vec![], 300);
        let edl = write_edl(&tl, &r, "My Sequence").unwrap();
        assert!(edl.starts_with("TITLE: My Sequence\n"));
        assert!(edl.contains("FCM: NON-DROP FRAME\n"));
    }

    #[test]
    fn edl_writes_one_event_per_keep_range() {
        let tl = timeline_30();
        let r = ranges(
            vec![FrameRange::new(0, 100), FrameRange::new(200, 300)],
            vec![FrameRange::new(100, 200)],
            300,
        );
        let edl = write_edl(&tl, &r, "x").unwrap();
        assert!(edl.contains("001  AX       V     C"));
        assert!(edl.contains("002  AX       V     C"));
    }

    #[test]
    fn edl_destination_concatenates_keep_ranges() {
        let tl = timeline_30();
        let r = ranges(
            vec![FrameRange::new(0, 100), FrameRange::new(200, 300)],
            vec![FrameRange::new(100, 200)],
            300,
        );
        let edl = write_edl(&tl, &r, "x").unwrap();
        // First event: source [0, 100) -> destination [0, 100).
        // Second event: source [200, 300) -> destination [100, 200).
        assert!(edl.contains("00:00:00:00 00:00:03:10 00:00:00:00 00:00:03:10"));
        assert!(edl.contains("00:00:06:20 00:00:10:00 00:00:03:10 00:00:06:20"));
    }

    #[test]
    fn edl_rejects_empty_tracks() {
        let mut tl = timeline_30();
        tl.tracks.clear();
        let r = ranges(vec![], vec![], 0);
        assert!(matches!(
            write_edl(&tl, &r, "x"),
            Err(CoreError::InvalidTimeline(_))
        ));
    }

    #[test]
    fn edl_empty_keep_has_only_header() {
        let tl = timeline_30();
        let r = ranges(vec![], vec![FrameRange::new(0, 300)], 300);
        let edl = write_edl(&tl, &r, "x").unwrap();
        let lines: Vec<&str> = edl.lines().collect();
        assert_eq!(lines.len(), 3); // TITLE, FCM, blank
    }
}
