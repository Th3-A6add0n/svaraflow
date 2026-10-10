//! Apply keep/silence ranges to a timeline.
//!
//! Consumes the [`SilenceRanges`] produced by
//! [`crate::silence::detect_silence`] and returns a **new** [`Timeline`].
//! The input is never mutated: no function in this module takes a
//! `&mut Timeline`.
//!
//! # Modes
//!
//! | Mode | Timeline length | Silence-range audio |
//! | --- | --- | --- |
//! | [`CutMode::Remove`] | shrinks — keep ranges placed back to back at frame 0 | removed |
//! | [`CutMode::Keep`] | unchanged — every range becomes its own clip at its original position | kept |
//! | [`CutMode::Mute`] | unchanged — as Keep, but silence-range clips on audio tracks get `muted: true` | disabled |
//! | [`CutMode::Gaps`] | unchanged — only keep ranges produce clips; silence ranges are empty | removed (no clip) |
//!
//! # Track selection
//!
//! Video tracks are always cut. Audio tracks are cut iff their `index`
//! appears in the caller's selection; `None` means "all audio tracks", and
//! `Some(&[])` means "no audio tracks". Unselected tracks are copied
//! verbatim, so a caller who only wants to touch audio cannot accidentally
//! shift the video.
//!
//! # Preconditions
//!
//! `ranges.keep` and `ranges.silences` must be sorted, disjoint, and
//! non-overlapping. [`crate::silence::detect_silence`] produces such ranges.
//! A caller who violates this precondition will get an invalid timeline; the
//! function does not panic.
//!
//! # Source mapping
//!
//! For a portion `[a, b)` of a clip with `speed v`, the new clip's source
//! range is `[source_in + round((a - clip.start) * v), source_in +
//! round((b - clip.start) * v))`. At `v = 1.0` this is exact; at other
//! speeds the two edges can be off by at most one frame, which is the same
//! approximation [`crate::audio::apply_speed`] makes.

use crate::silence::{FrameRange, SilenceRanges};
use crate::timeline::{Clip, Timeline, Track, TrackKind};

/// How keep/silence ranges are translated into a new timeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CutMode {
    /// Keep ranges are placed back to back starting at frame 0. The new
    /// timeline is shorter than the original.
    Remove,
    /// Every keep and silence range becomes its own clip, at its original
    /// position. The timeline is not shortened.
    Keep,
    /// As [`CutMode::Keep`], but silent-range clips on audio tracks get
    /// `muted: true`. Video clips in silence ranges are unaffected.
    Mute,
    /// Keep ranges stay at their original positions; silence ranges are
    /// left empty (no clips at all).
    Gaps,
}

/// Summary statistics for a cut operation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CutStats {
    /// Number of silence regions in the source ranges.
    pub cut_count: usize,
    /// Total frames of silence encoded in the source ranges. In Remove
    /// mode this equals the timeline shortening; in the other modes it is
    /// informational.
    pub silence_frames: i64,
    /// Percentage shorter if the timeline were cut with Remove mode.
    pub percent_shorter: f64,
    /// Number of clips in the resulting timeline.
    pub clips_created: usize,
    /// Number of original clips that produced no output. Zero in Keep and
    /// Mute modes; nonzero in Remove and Gaps when a clip lay entirely
    /// inside a silence region.
    pub clips_removed: usize,
    /// `Timeline::duration_frames()` of the input timeline.
    pub frames_before: i64,
    /// `Timeline::duration_frames()` of the output timeline.
    pub frames_after: i64,
}

/// Result of [`apply_cuts`].
#[derive(Debug, Clone, PartialEq)]
pub struct CutResult {
    /// The new timeline. The input is never mutated.
    pub timeline: Timeline,
    /// Summary statistics.
    pub stats: CutStats,
}

/// Applies `ranges` to `timeline` under `mode`.
///
/// Returns a new [`Timeline`] and a [`CutStats`]. See the module docs for
/// the mode semantics and the source-mapping rule.
#[must_use]
pub fn apply_cuts(
    timeline: &Timeline,
    ranges: &SilenceRanges,
    mode: CutMode,
    audio_indices: Option<&[u32]>,
) -> CutResult {
    let frames_before = timeline.duration_frames();

    let mut new_tracks = Vec::with_capacity(timeline.tracks.len());
    let mut clips_created = 0usize;
    let mut clips_removed = 0usize;

    for track in &timeline.tracks {
        if !is_selected(track, audio_indices) {
            new_tracks.push(track.clone());
            continue;
        }

        let (clips, removed) = cut_track(track, ranges, mode);
        clips_created += clips.len();
        clips_removed += removed;

        new_tracks.push(Track {
            kind: track.kind,
            index: track.index,
            clips,
        });
    }

    let new_timeline = Timeline {
        fps: timeline.fps,
        width: timeline.width,
        height: timeline.height,
        tracks: new_tracks,
    };
    let frames_after = new_timeline.duration_frames();

    CutResult {
        timeline: new_timeline,
        stats: CutStats {
            cut_count: ranges.stats.cut_count,
            silence_frames: ranges.stats.removed_frames,
            percent_shorter: ranges.stats.percent_shorter,
            clips_created,
            clips_removed,
            frames_before,
            frames_after,
        },
    }
}

/// True when `track` should be cut.
fn is_selected(track: &Track, audio_indices: Option<&[u32]>) -> bool {
    match track.kind {
        TrackKind::Video => true,
        TrackKind::Audio => match audio_indices {
            None => true,
            Some(list) => list.contains(&track.index),
        },
    }
}

/// Cuts one track according to `mode`.
fn cut_track(track: &Track, ranges: &SilenceRanges, mode: CutMode) -> (Vec<Clip>, usize) {
    let (clips, removed) = match mode {
        CutMode::Remove => cut_track_remove(track, &ranges.keep),
        CutMode::Keep => cut_track_split(track, ranges, false),
        CutMode::Mute => {
            let mute_silences = track.kind == TrackKind::Audio;
            cut_track_split(track, ranges, mute_silences)
        }
        CutMode::Gaps => cut_track_gaps(track, &ranges.keep),
    };
    (clips, removed)
}

/// Remove mode: keep ranges concatenated from frame 0.
fn cut_track_remove(track: &Track, keep: &[FrameRange]) -> (Vec<Clip>, usize) {
    // New position of each keep range's start in the shortened timeline.
    let mut new_positions = Vec::with_capacity(keep.len());
    let mut cursor = 0i64;
    for kr in keep {
        new_positions.push(cursor);
        cursor += kr.len();
    }

    let mut produced: Vec<Clip> = Vec::new();
    let mut removed = 0usize;

    for clip in &track.clips {
        let mut any = false;
        for (i, kr) in keep.iter().enumerate() {
            let a = kr.start.max(clip.start);
            let b = kr.end.min(clip.end());
            if b <= a {
                continue;
            }
            let new_start = new_positions[i] + (a - kr.start);
            if let Some(portion) = extract_portion(clip, a, b, new_start, produced.len()) {
                produced.push(portion);
                any = true;
            }
        }
        if !any {
            removed += 1;
        }
    }

    produced.sort_by_key(|c| (c.start, c.end()));
    (produced, removed)
}

/// Keep / Mute mode: every range (keep and silence) becomes its own clip at
/// its original position. `mute_silences` controls whether silence-range
/// clips are muted.
fn cut_track_split(
    track: &Track,
    ranges: &SilenceRanges,
    mute_silences: bool,
) -> (Vec<Clip>, usize) {
    let interleaved = interleave(ranges);
    let mut produced: Vec<Clip> = Vec::new();
    let mut removed = 0usize;

    for clip in &track.clips {
        let mut any = false;
        for &(is_silence, range) in &interleaved {
            let a = range.start.max(clip.start);
            let b = range.end.min(clip.end());
            if b <= a {
                continue;
            }
            if let Some(mut portion) = extract_portion(clip, a, b, a, produced.len()) {
                if is_silence && mute_silences {
                    portion.muted = true;
                }
                produced.push(portion);
                any = true;
            }
        }
        if !any {
            removed += 1;
        }
    }

    produced.sort_by_key(|c| (c.start, c.end()));
    (produced, removed)
}

/// Gaps mode: only keep ranges produce clips, at their original positions.
fn cut_track_gaps(track: &Track, keep: &[FrameRange]) -> (Vec<Clip>, usize) {
    let mut produced: Vec<Clip> = Vec::new();
    let mut removed = 0usize;

    for clip in &track.clips {
        let mut any = false;
        for kr in keep {
            let a = kr.start.max(clip.start);
            let b = kr.end.min(clip.end());
            if b <= a {
                continue;
            }
            if let Some(portion) = extract_portion(clip, a, b, a, produced.len()) {
                produced.push(portion);
                any = true;
            }
        }
        if !any {
            removed += 1;
        }
    }

    produced.sort_by_key(|c| (c.start, c.end()));
    (produced, removed)
}

/// Interleaves keep and silence ranges by start frame.
fn interleave(ranges: &SilenceRanges) -> Vec<(bool, FrameRange)> {
    let mut all: Vec<(bool, FrameRange)> = ranges
        .keep
        .iter()
        .map(|r| (false, *r))
        .chain(ranges.silences.iter().map(|r| (true, *r)))
        .collect();
    all.sort_by_key(|(_, r)| (r.start, r.end));
    all
}

/// Extracts the portion of `clip` that overlaps `[tl_start, tl_end)`,
/// placed at `new_start`.
///
/// `segment` is used to make the new clip's id unique when a single source
/// clip is cut into multiple pieces.
fn extract_portion(
    clip: &Clip,
    tl_start: i64,
    tl_end: i64,
    new_start: i64,
    segment: usize,
) -> Option<Clip> {
    if tl_end <= tl_start {
        return None;
    }
    let a = tl_start.max(clip.start);
    let b = tl_end.min(clip.end());
    if b <= a {
        return None;
    }

    let offset_start = (a - clip.start) as f64;
    let offset_end = (b - clip.start) as f64;

    let source_in = clip.source_in + (offset_start * clip.speed).round() as i64;
    let source_out = clip.source_in + (offset_end * clip.speed).round() as i64;

    if source_out <= source_in {
        return None;
    }

    Some(Clip {
        id: format!("{}__{}", clip.id, segment),
        source: clip.source.clone(),
        source_in,
        source_out,
        start: new_start,
        speed: clip.speed,
        has_audio: clip.has_audio,
        muted: clip.muted,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::silence::{SilenceStats, SilenceWarning};

    const FPS30: crate::timeline::Fps = crate::timeline::Fps { num: 30, den: 1 };

    fn clip(id: &str, start: i64, source_in: i64, source_out: i64) -> Clip {
        Clip {
            id: id.to_string(),
            source: format!("C:\\media\\{id}.mov"),
            source_in,
            source_out,
            start,
            speed: 1.0,
            has_audio: true,
            muted: false,
        }
    }

    fn video_track(index: u32, clips: Vec<Clip>) -> Track {
        Track {
            kind: TrackKind::Video,
            index,
            clips,
        }
    }

    fn audio_track(index: u32, clips: Vec<Clip>) -> Track {
        Track {
            kind: TrackKind::Audio,
            index,
            clips,
        }
    }

    fn timeline(tracks: Vec<Track>) -> Timeline {
        Timeline {
            fps: FPS30,
            width: 1920,
            height: 1080,
            tracks,
        }
    }

    fn ranges(keep: Vec<FrameRange>, silences: Vec<FrameRange>, total: i64) -> SilenceRanges {
        let kept_frames: i64 = keep.iter().map(FrameRange::len).sum();
        let removed_frames: i64 = silences.iter().map(FrameRange::len).sum();
        let percent_shorter = if total > 0 {
            (removed_frames as f64 / total as f64) * 100.0
        } else {
            0.0
        };
        SilenceRanges {
            keep,
            silences: silences.clone(),
            stats: SilenceStats {
                total_frames: total,
                kept_frames,
                removed_frames,
                cut_count: silences.len(),
                percent_shorter,
            },
            noise_db: -50.0,
            warnings: Vec::<SilenceWarning>::new(),
        }
    }

    // --- Remove mode ------------------------------------------------------

    #[test]
    fn remove_places_keeps_back_to_back() {
        // One clip [0, 300), keep [0, 100) and [200, 300).
        let tl = timeline(vec![audio_track(1, vec![clip("a", 0, 0, 300)])]);
        let r = ranges(
            vec![FrameRange::new(0, 100), FrameRange::new(200, 300)],
            vec![FrameRange::new(100, 200)],
            300,
        );

        let result = apply_cuts(&tl, &r, CutMode::Remove, None);
        let clips = &result.timeline.tracks[0].clips;
        assert_eq!(clips.len(), 2);
        assert_eq!(clips[0].start, 0);
        assert_eq!(clips[0].source_in, 0);
        assert_eq!(clips[0].source_out, 100);
        assert_eq!(clips[1].start, 100);
        assert_eq!(clips[1].source_in, 200);
        assert_eq!(clips[1].source_out, 300);
        assert_eq!(result.stats.frames_before, 300);
        assert_eq!(result.stats.frames_after, 200);
        assert_eq!(result.stats.clips_created, 2);
        assert_eq!(result.stats.clips_removed, 0);
    }

    #[test]
    fn remove_shortens_timeline_proportionally() {
        let tl = timeline(vec![audio_track(1, vec![clip("a", 0, 0, 300)])]);
        let r = ranges(
            vec![FrameRange::new(0, 120), FrameRange::new(180, 240)],
            vec![FrameRange::new(120, 180), FrameRange::new(240, 300)],
            300,
        );
        let result = apply_cuts(&tl, &r, CutMode::Remove, None);
        assert_eq!(result.timeline.duration_frames(), 180);
        assert_eq!(result.stats.silence_frames, 120);
    }

    #[test]
    fn remove_keeps_sync_between_video_and_audio() {
        let tl = timeline(vec![
            video_track(1, vec![clip("v", 0, 0, 300)]),
            audio_track(1, vec![clip("a", 0, 0, 300)]),
        ]);
        let r = ranges(
            vec![FrameRange::new(0, 100), FrameRange::new(200, 300)],
            vec![FrameRange::new(100, 200)],
            300,
        );
        let result = apply_cuts(&tl, &r, CutMode::Remove, None);

        for track in &result.timeline.tracks {
            assert_eq!(track.clips.len(), 2);
            assert_eq!(track.clips[0].start, 0);
            assert_eq!(track.clips[0].end(), 100);
            assert_eq!(track.clips[1].start, 100);
            assert_eq!(track.clips[1].end(), 200);
        }
    }

    #[test]
    fn remove_drops_clip_entirely_inside_silence() {
        let tl = timeline(vec![audio_track(
            1,
            vec![clip("keep", 0, 0, 100), clip("gone", 150, 0, 50)],
        )]);
        let r = ranges(
            vec![FrameRange::new(0, 100), FrameRange::new(200, 300)],
            vec![FrameRange::new(100, 200)],
            300,
        );
        let result = apply_cuts(&tl, &r, CutMode::Remove, None);
        assert_eq!(result.timeline.tracks[0].clips.len(), 1);
        assert_eq!(result.stats.clips_removed, 1);
    }

    #[test]
    fn remove_with_full_keep_range_is_identity() {
        let tl = timeline(vec![audio_track(1, vec![clip("a", 0, 0, 300)])]);
        let r = ranges(vec![FrameRange::new(0, 300)], vec![], 300);
        let result = apply_cuts(&tl, &r, CutMode::Remove, None);
        let clips = &result.timeline.tracks[0].clips;
        assert_eq!(clips.len(), 1);
        assert_eq!(clips[0].start, 0);
        assert_eq!(clips[0].source_in, 0);
        assert_eq!(clips[0].source_out, 300);
    }

    // --- Keep mode --------------------------------------------------------

    #[test]
    fn keep_splits_at_every_boundary_without_shortening() {
        let tl = timeline(vec![audio_track(1, vec![clip("a", 0, 0, 300)])]);
        let r = ranges(
            vec![FrameRange::new(0, 100), FrameRange::new(200, 300)],
            vec![FrameRange::new(100, 200)],
            300,
        );
        let result = apply_cuts(&tl, &r, CutMode::Keep, None);
        let clips = &result.timeline.tracks[0].clips;
        assert_eq!(clips.len(), 3);
        assert_eq!(clips[0].start, 0);
        assert_eq!(clips[0].source_in, 0);
        assert_eq!(clips[0].source_out, 100);
        assert_eq!(clips[1].start, 100);
        assert_eq!(clips[1].source_in, 100);
        assert_eq!(clips[1].source_out, 200);
        assert_eq!(clips[2].start, 200);
        assert_eq!(clips[2].source_in, 200);
        assert_eq!(clips[2].source_out, 300);
        assert_eq!(result.timeline.duration_frames(), 300);
        assert!(clips.iter().all(|c| !c.muted));
    }

    // --- Mute mode --------------------------------------------------------

    #[test]
    fn mute_marks_silence_clips_muted_on_audio_tracks() {
        let tl = timeline(vec![audio_track(1, vec![clip("a", 0, 0, 300)])]);
        let r = ranges(
            vec![FrameRange::new(0, 100), FrameRange::new(200, 300)],
            vec![FrameRange::new(100, 200)],
            300,
        );
        let result = apply_cuts(&tl, &r, CutMode::Mute, None);
        let clips = &result.timeline.tracks[0].clips;
        assert_eq!(clips.len(), 3);
        assert!(!clips[0].muted);
        assert!(clips[1].muted, "silence clip on audio must be muted");
        assert!(!clips[2].muted);
    }

    #[test]
    fn mute_leaves_video_clips_alone() {
        let tl = timeline(vec![
            video_track(1, vec![clip("v", 0, 0, 300)]),
            audio_track(1, vec![clip("a", 0, 0, 300)]),
        ]);
        let r = ranges(
            vec![FrameRange::new(0, 100), FrameRange::new(200, 300)],
            vec![FrameRange::new(100, 200)],
            300,
        );
        let result = apply_cuts(&tl, &r, CutMode::Mute, None);

        let video = &result.timeline.tracks[0];
        assert_eq!(video.kind, TrackKind::Video);
        assert!(video.clips.iter().all(|c| !c.muted));

        let audio = &result.timeline.tracks[1];
        assert_eq!(audio.kind, TrackKind::Audio);
        assert!(audio.clips.iter().any(|c| c.muted));
    }

    // --- Gaps mode --------------------------------------------------------

    #[test]
    fn gaps_preserves_positions_and_drops_silence_clips() {
        let tl = timeline(vec![audio_track(1, vec![clip("a", 0, 0, 300)])]);
        let r = ranges(
            vec![FrameRange::new(0, 100), FrameRange::new(200, 300)],
            vec![FrameRange::new(100, 200)],
            300,
        );
        let result = apply_cuts(&tl, &r, CutMode::Gaps, None);
        let clips = &result.timeline.tracks[0].clips;
        assert_eq!(clips.len(), 2);
        assert_eq!(clips[0].start, 0);
        assert_eq!(clips[0].source_in, 0);
        assert_eq!(clips[0].source_out, 100);
        assert_eq!(clips[1].start, 200);
        assert_eq!(clips[1].source_in, 200);
        assert_eq!(clips[1].source_out, 300);
        assert_eq!(result.timeline.duration_frames(), 300);
    }

    // --- Track selection --------------------------------------------------

    #[test]
    fn unselected_audio_tracks_are_untouched() {
        let tl = timeline(vec![
            audio_track(1, vec![clip("a1", 0, 0, 300)]),
            audio_track(2, vec![clip("a2", 0, 0, 300)]),
        ]);
        let r = ranges(
            vec![FrameRange::new(0, 100), FrameRange::new(200, 300)],
            vec![FrameRange::new(100, 200)],
            300,
        );
        let result = apply_cuts(&tl, &r, CutMode::Remove, Some(&[1]));

        let a1 = &result.timeline.tracks[0];
        assert_eq!(a1.clips.len(), 2);

        let a2 = &result.timeline.tracks[1];
        assert_eq!(a2.clips.len(), 1);
        assert_eq!(a2.clips[0].source_in, 0);
        assert_eq!(a2.clips[0].source_out, 300);
    }

    #[test]
    fn empty_audio_selection_skips_all_audio() {
        let tl = timeline(vec![
            video_track(1, vec![clip("v", 0, 0, 300)]),
            audio_track(1, vec![clip("a", 0, 0, 300)]),
        ]);
        let r = ranges(
            vec![FrameRange::new(0, 100), FrameRange::new(200, 300)],
            vec![FrameRange::new(100, 200)],
            300,
        );
        let result = apply_cuts(&tl, &r, CutMode::Remove, Some(&[]));
        assert_eq!(result.timeline.tracks[0].clips.len(), 2);
        assert_eq!(result.timeline.tracks[1].clips.len(), 1);
    }

    // --- Stats and idempotency -------------------------------------------

    #[test]
    fn stats_match_the_result() {
        let tl = timeline(vec![audio_track(1, vec![clip("a", 0, 0, 300)])]);
        let r = ranges(
            vec![FrameRange::new(0, 100), FrameRange::new(200, 300)],
            vec![FrameRange::new(100, 200)],
            300,
        );
        let result = apply_cuts(&tl, &r, CutMode::Remove, None);
        assert_eq!(result.stats.cut_count, 1);
        assert_eq!(result.stats.silence_frames, 100);
        assert!((result.stats.percent_shorter - (100.0 / 3.0)).abs() < 1e-9);
        assert_eq!(result.stats.clips_created, 2);
        assert_eq!(result.stats.clips_removed, 0);
        assert_eq!(result.stats.frames_before, 300);
        assert_eq!(result.stats.frames_after, 200);
    }

    #[test]
    fn output_ids_are_unique_across_tracks() {
        let tl = timeline(vec![
            video_track(1, vec![clip("v", 0, 0, 300)]),
            audio_track(1, vec![clip("a", 0, 0, 300)]),
        ]);
        let r = ranges(
            vec![FrameRange::new(0, 100), FrameRange::new(200, 300)],
            vec![FrameRange::new(100, 200)],
            300,
        );
        let result = apply_cuts(&tl, &r, CutMode::Keep, None);

        let mut ids = std::collections::HashSet::new();
        for track in &result.timeline.tracks {
            for clip in &track.clips {
                assert!(
                    ids.insert(clip.id.clone()),
                    "duplicate id {} in output",
                    clip.id
                );
            }
        }
    }

    #[test]
    fn output_timeline_validates() {
        let tl = timeline(vec![
            video_track(1, vec![clip("v", 0, 0, 300)]),
            audio_track(1, vec![clip("a", 0, 0, 300)]),
        ]);
        let r = ranges(
            vec![FrameRange::new(0, 100), FrameRange::new(200, 300)],
            vec![FrameRange::new(100, 200)],
            300,
        );
        for mode in [CutMode::Remove, CutMode::Keep, CutMode::Mute, CutMode::Gaps] {
            let result = apply_cuts(&tl, &r, mode, None);
            assert!(
                result.timeline.validate().is_ok(),
                "{mode:?} produced invalid timeline"
            );
        }
    }

    #[test]
    fn input_timeline_is_not_mutated() {
        let tl = timeline(vec![audio_track(1, vec![clip("a", 0, 0, 300)])]);
        let snapshot = tl.clone();
        let r = ranges(
            vec![FrameRange::new(0, 100), FrameRange::new(200, 300)],
            vec![FrameRange::new(100, 200)],
            300,
        );
        for mode in [CutMode::Remove, CutMode::Keep, CutMode::Mute, CutMode::Gaps] {
            let _ = apply_cuts(&tl, &r, mode, None);
        }
        assert_eq!(tl, snapshot);
    }

    #[test]
    fn speed_is_preserved_on_each_portion() {
        let mut c = clip("a", 0, 0, 300);
        c.speed = 2.0;
        let tl = timeline(vec![audio_track(1, vec![c])]);
        // At speed 2.0 the clip's timeline duration is 150 frames.
        // Keep [0, 50), silence [50, 100), keep [100, 150).
        let r = ranges(
            vec![FrameRange::new(0, 50), FrameRange::new(100, 150)],
            vec![FrameRange::new(50, 100)],
            150,
        );
        let result = apply_cuts(&tl, &r, CutMode::Remove, None);
        let clips = &result.timeline.tracks[0].clips;
        assert_eq!(clips.len(), 2);
        assert!(clips.iter().all(|c| (c.speed - 2.0).abs() < 1e-12));
        // First portion: 50 timeline frames at speed 2.0 -> 100 source frames.
        assert_eq!(clips[0].source_in, 0);
        assert_eq!(clips[0].source_out, 100);
        assert_eq!(clips[0].start, 0);
        // Second portion: source frames 200..300, timeline [50, 100).
        assert_eq!(clips[1].source_in, 200);
        assert_eq!(clips[1].source_out, 300);
        assert_eq!(clips[1].start, 50);
    }

    #[test]
    fn full_keep_range_is_identity_in_every_mode() {
        let tl = timeline(vec![audio_track(1, vec![clip("a", 0, 0, 300)])]);
        let r = ranges(vec![FrameRange::new(0, 300)], vec![], 300);
        for mode in [CutMode::Remove, CutMode::Keep, CutMode::Mute, CutMode::Gaps] {
            let result = apply_cuts(&tl, &r, mode, None);
            let clips = &result.timeline.tracks[0].clips;
            assert_eq!(clips.len(), 1, "{mode:?}");
            assert_eq!(clips[0].source_in, 0, "{mode:?}");
            assert_eq!(clips[0].source_out, 300, "{mode:?}");
            assert!(!clips[0].muted, "{mode:?}");
        }
    }
}
