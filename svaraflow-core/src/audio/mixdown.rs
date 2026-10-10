// svaraflow-core/src/audio/mixdown.rs
//! Places decoded clips on the timeline and mixes the selected tracks.
//!
//! [`render_analysis_signal`] is deliberately ignorant of media files: it
//! asks a [`ClipDecoder`] for each clip's analysis-domain audio and overlays
//! the result. That keeps the placement arithmetic testable with synthetic
//! signals and no fixtures on disk.

use crate::error::CoreError;
use crate::timeline::{Clip, Fps, Timeline, Track, TrackKind};

use super::{MonoAudio, ANALYSIS_SAMPLE_RATE};

/// Decodes one clip into the analysis domain.
///
/// Implementors must return audio at exactly [`ANALYSIS_SAMPLE_RATE`],
/// already time-compressed by the clip's `speed`. [`FileDecoder`] does both.
///
/// [`FileDecoder`]: super::decode::FileDecoder
pub trait ClipDecoder {
    /// Returns the clip's audio at 16 kHz mono, `speed` already applied.
    ///
    /// # Errors
    ///
    /// Implementors should return [`CoreError::AudioDecode`] naming `clip.id`.
    fn decode_clip(&mut self, clip: &Clip) -> Result<MonoAudio, CoreError>;
}

/// Converts a frame count at `fps` to a sample count at `sample_rate`.
///
/// Uses the exact rational `fps`, so `30_000/1001` does not accumulate drift
/// over long timelines.
#[must_use]
pub fn frames_to_samples(frames: i64, fps: Fps, sample_rate: u32) -> i64 {
    if fps.num == 0 || fps.den == 0 {
        return 0;
    }
    let seconds = frames as f64 * f64::from(fps.den) / f64::from(fps.num);
    (seconds * f64::from(sample_rate)).round() as i64
}

/// Selects the audio tracks that feed the analysis signal.
///
/// `indices`, when `Some`, keeps only audio tracks whose `index` appears in
/// the slice. When `None`, every audio track is kept. Video tracks are never
/// selected.
#[must_use]
pub fn select_tracks<'a>(timeline: &'a Timeline, indices: Option<&[u32]>) -> Vec<&'a Track> {
    timeline
        .tracks
        .iter()
        .filter(|track| track.kind == TrackKind::Audio)
        .filter(|track| match indices {
            Some(list) => list.contains(&track.index),
            None => true,
        })
        .collect()
}

/// Renders the mono 16 kHz analysis signal for `tracks`.
///
/// The signal runs from frame 0 to the last frame of the last clip on any
/// selected track. Clips are placed at `start` and mixed by averaging across
/// the selected tracks, so duplicating a track does not change the level.
///
/// # Errors
///
/// - [`CoreError::Mixdown`] when `tracks` is empty, or when a decoder returns
///   audio at a rate other than [`ANALYSIS_SAMPLE_RATE`].
/// - Whatever [`ClipDecoder::decode_clip`] returns for an unreadable clip.
pub fn render_analysis_signal(
    timeline: &Timeline,
    tracks: &[&Track],
    decoder: &mut dyn ClipDecoder,
) -> Result<MonoAudio, CoreError> {
    if tracks.is_empty() {
        return Err(CoreError::Mixdown("no audio tracks selected".to_string()));
    }

    let end_frames = tracks.iter().map(|track| track.end()).max().unwrap_or(0);
    if end_frames <= 0 {
        return Ok(MonoAudio::empty(ANALYSIS_SAMPLE_RATE));
    }

    let total_samples =
        frames_to_samples(end_frames, timeline.fps, ANALYSIS_SAMPLE_RATE).max(0) as usize;
    let mut accumulator = vec![0.0f32; total_samples];

    for track in tracks {
        let mut clips: Vec<&Clip> = track.clips.iter().collect();
        clips.sort_by_key(|clip| clip.start);

        for clip in clips {
            let decoded = decoder.decode_clip(clip)?;

            if decoded.sample_rate != ANALYSIS_SAMPLE_RATE {
                return Err(CoreError::Mixdown(format!(
                    "clip `{}` produced {} Hz audio, expected {} Hz",
                    clip.id, decoded.sample_rate, ANALYSIS_SAMPLE_RATE
                )));
            }

            let destination = frames_to_samples(clip.start, timeline.fps, ANALYSIS_SAMPLE_RATE);
            if destination < 0 {
                continue;
            }
            let destination = destination as usize;
            if destination >= total_samples {
                continue;
            }

            let count = decoded.samples.len().min(total_samples - destination);
            for (offset, sample) in decoded.samples.iter().take(count).enumerate() {
                accumulator[destination + offset] += *sample;
            }
        }
    }

    let scale = 1.0f32 / tracks.len() as f32;
    if (scale - 1.0).abs() > f32::EPSILON {
        for sample in &mut accumulator {
            *sample *= scale;
        }
    }

    Ok(MonoAudio::new(accumulator, ANALYSIS_SAMPLE_RATE))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    struct FakeDecoder {
        canned: HashMap<String, MonoAudio>,
        default_len: usize,
        calls: Vec<String>,
    }

    impl FakeDecoder {
        fn new(default_len: usize) -> Self {
            Self {
                canned: HashMap::new(),
                default_len,
                calls: Vec::new(),
            }
        }

        fn with(mut self, id: &str, audio: MonoAudio) -> Self {
            self.canned.insert(id.to_string(), audio);
            self
        }
    }

    impl ClipDecoder for FakeDecoder {
        fn decode_clip(&mut self, clip: &Clip) -> Result<MonoAudio, CoreError> {
            self.calls.push(clip.id.clone());
            if let Some(audio) = self.canned.get(&clip.id) {
                return Ok(audio.clone());
            }
            Ok(MonoAudio::new(
                vec![0.0; self.default_len],
                ANALYSIS_SAMPLE_RATE,
            ))
        }
    }

    /// Test helper: `source_in = 0`, `start = start`, `source_out = source_out`.
    /// Note the third argument is the exclusive source out-point, not the
    /// clip's end frame on the timeline.
    fn clip(id: &str, start: i64, source_out: i64) -> Clip {
        Clip {
            id: id.to_string(),
            source: format!("C:\\media\\{id}.wav"),
            source_in: 0,
            source_out,
            start,
            speed: 1.0,
            has_audio: true,
            muted: false,
        }
    }

    fn timeline(fps: Fps, tracks: Vec<Track>) -> Timeline {
        Timeline {
            fps,
            width: 1920,
            height: 1080,
            tracks,
        }
    }

    fn audio_track(index: u32, clips: Vec<Clip>) -> Track {
        Track {
            kind: TrackKind::Audio,
            index,
            clips,
        }
    }

    fn at_30fps() -> Fps {
        Fps { num: 30, den: 1 }
    }

    #[test]
    fn frames_to_samples_at_integer_fps() {
        assert_eq!(frames_to_samples(30, at_30fps(), 16_000), 16_000);
        assert_eq!(frames_to_samples(15, at_30fps(), 16_000), 8_000);
        assert_eq!(frames_to_samples(0, at_30fps(), 16_000), 0);
    }

    #[test]
    fn frames_to_samples_at_ntsc_fps() {
        // 30000/1001 fps: 30 000 frames is 1001 seconds.
        let ntsc = Fps {
            num: 30_000,
            den: 1001,
        };
        assert_eq!(frames_to_samples(30_000, ntsc, 16_000), 1001 * 16_000);
    }

    #[test]
    fn frames_to_samples_with_degenerate_fps_is_zero() {
        let broken = Fps { num: 0, den: 0 };
        assert_eq!(frames_to_samples(100, broken, 16_000), 0);
    }

    #[test]
    fn select_tracks_keeps_only_audio() {
        let tl = timeline(
            at_30fps(),
            vec![
                Track {
                    kind: TrackKind::Video,
                    index: 1,
                    clips: vec![clip("v", 0, 30)],
                },
                audio_track(1, vec![clip("a1", 0, 30)]),
                audio_track(2, vec![clip("a2", 0, 30)]),
            ],
        );
        let all = select_tracks(&tl, None);
        assert_eq!(all.len(), 2);
        assert!(all.iter().all(|t| t.kind == TrackKind::Audio));

        let only_second = select_tracks(&tl, Some(&[2]));
        assert_eq!(only_second.len(), 1);
        assert_eq!(only_second[0].index, 2);

        let none = select_tracks(&tl, Some(&[7]));
        assert!(none.is_empty());
    }

    #[test]
    fn render_places_a_single_clip() {
        let tl = timeline(at_30fps(), vec![audio_track(1, vec![clip("a", 0, 30)])]);
        let tracks = select_tracks(&tl, None);
        let mut decoder = FakeDecoder::new(16_000)
            .with("a", MonoAudio::new(vec![1.0; 16_000], ANALYSIS_SAMPLE_RATE));

        let mixed = render_analysis_signal(&tl, &tracks, &mut decoder).unwrap();
        assert_eq!(mixed.sample_rate, ANALYSIS_SAMPLE_RATE);
        assert_eq!(mixed.len(), 16_000);
        assert!(mixed.samples.iter().all(|s| (*s - 1.0).abs() < 1e-6));
    }

    #[test]
    fn render_offsets_a_clip_by_its_start_frame() {
        // One 30-frame clip (1 s) starting at frame 30.
        let tl = timeline(at_30fps(), vec![audio_track(1, vec![clip("a", 30, 30)])]);
        let tracks = select_tracks(&tl, None);
        let mut decoder = FakeDecoder::new(16_000)
            .with("a", MonoAudio::new(vec![1.0; 16_000], ANALYSIS_SAMPLE_RATE));

        let mixed = render_analysis_signal(&tl, &tracks, &mut decoder).unwrap();
        // Track ends at frame 60 = sample 32 000.
        assert_eq!(mixed.len(), 32_000);
        assert!(mixed.samples[..16_000].iter().all(|s| *s == 0.0));
        assert!(mixed.samples[16_000..]
            .iter()
            .all(|s| (*s - 1.0).abs() < 1e-6));
    }

    #[test]
    fn render_averages_two_tracks() {
        let tl = timeline(
            at_30fps(),
            vec![
                audio_track(1, vec![clip("a", 0, 30)]),
                audio_track(2, vec![clip("b", 0, 30)]),
            ],
        );
        let tracks = select_tracks(&tl, None);
        let mut decoder = FakeDecoder::new(16_000)
            .with("a", MonoAudio::new(vec![1.0; 16_000], ANALYSIS_SAMPLE_RATE))
            .with("b", MonoAudio::new(vec![0.0; 16_000], ANALYSIS_SAMPLE_RATE));

        let mixed = render_analysis_signal(&tl, &tracks, &mut decoder).unwrap();
        assert!(mixed.samples.iter().all(|s| (*s - 0.5).abs() < 1e-6));
    }

    #[test]
    fn render_a_single_track_is_not_attenuated() {
        let tl = timeline(at_30fps(), vec![audio_track(1, vec![clip("a", 0, 30)])]);
        let tracks = select_tracks(&tl, None);
        let mut decoder = FakeDecoder::new(16_000)
            .with("a", MonoAudio::new(vec![0.75; 16_000], ANALYSIS_SAMPLE_RATE));

        let mixed = render_analysis_signal(&tl, &tracks, &mut decoder).unwrap();
        assert!(mixed.samples.iter().all(|s| (*s - 0.75).abs() < 1e-6));
    }

    #[test]
    fn render_truncates_a_clip_that_overruns_the_signal() {
        let tl = timeline(at_30fps(), vec![audio_track(1, vec![clip("a", 0, 30)])]);
        let tracks = select_tracks(&tl, None);
        let mut decoder = FakeDecoder::new(16_000)
            .with("a", MonoAudio::new(vec![1.0; 32_000], ANALYSIS_SAMPLE_RATE));

        let mixed = render_analysis_signal(&tl, &tracks, &mut decoder).unwrap();
        assert_eq!(mixed.len(), 16_000);
    }

    #[test]
    fn render_places_multiple_clips_at_their_start_frames() {
        // short: start 0, duration 30 frames, end 30.
        // late:  start 100, duration 30 frames, end 130.
        let tl = timeline(
            at_30fps(),
            vec![audio_track(
                1,
                vec![clip("short", 0, 30), clip("late", 100, 30)],
            )],
        );
        let tracks = select_tracks(&tl, None);
        let mut decoder = FakeDecoder::new(16_000)
            .with("short", MonoAudio::new(vec![1.0; 16_000], ANALYSIS_SAMPLE_RATE));

        let mixed = render_analysis_signal(&tl, &tracks, &mut decoder).unwrap();
        // Signal runs to frame 130 = sample 69 333.
        assert_eq!(
            mixed.len(),
            frames_to_samples(130, at_30fps(), 16_000) as usize
        );
        assert_eq!(decoder.calls, vec!["short", "late"]);
    }

    #[test]
    fn render_empty_track_selection_is_an_error() {
        let tl = timeline(at_30fps(), vec![audio_track(1, vec![clip("a", 0, 30)])]);
        let mut decoder = FakeDecoder::new(16_000);
        let result = render_analysis_signal(&tl, &[], &mut decoder);
        assert!(matches!(result, Err(CoreError::Mixdown(_))));
    }

    #[test]
    fn render_with_no_clips_is_an_empty_signal() {
        let tl = timeline(at_30fps(), vec![audio_track(1, Vec::new())]);
        let tracks = select_tracks(&tl, None);
        let mut decoder = FakeDecoder::new(0);
        let mixed = render_analysis_signal(&tl, &tracks, &mut decoder).unwrap();
        assert!(mixed.is_empty());
        assert_eq!(mixed.sample_rate, ANALYSIS_SAMPLE_RATE);
    }

    #[test]
    fn render_rejects_a_decoder_at_the_wrong_rate() {
        let tl = timeline(at_30fps(), vec![audio_track(1, vec![clip("a", 0, 30)])]);
        let tracks = select_tracks(&tl, None);
        let mut decoder =
            FakeDecoder::new(16_000).with("a", MonoAudio::new(vec![0.0; 16_000], 48_000));

        let result = render_analysis_signal(&tl, &tracks, &mut decoder);
        match result {
            Err(CoreError::Mixdown(message)) => assert!(message.contains("48000")),
            other => panic!("expected Mixdown error, got {other:?}"),
        }
    }

    #[test]
    fn render_propagates_decoder_errors() {
        struct Failing;
        impl ClipDecoder for Failing {
            fn decode_clip(&mut self, clip: &Clip) -> Result<MonoAudio, CoreError> {
                Err(CoreError::AudioDecode {
                    id: clip.id.clone(),
                    reason: "synthetic failure".to_string(),
                })
            }
        }

        let tl = timeline(at_30fps(), vec![audio_track(1, vec![clip("a", 0, 30)])]);
        let tracks = select_tracks(&tl, None);
        let mut decoder = Failing;
        match render_analysis_signal(&tl, &tracks, &mut decoder) {
            Err(CoreError::AudioDecode { id, reason }) => {
                assert_eq!(id, "a");
                assert!(reason.contains("synthetic failure"));
            }
            other => panic!("expected AudioDecode error, got {other:?}"),
        }
    }
}