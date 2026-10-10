//! Timeline model for SvaraFlow.
//!
//! Mirrors `docs/SCHEMAS.md`. A [`Timeline`] is a set of [`Track`]s, each
//! holding [`Clip`]s placed at integer frame positions. All frame counts are
//! at the timeline's [`Fps`]. `source_out` and the implicit end of a clip are
//! exclusive, matching FCP7 `xmeml` conventions.

use serde::{Deserialize, Serialize};

use crate::error::CoreError;

/// Frame rate expressed as an exact rational, e.g. `30000 / 1001` for 29.97.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Fps {
    /// Numerator (e.g. `30000`).
    pub num: u32,
    /// Denominator (e.g. `1001`).
    pub den: u32,
}

impl Fps {
    /// Constructs an [`Fps`], rejecting a zero numerator or denominator.
    pub fn new(num: u32, den: u32) -> Result<Self, CoreError> {
        if num == 0 || den == 0 {
            return Err(CoreError::InvalidFps);
        }
        Ok(Self { num, den })
    }

    /// Returns the frame rate as an `f64`.
    #[must_use]
    pub fn as_f64(self) -> f64 {
        f64::from(self.num) / f64::from(self.den)
    }

    /// Converts a duration in seconds to a frame count, rounding to nearest.
    #[must_use]
    pub fn seconds_to_frames(self, seconds: f64) -> i64 {
        (seconds * self.as_f64()).round() as i64
    }

    /// Converts a frame count to seconds.
    #[must_use]
    pub fn frames_to_seconds(self, frames: i64) -> f64 {
        frames as f64 / self.as_f64()
    }

    /// True for the standard NTSC fractional rates (23.976, 29.97, 59.94).
    #[must_use]
    pub fn is_ntsc(self) -> bool {
        self.den == 1001 && matches!(self.num, 24000 | 30000 | 60000)
    }

    /// The nearest integer frame rate (e.g. `30` for `30000/1001`).
    #[must_use]
    pub fn integer_rate(self) -> u32 {
        ((f64::from(self.num) / f64::from(self.den)).round()) as u32
    }

    /// True when the rational denominator is `1` (integer rates only).
    #[must_use]
    pub fn is_integer(self) -> bool {
        self.den == 1
    }
}

/// Whether a track carries picture or sound.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TrackKind {
    /// Video track.
    Video,
    /// Audio track.
    Audio,
}

/// A single media clip placed on a track.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Clip {
    /// Stable identifier, unique within the timeline.
    pub id: String,
    /// Absolute path to the source media file.
    pub source: String,
    /// Inclusive first source frame at the timeline fps.
    pub source_in: i64,
    /// Exclusive last source frame at the timeline fps.
    pub source_out: i64,
    /// Timeline frame at which the clip begins.
    pub start: i64,
    /// Playback rate. `1.0` is real time.
    #[serde(default = "default_speed")]
    pub speed: f64,
    /// True when the source carries an audio stream.
    #[serde(default)]
    pub has_audio: bool,
    /// True when this clip's audio is disabled.
    ///
    /// Set by [`crate::cut::apply_cuts`] in [`crate::cut::CutMode::Mute`] on
    /// silence-range clips in audio tracks. Never affects video playback.
    #[serde(default)]
    pub muted: bool,
}

fn default_speed() -> f64 {
    1.0
}

impl Clip {
    /// Number of source frames the clip consumes (`source_out - source_in`).
    #[must_use]
    pub fn source_duration(&self) -> i64 {
        self.source_out - self.source_in
    }

    /// Duration of the clip on the timeline, in frames, after `speed`.
    #[must_use]
    pub fn timeline_duration(&self) -> i64 {
        (self.source_duration() as f64 / self.speed).round() as i64
    }

    /// Exclusive timeline frame at which the clip ends.
    #[must_use]
    pub fn end(&self) -> i64 {
        self.start + self.timeline_duration()
    }

    /// Validates a single clip in isolation.
    pub fn validate(&self) -> Result<(), CoreError> {
        if self.source_out <= self.source_in {
            return Err(CoreError::InvalidClipRange {
                id: self.id.clone(),
                source_in: self.source_in,
                source_out: self.source_out,
            });
        }
        if !(self.speed.is_finite() && self.speed > 0.0) {
            return Err(CoreError::InvalidClipSpeed {
                id: self.id.clone(),
                speed: self.speed,
            });
        }
        Ok(())
    }
}

/// A video or audio track.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Track {
    /// Video or audio.
    pub kind: TrackKind,
    /// 1-based index within the track kind.
    ///
    /// Video track 1 and audio track 1 are both legal: uniqueness is
    /// per-kind. This matches the standard NLE convention (V1, V2, A1, A2,
    /// ...).
    pub index: u32,
    /// Clips on this track. Order in JSON is preserved but does not need to
    /// be sorted by `start`.
    pub clips: Vec<Clip>,
}

impl Track {
    /// Exclusive timeline frame of the last clip on this track, or `0`.
    #[must_use]
    pub fn end(&self) -> i64 {
        self.clips.iter().map(Clip::end).max().unwrap_or(0)
    }
}

/// A complete timeline: frame rate, raster size, and tracks.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Timeline {
    /// Frame rate.
    pub fps: Fps,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Tracks. Video and audio tracks share one array; disambiguate by
    /// `kind`.
    pub tracks: Vec<Track>,
}

impl Timeline {
    /// Parses a `timeline.json` string.
    pub fn from_json_str(s: &str) -> Result<Self, CoreError> {
        let tl: Self = serde_json::from_str(s)?;
        tl.validate()?;
        Ok(tl)
    }

    /// Serializes this timeline to compact JSON.
    pub fn to_json_string(&self) -> Result<String, CoreError> {
        Ok(serde_json::to_string(self)?)
    }

    /// Serializes this timeline to indented JSON (for `timeline.json` files).
    pub fn to_json_string_pretty(&self) -> Result<String, CoreError> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    /// Exclusive timeline frame of the last clip across all tracks.
    #[must_use]
    pub fn duration_frames(&self) -> i64 {
        self.tracks.iter().map(Track::end).max().unwrap_or(0)
    }

    /// Validates every invariant of the timeline.
    ///
    /// Checks: at least one track, non-zero dimensions, track indices start
    /// at 1 and are unique **within their kind**, clip ids are unique across
    /// the whole timeline, and every clip passes [`Clip::validate`].
    pub fn validate(&self) -> Result<(), CoreError> {
        if self.width == 0 || self.height == 0 {
            return Err(CoreError::InvalidTimeline(
                "width and height must be non-zero".into(),
            ));
        }
        if self.tracks.is_empty() {
            return Err(CoreError::InvalidTimeline("no tracks".into()));
        }

        // (kind, index) pairs must be unique: video 1 and audio 1 can
        // coexist, but two video tracks cannot both be index 1.
        let mut seen_kind_indexes = std::collections::HashSet::new();
        let mut seen_clip_ids = std::collections::HashSet::new();

        for track in &self.tracks {
            if track.index == 0 {
                return Err(CoreError::InvalidTrackIndex(0));
            }
            if !seen_kind_indexes.insert((track.kind, track.index)) {
                return Err(CoreError::DuplicateTrackIndex(track.index));
            }
            for clip in &track.clips {
                clip.validate()?;
                if !seen_clip_ids.insert(clip.id.clone()) {
                    return Err(CoreError::DuplicateClipId(clip.id.clone()));
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NTSC: Fps = Fps {
        num: 30000,
        den: 1001,
    };

    fn sample_timeline() -> Timeline {
        Timeline {
            fps: NTSC,
            width: 1920,
            height: 1080,
            tracks: vec![
                Track {
                    kind: TrackKind::Video,
                    index: 1,
                    clips: vec![Clip {
                        id: "v1".into(),
                        source: "C:\\media\\a.mov".into(),
                        source_in: 0,
                        source_out: 300,
                        start: 0,
                        speed: 1.0,
                        has_audio: true,
                        muted: false,
                    }],
                },
                Track {
                    kind: TrackKind::Audio,
                    index: 1,
                    clips: vec![Clip {
                        id: "a1".into(),
                        source: "C:\\media\\a.mov".into(),
                        source_in: 0,
                        source_out: 300,
                        start: 0,
                        speed: 1.0,
                        has_audio: true,
                        muted: false,
                    }],
                },
            ],
        }
    }

    #[test]
    fn fps_rejects_zero() {
        assert!(matches!(Fps::new(0, 1001), Err(CoreError::InvalidFps)));
        assert!(matches!(Fps::new(30000, 0), Err(CoreError::InvalidFps)));
    }

    #[test]
    fn fps_ntsc_classification() {
        assert!(NTSC.is_ntsc());
        assert!(!Fps::new(25, 1).unwrap().is_ntsc());
        assert_eq!(NTSC.integer_rate(), 30);
        assert!(!NTSC.is_integer());
        assert!(Fps::new(25, 1).unwrap().is_integer());
    }

    #[test]
    fn fps_seconds_and_frames_round_trip() {
        let frames = NTSC.seconds_to_frames(1.0);
        assert_eq!(frames, 30);
        let back = NTSC.frames_to_seconds(frames);
        assert!((back - 1.001).abs() < 1e-9);
    }

    #[test]
    fn clip_duration_and_end() {
        let c = Clip {
            id: "x".into(),
            source: "s".into(),
            source_in: 10,
            source_out: 110,
            start: 50,
            speed: 1.0,
            has_audio: false,
            muted: false,
        };
        assert_eq!(c.source_duration(), 100);
        assert_eq!(c.timeline_duration(), 100);
        assert_eq!(c.end(), 150);
    }

    #[test]
    fn clip_speed_halves_timeline_duration() {
        let c = Clip {
            id: "x".into(),
            source: "s".into(),
            source_in: 0,
            source_out: 100,
            start: 0,
            speed: 2.0,
            has_audio: false,
            muted: false,
        };
        assert_eq!(c.source_duration(), 100);
        assert_eq!(c.timeline_duration(), 50);
        assert_eq!(c.end(), 50);
    }

    #[test]
    fn clip_validate_rejects_bad_range() {
        let mut c = sample_timeline().tracks[0].clips[0].clone();
        c.source_out = c.source_in;
        assert!(matches!(
            c.validate(),
            Err(CoreError::InvalidClipRange { .. })
        ));
    }

    #[test]
    fn clip_validate_rejects_bad_speed() {
        let mut c = sample_timeline().tracks[0].clips[0].clone();
        c.speed = 0.0;
        assert!(matches!(
            c.validate(),
            Err(CoreError::InvalidClipSpeed { .. })
        ));
        c.speed = -1.0;
        assert!(matches!(
            c.validate(),
            Err(CoreError::InvalidClipSpeed { .. })
        ));
        c.speed = f64::NAN;
        assert!(matches!(
            c.validate(),
            Err(CoreError::InvalidClipSpeed { .. })
        ));
    }

    #[test]
    fn timeline_validate_accepts_sample() {
        // sample_timeline has video 1 and audio 1, which is legal under
        // per-kind index uniqueness.
        assert!(sample_timeline().validate().is_ok());
    }

    #[test]
    fn timeline_validate_rejects_duplicate_clip_ids() {
        let mut tl = sample_timeline();
        let dup = tl.tracks[0].clips[0].clone();
        tl.tracks[1].clips.push(dup);
        assert!(matches!(tl.validate(), Err(CoreError::DuplicateClipId(_))));
    }

    #[test]
    fn timeline_validate_rejects_duplicate_track_index_within_kind() {
        // Two video tracks both at index 1 is a real error.
        let mut tl = sample_timeline();
        tl.tracks[1].kind = TrackKind::Video;
        tl.tracks[0].index = 1;
        tl.tracks[1].index = 1;
        assert!(matches!(
            tl.validate(),
            Err(CoreError::DuplicateTrackIndex(1))
        ));
    }

    #[test]
    fn timeline_validate_allows_same_index_across_kinds() {
        // Video 1 and audio 1 coexist.
        let tl = sample_timeline();
        assert_eq!(tl.tracks[0].index, 1);
        assert_eq!(tl.tracks[1].index, 1);
        assert_eq!(tl.tracks[0].kind, TrackKind::Video);
        assert_eq!(tl.tracks[1].kind, TrackKind::Audio);
        assert!(tl.validate().is_ok());
    }

    #[test]
    fn timeline_validate_rejects_zero_index() {
        let mut tl = sample_timeline();
        tl.tracks[0].index = 0;
        assert!(matches!(tl.validate(), Err(CoreError::InvalidTrackIndex(0))));
    }

    #[test]
    fn timeline_validate_rejects_empty_tracks() {
        let mut tl = sample_timeline();
        tl.tracks.clear();
        assert!(matches!(tl.validate(), Err(CoreError::InvalidTimeline(_))));
    }

    #[test]
    fn timeline_validate_rejects_zero_dimensions() {
        let mut tl = sample_timeline();
        tl.width = 0;
        assert!(matches!(tl.validate(), Err(CoreError::InvalidTimeline(_))));
    }

    #[test]
    fn timeline_json_round_trip() {
        let tl = sample_timeline();
        let s = tl.to_json_string_pretty().unwrap();
        let back = Timeline::from_json_str(&s).unwrap();
        assert_eq!(tl, back);
    }

    #[test]
    fn timeline_json_defaults_speed_has_audio_muted() {
        let json = r#"{
          "fps": { "num": 25, "den": 1 },
          "width": 1920,
          "height": 1080,
          "tracks": [
            {
              "kind": "video",
              "index": 1,
              "clips": [
                { "id": "c", "source": "s", "source_in": 0, "source_out": 10, "start": 0 }
              ]
            }
          ]
        }"#;
        let tl = Timeline::from_json_str(json).unwrap();
        let c = &tl.tracks[0].clips[0];
        assert_eq!(c.speed, 1.0);
        assert!(!c.has_audio);
        assert!(!c.muted);
    }

    #[test]
    fn timeline_from_json_str_validates() {
        let json = r#"{
          "fps": { "num": 25, "den": 1 },
          "width": 1920,
          "height": 1080,
          "tracks": [
            {
              "kind": "video",
              "index": 1,
              "clips": [
                { "id": "c", "source": "s", "source_in": 5, "source_out": 5, "start": 0 }
              ]
            }
          ]
        }"#;
        assert!(matches!(
            Timeline::from_json_str(json),
            Err(CoreError::InvalidClipRange { .. })
        ));
    }

    #[test]
    fn timeline_duration_is_max_over_tracks() {
        let mut tl = sample_timeline();
        tl.tracks[1].clips[0].source_out = 500;
        assert_eq!(tl.duration_frames(), 500);
    }
}