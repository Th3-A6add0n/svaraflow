//! Subtitle and transcript exports.
//!
//! Four writers, all returning a `String`:
//!
//! - [`write_srt`] — SubRip `.srt`.
//! - [`write_vtt`] — WebVTT `.vtt`.
//! - [`write_ass`] — Advanced SubStation Alpha `.ass`, with `\kf`
//!   karaoke tags for word-by-word highlighting.
//! - [`write_json`] — the transcript JSON documented in
//!   `docs/SCHEMAS.md`.
//!
//! Timing granularity is milliseconds in SRT and VTT, and centiseconds in
//! ASS. Every writer rounds to nearest, so a word at `t = 1.2345` becomes
//! `1.235` in SRT and `1.23` in ASS.

use crate::transcribe::{Segment, Transcript};

/// Newline used in every export.
const NL: &str = "\n";

/// Formats `seconds` as `HH:MM:SS,mmm` for SRT.
#[must_use]
pub fn srt_time(seconds: f64) -> String {
    let total_ms = (seconds.max(0.0) * 1000.0).round() as u64;
    let ms = total_ms % 1000;
    let total_s = total_ms / 1000;
    let s = total_s % 60;
    let m = (total_s / 60) % 60;
    let h = total_s / 3600;
    format!("{h:02}:{m:02}:{s:02},{ms:03}")
}

/// Formats `seconds` as `HH:MM:SS.mmm` for VTT.
#[must_use]
pub fn vtt_time(seconds: f64) -> String {
    let total_ms = (seconds.max(0.0) * 1000.0).round() as u64;
    let ms = total_ms % 1000;
    let total_s = total_ms / 1000;
    let s = total_s % 60;
    let m = (total_s / 60) % 60;
    let h = total_s / 3600;
    format!("{h:02}:{m:02}:{s:02}.{ms:03}")
}

/// Formats `seconds` as `H:MM:SS.cc` for ASS.
#[must_use]
pub fn ass_time(seconds: f64) -> String {
    let total_cs = (seconds.max(0.0) * 100.0).round() as u64;
    let cs = total_cs % 100;
    let total_s = total_cs / 100;
    let s = total_s % 60;
    let m = (total_s / 60) % 60;
    let h = total_s / 3600;
    format!("{h}:{m:02}:{s:02}.{cs:02}")
}

/// Writes an SRT file.
#[must_use]
pub fn write_srt(transcript: &Transcript) -> String {
    let mut out = String::new();
    for (i, seg) in transcript.segments.iter().enumerate() {
        if seg.words.is_empty() && seg.text.trim().is_empty() {
            continue;
        }
        let idx = i + 1;
        out.push_str(&idx.to_string());
        out.push_str(NL);
        out.push_str(&srt_time(seg.start_s));
        out.push_str(" --> ");
        out.push_str(&srt_time(seg.end_s));
        out.push_str(NL);
        out.push_str(&cue_text(seg));
        out.push_str(NL);
        out.push_str(NL);
    }
    out
}

/// Writes a WebVTT file.
#[must_use]
pub fn write_vtt(transcript: &Transcript) -> String {
    let mut out = String::from("WEBVTT");
    out.push_str(NL);
    out.push_str(NL);
    for seg in &transcript.segments {
        if seg.words.is_empty() && seg.text.trim().is_empty() {
            continue;
        }
        out.push_str(&vtt_time(seg.start_s));
        out.push_str(" --> ");
        out.push_str(&vtt_time(seg.end_s));
        out.push_str(NL);
        out.push_str(&cue_text(seg));
        out.push_str(NL);
        out.push_str(NL);
    }
    out
}

/// Writes an ASS file with karaoke word highlighting.
///
/// Every word is prefixed with `{\kf<cs>}` where `<cs>` is the word's
/// duration in centiseconds. The last word's duration is extended so the
/// sum of the durations equals the dialogue line's total duration. That
/// keeps a player that advances karaoke on the sum in sync with the line:
/// the sweep ends exactly when the subtitle disappears.
///
/// # Characters
///
/// `{`, `}`, `\`, and newlines are stripped from word text, because those
/// are ASS control characters and would either break the line or be
/// interpreted as override tags.
#[must_use]
pub fn write_ass(transcript: &Transcript, title: &str) -> String {
    let mut out = String::new();

    out.push_str("[Script Info]");
    out.push_str(NL);
    out.push_str(&format!("Title: {title}"));
    out.push_str(NL);
    out.push_str("ScriptType: v4.00+");
    out.push_str(NL);
    out.push_str("WrapStyle: 0");
    out.push_str(NL);
    out.push_str("ScaledBorderAndShadow: yes");
    out.push_str(NL);
    out.push_str("PlayResX: 1920");
    out.push_str(NL);
    out.push_str("PlayResY: 1080");
    out.push_str(NL);
    out.push_str(NL);

    out.push_str("[V4+ Styles]");
    out.push_str(NL);
    out.push_str(
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, \
         OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, \
         ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, \
         Alignment, MarginL, MarginR, MarginV, Encoding",
    );
    out.push_str(NL);
    out.push_str(
        "Style: Default,Arial,48,&H00FFFFFF,&H0000FF00,&H00000000,\
         &H00000000,0,0,0,0,100,100,0,0,1,2,1,2,40,40,40,1",
    );
    out.push_str(NL);
    out.push_str(NL);

    out.push_str("[Events]");
    out.push_str(NL);
    out.push_str(
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, \
         MarginV, Effect, Text",
    );
    out.push_str(NL);

    for seg in &transcript.segments {
        if seg.words.is_empty() {
            if seg.text.trim().is_empty() {
                continue;
            }
            let total_cs = ((seg.end_s - seg.start_s).max(0.0) * 100.0).round() as u64;
            let text = ass_escape(seg.text.trim());
            out.push_str(&format!(
                "Dialogue: 0,{start},{end},Default,,0,0,0,,{{\\kf{total_cs}}}{text}",
                start = ass_time(seg.start_s),
                end = ass_time(seg.end_s),
            ));
            out.push_str(NL);
            continue;
        }

        let text = karaoke_text(seg);
        out.push_str(&format!(
            "Dialogue: 0,{start},{end},Default,,0,0,0,,{text}",
            start = ass_time(seg.start_s),
            end = ass_time(seg.end_s),
        ));
        out.push_str(NL);
    }

    out
}

/// The plain-text form of a segment for SRT and VTT.
fn cue_text(seg: &Segment) -> String {
    if !seg.words.is_empty() {
        seg.words
            .iter()
            .map(|w| w.text.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    } else {
        seg.text.trim().to_string()
    }
}

/// Renders a segment's words as a sequence of `{\kfN}word` pairs.
///
/// The total karaoke duration is the segment's span (`end_s - start_s`),
/// not the sum of the words' individual durations. The last word is
/// extended to absorb the difference, so the karaoke sweep ends when the
/// subtitle line does.
fn karaoke_text(seg: &Segment) -> String {
    let words = &seg.words;
    if words.is_empty() {
        return String::new();
    }

    let total_cs = ((seg.end_s - seg.start_s).max(0.0) * 100.0).round() as i64;

    let mut durations: Vec<i64> = words
        .iter()
        .map(|w| ((w.end_s - w.start_s).max(0.0) * 100.0).round() as i64)
        .collect();
    let sum: i64 = durations.iter().sum();
    if let Some(last) = durations.last_mut() {
        *last = (*last + total_cs - sum).max(1);
    }

    let mut out = String::new();
    for (w, d) in words.iter().zip(durations.iter()) {
        out.push_str(&format!("{{\\kf{d}}}{} ", ass_escape(&w.text)));
    }
    out.trim_end().to_string()
}

/// Strips characters that would confuse an ASS parser.
fn ass_escape(s: &str) -> String {
    s.chars()
        .filter(|c| !matches!(c, '{' | '}' | '\\' | '\n' | '\r'))
        .collect()
}

/// Writes the transcript JSON documented in `docs/SCHEMAS.md`.
///
/// # Errors
///
/// [`crate::error::AsrError::Json`] on serialization failure.
pub fn write_json(transcript: &Transcript) -> Result<String, crate::error::AsrError> {
    use serde::Serialize;

    #[derive(Serialize)]
    struct JsonWord<'a> {
        text: &'a str,
        start_s: f64,
        end_s: f64,
        probability: f32,
        estimated: bool,
    }

    #[derive(Serialize)]
    struct JsonSegment<'a> {
        text: &'a str,
        start_s: f64,
        end_s: f64,
        words: Vec<JsonWord<'a>>,
    }

    #[derive(Serialize)]
    struct JsonTranscript<'a> {
        version: u32,
        model: &'a str,
        tier: &'a str,
        language: &'a str,
        audio_duration_s: f64,
        elapsed_s: f64,
        real_time_factor: f64,
        segments: Vec<JsonSegment<'a>>,
        words: Vec<JsonWord<'a>>,
    }

    let segments: Vec<JsonSegment<'_>> = transcript
        .segments
        .iter()
        .map(|s| JsonSegment {
            text: s.text.as_str(),
            start_s: s.start_s,
            end_s: s.end_s,
            words: s
                .words
                .iter()
                .map(|w| JsonWord {
                    text: w.text.as_str(),
                    start_s: w.start_s,
                    end_s: w.end_s,
                    probability: w.probability,
                    estimated: w.estimated,
                })
                .collect(),
        })
        .collect();

    let words: Vec<JsonWord<'_>> = transcript
        .segments
        .iter()
        .flat_map(|s| s.words.iter())
        .map(|w| JsonWord {
            text: w.text.as_str(),
            start_s: w.start_s,
            end_s: w.end_s,
            probability: w.probability,
            estimated: w.estimated,
        })
        .collect();

    let payload = JsonTranscript {
        version: 1,
        model: &transcript.model,
        tier: match transcript.tier {
            crate::models::Tier::Light => "light",
            crate::models::Tier::Standard => "standard",
        },
        language: &transcript.language,
        audio_duration_s: transcript.audio_duration_s,
        elapsed_s: transcript.elapsed_s,
        real_time_factor: transcript.real_time_factor,
        segments,
        words,
    };

    Ok(serde_json::to_string_pretty(&payload)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::Tier;
    use crate::transcribe::{Segment, Word};

    fn word(text: &str, start: f64, end: f64) -> Word {
        Word {
            text: text.to_string(),
            start_s: start,
            end_s: end,
            probability: 0.9,
            estimated: false,
        }
    }

    fn sample() -> Transcript {
        Transcript {
            segments: vec![
                Segment {
                    text: "Hello world".into(),
                    start_s: 0.0,
                    end_s: 2.5,
                    words: vec![word("Hello", 0.0, 1.0), word("world", 1.0, 2.5)],
                    low_confidence: false,
                },
                Segment {
                    text: "Goodbye".into(),
                    start_s: 3.0,
                    end_s: 4.0,
                    words: vec![word("Goodbye", 3.0, 4.0)],
                    low_confidence: false,
                },
            ],
            language: "hi".into(),
            tier: Tier::Standard,
            model: "Whisper-Hindi2Hinglish-Apex".into(),
            audio_duration_s: 4.0,
            elapsed_s: 2.0,
            real_time_factor: 2.0,
        }
    }

    #[test]
    fn srt_time_formats_known_values() {
        assert_eq!(srt_time(0.0), "00:00:00,000");
        assert_eq!(srt_time(1.0), "00:00:01,000");
        assert_eq!(srt_time(1.5), "00:00:01,500");
        assert_eq!(srt_time(61.5), "00:01:01,500");
        assert_eq!(srt_time(3661.001), "01:01:01,001");
        // 1.2346 * 1000 = 1234.6, round() rounds half away from zero.
        assert_eq!(srt_time(1.2346), "00:00:01,235");
        // 1.2344 * 1000 = 1234.4, rounds down.
        assert_eq!(srt_time(1.2344), "00:00:01,234");
    }

    #[test]
    fn vtt_time_uses_period() {
        assert_eq!(vtt_time(1.5), "00:00:01.500");
        assert_eq!(vtt_time(0.0), "00:00:00.000");
    }

    #[test]
    fn ass_time_uses_centiseconds() {
        assert_eq!(ass_time(0.0), "0:00:00.00");
        assert_eq!(ass_time(1.5), "0:00:01.50");
        assert_eq!(ass_time(61.5), "0:01:01.50");
    }

    #[test]
    fn srt_has_two_cues() {
        let out = write_srt(&sample());
        assert!(out.contains("1\n"));
        assert!(out.contains("00:00:00,000 --> 00:00:02,500"));
        assert!(out.contains("Hello world"));
        assert!(out.contains("2\n"));
        assert!(out.contains("00:00:03,000 --> 00:00:04,000"));
    }

    #[test]
    fn vtt_starts_with_header() {
        let out = write_vtt(&sample());
        assert!(out.starts_with("WEBVTT\n\n"), "{out}");
        assert!(out.contains("00:00:00.000 --> 00:00:02.500"));
    }

    #[test]
    fn ass_has_script_info_and_events() {
        let out = write_ass(&sample(), "Test");
        assert!(out.contains("[Script Info]"), "{out}");
        assert!(out.contains("Title: Test"), "{out}");
        assert!(out.contains("[V4+ Styles]"), "{out}");
        assert!(out.contains("[Events]"), "{out}");
        assert!(
            out.contains("Dialogue: 0,0:00:00.00,0:00:02.50,Default"),
            "{out}"
        );
    }

    #[test]
    fn ass_karaoke_durations_sum_to_line_duration() {
        let out = write_ass(&sample(), "Test");
        // First line spans 0.0 to 2.5 s = 250 cs. "Hello" is 100 cs,
        // "world" would be 150 cs, sum = 250 = line duration.
        assert!(out.contains("{\\kf100}Hello {\\kf150}world"), "{out}");
    }

    #[test]
    fn ass_escapes_control_characters() {
        let mut t = sample();
        t.segments[0].words[0].text = "He{llo}\\ world".into();
        let out = write_ass(&t, "Test");
        assert!(!out.contains("He{llo}"), "{out}");
        assert!(out.contains("Hello world"), "{out}");
    }

    #[test]
    fn ass_karaoke_adjusts_last_word_to_match_total() {
        // A segment whose line span (3.0 s) is longer than the words'
        // span (0.0 to 1.0 s). The last word must be extended so the
        // karaoke sum matches the line: 50 cs for "a", 250 cs for "b".
        let mut t = sample();
        t.segments[0].words = vec![word("a", 0.0, 0.5), word("b", 0.5, 1.0)];
        t.segments[0].start_s = 0.0;
        t.segments[0].end_s = 3.0;
        let out = write_ass(&t, "Test");
        assert!(out.contains("{\\kf50}a {\\kf250}b"), "{out}");
    }

    #[test]
    fn json_has_version_and_both_arrays() {
        let out = write_json(&sample()).unwrap();
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["version"], 1);
        assert_eq!(v["segments"].as_array().unwrap().len(), 2);
        assert_eq!(v["words"].as_array().unwrap().len(), 3);
        assert_eq!(v["tier"], "standard");
    }

    #[test]
    fn json_words_include_estimated_flag() {
        let out = write_json(&sample()).unwrap();
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["words"][0]["estimated"], false);
    }

    #[test]
    fn empty_segments_produce_empty_srt() {
        let t = Transcript {
            segments: vec![],
            language: "hi".into(),
            tier: Tier::Standard,
            model: "m".into(),
            audio_duration_s: 0.0,
            elapsed_s: 0.0,
            real_time_factor: 0.0,
        };
        assert_eq!(write_srt(&t), "");
    }
}
