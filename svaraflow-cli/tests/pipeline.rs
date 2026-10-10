//! End-to-end tests for `svaraflow-cli silence analyze` and
//! `svaraflow-cli silence apply`.
//!
//! These tests exercise the compiled binary as a subprocess. A synthetic
//! WAV is generated at test time and referenced from a generated
//! `timeline.json`. No external media, no network, no fixtures on disk.
//!
//! The signal is 1 s silence + 1 s 1 kHz tone at 0.9 amplitude + 1 s
//! silence, 16 kHz mono. At 30 fps that is 90 timeline frames. The
//! detector identifies the middle second as the sole keep range.

mod common;

use std::path::{Path, PathBuf};
use std::process::Command;

use common::{append_silence, append_sine, escape_for_json, tmp_path, write_wav};

const SAMPLE_RATE: u32 = 16_000;
const FPS_NUM: u32 = 30;
const FPS_DEN: u32 = 1;

fn bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_svaraflow-cli"))
}

/// Builds the test WAV and returns its path.
fn build_wav(name: &str) -> PathBuf {
    let path = tmp_path(name);
    let mut samples = Vec::with_capacity(SAMPLE_RATE as usize * 3);
    append_silence(&mut samples, 1.0, SAMPLE_RATE);
    append_sine(&mut samples, 1.0, 0.9, SAMPLE_RATE);
    append_silence(&mut samples, 1.0, SAMPLE_RATE);
    write_wav(&path, &samples, SAMPLE_RATE).expect("write wav");
    path
}

/// Builds a `timeline.json` referencing `wav`. One audio track, one clip,
/// `source_out` = 90 frames (3 s at 30 fps, exclusive).
fn build_timeline_json(name: &str, wav: &Path) -> PathBuf {
    let path = tmp_path(name);
    let source = escape_for_json(wav);
    let json = format!(
        r#"{{
  "fps": {{ "num": {FPS_NUM}, "den": {FPS_DEN} }},
  "width": 1920,
  "height": 1080,
  "tracks": [
    {{
      "kind": "audio",
      "index": 1,
      "clips": [
        {{
          "id": "a1",
          "source": "{source}",
          "source_in": 0,
          "source_out": 90,
          "start": 0,
          "speed": 1.0,
          "has_audio": true
        }}
      ]
    }}
  ]
}}
"#
    );
    std::fs::write(&path, json).expect("write timeline.json");
    path
}

/// Converts a `&str` slice to `OsString` so `Command::args` can take mixed
/// paths and flags uniformly.
fn args<'a>(items: &'a [&'a str]) -> Vec<std::ffi::OsString> {
    items.iter().map(std::ffi::OsString::from).collect()
}

// ---------------------------------------------------------------------------
// analyze
// ---------------------------------------------------------------------------

#[test]
fn silence_analyze_json_reports_expected_shape() {
    let wav = build_wav("analyze_signal.wav");
    let timeline = build_timeline_json("analyze_timeline.json", &wav);

    let mut argv: Vec<std::ffi::OsString> = args(&["silence", "analyze"]);
    argv.push(timeline.clone().into_os_string());
    argv.push("--json".into());

    let out = Command::new(bin())
        .args(&argv)
        .output()
        .expect("spawn svaraflow-cli");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "analyze exited {}: stderr = {stderr}\nstdout = {stdout}",
        out.status
    );

    // Method is the default.
    assert!(
        stdout.contains("\"method\": \"threshold\""),
        "stdout did not report the method: {stdout}"
    );

    // Exactly two silence regions (one on each side of the tone).
    assert!(
        stdout.contains("\"cut_count\": 2"),
        "expected 2 cuts: {stdout}"
    );
    assert!(stdout.contains("\"keep\": ["), "no keep array: {stdout}");

    // The kept+removed invariant holds.
    let total = extract_int(&stdout, "\"total_frames\": ").expect("total_frames");
    let kept = extract_int(&stdout, "\"kept_frames\": ").expect("kept_frames");
    let removed = extract_int(&stdout, "\"removed_frames\": ").expect("removed_frames");
    assert_eq!(kept + removed, total, "stats invariant violated: {stdout}");
}

#[test]
fn silence_analyze_human_output_reports_a_keep_range() {
    let wav = build_wav("human_signal.wav");
    let timeline = build_timeline_json("human_timeline.json", &wav);

    let mut argv: Vec<std::ffi::OsString> = args(&["silence", "analyze"]);
    argv.push(timeline.into_os_string());

    let out = Command::new(bin())
        .args(&argv)
        .output()
        .expect("spawn svaraflow-cli");
    assert!(out.status.success(), "analyze exited {}", out.status);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("Keep ranges (1):"), "stdout: {stdout}");
    assert!(
        stdout.contains("method:      threshold"),
        "stdout: {stdout}"
    );
}

// ---------------------------------------------------------------------------
// apply
// ---------------------------------------------------------------------------

#[test]
fn silence_apply_remove_writes_xml_and_edl() {
    let wav = build_wav("apply_signal.wav");
    let timeline = build_timeline_json("apply_timeline.json", &wav);
    let xml_path = tmp_path("apply_out.xml");
    let edl_path = tmp_path("apply_out.edl");

    let mut argv: Vec<std::ffi::OsString> = args(&[
        "silence",
        "apply",
        "--mode",
        "remove",
        "--sequence-name",
        "Pipeline",
    ]);
    argv.push(timeline.into_os_string());
    argv.push("--xml".into());
    argv.push(xml_path.clone().into_os_string());
    argv.push("--edl".into());
    argv.push(edl_path.clone().into_os_string());

    let out = Command::new(bin())
        .args(&argv)
        .output()
        .expect("spawn svaraflow-cli");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "apply exited {}: stderr = {stderr}\nstdout = {stdout}",
        out.status
    );

    // XML exists and has the expected shape.
    let xml = std::fs::read_to_string(&xml_path).expect("read xml");
    assert!(
        xml.starts_with("<?xml"),
        "xml does not start with declaration"
    );
    assert!(xml.contains("<xmeml version=\"4\">"));
    assert!(xml.contains("<name>Pipeline - SvaraFlow cut</name>"));
    assert!(xml.contains("<ntsc>FALSE</ntsc>"));
    assert!(xml.contains("<timebase>30</timebase>"));
    assert!(
        xml.contains("<pathurl>file://localhost/"),
        "pathurl missing: {xml}"
    );
    // Remove mode produces exactly one clip on the audio track: the keep
    // range, placed back to back.
    let clipitems = xml.matches("<clipitem id=").count();
    assert_eq!(clipitems, 1, "expected 1 clipitem: {xml}");

    // EDL exists and has the header and one event.
    let edl = std::fs::read_to_string(&edl_path).expect("read edl");
    assert!(edl.starts_with("TITLE: Pipeline\n"), "edl: {edl}");
    assert!(edl.contains("FCM: NON-DROP FRAME"));
    assert!(edl.contains("001  AX       V     C"));
    assert_eq!(edl.matches("001  AX").count(), 1);
}

#[test]
fn silence_apply_keep_preserves_timeline_length() {
    let wav = build_wav("keep_signal.wav");
    let timeline = build_timeline_json("keep_timeline.json", &wav);
    let xml_path = tmp_path("keep_out.xml");

    let mut argv: Vec<std::ffi::OsString> = args(&["silence", "apply", "--mode", "keep"]);
    argv.push(timeline.into_os_string());
    argv.push("--xml".into());
    argv.push(xml_path.clone().into_os_string());

    let out = Command::new(bin())
        .args(&argv)
        .output()
        .expect("spawn svaraflow-cli");
    assert!(out.status.success(), "apply exited {}", out.status);

    let xml = std::fs::read_to_string(&xml_path).expect("read xml");
    // Keep mode splits at every boundary: three pieces (two silence ranges,
    // one keep range).
    let clipitems = xml.matches("<clipitem id=").count();
    assert_eq!(clipitems, 3, "keep mode should produce 3 clipitems: {xml}");

    // The sequence duration is the input's 90 frames.
    assert!(xml.contains("<duration>90</duration>"), "xml: {xml}");
}

#[test]
fn silence_apply_mute_marks_silence_clips_disabled() {
    let wav = build_wav("mute_signal.wav");
    let timeline = build_timeline_json("mute_timeline.json", &wav);
    let xml_path = tmp_path("mute_out.xml");

    let mut argv: Vec<std::ffi::OsString> = args(&["silence", "apply", "--mode", "mute"]);
    argv.push(timeline.into_os_string());
    argv.push("--xml".into());
    argv.push(xml_path.clone().into_os_string());

    let out = Command::new(bin())
        .args(&argv)
        .output()
        .expect("spawn svaraflow-cli");
    assert!(out.status.success(), "apply exited {}", out.status);

    let xml = std::fs::read_to_string(&xml_path).expect("read xml");
    // Three clipitems, at least one with `enabled FALSE`.
    let clipitems = xml.matches("<clipitem id=").count();
    assert_eq!(clipitems, 3, "mute mode should produce 3 clipitems: {xml}");
    assert!(
        xml.contains("<enabled>FALSE</enabled>"),
        "mute mode did not disable any clip: {xml}"
    );
    assert!(
        xml.contains("<enabled>TRUE</enabled>"),
        "mute mode disabled every clip: {xml}"
    );
}

#[test]
fn silence_apply_without_xml_or_edl_fails() {
    let wav = build_wav("noout_signal.wav");
    let timeline = build_timeline_json("noout_timeline.json", &wav);

    let mut argv: Vec<std::ffi::OsString> = args(&["silence", "apply"]);
    argv.push(timeline.into_os_string());

    let out = Command::new(bin())
        .args(&argv)
        .output()
        .expect("spawn svaraflow-cli");
    assert!(
        !out.status.success(),
        "apply without output paths should fail"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("--xml") && stderr.contains("--edl"),
        "stderr did not name the missing flags: {stderr}"
    );
}

// ---------------------------------------------------------------------------
// error paths
// ---------------------------------------------------------------------------

#[test]
fn silence_analyze_missing_timeline_reports_the_path() {
    let bogus = tmp_path("this_does_not_exist.json");
    let _ = std::fs::remove_file(&bogus);

    let mut argv: Vec<std::ffi::OsString> = args(&["silence", "analyze"]);
    argv.push(bogus.into_os_string());

    let out = Command::new(bin())
        .args(&argv)
        .output()
        .expect("spawn svaraflow-cli");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.to_lowercase().contains("reading"),
        "stderr did not mention the read failure: {stderr}"
    );
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

/// Extracts the integer following `prefix` in `text`. Returns `None` when
/// the prefix is missing or the value is not an integer.
fn extract_int(text: &str, prefix: &str) -> Option<i64> {
    let start = text.find(prefix)? + prefix.len();
    let rest = &text[start..];
    let end = rest
        .find(|c: char| !c.is_ascii_digit() && c != '-')
        .unwrap_or(rest.len());
    rest[..end].parse().ok()
}