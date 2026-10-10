//! End-to-end smoke tests for the SvaraFlow CLI.
//!
//! These spawn the compiled binary (Cargo's `CARGO_BIN_EXE_svaraflow-cli`
//! env var) as a subprocess. No external media or network access is
//! required: the tests exercise argument parsing, version/help output, and
//! error paths for missing inputs.

use std::path::PathBuf;
use std::process::Command;

fn bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_svaraflow-cli"))
}

#[test]
fn version_flag_prints_binary_and_version() {
    let output = Command::new(bin())
        .arg("--version")
        .output()
        .expect("spawn svaraflow-cli");
    assert!(output.status.success(), "--version exited nonzero");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("svaraflow-cli"),
        "stdout did not mention the binary name: {stdout}"
    );
}

#[test]
fn help_flag_lists_subcommands() {
    let output = Command::new(bin())
        .arg("--help")
        .output()
        .expect("spawn svaraflow-cli");
    assert!(output.status.success(), "--help exited nonzero");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("timeline"), "stdout: {stdout}");
    assert!(stdout.contains("silence"), "stdout: {stdout}");
}

#[test]
fn unknown_subcommand_fails() {
    let output = Command::new(bin())
        .arg("nonsense")
        .output()
        .expect("spawn svaraflow-cli");
    assert!(!output.status.success());
}

#[test]
fn timeline_from_media_missing_file_fails_clearly() {
    let output = Command::new(bin())
        .args(["timeline", "from-media", "C:\\does\\not\\exist.mov"])
        .args(["--out", "C:\\does\\not\\exist.json"])
        .output()
        .expect("spawn svaraflow-cli");
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("does not exist"),
        "stderr did not explain the missing file: {stderr}"
    );
}

#[test]
fn timeline_from_media_requires_out_flag() {
    let output = Command::new(bin())
        .args(["timeline", "from-media", "a.mov"])
        .output()
        .expect("spawn svaraflow-cli");
    assert!(!output.status.success());
}

#[test]
fn silence_analyze_missing_timeline_fails_with_context() {
    let output = Command::new(bin())
        .args(["silence", "analyze", "C:\\does\\not\\exist.json"])
        .output()
        .expect("spawn svaraflow-cli");
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.to_lowercase().contains("reading"),
        "stderr did not name the read failure: {stderr}"
    );
}

#[test]
fn silence_apply_without_xml_or_edl_fails() {
    let output = Command::new(bin())
        .args(["silence", "apply", "C:\\does\\not\\exist.json"])
        .output()
        .expect("spawn svaraflow-cli");
    assert!(!output.status.success());
}
