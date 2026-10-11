//! Performance gates for the analysis pipeline.
//!
//! Two tests:
//!
//! - `analysis_of_short_signal_is_fast` runs by default with a 10 s
//!   signal and asserts the envelope + detector complete within a wide
//!   budget. This catches catastrophic regressions on every
//!   `cargo test`.
//! - `one_hour_synthetic_timeline_analyzed_in_under_30s` is the spec's
//!   gate. It is `#[ignore]`d.
//!
//! Run the slow one with:
//!
//! ```powershell
//! cargo test --release -p svaraflow-core --test perf -- --ignored
//! ```

use std::time::Instant;

use svaraflow_core::audio::{compute_envelope, EnvelopeParams, MonoAudio, ANALYSIS_SAMPLE_RATE};
use svaraflow_core::silence::{detect_silence, SilenceParams};
use svaraflow_core::timeline::Fps;

/// Builds `total_seconds` of alternating 5 s "loud" (0.4 constant) and 3 s
/// silence at 16 kHz.
fn synthetic_signal(total_seconds: u32) -> Vec<f32> {
    let rate = ANALYSIS_SAMPLE_RATE as usize;
    let total = rate * total_seconds as usize;
    let five_seconds = rate * 5;
    let eight_seconds = rate * 8;

    let mut samples = Vec::with_capacity(total);
    while samples.len() < total {
        samples.resize(samples.len() + five_seconds, 0.4);
        samples.resize(samples.len() + (eight_seconds - five_seconds), 0.0);
    }
    samples.truncate(total);
    samples
}

/// Runs the full analysis path once and returns the number of elapsed
/// seconds.
fn time_analysis(total_seconds: u32) -> f64 {
    let samples = synthetic_signal(total_seconds);
    let audio = MonoAudio::new(samples, ANALYSIS_SAMPLE_RATE);

    let start = Instant::now();
    let envelope = compute_envelope(&audio, EnvelopeParams::default());
    let params = SilenceParams::default();
    let ranges = detect_silence(&envelope, &params, Fps { num: 30, den: 1 })
        .expect("detect_silence");
    let elapsed = start.elapsed().as_secs_f64();

    assert!(!ranges.keep.is_empty(), "no keep ranges found");
    assert!(ranges.stats.total_frames > 0);
    elapsed
}

#[test]
fn analysis_of_short_signal_is_fast() {
    let elapsed = time_analysis(10);
    assert!(
        elapsed < 2.0,
        "10 s analysis took {elapsed:.2} s; expected < 2.0 s"
    );
}

#[test]
#[ignore = "slow (one hour of audio); run with `cargo test --release -p svaraflow-core --test perf -- --ignored`"]
fn one_hour_synthetic_timeline_analyzed_in_under_30s() {
    let elapsed = time_analysis(3600);
    assert!(
        elapsed < 30.0,
        "1-hour analysis took {elapsed:.2} s; spec ceiling is 30.0 s"
    );
}