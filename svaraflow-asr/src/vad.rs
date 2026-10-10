//! Silero VAD v5 through ONNX Runtime.
//!
//! Wraps `ort`'s `Session` in the engine-agnostic [`svaraflow_core::vad::Vad`]
//! trait so the detector in `svaraflow-core` does not need to know about
//! ONNX. The model is the standard Silero VAD v5 export: 512-sample windows
//! at 16 kHz, `input`/`sr`/`state` in, `output`/`stateN` out.
//!
//! # Tensor layout
//!
//! | Name | Shape | Type | Meaning |
//! | --- | --- | --- | --- |
//! | `input` | `[1, 512]` | f32 | One window of audio |
//! | `sr` | `[1]` | i64 | Sample rate, always 16000 |
//! | `state` | `[2, 1, 128]` | f32 | LSTM hidden state |
//!
//! | Name | Shape | Type | Meaning |
//! | --- | --- | --- | --- |
//! | `output` | `[1, 1]` | f32 | Speech probability for this window |
//! | `stateN` | `[2, 1, 128]` | f32 | Updated LSTM state |
//!
//! The state is fed back between calls, so the windows of one analysis
//! signal must be processed in order. [`SileroVad::speech_probabilities`]
//! resets the state to zeros at the start of each call.
//!
//! # Building
//!
//! The `vad` feature (default) enables this module. `cargo build
//! -p svaraflow-asr --no-default-features` compiles the crate without
//! ONNX Runtime; [`SileroVad::from_file`] is then replaced by a stub that
//! always returns [`AsrError::VadFeatureDisabled`].

#![cfg(feature = "vad")]

use std::path::Path;

use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;
use svaraflow_core::vad::{Vad, VadProbabilities, VAD_SAMPLE_RATE, VAD_WINDOW_SAMPLES};

use crate::error::AsrError;

/// LSTM state shape, as two `[1, 128]` blocks.
const STATE_SHAPE: [usize; 3] = [2, 1, 128];

/// Silero VAD v5 running through ONNX Runtime.
pub struct SileroVad {
    session: Session,
    /// Input name for the audio window. Cached so we do not re-scan
    /// `session.inputs` on every window.
    input_name: String,
    /// Input name for the sample rate.
    sr_name: String,
    /// Input name for the LSTM state.
    state_name: String,
    /// Output name for the probability.
    output_name: String,
    /// Output name for the updated LSTM state.
    state_out_name: String,
}

impl SileroVad {
    /// Loads the Silero VAD ONNX model from `path`.
    ///
    /// The model must be the standard v5 export: inputs `input` (`[1,512]`
    /// f32), `sr` (`[1]` i64), `state` (`[2,1,128]` f32); outputs `output`
    /// (`[1,1]` f32) and `stateN` (`[2,1,128]` f32). Input and output names
    /// are discovered from the session, not hard-coded, so the older v4
    /// export (which uses `h`/`c` instead of `state`) will also load — the
    /// state name is optional.
    ///
    /// # Errors
    ///
    /// - [`AsrError::VadModelIo`] when `path` cannot be read.
    /// - [`AsrError::VadSession`] when ONNX Runtime rejects the model.
    /// - [`AsrError::VadOutputShape`] when a required input or output is
    ///   missing from the graph.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, AsrError> {
        let path_ref = path.as_ref();
        if !path_ref.exists() {
            return Err(AsrError::VadModelIo {
                path: path_ref.display().to_string(),
                reason: "file not found".to_string(),
            });
        }

        let session = Session::builder()
            .map_err(|e| AsrError::VadSession(e.to_string()))?
            .with_optimization_level(GraphOptimizationLevel::Level1)
            .map_err(|e| AsrError::VadSession(e.to_string()))?
            .with_intra_threads(1)
            .map_err(|e| AsrError::VadSession(e.to_string()))?
            .commit_from_file(path_ref)
            .map_err(|e| AsrError::VadSession(format!("{}: {e}", path_ref.display())))?;

        let names: Vec<String> = session
            .inputs
            .iter()
            .map(|i| i.name.to_string())
            .collect();
        let input_name = pick_name(&names, &["input", "input_1", "audio"])
            .ok_or_else(|| AsrError::VadOutputShape {
                name: "input".to_string(),
                reason: format!("no input named `input`; session has {names:?}"),
            })?;
        let sr_name = pick_name(&names, &["sr", "sample_rate"]).unwrap_or_default();
        let state_name = pick_name(&names, &["state", "h"]).unwrap_or_default();

        let out_names: Vec<String> = session
            .outputs
            .iter()
            .map(|o| o.name.to_string())
            .collect();
        let output_name =
            pick_name(&out_names, &["output", "prob"]).ok_or_else(|| {
                AsrError::VadOutputShape {
                    name: "output".to_string(),
                    reason: format!("no output named `output`; session has {out_names:?}"),
                }
            })?;
        let state_out_name =
            pick_name(&out_names, &["stateN", "hn"]).unwrap_or_default();

        Ok(Self {
            session,
            input_name,
            sr_name,
            state_name,
            output_name,
            state_out_name,
        })
    }

    /// Runs inference over `samples`, one 512-sample window at a time.
    ///
    /// The final partial window is zero-padded to 512 samples. Windows are
    /// processed in order with the LSTM state fed back between them, so the
    /// probabilities reflect the model's context.
    fn run_windows(&mut self, samples: &[f32]) -> Result<Vec<f32>, AsrError> {
        let window = VAD_WINDOW_SAMPLES as usize;
        let count = samples.len().div_ceil(window);
        let mut probs = Vec::with_capacity(count);

        let mut state_data = vec![0.0f32; STATE_SHAPE.iter().product()];
        let mut window_buf = vec![0.0f32; window];

        for i in 0..count {
            let start = i * window;
            let end = (start + window).min(samples.len());
            window_buf[..end - start].copy_from_slice(&samples[start..end]);
            for s in &mut window_buf[end - start..] {
                *s = 0.0;
            }

            let input_tensor = Tensor::from_array(([1usize, window], window_buf.clone()))
                .map_err(|e| AsrError::VadInference(e.to_string()))?;
            let sr_data = [i64::from(VAD_SAMPLE_RATE)];
            let sr_tensor = Tensor::from_array(([1usize], sr_data.to_vec()))
                .map_err(|e| AsrError::VadInference(e.to_string()))?;
            let state_tensor =
                Tensor::from_array((STATE_SHAPE, state_data.clone()))
                    .map_err(|e| AsrError::VadInference(e.to_string()))?;

            let mut inputs = ort::inputs![];
            inputs.insert(self.input_name.clone().into(), input_tensor.into());
            if !self.sr_name.is_empty() {
                inputs.insert(self.sr_name.clone().into(), sr_tensor.into());
            }
            if !self.state_name.is_empty() {
                inputs.insert(self.state_name.clone().into(), state_tensor.into());
            }

            let outputs = self
                .session
                .run(inputs)
                .map_err(|e| AsrError::VadInference(e.to_string()))?;

            let (_, prob_slice) = outputs[self.output_name.as_str()]
                .try_extract_tensor::<f32>()
                .map_err(|e| AsrError::VadInference(e.to_string()))?;
            probs.push(prob_slice.first().copied().unwrap_or(0.0));

            if !self.state_out_name.is_empty() {
                let (_, next_state) = outputs[self.state_out_name.as_str()]
                    .try_extract_tensor::<f32>()
                    .map_err(|e| AsrError::VadInference(e.to_string()))?;
                if next_state.len() == state_data.len() {
                    state_data.copy_from_slice(next_state);
                } else {
                    return Err(AsrError::VadOutputShape {
                        name: self.state_out_name.clone(),
                        reason: format!(
                            "expected {} state values, got {}",
                            state_data.len(),
                            next_state.len()
                        ),
                    });
                }
            }
        }

        Ok(probs)
    }
}

impl Vad for SileroVad {
    fn name(&self) -> &'static str {
        "silero-vad-v5"
    }

    fn window_samples(&self) -> u32 {
        VAD_WINDOW_SAMPLES
    }

    fn sample_rate(&self) -> u32 {
        VAD_SAMPLE_RATE
    }

    fn speech_probabilities(
        &self,
        samples: &[f32],
    ) -> Result<VadProbabilities, svaraflow_core::error::CoreError> {
        // The `Vad` trait takes `&self` so it can be shared behind `&dyn`.
        // `ort::Session::run` takes `&self` too, so no interior mutability
        // is needed here; the per-call state lives in `run_windows`.
        let mut this = UnsafeRef::new(self);
        this.run_windows(samples).map_err(|e| {
            svaraflow_core::error::CoreError::VadFailed(e.to_string())
        })
    }
}

/// Minimal shim so `run_windows` (which needs `&mut self` for clarity of
/// ownership at the call site) can be called from `&self`.
///
/// `run_windows` only mutates local state; it does not write through the
/// session. The shim is safe by construction: no aliasing `&mut self`
/// escapes this function.
struct UnsafeRef<'a> {
    inner: &'a SileroVad,
}

impl<'a> UnsafeRef<'a> {
    fn new(inner: &'a SileroVad) -> Self {
        Self { inner }
    }

    fn run_windows(&mut self, samples: &[f32]) -> Result<Vec<f32>, AsrError> {
        // Call through `&self`; the method only allocates locals.
        self.inner.run_windows_immut(samples)
    }
}

impl SileroVad {
    /// The real implementation, taking `&self`.
    fn run_windows_immut(&self, samples: &[f32]) -> Result<Vec<f32>, AsrError> {
        // Same body as `run_windows` above but through `&self`. The session
        // is `Sync`-safe for `run`; per-call state is local.
        let window = VAD_WINDOW_SAMPLES as usize;
        let count = samples.len().div_ceil(window);
        let mut probs = Vec::with_capacity(count);

        let mut state_data = vec![0.0f32; STATE_SHAPE.iter().product()];
        let mut window_buf = vec![0.0f32; window];

        for i in 0..count {
            let start = i * window;
            let end = (start + window).min(samples.len());
            window_buf[..end - start].copy_from_slice(&samples[start..end]);
            for s in &mut window_buf[end - start..] {
                *s = 0.0;
            }

            let input_tensor =
                Tensor::from_array(([1usize, window], window_buf.clone()))
                    .map_err(|e| AsrError::VadInference(e.to_string()))?;
            let sr_data = [i64::from(VAD_SAMPLE_RATE)];
            let sr_tensor = Tensor::from_array(([1usize], sr_data.to_vec()))
                .map_err(|e| AsrError::VadInference(e.to_string()))?;
            let state_tensor =
                Tensor::from_array((STATE_SHAPE, state_data.clone()))
                    .map_err(|e| AsrError::VadInference(e.to_string()))?;

            let mut inputs = ort::inputs![];
            inputs.insert(self.input_name.clone().into(), input_tensor.into());
            if !self.sr_name.is_empty() {
                inputs.insert(self.sr_name.clone().into(), sr_tensor.into());
            }
            if !self.state_name.is_empty() {
                inputs.insert(self.state_name.clone().into(), state_tensor.into());
            }

            let outputs = self
                .session
                .run(inputs)
                .map_err(|e| AsrError::VadInference(e.to_string()))?;

            let (_, prob_slice) = outputs[self.output_name.as_str()]
                .try_extract_tensor::<f32>()
                .map_err(|e| AsrError::VadInference(e.to_string()))?;
            probs.push(prob_slice.first().copied().unwrap_or(0.0));

            if !self.state_out_name.is_empty() {
                let (_, next_state) = outputs[self.state_out_name.as_str()]
                    .try_extract_tensor::<f32>()
                    .map_err(|e| AsrError::VadInference(e.to_string()))?;
                if next_state.len() == state_data.len() {
                    state_data.copy_from_slice(next_state);
                } else {
                    return Err(AsrError::VadOutputShape {
                        name: self.state_out_name.clone(),
                        reason: format!(
                            "expected {} state values, got {}",
                            state_data.len(),
                            next_state.len()
                        ),
                    });
                }
            }
        }

        Ok(probs)
    }
}

/// Returns the first name in `candidates` that appears in `available`.
fn pick_name(available: &[String], candidates: &[&str]) -> Option<String> {
    for c in candidates {
        if let Some(found) = available.iter().find(|n| n.eq_ignore_ascii_case(c)) {
            return Some(found.clone());
        }
    }
    None
}

/// Returns the ASR crate's crate version as declared by Cargo.
#[must_use]
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pick_name_matches_exact() {
        let available = vec!["input".to_string(), "sr".to_string()];
        assert_eq!(pick_name(&available, &["input"]), Some("input".to_string()));
    }

    #[test]
    fn pick_name_matches_case_insensitive() {
        let available = vec!["Input".to_string()];
        assert_eq!(pick_name(&available, &["input"]), Some("Input".to_string()));
    }

    #[test]
    fn pick_name_returns_none_when_absent() {
        let available = vec!["foo".to_string()];
        assert_eq!(pick_name(&available, &["input"]), None);
    }

    #[test]
    fn pick_name_prefers_first_candidate() {
        let available = vec!["h".to_string(), "state".to_string()];
        assert_eq!(
            pick_name(&available, &["state", "h"]),
            Some("state".to_string())
        );
    }

    #[test]
    fn state_shape_has_256_elements() {
        let n: usize = STATE_SHAPE.iter().product();
        assert_eq!(n, 256);
    }

    #[test]
    fn version_is_non_empty() {
        assert!(!version().is_empty());
    }
}