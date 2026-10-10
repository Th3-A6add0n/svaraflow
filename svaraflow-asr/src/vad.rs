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
//! signal must be processed in order. `speech_probabilities` resets the
//! state to zeros at the start of each call.
//!
//! # ONNX Runtime discovery
//!
//! `svaraflow-asr` links ONNX Runtime **dynamically** (`load-dynamic`).
//! The runtime is loaded on first use of a `SileroVad`, from
//! `ORT_DYLIB_PATH` if set, then from the platform's dynamic loader search
//! path (`PATH` on Windows, `LD_LIBRARY_PATH` on Linux). If neither finds
//! it, session creation fails with [`AsrError::VadSession`] naming the
//! variable.

use std::path::Path;

use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;
use svaraflow_core::error::CoreError;
use svaraflow_core::vad::{Vad, VadProbabilities, VAD_SAMPLE_RATE, VAD_WINDOW_SAMPLES};

use crate::error::AsrError;

/// LSTM state shape, as two `[1, 128]` blocks.
const STATE_SHAPE: [usize; 3] = [2, 1, 128];

/// Silero VAD v5 running through ONNX Runtime.
///
/// `Session::run` takes `&mut self`, so `speech_probabilities` in the `Vad`
/// trait takes `&mut self` too. See ADR-0019.
pub struct SileroVad {
    session: Session,
    input_name: String,
    sr_name: String,
    state_name: String,
    output_name: String,
    state_out_name: String,
}

impl SileroVad {
    /// Loads the Silero VAD ONNX model from `path`.
    ///
    /// The model must be the standard v5 export: inputs `input` (`[1,512]`
    /// f32), `sr` (`[1]` i64), `state` (`[2,1,128]` f32); outputs `output`
    /// (`[1,1]` f32) and `stateN` (`[2,1,128]` f32). Input and output names
    /// are discovered from the session, not hard-coded, so the older v4
    /// export (which uses `h`/`c` instead of `state`) will also load.
    ///
    /// # Errors
    ///
    /// - [`AsrError::VadModelIo`] when `path` cannot be read.
    /// - [`AsrError::VadSession`] when ONNX Runtime rejects the model or
    ///   cannot be found. Set `ORT_DYLIB_PATH` to point at the shared
    ///   library if the loader cannot find it.
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
            .inputs()
            .iter()
            .map(|i| i.name().to_string())
            .collect();
        let input_name = pick_name(&names, &["input", "input_1", "audio"]).ok_or_else(|| {
            AsrError::VadOutputShape {
                name: "input".to_string(),
                reason: format!("no input named `input`; session has {names:?}"),
            }
        })?;
        let sr_name = pick_name(&names, &["sr", "sample_rate"]).unwrap_or_default();
        let state_name = pick_name(&names, &["state", "h"]).unwrap_or_default();

        let out_names: Vec<String> = session
            .outputs()
            .iter()
            .map(|o| o.name().to_string())
            .collect();
        let output_name =
            pick_name(&out_names, &["output", "prob"]).ok_or_else(|| AsrError::VadOutputShape {
                name: "output".to_string(),
                reason: format!("no output named `output`; session has {out_names:?}"),
            })?;
        let state_out_name = pick_name(&out_names, &["stateN", "hn"]).unwrap_or_default();

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
    ///
    /// `&mut self` because `Session::run` takes a mutable receiver.
    fn run(&mut self, samples: &[f32]) -> Result<VadProbabilities, AsrError> {
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
            let sr_tensor = Tensor::from_array(([1usize], vec![i64::from(VAD_SAMPLE_RATE)]))
                .map_err(|e| AsrError::VadInference(e.to_string()))?;
            let state_tensor = Tensor::from_array((STATE_SHAPE, state_data.clone()))
                .map_err(|e| AsrError::VadInference(e.to_string()))?;

            // `ort::inputs!` requires at least one expression. Seed it with
            // the mandatory `input` tensor, then `push` the optional ones.
            let mut inputs = ort::inputs![self.input_name.clone() => input_tensor];
            if !self.sr_name.is_empty() {
                inputs.push((self.sr_name.clone().into(), sr_tensor.into()));
            }
            if !self.state_name.is_empty() {
                inputs.push((self.state_name.clone().into(), state_tensor.into()));
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

        Ok(VadProbabilities {
            values: probs,
            hop_samples: VAD_WINDOW_SAMPLES,
            sample_rate: VAD_SAMPLE_RATE,
        })
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

    fn speech_probabilities(&mut self, samples: &[f32]) -> Result<VadProbabilities, CoreError> {
        self.run(samples)
            .map_err(|e| CoreError::VadFailed(e.to_string()))
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