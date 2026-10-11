//! Transcript cache.
//!
//! Keyed on three things: a hash of the audio samples, the model name, and
//! a hash of the settings that affect output. A cache hit returns the
//! stored transcript without touching the VAD or whisper.cpp.
//!
//! The cache lives outside the repository, next to the models:
//!
//! - Windows: `%LOCALAPPDATA%\SvaraFlow\cache`
//! - Linux:   `$XDG_CACHE_HOME/svaraflow`, or `~/.cache/svaraflow`
//!
//! Every entry is one JSON file. The schema is documented in
//! `docs/SCHEMAS.md` and pinned by [`CACHE_VERSION`]. A version mismatch
//! invalidates the whole cache silently: the caller simply sees a miss.

use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use svaraflow_core::audio::MonoAudio;

use crate::error::AsrError;
use crate::models::Tier;
use crate::transcribe::{DecodeStrategy, Segment, TranscribeConfig, Transcript, Word};

/// Cache schema version. Bump when the on-disk shape changes.
pub const CACHE_VERSION: u32 = 1;

/// How many hex characters of each hash appear in the filename.
const FILENAME_HASH_PREFIX: usize = 16;

/// A cache key: audio identity plus everything that affects output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheKey {
    /// Lowercase hex SHA-256 of the audio samples and sample rate.
    pub audio_hash: String,
    /// Lowercase hex SHA-256 of the model name and settings.
    pub settings_hash: String,
}

impl CacheKey {
    /// The filename for this key, without directory or extension.
    #[must_use]
    pub fn as_filename(&self) -> String {
        let a = &self.audio_hash[..FILENAME_HASH_PREFIX.min(self.audio_hash.len())];
        let s = &self.settings_hash[..FILENAME_HASH_PREFIX.min(self.settings_hash.len())];
        format!("{a}-{s}")
    }
}

/// Hashes the audio samples and sample rate.
#[must_use]
pub fn hash_audio(audio: &MonoAudio) -> String {
    let mut hasher = Sha256::new();
    hasher.update(audio.sample_rate.to_le_bytes());
    hasher.update((audio.samples.len() as u64).to_le_bytes());
    for chunk in audio.samples.chunks(16_384) {
        for sample in chunk {
            hasher.update(sample.to_le_bytes());
        }
    }
    hex::encode(hasher.finalize())
}

/// Hashes the model name and every setting that changes the output.
#[must_use]
pub fn hash_settings(model: &str, cfg: &TranscribeConfig) -> String {
    let decode = match cfg.decode {
        DecodeStrategy::Greedy => "greedy".to_string(),
        DecodeStrategy::BeamSearch { beam_size } => format!("beam:{beam_size}"),
    };
    let language = cfg
        .language
        .as_whisper_str()
        .unwrap_or("auto")
        .to_string();

    let payload = serde_json::json!({
        "v": CACHE_VERSION,
        "model": model,
        "language": language,
        "decode": decode,
        "use_dtw": cfg.use_dtw,
        "no_speech_threshold": cfg.no_speech_threshold,
        "repetition_guard": cfg.repetition_guard,
        "temperature_fallback": cfg.temperature_fallback,
        "max_chunk_s": cfg.max_chunk_s,
        "overlap_s": cfg.overlap_s,
        "prompt": cfg.effective_prompt(),
    });
    let bytes = serde_json::to_vec(&payload).unwrap_or_default();

    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    hex::encode(hasher.finalize())
}

/// Builds the cache key for one run.
#[must_use]
pub fn key_for(audio: &MonoAudio, model: &str, cfg: &TranscribeConfig) -> CacheKey {
    CacheKey {
        audio_hash: hash_audio(audio),
        settings_hash: hash_settings(model, cfg),
    }
}

/// The cache directory on this machine.
#[must_use]
pub fn cache_dir() -> PathBuf {
    cache_dir_result().unwrap_or_else(|_| PathBuf::from("cache"))
}

/// The cache directory on this machine.
///
/// # Errors
///
/// [`AsrError::ModelFolder`] when the platform's cache variables are
/// missing.
pub fn cache_dir_result() -> Result<PathBuf, AsrError> {
    #[cfg(target_os = "windows")]
    {
        let base = std::env::var_os("LOCALAPPDATA")
            .map(PathBuf::from)
            .ok_or_else(|| AsrError::ModelFolder("%LOCALAPPDATA% is not set".into()))?;
        Ok(base.join("SvaraFlow").join("cache"))
    }

    #[cfg(not(target_os = "windows"))]
    {
        let base = if let Some(x) = std::env::var_os("XDG_CACHE_HOME") {
            PathBuf::from(x)
        } else if let Some(h) = std::env::var_os("HOME") {
            PathBuf::from(h).join(".cache")
        } else {
            return Err(AsrError::ModelFolder(
                "neither XDG_CACHE_HOME nor HOME is set".into(),
            ));
        };
        Ok(base.join("svaraflow"))
    }
}

/// The file that holds the entry for `key`.
#[must_use]
pub fn path_for(key: &CacheKey) -> PathBuf {
    cache_dir().join(format!("{}.json", key.as_filename()))
}

/// The on-disk form of one word.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedWord {
    /// Word text.
    pub text: String,
    /// Start time in seconds.
    pub start_s: f64,
    /// End time in seconds.
    pub end_s: f64,
    /// Probability.
    pub probability: f32,
    /// True when the timing is proportional, not DTW.
    pub estimated: bool,
}

/// The on-disk form of one segment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedSegment {
    /// Segment text.
    pub text: String,
    /// Start time in seconds.
    pub start_s: f64,
    /// End time in seconds.
    pub end_s: f64,
    /// Words.
    pub words: Vec<CachedWord>,
}

/// The on-disk form of a transcript.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedTranscript {
    /// Schema version. Must equal [`CACHE_VERSION`].
    pub version: u32,
    /// SHA-256 of the audio.
    pub audio_hash: String,
    /// SHA-256 of model + settings.
    pub settings_hash: String,
    /// Manifest name of the model.
    pub model: String,
    /// Tier name.
    pub tier: String,
    /// Language code.
    pub language: String,
    /// Audio duration in seconds.
    pub audio_duration_s: f64,
    /// Wall-clock time spent transcribing.
    pub elapsed_s: f64,
    /// `audio_duration_s / elapsed_s`.
    pub real_time_factor: f64,
    /// The segments.
    pub segments: Vec<CachedSegment>,
    /// Unix seconds when the entry was written.
    pub created_at_unix: u64,
}

impl CachedTranscript {
    /// Wraps a [`Transcript`] plus its key.
    #[must_use]
    pub fn from_transcript(t: &Transcript, key: &CacheKey) -> Self {
        Self {
            version: CACHE_VERSION,
            audio_hash: key.audio_hash.clone(),
            settings_hash: key.settings_hash.clone(),
            model: t.model.clone(),
            tier: match t.tier {
                Tier::Light => "light".to_string(),
                Tier::Standard => "standard".to_string(),
            },
            language: t.language.clone(),
            audio_duration_s: t.audio_duration_s,
            elapsed_s: t.elapsed_s,
            real_time_factor: t.real_time_factor,
            segments: t
                .segments
                .iter()
                .map(|s| CachedSegment {
                    text: s.text.clone(),
                    start_s: s.start_s,
                    end_s: s.end_s,
                    words: s
                        .words
                        .iter()
                        .map(|w| CachedWord {
                            text: w.text.clone(),
                            start_s: w.start_s,
                            end_s: w.end_s,
                            probability: w.probability,
                            estimated: w.estimated,
                        })
                        .collect(),
                })
                .collect(),
            created_at_unix: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
        }
    }

    /// Rebuilds the in-memory [`Transcript`].
    #[must_use]
    pub fn to_transcript(&self) -> Transcript {
        let tier = match self.tier.as_str() {
            "light" => Tier::Light,
            _ => Tier::Standard,
        };
        Transcript {
            segments: self
                .segments
                .iter()
                .map(|s| Segment {
                    text: s.text.clone(),
                    start_s: s.start_s,
                    end_s: s.end_s,
                    words: s
                        .words
                        .iter()
                        .map(|w| Word {
                            text: w.text.clone(),
                            start_s: w.start_s,
                            end_s: w.end_s,
                            probability: w.probability,
                            estimated: w.estimated,
                        })
                        .collect(),
                })
                .collect(),
            language: self.language.clone(),
            tier,
            model: self.model.clone(),
            audio_duration_s: self.audio_duration_s,
            elapsed_s: self.elapsed_s,
            real_time_factor: self.real_time_factor,
        }
    }
}

/// Loads a cache entry for `key`, or `None` on any miss.
#[must_use]
pub fn load(key: &CacheKey) -> Option<Transcript> {
    let path = path_for(key);
    let text = fs::read_to_string(&path).ok()?;
    let cached: CachedTranscript = serde_json::from_str(&text).ok()?;
    if cached.version != CACHE_VERSION {
        return None;
    }
    if cached.audio_hash != key.audio_hash || cached.settings_hash != key.settings_hash {
        return None;
    }
    Some(cached.to_transcript())
}

/// Writes a cache entry.
///
/// # Errors
///
/// [`AsrError::Io`] when the directory or file cannot be written.
pub fn save(key: &CacheKey, transcript: &Transcript) -> Result<(), AsrError> {
    let dir = cache_dir_result()?;
    fs::create_dir_all(&dir)?;
    let cached = CachedTranscript::from_transcript(transcript, key);
    let json = serde_json::to_string(&cached)?;
    let path = dir.join(format!("{}.json", key.as_filename()));
    let tmp = path.with_extension("json.part");
    fs::write(&tmp, json)?;
    fs::rename(&tmp, &path)?;
    Ok(())
}

/// Removes every cache entry.
///
/// Returns the number of files removed.
///
/// # Errors
///
/// [`AsrError::Io`] when the directory cannot be read or a file cannot be
/// removed.
pub fn clear() -> Result<usize, AsrError> {
    let dir = match cache_dir_result() {
        Ok(d) => d,
        Err(_) => return Ok(0),
    };
    if !dir.exists() {
        return Ok(0);
    }
    let mut removed = 0usize;
    for entry in fs::read_dir(&dir)? {
        let entry = entry?;
        if entry.path().extension().and_then(|e| e.to_str()) == Some("json") {
            fs::remove_file(entry.path())?;
            removed += 1;
        }
    }
    Ok(removed)
}

/// Number of entries in the cache.
#[must_use]
pub fn entry_count() -> usize {
    let dir = match cache_dir_result() {
        Ok(d) => d,
        Err(_) => return 0,
    };
    let Ok(iter) = fs::read_dir(&dir) else {
        return 0;
    };
    iter.filter_map(Result::ok)
        .filter(|e| e.path().extension().and_then(|x| x.to_str()) == Some("json"))
        .count()
}

/// Approximate cache size on disk, in bytes.
#[must_use]
pub fn size_bytes() -> u64 {
    let dir = match cache_dir_result() {
        Ok(d) => d,
        Err(_) => return 0,
    };
    let Ok(iter) = fs::read_dir(&dir) else {
        return 0;
    };
    iter.filter_map(Result::ok)
        .filter_map(|e| e.metadata().ok())
        .map(|m| m.len())
        .sum()
}

/// Path of a specific cache entry, for debugging.
#[must_use]
pub fn entry_path(key: &CacheKey) -> PathBuf {
    path_for(key)
}

/// Deletes one entry. Returns `true` if it existed.
#[must_use]
pub fn remove(key: &CacheKey) -> bool {
    fs::remove_file(path_for(key)).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn audio(seconds: f64) -> MonoAudio {
        let n = (seconds * 16_000.0) as usize;
        MonoAudio::new(vec![0.1; n], 16_000)
    }

    fn cfg() -> TranscribeConfig {
        TranscribeConfig::default()
    }

    #[test]
    fn hash_audio_is_deterministic() {
        let a = audio(0.5);
        let h1 = hash_audio(&a);
        let h2 = hash_audio(&a);
        assert_eq!(h1, h2);
        assert_eq!(h1.len(), 64);
    }

    #[test]
    fn hash_audio_changes_with_content() {
        let a = audio(0.5);
        let mut b = audio(0.5);
        b.samples[0] = 0.2;
        assert_ne!(hash_audio(&a), hash_audio(&b));
    }

    #[test]
    fn hash_audio_changes_with_rate() {
        let a = MonoAudio::new(vec![0.0; 100], 16_000);
        let b = MonoAudio::new(vec![0.0; 100], 48_000);
        assert_ne!(hash_audio(&a), hash_audio(&b));
    }

    #[test]
    fn hash_settings_includes_model() {
        let c = cfg();
        assert_ne!(hash_settings("A", &c), hash_settings("B", &c));
    }

    #[test]
    fn hash_settings_changes_with_language() {
        let mut c = cfg();
        let h1 = hash_settings("A", &c);
        c.language = crate::transcribe::LanguageHint::Auto;
        let h2 = hash_settings("A", &c);
        assert_ne!(h1, h2);
    }

    #[test]
    fn hash_settings_changes_with_prompt() {
        let mut c = cfg();
        let h1 = hash_settings("A", &c);
        c.initial_prompt = Some("Hello".into());
        let h2 = hash_settings("A", &c);
        assert_ne!(h1, h2);
    }

    #[test]
    fn hash_settings_changes_with_beam() {
        let mut c = cfg();
        let h1 = hash_settings("A", &c);
        c.decode = DecodeStrategy::BeamSearch { beam_size: 5 };
        let h2 = hash_settings("A", &c);
        assert_ne!(h1, h2);
    }

    #[test]
    fn cache_key_filename_is_short_and_hex() {
        let k = key_for(&audio(0.1), "model", &cfg());
        let name = k.as_filename();
        assert_eq!(name.len(), 33, "{name}");
        assert!(name.chars().all(|c| c.is_ascii_hexdigit() || c == '-'));
    }

    #[test]
    fn cached_round_trip_preserves_words() {
        let t = Transcript {
            segments: vec![Segment {
                text: "hello world".into(),
                start_s: 0.0,
                end_s: 1.0,
                words: vec![
                    Word {
                        text: "hello".into(),
                        start_s: 0.0,
                        end_s: 0.5,
                        probability: 0.9,
                        estimated: false,
                    },
                    Word {
                        text: "world".into(),
                        start_s: 0.5,
                        end_s: 1.0,
                        probability: 0.8,
                        estimated: true,
                    },
                ],
            }],
            language: "hi".into(),
            tier: Tier::Standard,
            model: "Whisper-Hindi2Hinglish-Apex".into(),
            audio_duration_s: 1.0,
            elapsed_s: 0.3,
            real_time_factor: 3.33,
        };
        let k = CacheKey {
            audio_hash: "a".repeat(64),
            settings_hash: "b".repeat(64),
        };
        let cached = CachedTranscript::from_transcript(&t, &k);
        let back = cached.to_transcript();
        assert_eq!(back.segments.len(), 1);
        assert_eq!(back.segments[0].words.len(), 2);
        assert_eq!(back.segments[0].words[0].text, "hello");
        assert!(!back.segments[0].words[0].estimated);
        assert!(back.segments[0].words[1].estimated);
        assert_eq!(back.tier, Tier::Standard);
    }

    #[test]
    fn cache_dir_does_not_panic() {
        let _ = cache_dir();
    }
}