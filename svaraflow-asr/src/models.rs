//! Model manifest and downloader.
//!
//! The manifest is `models.json` at the repository root, embedded into the
//! binary at compile time with [`include_str!`]. It declares the two
//! speech-to-text tiers SvaraFlow ships: `light` (Whisper-base-sized) and
//! `standard` (large-v3-turbo-sized, Q5_0).
//!
//! Model files live outside the repository, in a per-user folder:
//!
//! - Windows: `%LOCALAPPDATA%\SvaraFlow\models`
//! - Linux: `$XDG_DATA_HOME/svaraflow/models`, or
//!   `~/.local/share/svaraflow/models`
//!
//! The downloader resumes interrupted transfers with HTTP `Range` requests
//! and verifies the file's SHA-256 against the manifest when the manifest
//! carries a real hash. A manifest entry whose `sha256` is the literal
//! string `"TODO"` skips verification, so the pipeline is testable before
//! the hashes are filled in.

use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::error::AsrError;

/// The manifest, embedded at compile time.
///
/// The path is relative to this file's directory, so `../../` reaches the
/// workspace root from `svaraflow-asr/src/`.
const MANIFEST_JSON: &str = include_str!("../../models.json");

/// The literal that marks a manifest field as not yet filled in.
pub const TODO_SENTINEL: &str = "TODO";

/// One of the two speed/accuracy tiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Tier {
    /// Whisper-base-sized model. Fast, lower accuracy.
    Light,
    /// Large-v3-turbo-sized model, Q5_0. Slower, higher accuracy.
    Standard,
}

impl Tier {
    /// The manifest's tier string.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Tier::Light => "light",
            Tier::Standard => "standard",
        }
    }

    /// Parses a tier name. Case-insensitive.
    pub fn parse(s: &str) -> Result<Self, AsrError> {
        match s.trim().to_ascii_lowercase().as_str() {
            "light" => Ok(Tier::Light),
            "standard" | "std" => Ok(Tier::Standard),
            other => Err(AsrError::UnknownTier(other.to_string())),
        }
    }
}

/// One model entry in the manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEntry {
    /// Human-readable name, e.g. `Whisper-Hindi2Hinglish-Apex`.
    pub name: String,
    /// Which tier this model serves.
    pub tier: Tier,
    /// One-line description.
    pub description: String,
    /// File name on disk, e.g. `ggml-hindi2hinglish-apex-q5_0.bin`.
    pub filename: String,
    /// Download URL. `"TODO"` until the manifest is filled in.
    pub url: String,
    /// Revision or tag the URL points at.
    pub revision: String,
    /// Expected size in bytes. `0` means unknown.
    pub size_bytes: u64,
    /// Lowercase hex SHA-256. `"TODO"` until the manifest is filled in.
    pub sha256: String,
    /// SPDX license identifier.
    pub license: String,
    /// Quantization string, e.g. `q5_0`.
    pub quantization: String,
    /// Upstream Hugging Face repo the model came from.
    pub source_repo: String,
}

impl ModelEntry {
    /// True when the manifest carries a real SHA-256.
    #[must_use]
    pub fn has_real_hash(&self) -> bool {
        !self.sha256.eq_ignore_ascii_case(TODO_SENTINEL) && !self.sha256.is_empty()
    }

    /// True when the manifest carries a real download URL.
    #[must_use]
    pub fn has_real_url(&self) -> bool {
        !self.url.eq_ignore_ascii_case(TODO_SENTINEL) && !self.url.is_empty()
    }
}

/// The whole manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    /// Schema version. Must be `1`.
    pub version: u32,
    /// The models.
    pub models: Vec<ModelEntry>,
}

impl Manifest {
    /// Parses the embedded manifest.
    ///
    /// # Errors
    ///
    /// [`AsrError::Manifest`] when the JSON is malformed or the version is
    /// not `1`.
    pub fn load_embedded() -> Result<Self, AsrError> {
        let m: Manifest = serde_json::from_str(MANIFEST_JSON)
            .map_err(|e| AsrError::Manifest(e.to_string()))?;
        if m.version != 1 {
            return Err(AsrError::Manifest(format!(
                "unsupported manifest version {}; this build reads version 1",
                m.version
            )));
        }
        Ok(m)
    }

    /// Finds the entry for a tier.
    ///
    /// # Errors
    ///
    /// [`AsrError::UnknownTier`] when no entry serves that tier.
    pub fn for_tier(&self, tier: Tier) -> Result<&ModelEntry, AsrError> {
        self.models
            .iter()
            .find(|m| m.tier == tier)
            .ok_or(AsrError::UnknownTier(tier.as_str().to_string()))
    }

    /// Finds an entry by manifest name. Case-insensitive.
    pub fn by_name(&self, name: &str) -> Option<&ModelEntry> {
        self.models
            .iter()
            .find(|m| m.name.eq_ignore_ascii_case(name))
    }

    /// A map from tier to entry, for quick lookup.
    #[must_use]
    pub fn by_tier(&self) -> HashMap<Tier, ModelEntry> {
        self.models.iter().map(|m| (m.tier, m.clone())).collect()
    }
}

/// Where models live on this machine.
///
/// Never fails: falls back to `./models` relative to the working directory
/// when neither environment variable is set.
#[must_use]
pub fn storage_dir() -> PathBuf {
    storage_dir_result().unwrap_or_else(|_| PathBuf::from("models"))
}

/// Where models live on this machine.
///
/// # Errors
///
/// [`AsrError::ModelFolder`] when the platform's data directory variables
/// are missing.
pub fn storage_dir_result() -> Result<PathBuf, AsrError> {
    #[cfg(target_os = "windows")]
    {
        let base = std::env::var_os("LOCALAPPDATA")
            .map(PathBuf::from)
            .ok_or_else(|| AsrError::ModelFolder("%LOCALAPPDATA% is not set".into()))?;
        Ok(base.join("SvaraFlow").join("models"))
    }

    #[cfg(not(target_os = "windows"))]
    {
        let base = if let Some(x) = std::env::var_os("XDG_DATA_HOME") {
            PathBuf::from(x)
        } else if let Some(h) = std::env::var_os("HOME") {
            PathBuf::from(h).join(".local").join("share")
        } else {
            return Err(AsrError::ModelFolder(
                "neither XDG_DATA_HOME nor HOME is set".into(),
            ));
        };
        Ok(base.join("svaraflow").join("models"))
    }
}

/// Ensures the model folder exists and returns it.
///
/// # Errors
///
/// [`AsrError::ModelFolder`] when the folder cannot be created.
pub fn ensure_storage_dir() -> Result<PathBuf, AsrError> {
    let dir = storage_dir_result()?;
    fs::create_dir_all(&dir)
        .map_err(|e| AsrError::ModelFolder(format!("creating {}: {e}", dir.display())))?;
    Ok(dir)
}

/// The on-disk path for a manifest entry.
#[must_use]
pub fn path_for(entry: &ModelEntry) -> PathBuf {
    storage_dir().join(&entry.filename)
}

/// Computes the SHA-256 of a file, streaming it so memory stays flat.
///
/// # Errors
///
/// [`AsrError::Io`] on any read failure.
pub fn hash_file(path: &Path) -> Result<String, AsrError> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex::encode(hasher.finalize()))
}

/// Progress callback signature for downloads.
///
/// `downloaded` is bytes written so far, `total` is the expected total when
/// the server reported it.
pub type ProgressFn<'a> = dyn FnMut(u64, Option<u64>) + 'a;

/// Ensures a model file is present locally, downloading and verifying it
/// if necessary.
///
/// A file that exists and hashes correctly is returned immediately. A file
/// that exists but hashes wrong is deleted and re-downloaded. A `.part`
/// file is resumed with an HTTP `Range` request.
///
/// # Errors
///
/// - [`AsrError::Download`] on network failure or a non-2xx response.
/// - [`AsrError::ModelHashMismatch`] when the downloaded bytes do not match
///   the manifest.
/// - [`AsrError::ModelFolder`] when the folder cannot be created.
pub fn ensure_model(
    entry: &ModelEntry,
    progress: &mut ProgressFn<'_>,
) -> Result<PathBuf, AsrError> {
    let dir = ensure_storage_dir()?;
    let final_path = dir.join(&entry.filename);
    let part_path = dir.join(format!("{}.part", entry.filename));

    if final_path.exists() {
        if !entry.has_real_hash() {
            return Ok(final_path);
        }
        let actual = hash_file(&final_path)?;
        if actual.eq_ignore_ascii_case(&entry.sha256) {
            return Ok(final_path);
        }
        let _ = fs::remove_file(&final_path);
    }

    if !entry.has_real_url() {
        return Err(AsrError::Download {
            name: entry.name.clone(),
            url: entry.url.clone(),
            reason: format!(
                "the manifest's `url` for this model is `{TODO_SENTINEL}`. \
                 Fill it in models.json, or place the file by hand at {}.",
                final_path.display()
            ),
        });
    }

    let start_byte = fs::metadata(&part_path).map(|m| m.len()).unwrap_or(0);
    download_resumable(entry, &part_path, start_byte, progress)?;

    if entry.has_real_hash() {
        let actual = hash_file(&part_path)?;
        if !actual.eq_ignore_ascii_case(&entry.sha256) {
            let _ = fs::remove_file(&part_path);
            return Err(AsrError::ModelHashMismatch {
                name: entry.name.clone(),
                path: part_path.display().to_string(),
                actual,
                expected: entry.sha256.clone(),
            });
        }
    }

    fs::rename(&part_path, &final_path)?;
    Ok(final_path)
}

/// Runs the download loop with resume support.
fn download_resumable(
    entry: &ModelEntry,
    part_path: &Path,
    start_byte: u64,
    progress: &mut ProgressFn<'_>,
) -> Result<(), AsrError> {
    let mut request = ureq::get(&entry.url);
    if start_byte > 0 {
        request = request.header("Range", &format!("bytes={start_byte}-"));
    }

    let response = request.call().map_err(|e| AsrError::Download {
        name: entry.name.clone(),
        url: entry.url.clone(),
        reason: e.to_string(),
    })?;

    let status = response.status().as_u16();
    let (append, total) = match status {
        206 => {
            let total = response
                .headers()
                .get("content-range")
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.rsplit('/').next())
                .and_then(|s| s.trim().parse::<u64>().ok());
            (true, total)
        }
        200 => {
            let total = response
                .headers()
                .get("content-length")
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.trim().parse::<u64>().ok());
            (false, total)
        }
        other => {
            return Err(AsrError::Download {
                name: entry.name.clone(),
                url: entry.url.clone(),
                reason: format!("server returned HTTP {other}"),
            });
        }
    };

    let mut out = if append {
        OpenOptions::new().append(true).open(part_path)?
    } else {
        File::create(part_path)?
    };

    let mut reader = response.into_body().into_reader();
    let mut buf = [0u8; 256 * 1024];
    let mut written = if append { start_byte } else { 0 };

    loop {
        let n = reader.read(&mut buf).map_err(|e| AsrError::Download {
            name: entry.name.clone(),
            url: entry.url.clone(),
            reason: format!("reading response body: {e}"),
        })?;
        if n == 0 {
            break;
        }
        out.write_all(&buf[..n])?;
        written += n as u64;
        progress(written, total);
    }
    out.sync_all()?;

    if entry.size_bytes > 0 && written != entry.size_bytes {
        return Err(AsrError::Download {
            name: entry.name.clone(),
            url: entry.url.clone(),
            reason: format!(
                "downloaded {written} bytes but the manifest declares {}",
                entry.size_bytes
            ),
        });
    }
    Ok(())
}

/// Deletes a model file if present. Also deletes a leftover `.part`.
///
/// Returns `true` when something was removed.
///
/// # Errors
///
/// [`AsrError::Io`] when a deletion fails for a reason other than
/// "not found".
pub fn remove_model(entry: &ModelEntry) -> Result<bool, AsrError> {
    let dir = ensure_storage_dir()?;
    let mut removed = false;
    for name in [&entry.filename, &format!("{}.part", entry.filename)] {
        let path = dir.join(name);
        match fs::remove_file(&path) {
            Ok(()) => removed = true,
            Err(e) if e.kind() == io::ErrorKind::NotFound => {}
            Err(e) => return Err(AsrError::Io(e)),
        }
    }
    Ok(removed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_manifest_parses() {
        let m = Manifest::load_embedded().expect("manifest parses");
        assert_eq!(m.version, 1);
        assert_eq!(m.models.len(), 2);
    }

    #[test]
    fn manifest_has_both_tiers() {
        let m = Manifest::load_embedded().unwrap();
        assert!(m.for_tier(Tier::Light).is_ok());
        assert!(m.for_tier(Tier::Standard).is_ok());
    }

    #[test]
    fn tier_parse_accepts_both_and_aliases() {
        assert_eq!(Tier::parse("light").unwrap(), Tier::Light);
        assert_eq!(Tier::parse("LIGHT").unwrap(), Tier::Light);
        assert_eq!(Tier::parse("standard").unwrap(), Tier::Standard);
        assert_eq!(Tier::parse("std").unwrap(), Tier::Standard);
    }

    #[test]
    fn tier_parse_rejects_garbage() {
        assert!(Tier::parse("turbo").is_err());
        assert!(Tier::parse("").is_err());
    }

    #[test]
    fn light_model_filename_mentions_swift() {
        let m = Manifest::load_embedded().unwrap();
        let e = m.for_tier(Tier::Light).unwrap();
        assert!(e.filename.contains("swift"), "{}", e.filename);
        assert_eq!(e.quantization, "f16");
    }

    #[test]
    fn standard_model_filename_mentions_apex_and_q5_0() {
        let m = Manifest::load_embedded().unwrap();
        let e = m.for_tier(Tier::Standard).unwrap();
        assert!(e.filename.contains("apex"), "{}", e.filename);
        assert!(e.filename.contains("q5_0"), "{}", e.filename);
        assert_eq!(e.quantization, "q5_0");
    }

    #[test]
    fn todo_hashes_are_detected() {
        let m = Manifest::load_embedded().unwrap();
        for e in &m.models {
            assert!(!e.has_real_hash(), "{} carries a real hash", e.name);
            assert!(!e.has_real_url(), "{} carries a real url", e.name);
        }
    }

    #[test]
    fn by_name_is_case_insensitive() {
        let m = Manifest::load_embedded().unwrap();
        assert!(m.by_name("whisper-hindi2hinglish-apex").is_some());
        assert!(m.by_name("Whisper-Hindi2Hinglish-Swift").is_some());
        assert!(m.by_name("nonexistent").is_none());
    }

    #[test]
    fn by_tier_produces_a_map() {
        let m = Manifest::load_embedded().unwrap();
        let map = m.by_tier();
        assert_eq!(map.len(), 2);
        assert!(map.contains_key(&Tier::Light));
        assert!(map.contains_key(&Tier::Standard));
    }

    #[test]
    fn storage_dir_does_not_panic() {
        let _ = storage_dir();
    }

    #[test]
    fn hash_file_matches_known_vector() {
        let tmp = std::env::temp_dir().join("svaraflow-asr-empty-hash-test.bin");
        std::fs::write(&tmp, b"").unwrap();
        let h = hash_file(&tmp).unwrap();
        assert_eq!(
            h,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn hash_file_matches_abc() {
        let tmp = std::env::temp_dir().join("svaraflow-asr-abc-hash-test.bin");
        std::fs::write(&tmp, b"abc").unwrap();
        let h = hash_file(&tmp).unwrap();
        assert_eq!(
            h,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        let _ = std::fs::remove_file(&tmp);
    }
}