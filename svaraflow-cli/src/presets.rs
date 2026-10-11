//! User preset storage.
//!
//! Presets live in a JSON file outside the timeline:
//!
//! - Windows: `%APPDATA%\SvaraFlow\presets\silence.json`
//! - Linux: `$XDG_CONFIG_HOME/svaraflow/presets/silence.json`, or
//!   `~/.config/svaraflow/presets/silence.json`
//!
//! Built-in names are reserved and cannot be saved over. The file schema
//! is documented in `docs/SCHEMAS.md`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use svaraflow_core::silence::{BuiltinPreset, PresetValues};

/// On-disk file version.
pub const PRESET_FILE_VERSION: u32 = 1;

/// One preset's values, as stored on disk.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct StoredPreset {
    /// Silence removal threshold, milliseconds.
    pub remove_silences_ms: u32,
    /// Minimum kept speech, milliseconds.
    pub keep_talks_ms: u32,
    /// Padding before each kept run, milliseconds.
    pub margin_before_ms: u32,
    /// Padding after each kept run, milliseconds.
    pub margin_after_ms: u32,
    /// Minimum gap between cuts. `0` disables the rule.
    pub min_between_cuts_ms: u32,
}

impl From<PresetValues> for StoredPreset {
    fn from(v: PresetValues) -> Self {
        Self {
            remove_silences_ms: v.remove_silences_ms,
            keep_talks_ms: v.keep_talks_ms,
            margin_before_ms: v.margin_before_ms,
            margin_after_ms: v.margin_after_ms,
            min_between_cuts_ms: v.min_between_cuts_ms,
        }
    }
}

impl From<StoredPreset> for PresetValues {
    fn from(s: StoredPreset) -> Self {
        Self {
            remove_silences_ms: s.remove_silences_ms,
            keep_talks_ms: s.keep_talks_ms,
            margin_before_ms: s.margin_before_ms,
            margin_after_ms: s.margin_after_ms,
            min_between_cuts_ms: s.min_between_cuts_ms,
        }
    }
}

/// The whole preset file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PresetFile {
    /// Schema version. Must be `1`.
    pub version: u32,
    /// Preset name → values.
    #[serde(default)]
    pub presets: BTreeMap<String, StoredPreset>,
}

impl Default for PresetFile {
    fn default() -> Self {
        Self {
            version: PRESET_FILE_VERSION,
            presets: BTreeMap::new(),
        }
    }
}

/// Location of the preset file on this machine.
pub fn storage_path() -> Result<PathBuf> {
    #[cfg(target_os = "windows")]
    let base = std::env::var_os("APPDATA")
        .map(PathBuf::from)
        .ok_or_else(|| anyhow!("%APPDATA% is not set"))?
        .join("SvaraFlow");

    #[cfg(not(target_os = "windows"))]
    let base = {
        let xdg = std::env::var_os("XDG_CONFIG_HOME").map(PathBuf::from);
        let home = std::env::var_os("HOME").map(PathBuf::from);
        let dir = xdg
            .or_else(|| home.map(|h| h.join(".config")))
            .ok_or_else(|| anyhow!("neither XDG_CONFIG_HOME nor HOME is set"))?;
        dir.join("svaraflow")
    };

    Ok(base.join("presets").join("silence.json"))
}

/// Loads the file. A missing file yields an empty [`PresetFile`].
pub fn load() -> Result<PresetFile> {
    load_from(&storage_path()?)
}

/// Loads the file at `path`. A missing file yields an empty [`PresetFile`].
pub fn load_from(path: &Path) -> Result<PresetFile> {
    if !path.exists() {
        return Ok(PresetFile::default());
    }
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let file: PresetFile =
        serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
    if file.version != PRESET_FILE_VERSION {
        return Err(anyhow!(
            "{}: unsupported preset file version {}; this build writes version {}",
            path.display(),
            file.version,
            PRESET_FILE_VERSION
        ));
    }
    Ok(file)
}

/// Writes the file at `path`, creating parent directories.
pub fn save_to(path: &Path, file: &PresetFile) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }
    let text = serde_json::to_string_pretty(file)?;
    std::fs::write(path, text).with_context(|| format!("writing {}", path.display()))
}

/// Writes the file at the default location, creating parent directories.
pub fn save(file: &PresetFile) -> Result<()> {
    save_to(&storage_path()?, file)
}

/// True when `name` collides with a built-in.
#[must_use]
pub fn is_builtin(name: &str) -> bool {
    BuiltinPreset::ALL
        .iter()
        .any(|p| p.name().eq_ignore_ascii_case(name))
}

/// Saves a user preset, refusing built-in names.
pub fn save_user(name: &str, values: PresetValues) -> Result<()> {
    if is_builtin(name) {
        return Err(anyhow!(
            "`{name}` is a built-in preset and is read-only; pick a different name"
        ));
    }
    if name.trim().is_empty() {
        return Err(anyhow!("preset name must not be empty"));
    }
    let mut file = load()?;
    file.presets.insert(name.to_string(), values.into());
    save(&file)
}

/// Renames a user preset. Refuses built-ins on either side.
pub fn rename_user(old: &str, new: &str) -> Result<()> {
    if is_builtin(old) {
        return Err(anyhow!(
            "`{old}` is a built-in preset and cannot be renamed"
        ));
    }
    if is_builtin(new) {
        return Err(anyhow!(
            "`{new}` is a built-in preset name; pick a different name"
        ));
    }
    if new.trim().is_empty() {
        return Err(anyhow!("new preset name must not be empty"));
    }
    let mut file = load()?;
    let values = file
        .presets
        .remove(old)
        .ok_or_else(|| anyhow!("no user preset named `{old}`"))?;
    file.presets.insert(new.to_string(), values);
    save(&file)
}

/// Deletes a user preset. Refuses built-ins.
pub fn delete_user(name: &str) -> Result<()> {
    if is_builtin(name) {
        return Err(anyhow!(
            "`{name}` is a built-in preset and cannot be deleted"
        ));
    }
    let mut file = load()?;
    if file.presets.remove(name).is_none() {
        return Err(anyhow!("no user preset named `{name}`"));
    }
    save(&file)
}

/// Lists every user preset as `(name, values)`, sorted by name.
pub fn list_user() -> Result<Vec<(String, PresetValues)>> {
    let file = load()?;
    Ok(file
        .presets
        .into_iter()
        .map(|(k, v)| (k, v.into()))
        .collect())
}

/// Imports presets from `path`, merging into the current file.
pub fn import(path: &Path) -> Result<usize> {
    let incoming = load_from(path)?;
    let mut current = load()?;
    let mut imported = 0usize;
    for (name, values) in incoming.presets {
        if is_builtin(&name) {
            continue;
        }
        current.presets.insert(name, values);
        imported += 1;
    }
    save(&current)?;
    Ok(imported)
}

/// Exports user presets to `path`.
pub fn export(path: &Path) -> Result<usize> {
    let current = load()?;
    let count = current.presets.len();
    save_to(path, &current)?;
    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_file(name: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "svaraflow-presets-{}-{}.json",
            name,
            std::process::id()
        ));
        let _ = std::fs::remove_file(&p);
        p
    }

    #[test]
    fn stored_preset_round_trips_through_values() {
        let v = PresetValues {
            remove_silences_ms: 220,
            keep_talks_ms: 110,
            margin_before_ms: 90,
            margin_after_ms: 90,
            min_between_cuts_ms: 180,
        };
        let s: StoredPreset = v.into();
        let back: PresetValues = s.into();
        assert_eq!(v, back);
    }

    #[test]
    fn missing_file_yields_empty() {
        let path = tmp_file("missing");
        let file = load_from(&path).unwrap();
        assert!(file.presets.is_empty());
        assert_eq!(file.version, PRESET_FILE_VERSION);
    }

    #[test]
    fn save_and_load_round_trip() {
        let path = tmp_file("roundtrip");
        let mut file = PresetFile::default();
        file.presets.insert(
            "custom".to_string(),
            StoredPreset {
                remove_silences_ms: 220,
                keep_talks_ms: 110,
                margin_before_ms: 90,
                margin_after_ms: 90,
                min_between_cuts_ms: 0,
            },
        );
        save_to(&path, &file).unwrap();
        let back = load_from(&path).unwrap();
        assert_eq!(back.presets.len(), 1);
        assert_eq!(back.presets["custom"].remove_silences_ms, 220);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn built_in_names_are_reserved() {
        assert!(is_builtin("natural"));
        assert!(is_builtin("Natural"));
        assert!(!is_builtin("mycustom"));
    }

    #[test]
    fn save_user_refuses_builtin_names() {
        let err = save_user("natural", BuiltinPreset::Natural.values()).unwrap_err();
        assert!(err.to_string().contains("read-only"), "{err}");
    }

    #[test]
    fn unknown_version_is_rejected() {
        let path = tmp_file("badversion");
        std::fs::write(&path, r#"{"version": 99, "presets": {}}"#).unwrap();
        let err = load_from(&path).unwrap_err();
        assert!(err.to_string().contains("unsupported preset file version"));
        let _ = std::fs::remove_file(&path);
    }
}
