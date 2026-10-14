// svaraflow-asr/build.rs
//! Build-time check that whisper-rs-sys was built with optimization.
//!
//! # Why
//!
//! whisper-rs-sys 0.15.0 invokes cmake without an explicit
//! `CMAKE_BUILD_TYPE`, so on Windows/MSVC the Visual Studio generator
//! leaves `CMAKE_C_FLAGS_RELEASE = " -nologo -MD -Brepro -W0"` — no
//! `/O2`, no `/DNDEBUG`. whisper.cpp then runs 8-10x slower than the
//! same source built by upstream. The fix is `.cargo/config.toml` at
//! the workspace root.
//!
//! This build script verifies the fix landed. In release builds it
//! reads `target/<profile>/build/whisper-rs-sys-*/out/build/CMakeCache.txt`
//! and fails the build if the optimization flags are missing.
//!
//! # Escaping
//!
//! Set `SVARAFLOW_SKIP_BUILD_CHECK=1` to turn the failure into a warning.
//! Useful when the cache is somewhere non-standard or when bisecting.
//!
//! # Output
//!
//! Emits `cargo:rustc-env=SVARAFLOW_BUILD_OPTIMIZED={0,1}`, which
//! `transcribe.rs` reads with `option_env!` at compile time and prints
//! as a startup warning at runtime.

use std::env;
use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    let profile = env::var("PROFILE").unwrap_or_default();
    let target_env = env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();

    // Debug builds are expected to be unoptimized. Do not check them.
    if profile != "release" {
        println!("cargo:rustc-env=SVARAFLOW_BUILD_OPTIMIZED=0");
        return;
    }

    if env::var("SVARAFLOW_SKIP_BUILD_CHECK").ok().as_deref() == Some("1") {
        println!(
            "cargo:warning=SVARAFLOW_SKIP_BUILD_CHECK=1; skipping whisper-rs-sys optimization check"
        );
        println!("cargo:rustc-env=SVARAFLOW_BUILD_OPTIMIZED=0");
        return;
    }

    // MSVC: /O2, /O3, or /Ox. gcc/clang: -O2 or -O3.
    let needles: &[&str] = if target_env == "msvc" {
        &["/O2", "/O3", "/Ox"]
    } else {
        &["-O2", "-O3"]
    };

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let target_dir = manifest_dir
        .parent()
        .map(|p| p.join("target"))
        .unwrap_or_else(|| PathBuf::from("target"));

    let Some(cache_path) = find_cmake_cache(&target_dir, &profile) else {
        println!(
            "cargo:warning=whisper-rs-sys CMakeCache.txt not found under {}. \
             Cannot verify optimization flags. See docs/BUILD.md.",
            target_dir.display()
        );
        // Do NOT emit SVARAFLOW_BUILD_OPTIMIZED. Absence means "unknown",
        // which the runtime treats as a pass.
        return;
    };

    let content = std::fs::read_to_string(&cache_path).unwrap_or_default();

    let flags: Option<String> = content.lines().find_map(|line| {
        line.strip_prefix("CMAKE_C_FLAGS_RELEASE:")
            .and_then(|rest| rest.split_once('='))
            .map(|(_, value)| value.trim().to_string())
    });

    let Some(flags) = flags else {
        println!(
            "cargo:warning={} has no CMAKE_C_FLAGS_RELEASE entry. \
             Cannot verify optimization flags. See docs/BUILD.md.",
            cache_path.display()
        );
        println!("cargo:rustc-env=SVARAFLOW_BUILD_OPTIMIZED=0");
        return;
    };

    let optimized = needles.iter().any(|n| flags.contains(n));

    if optimized {
        println!("cargo:rustc-env=SVARAFLOW_BUILD_OPTIMIZED=1");
        return;
    }

    println!("cargo:rustc-env=SVARAFLOW_BUILD_OPTIMIZED=0");
    panic!(
        "whisper-rs-sys was built without optimization.\n\
         \n\
         CMAKE_C_FLAGS_RELEASE = {flags}\n\
         \n\
         Expected one of {needles:?}. Without /O2 (MSVC) or -O2/-O3 \
         (gcc/clang), whisper.cpp runs 8-10x slower than upstream. This \
         is a correctness requirement for the CPU path, not a \
         performance preference.\n\
         \n\
         See docs/BUILD.md for the full explanation. In short:\n\
         \n\
         1. Confirm .cargo/config.toml at the workspace root carries the\n\
            [target.<triple>.env] block for your host.\n\
         2. Force a whisper-rs-sys rebuild:\n\
         \n\
                Get-ChildItem target -Recurse -Directory -Filter 'whisper-rs-sys-*' |\n\
                    Remove-Item -Recurse -Force\n\
                cargo build --release -p svaraflow-cli --bin bench-asr\n\
         \n\
         Set SVARAFLOW_SKIP_BUILD_CHECK=1 to bypass and accept a slow build.",
        flags = flags,
        needles = needles
    );
}

/// Finds the newest whisper-rs-sys CMakeCache under
/// `target/<profile>/build/whisper-rs-sys-*/out/build/`.
///
/// Returns `None` if the directory does not exist or no matching entry
/// has a cache file. Both are non-fatal: the runtime warning covers them.
fn find_cmake_cache(target_dir: &Path, profile: &str) -> Option<PathBuf> {
    let build_root = target_dir.join(profile).join("build");
    if !build_root.is_dir() {
        return None;
    }
    let entries = std::fs::read_dir(&build_root).ok()?;
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.starts_with("whisper-rs-sys-") {
            let cache = entry
                .path()
                .join("out")
                .join("build")
                .join("CMakeCache.txt");
            if cache.is_file() {
                return Some(cache);
            }
        }
    }
    None
}
