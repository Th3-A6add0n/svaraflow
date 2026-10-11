# Repository Tree

```text
svaraflow
├── app
├── docs
│   ├── BUILD.md
│   ├── DECISIONS.md
│   ├── Last Error.md
│   ├── PROJECT_STATE.md
│   └── SCHEMAS.md
├── hosts
│   ├── premiere
│   └── resolve
├── svaraflow-asr
│   ├── src
│   │   ├── bench.rs
│   │   ├── cache.rs
│   │   ├── error.rs
│   │   ├── exports.rs
│   │   ├── lib.rs
│   │   ├── models.rs
│   │   ├── transcribe.rs
│   │   └── vad.rs
│   └── Cargo.toml
├── svaraflow-cli
│   ├── src
│   │   ├── cli.rs
│   │   ├── main.rs
│   │   ├── presets.rs
│   │   ├── silence_cmd.rs
│   │   ├── timeline_cmd.rs
│   │   └── transcribe_cmd.rs
│   ├── tests
│   │   ├── common
│   │   │   └── mod.rs
│   │   ├── cli_smoke.rs
│   │   └── pipeline.rs
│   └── Cargo.toml
├── svaraflow-core
│   ├── src
│   │   ├── audio
│   │   │   ├── decode.rs
│   │   │   ├── envelope.rs
│   │   │   ├── mixdown.rs
│   │   │   └── mod.rs
│   │   ├── cut.rs
│   │   ├── edl.rs
│   │   ├── error.rs
│   │   ├── lib.rs
│   │   ├── silence.rs
│   │   ├── timeline.rs
│   │   ├── vad.rs
│   │   └── xmeml.rs
│   ├── tests
│   │   ├── fixtures
│   │   │   ├── golden.edl
│   │   │   └── golden.xml
│   │   ├── perf.rs
│   │   └── xmeml_golden.rs
│   └── Cargo.toml
├── svaraflow-render
│   ├── src
│   │   └── lib.rs
│   └── Cargo.toml
├── svaraflow-rpc
│   ├── src
│   │   └── lib.rs
│   └── Cargo.toml
├── svaraflow-text
│   ├── src
│   │   └── lib.rs
│   └── Cargo.toml
├── tests
│   └── audio
│       ├── fixtures
│       │   └── README.md
│       ├── benchmark20s.wav
│       ├── speech1.mp4
│       ├── test1.prproj
│       └── test2.mp4
├── tools
│   ├── check-env.ps1
│   ├── convert-model.ps1
│   └── README.md
├── Cargo.lock
├── Cargo.toml
├── CHANGELOG.md
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── deny.toml
├── generate-Repo-Tree.py
├── LICENSE
├── models.json
├── README.md
├── repo-tree.md
├── rustfmt.toml
└── SECURITY.md
```
