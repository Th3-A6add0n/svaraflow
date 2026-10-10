# Contributing to SvaraFlow

Thanks for your interest in SvaraFlow. This project is MIT-licensed and welcomes
issues and pull requests. By participating you agree to the
[Code of Conduct](CODE_OF_CONDUCT.md).

## Ground rules

1. **Clean-room.** Do not copy code, assets, strings, names or branding from any
   commercial plugin or closed-source project. Work only from the public
   specification, public documentation and permissively licensed open-source
   libraries.
2. **Offline-first.** All AI runs locally. No feature may phone home, no model
   may be fetched at runtime without the user's explicit action, and no
   telemetry is added.
3. **Small models.** The Light and Standard ASR tiers must stay within their
   declared sizes. Do not pull in a larger model to make a test pass.
4. **Non-destructive.** Edits are always applied by generating a new timeline
   (FCP7 `xmeml`) from a `timeline.json`. The original timeline is never
   mutated.
5. **Typed errors.** Libraries use `thiserror`; binaries use `anyhow`. No
   `unwrap` or `expect` outside tests.
6. **Small modules, documented public functions.** Every `pub` item carries a
   doc comment.

## Getting set up

1. Install the toolchain and verify it:

   ```powershell
   pwsh tools/check-env.ps1