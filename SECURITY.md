# Security Policy

## Supported versions

SvaraFlow is pre-1.0. Only the latest commit on `main` is supported. There are
no maintained release branches yet.

## Reporting a vulnerability

Please do **not** open a public issue for security problems. Instead, use
GitHub's private vulnerability reporting:

1. Go to <https://github.com/Th3-A6add0n/svaraflow/security/advisories/new>.
2. Describe the issue, the affected crate or component, and the steps to
   reproduce it.

If private reporting is unavailable to you, open a minimal public issue that
says only that you have a security report and would like a private channel.
Do not include exploit details in that issue.

You can expect an acknowledgement within seven days. Fixes are released on
`main`; there is no backport policy yet.

## Design commitments

SvaraFlow is built so that the following properties hold. If you find a way to
violate any of them, that is a security bug:

- **Local only.** The JSON-RPC server binds to `127.0.0.1` and never to a
  routable interface. It requires a one-time token that the desktop app
  generates per launch.
- **Offline AI.** No audio, transcript or timeline data leaves the machine.
  There is no telemetry and no remote model download at runtime.
- **No shell execution from the engine.** Host adapters invoke documented
  Premiere Pro UXP APIs and DaVinci Resolve scripting APIs only; the engine
  does not construct shell command lines from user input.
- **Non-destructive editing.** The original timeline is never modified. A
  compromised engine cannot damage a user's existing project by design.

## Out of scope

- Vulnerabilities in Adobe Premiere Pro, DaVinci Resolve, or their scripting
  runtimes. Report those to the respective vendor.
- Vulnerabilities in `whisper.cpp`, Silero VAD, or other upstream dependencies
  that are already publicly known and unfixed upstream. Report those upstream
  first; we will track the fix here.
- Malicious local attackers with write access to the user's disk.