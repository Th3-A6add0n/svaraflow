# SvaraFlow ASR benchmark

Skipped (not on disk): Swift Q5, Swift Q8, Apex Q8_0.

| Model | Mode | xRT (wall) | Load (s) | Peak RAM (MB) | Peak VRAM (MB) | WER | Median word offset (ms) | Estimated words | DTW |
|---|---|---|---|---|---|---|---|---|---|
| Swift F16 | GPU | 23.97 | 0.55 | 491 | 2062 | n/a | n/a | 100% (2233/2233) | no |
| Apex Q5_0 | GPU | 8.28 | 0.88 | 507 | 2706 | n/a | n/a | 100% (1324/1324) | no |
