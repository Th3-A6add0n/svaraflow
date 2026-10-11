# SvaraFlow ASR benchmark

Skipped (not on disk): Swift Q5, Swift Q8, Apex Q8_0.

| Model | Mode | xRT (wall) | Load (s) | Peak RAM (MB) | Peak VRAM (MB) | WER | Median word offset (ms) | Estimated words | DTW |
|---|---|---|---|---|---|---|---|---|---|
| Swift F16 | GPU | 23.94 | 0.56 | 483 | 2081 | n/a | n/a | 100% (2233/2233) | no |
| Swift F16 | CPU | 3.44 | 0.43 | 430 | n/a | n/a | n/a | 100% (2217/2217) | no |
| Apex Q5_0 | GPU | 8.22 | 0.89 | 506 | 2704 | n/a | n/a | 100% (1324/1324) | no |
| Apex Q5_0 | CPU | 0.18 | 0.81 | 995 | n/a | n/a | n/a | 100% (67/67) | no |
