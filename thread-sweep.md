# SvaraFlow CPU thread sweep

Clip: `tests\audio\benchmark20s.wav`  
Audio duration: 1.00  
Machine: THE-ABADDON  

| Model | Threads | Flash | RTF (wall) | Total (s) | whisper_full (s) | Tokens | Status |
|---|---|---|---|---|---|---|---|
| Apex Q5_0 | 4 | off | 0.09 | 309.46 | 308.88 | 94 | ok |
| Apex Q5_0 | 4 | on | 0.09 | 315.81 | 315.23 | 94 | ok |
| Apex Q5_0 | 8 | off | 0.17 | 162.15 | 161.55 | 94 | ok |
| Apex Q5_0 | 8 | on | 0.17 | 165.80 | 165.23 | 94 | ok |
| Apex Q5_0 | 12 | off | 0.21 | 130.43 | 129.82 | 94 | ok |
| Apex Q5_0 | 12 | on | 0.22 | 124.67 | 124.07 | 94 | ok |
| Apex Q5_0 | 24 | off | 0.27 | 103.67 | 103.07 | 94 | ok |
| Apex Q5_0 | 24 | on | 0.30 | 94.40 | 93.81 | 94 | ok |
| Apex Q8_0 | 4 | off | 0.13 | 219.73 | 218.91 | 94 | ok |
| Apex Q8_0 | 4 | on | 0.12 | 223.48 | 222.65 | 94 | ok |
| Apex Q8_0 | 8 | off | 0.23 | 119.98 | 119.13 | 94 | ok |
| Apex Q8_0 | 8 | on | 0.23 | 120.81 | 119.95 | 94 | ok |
| Apex Q8_0 | 12 | off | 0.30 | 93.53 | 92.67 | 94 | ok |
| Apex Q8_0 | 12 | on | 0.29 | 94.95 | 94.13 | 94 | ok |
| Apex Q8_0 | 24 | off | 0.38 | 73.40 | 72.53 | 94 | ok |
| Apex Q8_0 | 24 | on | 0.39 | 71.93 | 71.09 | 94 | ok |
