# Upstream whisper-cli sweep

CLI: `C:\Users\The Abaddon\Desktop\DESKTOP\Personal CODING Projects\whisper.cpp\build\bin\Release\whisper-cli.exe`  
Model: `C:\Users\The Abaddon\AppData\Local\SvaraFlow\models\ggml-hinglish-apex-q5_0.bin`  
Clip: `tests\audio\benchmark20s.wav`  
Language: hi  

| Threads | Flash | Load (ms) | Encode (ms) | Decode (ms) | batchd (ms) | Total (ms) | Fallbacks | Status |
|---|---|---|---|---|---|---|---|---|
| 4 | on | 453.22 | 15946.02 | 803.84 | 19.95 | 17310.59 | 0 | ok |
| 4 | off | 494.35 | 22159.41 | 583.68 | 14.73 | 23345.12 | 0 | ok |
| 8 | on | 464.48 | 8646.96 | 669.33 | 12.88 | 9875.96 | 0 | ok |
| 8 | off | 470.61 | 12715.11 | 588.99 | 10.35 | 13869.77 | 0 | ok |
| 12 | on | 453.69 | 6437.96 | 642.58 | 11.32 | 7630.76 | 0 | ok |
| 12 | off | 450.52 | 9787.32 | 607.73 | 9.16 | 10941.23 | 0 | ok |
| 24 | on | 446.99 | 5629.13 | 663.97 | 12.57 | 6854.41 | 0 | ok |
| 24 | off | 455.76 | 8442.91 | 655.72 | 9.64 | 9664.11 | 0 | ok |
