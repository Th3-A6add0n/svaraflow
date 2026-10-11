<!-- tests/timing/README.md -->
# Word-timing labels

Place one JSON file per clip here, named after the audio stem
(e.g. `speech1.json` for `speech1.mp4`).

The file must be a JSON array of objects:

```json
[
  { "word": "nahin", "start": 1.23, "end": 1.55 },
  { "word": "yaar",  "start": 1.56, "end": 1.80 }
]