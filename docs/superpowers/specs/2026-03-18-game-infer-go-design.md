# game-infer: Go ONNX Inference Tool — Design Spec

**Date:** 2026-03-18
**Project:** GAME (Generative Adaptive MIDI Extractor)
**Goal:** Reimplement `infer_onnx.py` in Go so that users receive a compiled binary requiring no Python environment.

---

## 1. Overview

`game-infer` is a standalone CLI tool that runs GAME's exported ONNX models to transcribe singing voice into MIDI. It mirrors the interface of `infer.py` (`extract` and `align` subcommands) and ships as a binary + onnxruntime shared library, with no other runtime dependencies.

**Target platforms:**
- macOS arm64 (Apple Silicon)
- Windows x86_64

**Audio input:** WAV only (pure-Go decoding, zero external dependencies). Stereo WAV files are mixed down to mono by averaging channels before inference.

---

## 2. Architecture

### 2.1 Project Structure

```
game-infer/
├── main.go                  # Entry point, cobra root command
├── cmd/
│   ├── extract.go           # `extract` subcommand
│   └── align.go             # `align` subcommand
├── onnx/
│   ├── session.go           # OnnxSessions: load all models + config.json, EP selection
│   └── infer.go             # Full inference pipeline
├── audio/
│   ├── wav.go               # WAV decode (pure Go, go-audio/wav); stereo → mono mix-down
│   └── slicer.go            # Silence-based audio slicer (ported from slicer2.py)
├── output/
│   ├── midi.go              # MIDI file writer (gomidi)
│   └── text.go              # TXT / CSV writer
├── go.mod
└── go.sum
```

> **Note:** There is no `mel/` package. The encoder ONNX model accepts raw waveforms (`waveform [B, L] float32`) and performs Mel spectrogram computation internally. No Mel implementation is required in Go.

### 2.2 Dependencies

| Package | Purpose |
|---|---|
| `github.com/yalue/onnxruntime_go` | CGo bindings to onnxruntime C API |
| `github.com/spf13/cobra` | CLI framework |
| `github.com/go-audio/wav` | Pure-Go WAV decoding |
| `gitlab.com/gomidi/midi/v2` | MIDI file writing |

### 2.3 onnxruntime Library Loading

At startup, `onnx/session.go` calls `onnxruntime_go.SetSharedLibraryPath()` pointing to the `.dylib` / `.dll` located in the same directory as the binary. If the file is not found, the program exits immediately with a clear message directing the user to place it alongside the binary.

---

## 3. Inference Pipeline

Mirrors `infer_onnx.py` exactly.

```
WAV file
  └─► audio.LoadWAV()           → []float32 waveform [L]  (resampled to 44100 Hz if needed)
  └─► audio.Slice()             → []Chunk{waveform []float32, offset float64}

Per batch of chunks:
  Zero-pad waveforms to same length → waveformBatch [B, L_max]
  Derive durations from actual lengths → durations [B] float32

  onnx.RunEncoder(waveformBatch, durations)
    → x_seg [B, T, 128] float32
    → x_est [B, T, 128] float32
    → maskT [B, T] bool

  onnx.RunSegmenter(x_seg, maskT, knownBoundaries, prev_boundaries, language, t, threshold, radius)
    → boundaries [B, T] bool   // D3PM loop iterates, feeding output back as prev_boundaries

  onnx.RunBd2Dur(boundaries, maskT)
    → noteDurations [B, N] float32
    → maskN [B, N] bool

  onnx.RunEstimator(x_est, boundaries, maskT, maskN, threshold)
    → presence [B, N] bool
    → scores   [B, N] float32

  reconstructNoteTimestamps(noteDurations, presence, scores, chunkOffsets)
    → []NoteInfo{onset, offset, pitch}

  mergeAndSortNotes(allNotes)
    → []NoteInfo (sorted, deduplicated, no zero-duration notes)

output.WriteMIDI() / WriteText()
```

### 3.1 WAV Loading and Resampling

`audio.LoadWAV()` decodes PCM WAV files using `go-audio/wav`. If the file's sample rate differs from the model's required rate (44100 Hz), the samples are resampled using linear interpolation. Non-PCM formats (e.g. float WAV, ADPCM) are rejected with a clear error. Stereo and multi-channel files are mixed down to mono by averaging all channels.

### 3.2 Audio Slicer (`audio/slicer.go`)

Ported from `slicer2.py`. Fixed parameters (not user-configurable):

| Parameter | Value |
|---|---|
| threshold | -40 dB |
| min_length | 1000 ms |
| min_interval | 200 ms |
| max_sil_kept | 100 ms |
| hop_size | 20 ms |

Each returned `Chunk` carries `waveform []float32` and `offset float64` (start time in seconds within the original file).

### 3.3 Segmenter — D3PM Loop

The segmenter is called in a loop when `config.loop == true`:

```
knownBoundaries = zeros([B, T], bool)   // no prior knowledge in extract mode

// Initialization: start with known_boundaries (matches Python)
boundaries = knownBoundaries.copy()

for each ti in resolveTs(t0, nsteps, customTs):
    t_arr = [ti, ti, ..., ti]  // shape [B] float32
    boundaries = runSegmenterOnce(
        x_seg,
        language   = [languageID, ...],   // [B] int64
        known_boundaries = knownBoundaries,
        prev_boundaries  = boundaries,    // previous iteration output
        t          = t_arr,               // [B] float32
        maskT      = maskT,
        threshold  = scalar float32,
        radius     = scalar int64,
    )

// Non-loop branch (config.loop == false):
boundaries = runSegmenterOnce(
    x_seg,
    language         = [languageID, ...],
    known_boundaries = knownBoundaries,
    prev_boundaries  = zeros([B, T], bool),  // NOT knownBoundaries
    t                = zeros([B], float32),
    maskT            = maskT,
    threshold        = scalar float32,
    radius           = scalar int64,
)
```

Time schedule:

```go
// Mirrors ValidationConfig.d3pm_sample_ts_resolved
func resolveTs(t0 float32, nsteps int, custom []float32) []float32
// Default (t0=0.0, nsteps=8): [0.000, 0.125, 0.250, 0.375, 0.500, 0.625, 0.750, 0.875]
```

Segmenter input shapes:

| Name | Shape | dtype |
|---|---|---|
| `x_seg` | `[B, T, 128]` | float32 |
| `language` | `[B]` | int64 |
| `known_boundaries` | `[B, T]` | bool |
| `prev_boundaries` | `[B, T]` | bool |
| `t` | `[B]` | float32 |
| `maskT` | `[B, T]` | bool |
| `threshold` | `[]` (scalar) | float32 |
| `radius` | `[]` (scalar) | int64 |

> Note: `language` and `t` and `prev_boundaries` are omitted from the feed if the model's config indicates they are not present. Use the segmenter's actual input name list at load time to conditionally include them (mirrors `sess._seg_input_names` in Python).

### 3.4 Note Timestamp Reconstruction

After `RunBd2Dur` + `RunEstimator`, each item `i` in the batch produces:

```go
// noteDurations[i]: []float32 of valid note durations (masked by maskN[i])
// presence[i]:      []bool
// scores[i]:        []float32  (MIDI pitch, A4=69)
// chunkOffset[i]:   float64    (seconds from start of original file)

onsetAcc := 0.0
for j := range noteDurations[i] {
    onset  := onsetAcc + chunkOffset[i]
    offset := onsetAcc + float64(noteDurations[i][j]) + chunkOffset[i]
    onsetAcc += float64(noteDurations[i][j])
    if offset <= onset { continue }
    if !presence[i][j]  { continue }
    allNotes = append(allNotes, NoteInfo{onset, offset, scores[i][j]})
}
```

### 3.5 Note Post-Processing (`mergeAndSortNotes`)

Applied once per file after all chunks are processed. Mirrors `_merge_and_sort_notes` in `infer_onnx.py`:

1. Initialise `lastTime = 0.0`
2. Sort by `(onset, offset, pitch)`
3. For each note in order: clamp `onset = max(onset, lastTime)`, clamp `offset = max(offset, onset)`
4. Drop notes where `offset <= onset`
5. Set `lastTime = offset`

### 3.6 `dur2bd` Model

`dur2bd.onnx` is loaded in `OnnxSessions` for completeness but is **not called** in the `extract` or `align` inference paths. It is available for potential future use (e.g., a `refine` command that accepts known word durations and converts them to boundary masks). No pipeline step currently invokes it.

---

## 4. GPU Support

Controlled by `--device` flag:

| `--device` | Platform | Execution Provider |
|---|---|---|
| `cpu` | all | CPUExecutionProvider (default) |
| `coreml` | macOS arm64 | CoreMLExecutionProvider |
| `cuda` | Windows x64 | CUDAExecutionProvider |

onnxruntime automatically falls back to CPU for any operator the chosen EP does not support.

---

## 5. CLI Interface

### 5.1 `extract`

```
game-infer extract <path> -d <onnx-dir> [flags]

  <path>  Audio file or directory.

Flags:
  -d, --onnx-dir       ONNX model directory (with config.json)  [required]
  -l, --language       Language code: zh, en, ja, yue
      --device         cpu | coreml | cuda              [default: cpu]
      --batch-size     Chunks per inference batch        [default: 4]
      --seg-threshold  Boundary decoding threshold       [default: 0.2]
      --seg-radius     Boundary decoding radius in seconds, converted to frames
                       via round(seg_radius / timestep); default 0.02s → 2 frames
                       [default: 0.02]
      --t0             D3PM start T                      [default: 0.0]
      --nsteps         D3PM steps                        [default: 8]
      --ts             Custom T values, comma-separated
      --est-threshold  Note presence threshold           [default: 0.2]
      --output-formats mid,txt,csv                       [default: mid]
      --tempo          MIDI BPM                          [default: 120]
      --pitch-format   number | name                     [default: name]
      --round-pitch    Round pitch to nearest semitone
      --output-dir     Output directory (default: same as input)
      --input-formats  Extensions when PATH is a dir     [default: wav]
      --glob           Glob pattern for file filtering in directory mode
```

### 5.2 `align`

All `extract` flags, plus:

```
  <paths...>  One or more transcription CSV files (or glob patterns).

      --save-path      Full output file path (single-input only; overrides --save-name)
      --save-name      Output filename (used when processing multiple files)
      --overwrite      Overwrite input file in place (requires explicit flag)
      --uv-vocab       Unvoiced phoneme list             [default: AP,SP,br,sil]
      --uv-vocab-path  Path to file of unvoiced phonemes (overrides --uv-vocab)
      --uv-word-cond   lead | all                        [default: lead]
      --uv-note-cond   predict | follow                  [default: predict]
      --no-wb          Disable word-note alignment
```

`align` accepts multiple path arguments and glob patterns, matching `infer.py align` behavior.

---

## 6. Error Handling

| Scenario | Behavior |
|---|---|
| onnxruntime library not found | Exit immediately, print: "Place libonnxruntime.dylib / onnxruntime.dll in the same directory as game-infer" |
| `config.json` missing or malformed | Exit immediately, print path |
| Non-PCM WAV format | Skip file, print warning, continue |
| WAV sample rate mismatch | Resample to 44100 Hz using linear interpolation, print info |
| Stereo / multi-channel WAV | Mix down to mono silently |
| ONNX runtime error in a chunk | Skip chunk, print error with chunk index, continue |
| Output directory not writable | Exit immediately before processing |

**Principle:** single-file failure never aborts a directory run. All skips are logged.

### 6.1 Progress Output Format

```
Loading ONNX models from: checkpoints/GAME-1.0.3-small-onnx
  device=coreml  samplerate=44100  loop=true
[1/3] Processing: song1.wav
  chunks=4  notes=52  → song1.mid
[2/3] Processing: song2.wav  [WARN: skipped chunk 2: runtime error]
  chunks=3  notes=31  → song2.mid
[3/3] Processing: song3.wav
  chunks=6  notes=78  → song3.mid
Done. 3 files, 161 notes total.
```

---

## 7. MIDI Output Format

MIDI tick formula (mirrors `infer_onnx.py`):

```
ticks_per_beat = 480   (standard MIDI default)
onset_tick  = round(onset_sec  * tempo_bpm / 60 * ticks_per_beat)
offset_tick = round(offset_sec * tempo_bpm / 60 * ticks_per_beat)
```

At default 120 BPM: `tick = round(seconds * 120 / 60 * 480) = round(seconds * 960)`.

This matches the Python: `round(onset * tempo * 8)` at 120 BPM = `round(onset * 960)`.

Pitch is clamped to `[0, 127]` and rounded to the nearest integer.

---

## 8. Distribution

### 8.1 Release Artifacts

```
game-infer-macos-arm64.zip
  ├── game-infer
  └── libonnxruntime.dylib              (~15 MB)

game-infer-windows-x64-cpu.zip
  ├── game-infer.exe
  └── onnxruntime.dll                   (~15 MB)

game-infer-windows-x64-cuda.zip        (optional)
  ├── game-infer.exe
  ├── onnxruntime_gpu.dll               (~100 MB+)
  └── README-cuda.txt                   # notes that CUDA Toolkit must be installed separately
```

> **CUDA note:** `onnxruntime_gpu.dll` requires CUDA runtime DLLs (`cudart64_*.dll`, `cublas64_*.dll`, `cublasLt64_*.dll`) which are part of the NVIDIA CUDA Toolkit. These are not bundled due to size and licensing; the README instructs users to install the CUDA Toolkit separately.

### 8.2 Build Matrix

| Target | GOOS | GOARCH | Build environment |
|---|---|---|---|
| macOS arm64 | `darwin` | `arm64` | macOS arm64 runner |
| Windows x64 | `windows` | `amd64` | Windows x64 runner |

CGo requires a native or cross-compile toolchain. Recommended: build on each target platform (or GitHub Actions matrix).

---

## 9. Testing Strategy

- **`audio/wav_test.go`**: verify stereo mix-down, sample rate resampling, PCM decoding.
- **`audio/slicer_test.go`**: verify slice boundaries on synthetic silence-padded WAV.
- **`onnx/infer_test.go`**: end-to-end test on `sample/generated_zh.wav` — compare note count and pitch range against known-good output from `infer_onnx.py`.
- **Manual**: diff MIDI output of `game-infer extract` vs `infer_onnx.py extract` on the same file.
