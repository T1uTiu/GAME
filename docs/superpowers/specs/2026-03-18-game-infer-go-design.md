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

**Audio input:** WAV only (pure-Go decoding, zero external dependencies).

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
│   ├── wav.go               # WAV decode (pure Go, go-audio/wav)
│   └── slicer.go            # Silence-based audio slicer (ported from slicer2.py)
├── mel/
│   └── mel.go               # Mel spectrogram (ported from lib/feature/mel.py)
├── output/
│   ├── midi.go              # MIDI file writer (gomidi)
│   └── text.go              # TXT / CSV writer
├── go.mod
└── go.sum
```

### 2.2 Dependencies

| Package | Purpose |
|---|---|
| `github.com/yalue/onnxruntime_go` | CGo bindings to onnxruntime C API |
| `github.com/spf13/cobra` | CLI framework |
| `github.com/go-audio/wav` | Pure-Go WAV decoding |
| `gonum.org/v1/gonum/dsp/fourier` | FFT for Mel spectrogram |
| `gitlab.com/gomidi/midi/v2` | MIDI file writing |

### 2.3 onnxruntime Library Loading

At startup, `onnx/session.go` calls `onnxruntime_go.SetSharedLibraryPath()` pointing to the `.dylib` / `.dll` located in the same directory as the binary. If the file is not found, the program exits immediately with a clear message.

---

## 3. Inference Pipeline

Mirrors `infer_onnx.py` exactly:

```
WAV file
  └─► audio.LoadWAV()           → []float32 waveform [L]
  └─► audio.Slice()             → []Chunk{waveform, offset}

Per batch:
  mel.Compute(waveform)         → spectrogram [T, 128]   (inside encoder)
  onnx.RunEncoder(waveforms, durations)
    → x_seg [B,T,128], x_est [B,T,128], maskT [B,T]

  onnx.RunSegmenter(...)        D3PM loop over ts[]
    → boundaries [B,T]

  onnx.RunBd2Dur(boundaries, maskT)
    → durations [B,N], maskN [B,N]

  onnx.RunEstimator(x_est, boundaries, maskT, maskN, threshold)
    → presence [B,N], scores [B,N]

results → output.WriteMIDI() / WriteText()
```

### 3.1 Mel Spectrogram (`mel/mel.go`)

Ported from `lib/feature/mel.py` (`StretchableMelSpectrogram`):

1. Hann window, n_fft=2048, hop_length=441
2. STFT via `gonum/fourier`, power spectrum
3. Mel filterbank: 128 bins, fmin=0 Hz, fmax=8000 Hz, HTK scale
4. Log compression: `log(max(x, 1e-5))`
5. Output shape: `[T, 128]`

Correctness verified by unit tests comparing output against Python/librosa reference (tolerance ≤ 1e-4).

### 3.2 D3PM Time Schedule

```go
// Mirrors ValidationConfig.d3pm_sample_ts_resolved
func resolveTs(t0 float32, nsteps int, custom []float32) []float32
// Default: t0=0.0, nsteps=8 → [0.000, 0.125, 0.250, ..., 0.875]
```

### 3.3 Batching

Waveforms within a batch are zero-padded to the same length. `maskT` is derived from each chunk's actual duration, masking padding frames before segmentation and estimation.

---

## 4. GPU Support

Controlled by `--device` flag:

| `--device` | Platform | Execution Provider |
|---|---|---|
| `cpu` | all | CPUExecutionProvider (default) |
| `coreml` | macOS arm64 | CoreMLExecutionProvider |
| `cuda` | Windows x64 | CUDAExecutionProvider |

onnxruntime automatically falls back to CPU for any operator the chosen EP does not support. No user-facing handling is required.

---

## 5. CLI Interface

### 5.1 `extract`

```
game-infer extract <path> -d <onnx-dir> [flags]

Flags:
  -d, --onnx-dir       ONNX model directory (with config.json)  [required]
  -l, --language       Language code: zh, en, ja, yue
      --device         cpu | coreml | cuda              [default: cpu]
      --batch-size     Chunks per inference batch        [default: 4]
      --seg-threshold  Boundary decoding threshold       [default: 0.2]
      --seg-radius     Boundary decoding radius (sec)    [default: 0.02]
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
```

### 5.2 `align`

All `extract` flags, plus:

```
      --save-name      Output filename
      --overwrite      Overwrite input file in place
      --uv-vocab       Unvoiced phoneme list             [default: AP,SP,br,sil]
      --uv-word-cond   lead | all                        [default: lead]
      --uv-note-cond   predict | follow                  [default: predict]
      --no-wb          Disable word-note alignment
```

---

## 6. Error Handling

| Scenario | Behavior |
|---|---|
| onnxruntime library not found | Exit immediately, print path hint |
| `config.json` missing or malformed | Exit immediately, print path |
| Unsupported WAV format (non-PCM) | Skip file, print warning, continue |
| ONNX runtime error in a chunk | Skip chunk, print error, continue |
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

## 7. Distribution

### 7.1 Release Artifacts

```
game-infer-macos-arm64.zip
  ├── game-infer
  └── libonnxruntime.dylib        (~15 MB)

game-infer-windows-x64-cpu.zip
  ├── game-infer.exe
  └── onnxruntime.dll             (~15 MB)

game-infer-windows-x64-cuda.zip  (optional)
  ├── game-infer.exe
  └── onnxruntime_gpu.dll         (~100 MB+)
```

### 7.2 Build Matrix

| Target | GOOS | GOARCH | Build environment |
|---|---|---|---|
| macOS arm64 | `darwin` | `arm64` | macOS arm64 runner |
| Windows x64 | `windows` | `amd64` | Windows x64 runner |

CGo requires a native or cross-compile toolchain. Recommended: build on each target platform (or GitHub Actions matrix).

---

## 8. Testing Strategy

- **`mel/mel_test.go`**: compare Mel spectrogram output against pre-generated NumPy reference arrays (tolerance 1e-4).
- **`audio/slicer_test.go`**: verify slice boundaries on synthetic silence-padded WAV.
- **`onnx/infer_test.go`**: end-to-end test on `sample/generated_zh.wav` against known MIDI note count.
- **Manual**: compare `game-infer extract` output MIDI against `infer_onnx.py extract` output on the same file.
