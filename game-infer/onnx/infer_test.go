package onnx_test

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"

	ort "github.com/yalue/onnxruntime_go"
	"github.com/openvpi/game-infer/audio"
	"github.com/openvpi/game-infer/onnx"
	"github.com/openvpi/game-infer/output"
)

func TestEndToEnd_ExtractSample(t *testing.T) {
	libPath, err := findOnnxLib()
	if err != nil {
		t.Skipf("onnxruntime lib not found, skipping: %v", err)
	}

	ort.SetSharedLibraryPath(libPath)
	if err := ort.InitializeEnvironment(); err != nil {
		t.Fatalf("ort init: %v", err)
	}
	defer ort.DestroyEnvironment()

	// Paths relative to game-infer/onnx/ (test runs from package dir)
	repoRoot := "../../"
	onnxDir := filepath.Join(repoRoot, "checkpoints/GAME-1.0.3-small-onnx")
	wavPath := filepath.Join(repoRoot, "sample/generated_zh.wav")

	sess, err := onnx.Load(onnxDir, "cpu")
	if err != nil {
		t.Fatalf("load sessions: %v", err)
	}

	waveform, err := audio.LoadWAV(wavPath, sess.Config.Samplerate)
	if err != nil {
		t.Fatalf("load wav: %v", err)
	}

	chunks := audio.Slice(waveform, sess.Config.Samplerate)
	params := onnx.InferParams{
		LanguageID:   0,
		SegThreshold: 0.2,
		SegRadius:    2,
		EstThreshold: 0.2,
		T0:           0.0,
		NSteps:       8,
	}

	var allNotes []output.NoteInfo
	for bStart := 0; bStart < len(chunks); bStart += 4 {
		bEnd := bStart + 4
		if bEnd > len(chunks) {
			bEnd = len(chunks)
		}
		batch := chunks[bStart:bEnd]
		waveforms := make([][]float32, len(batch))
		for i, c := range batch {
			waveforms[i] = c.Waveform
		}

		results, err := onnx.InferBatch(sess, waveforms, params)
		if err != nil {
			t.Fatalf("infer batch %d: %v", bStart, err)
		}
		for i, res := range results {
			off := batch[i].Offset
			for j := range res {
				res[j].Onset += off
				res[j].Offset += off
			}
			allNotes = append(allNotes, res...)
		}
	}
	allNotes = output.MergeAndSortNotes(allNotes)

	t.Logf("Total notes: %d", len(allNotes))
	for i, n := range allNotes {
		t.Logf("note[%d]: onset=%.3f offset=%.3f pitch=%.2f", i, n.Onset, n.Offset, n.Pitch)
	}

	// Ground truth: Python infer_onnx.py → 18 notes, first note C#3 (MIDI ~49)
	if len(allNotes) != 18 {
		t.Errorf("want 18 notes, got %d", len(allNotes))
	}
	if len(allNotes) > 0 {
		first := allNotes[0]
		if first.Pitch < 48 || first.Pitch > 51 {
			t.Errorf("first note pitch want ~49 (C#3), got %.2f", first.Pitch)
		}
	}
}

func findOnnxLib() (string, error) {
	candidates := []string{
		"../libonnxruntime.dylib",
		"../../libonnxruntime.dylib",
		"libonnxruntime.dylib",
		"/opt/homebrew/lib/libonnxruntime.dylib",
	}
	for _, p := range candidates {
		if _, err := os.Stat(p); err == nil {
			return filepath.Abs(p)
		}
	}
	return "", fmt.Errorf("libonnxruntime.dylib not found")
}
