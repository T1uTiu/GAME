package output_test

import (
	"os"
	"testing"
	"github.com/openvpi/game-infer/output"
)

func TestWriteMIDI_CreatesFile(t *testing.T) {
	notes := []output.NoteInfo{
		{Onset: 0.15, Offset: 0.43, Pitch: 49.43},
		{Onset: 0.43, Offset: 0.67, Pitch: 51.98},
	}
	path := t.TempDir() + "/test.mid"
	if err := output.WriteMIDI(notes, path, 120); err != nil {
		t.Fatalf("WriteMIDI error: %v", err)
	}
	info, err := os.Stat(path)
	if err != nil { t.Fatalf("file not created: %v", err) }
	if info.Size() < 10 { t.Errorf("MIDI file suspiciously small: %d bytes", info.Size()) }
}
