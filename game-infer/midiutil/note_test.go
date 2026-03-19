package midiutil_test

import (
	"testing"
	"github.com/openvpi/game-infer/midiutil"
)

func TestMIDIToNoteNameCents(t *testing.T) {
	// Reference: Python infer_onnx.py first note pitch → "C#3+43"
	// C#3 = MIDI 49. 49 + 0.43 = 49.43
	got := midiutil.MIDIToNoteNameCents(49.43)
	if got != "C#3+43" {
		t.Errorf("want C#3+43, got %s", got)
	}
	// Negative cents: E3-7 = MIDI 52 - 0.07 = 51.93
	got2 := midiutil.MIDIToNoteNameCents(51.93)
	if got2 != "E3-7" {
		t.Errorf("want E3-7, got %s", got2)
	}
}
