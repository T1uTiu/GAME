package midiutil

import (
	"fmt"
	"math"
)

var noteNames = []string{"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"}

// MIDIToNoteNameCents formats a fractional MIDI pitch as "C#3+43" or "E3-7".
// Mirrors librosa.midi_to_note(pitch, unicode=False, cents=True).
func MIDIToNoteNameCents(pitch float32) string {
	semitone := int(math.Round(float64(pitch)))
	cents    := int(math.Round((float64(pitch) - float64(semitone)) * 100))
	// Guard: semitone may be negative if pitch is very low
	if semitone < 0 { semitone = 0 }
	octave := semitone/12 - 1
	name   := noteNames[semitone%12]
	if cents >= 0 {
		return fmt.Sprintf("%s%d+%d", name, octave, cents)
	}
	return fmt.Sprintf("%s%d%d", name, octave, cents) // negative already has minus
}

// MIDIToNoteName formats an integer MIDI pitch as "C4", "F#3".
func MIDIToNoteName(midi int) string {
	if midi < 0 { midi = 0 }
	octave := midi/12 - 1
	return fmt.Sprintf("%s%d", noteNames[midi%12], octave)
}
