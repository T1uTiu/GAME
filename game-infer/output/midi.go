package output

import (
	"fmt"
	"math"
	"os"
	"path/filepath"

	"gitlab.com/gomidi/midi/v2"
	"gitlab.com/gomidi/midi/v2/smf"
)

// WriteMIDI writes notes to a Standard MIDI File.
// Tick formula: round(sec * tempoBPM / 60 * 480). Matches Python: round(sec * tempo * 8) at 120 BPM.
func WriteMIDI(notes []NoteInfo, outPath string, tempoBPM float64) error {
	if err := os.MkdirAll(filepath.Dir(outPath), 0o755); err != nil {
		return fmt.Errorf("mkdir: %w", err)
	}

	const tpb = smf.MetricTicks(480)
	tickOf := func(sec float64) uint32 {
		return uint32(math.Round(sec * tempoBPM / 60.0 * 480.0))
	}

	var tr smf.Track
	tr.Add(0, smf.MetaTempo(tempoBPM))

	lastTick := uint32(0)
	for _, n := range notes {
		pitch := int(math.Round(float64(n.Pitch)))
		if pitch < 0   { pitch = 0 }
		if pitch > 127 { pitch = 127 }
		onTick  := tickOf(n.Onset)
		offTick := tickOf(n.Offset)
		if offTick <= onTick { continue }

		tr.Add(onTick-lastTick,  midi.NoteOn(0, uint8(pitch), 80))
		tr.Add(offTick-onTick,   midi.NoteOff(0, uint8(pitch)))
		lastTick = offTick
	}
	tr.Close(0)

	s := smf.New()
	s.TimeFormat = tpb
	if err := s.Add(tr); err != nil {
		return fmt.Errorf("smf.Add: %w", err)
	}
	return s.WriteFile(outPath)
}
