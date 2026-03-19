package output

import (
	"encoding/csv"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/openvpi/game-infer/midiutil"
)

func formatPitch(pitch float32, format string, round bool) string {
	if round {
		p := int(math.Round(float64(pitch)))
		if format == "name" { return midiutil.MIDIToNoteName(p) }
		return fmt.Sprintf("%d", p)
	}
	if format == "name" { return midiutil.MIDIToNoteNameCents(pitch) }
	return fmt.Sprintf("%.3f", pitch)
}

// WriteTXT writes onset/offset/pitch as tab-separated lines.
func WriteTXT(notes []NoteInfo, outPath, pitchFormat string, roundPitch bool) error {
	if err := os.MkdirAll(filepath.Dir(outPath), 0o755); err != nil { return err }
	f, err := os.Create(outPath)
	if err != nil { return err }
	defer f.Close()
	for _, n := range notes {
		fmt.Fprintf(f, "%.3f\t%.3f\t%s\n", n.Onset, n.Offset, formatPitch(n.Pitch, pitchFormat, roundPitch))
	}
	return nil
}

// WriteCSV writes onset/offset/pitch as CSV with header row.
func WriteCSV(notes []NoteInfo, outPath, pitchFormat string, roundPitch bool) error {
	if err := os.MkdirAll(filepath.Dir(outPath), 0o755); err != nil { return err }
	f, err := os.Create(outPath)
	if err != nil { return err }
	defer f.Close()
	w := csv.NewWriter(f)
	_ = w.Write([]string{"onset", "offset", "pitch"})
	for _, n := range notes {
		_ = w.Write([]string{
			fmt.Sprintf("%.3f", n.Onset),
			fmt.Sprintf("%.3f", n.Offset),
			formatPitch(n.Pitch, pitchFormat, roundPitch),
		})
	}
	w.Flush()
	return w.Error()
}
