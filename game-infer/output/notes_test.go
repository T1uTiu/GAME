package output_test

import (
	"testing"
	"github.com/openvpi/game-infer/output"
)

func abs64(x float64) float64 { if x < 0 { return -x }; return x }

func TestReconstructTimestamps(t *testing.T) {
	durs     := []float32{0.5, 0.3}
	presence := []bool{true, true}
	scores   := []float32{60.0, 62.0}

	notes := output.ReconstructTimestamps(durs, presence, scores, 1.0)
	if len(notes) != 2 { t.Fatalf("want 2 notes, got %d", len(notes)) }
	if abs64(notes[0].Onset-1.0)  > 0.001 { t.Errorf("note[0].Onset: want 1.0, got %f", notes[0].Onset) }
	if abs64(notes[0].Offset-1.5) > 0.001 { t.Errorf("note[0].Offset: want 1.5, got %f", notes[0].Offset) }
	if abs64(notes[1].Onset-1.5)  > 0.001 { t.Errorf("note[1].Onset: want 1.5, got %f", notes[1].Onset) }
	if abs64(notes[1].Offset-1.8) > 0.001 { t.Errorf("note[1].Offset: want 1.8, got %f", notes[1].Offset) }
}

func TestReconstructTimestamps_SkipsUnvoiced(t *testing.T) {
	notes := output.ReconstructTimestamps(
		[]float32{0.5, 0.3}, []bool{false, true}, []float32{60, 62}, 0.0,
	)
	if len(notes) != 1 { t.Fatalf("want 1 note (unvoiced skipped), got %d", len(notes)) }
	if abs64(notes[0].Onset-0.5) > 0.001 { t.Errorf("want onset 0.5, got %f", notes[0].Onset) }
}

func TestMergeAndSortNotes(t *testing.T) {
	notes := []output.NoteInfo{
		{Onset: 1.0, Offset: 1.5, Pitch: 60},
		{Onset: 0.5, Offset: 1.2, Pitch: 62}, // overlaps: onset 1.0 clamps to 1.5
		{Onset: 2.0, Offset: 2.0, Pitch: 64}, // zero duration: dropped
	}
	merged := output.MergeAndSortNotes(notes)
	if len(merged) != 2 { t.Fatalf("want 2 notes, got %d", len(merged)) }
	// First note starts at 0.5
	if abs64(merged[0].Onset-0.5) > 0.001 { t.Errorf("first onset want 0.5, got %f", merged[0].Onset) }
	// Second note onset clamped to first note's offset (1.2)
	if merged[1].Onset < merged[0].Offset-0.001 {
		t.Errorf("second onset %f should be >= first offset %f", merged[1].Onset, merged[0].Offset)
	}
}
