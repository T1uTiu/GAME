package output

import "sort"

// NoteInfo holds a single detected note.
type NoteInfo struct {
	Onset  float64 // seconds
	Offset float64 // seconds
	Pitch  float32 // MIDI pitch (A4=69), fractional
}

// ReconstructTimestamps converts per-note durations + chunk offset to absolute timestamps.
// Unvoiced (presence=false) and zero-duration notes are dropped.
func ReconstructTimestamps(durations []float32, presence []bool, scores []float32, chunkOffset float64) []NoteInfo {
	var notes []NoteInfo
	onsetAcc := 0.0
	for j := range durations {
		onset  := onsetAcc + chunkOffset
		offset := onsetAcc + float64(durations[j]) + chunkOffset
		onsetAcc += float64(durations[j])
		if offset <= onset { continue }
		if !presence[j]   { continue }
		notes = append(notes, NoteInfo{Onset: onset, Offset: offset, Pitch: scores[j]})
	}
	return notes
}

// MergeAndSortNotes sorts notes, clamps overlaps, and drops zero-duration notes.
// Mirrors _merge_and_sort_notes in infer_onnx.py.
func MergeAndSortNotes(notes []NoteInfo) []NoteInfo {
	sort.Slice(notes, func(i, j int) bool {
		if notes[i].Onset  != notes[j].Onset  { return notes[i].Onset  < notes[j].Onset }
		if notes[i].Offset != notes[j].Offset { return notes[i].Offset < notes[j].Offset }
		return notes[i].Pitch < notes[j].Pitch
	})
	lastTime := 0.0
	out := notes[:0]
	for _, n := range notes {
		if n.Onset  < lastTime { n.Onset  = lastTime }
		if n.Offset < n.Onset  { n.Offset = n.Onset  }
		if n.Offset <= n.Onset { continue }
		out = append(out, n)
		lastTime = n.Offset
	}
	return out
}
