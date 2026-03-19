package cmd

import (
	"encoding/csv"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/spf13/cobra"
	"github.com/openvpi/game-infer/align"
	"github.com/openvpi/game-infer/audio"
	"github.com/openvpi/game-infer/midiutil"
	"github.com/openvpi/game-infer/onnx"
	"github.com/openvpi/game-infer/output"
)

var (
	alignFlags       InferFlags
	alignSavePath    string
	alignSaveName    string
	alignOverwrite   bool
	alignUVVocab     string
	alignUVVocabPath string
	alignUVWordCond  string
	alignUVNoteCond  string
	alignNoWB        bool
)

var alignCmd = &cobra.Command{
	Use:   "align <paths...>",
	Short: "Generate aligned note labels for DiffSinger transcriptions",
	Args:  cobra.MinimumNArgs(1),
	RunE:  runAlign,
}

func init() {
	rootCmd.AddCommand(alignCmd)
	f := alignCmd.Flags()
	f.StringVarP(&alignFlags.OnnxDir,      "onnx-dir",       "d", "",    "ONNX model directory [required]")
	f.StringVarP(&alignFlags.Language,     "language",       "l", "",    "Language code")
	f.StringVar( &alignFlags.Device,       "device",              "cpu", "cpu | coreml | cuda")
	f.IntVar(    &alignFlags.BatchSize,    "batch-size",          4,     "Items per batch")
	f.Float64Var(&alignFlags.SegThreshold, "seg-threshold",        0.2,  "Boundary threshold")
	f.Float64Var(&alignFlags.SegRadius,    "seg-radius",           0.02, "Boundary radius (sec)")
	f.Float64Var(&alignFlags.T0,           "t0",                   0.0,  "D3PM start T")
	f.IntVar(    &alignFlags.NSteps,       "nsteps",               8,    "D3PM steps")
	f.StringVar( &alignFlags.CustomTs,     "ts",                   "",   "Custom T values")
	f.Float64Var(&alignFlags.EstThreshold, "est-threshold",        0.2,  "Note presence threshold")
	f.StringVar( &alignSavePath,           "save-path",            "",   "Output file path (single input)")
	f.StringVar( &alignSaveName,           "save-name",            "",   "Output filename")
	f.BoolVar(   &alignOverwrite,          "overwrite",            false,"Overwrite in place")
	f.StringVar( &alignUVVocab,            "uv-vocab",        "AP,SP,br,sil", "Unvoiced phonemes")
	f.StringVar( &alignUVVocabPath,        "uv-vocab-path",        "",   "File of unvoiced phonemes")
	f.StringVar( &alignUVWordCond,         "uv-word-cond",         "lead","lead | all")
	f.StringVar( &alignUVNoteCond,         "uv-note-cond",         "predict","predict | follow")
	f.BoolVar(   &alignNoWB,               "no-wb",                false,"Disable word-note alignment")
	_ = alignCmd.MarkFlagRequired("onnx-dir")
}

func runAlign(_ *cobra.Command, args []string) error {
	// Resolve input paths (glob expansion)
	var inputPaths []string
	for _, a := range args {
		matches, err := filepath.Glob(a)
		if err != nil || len(matches) == 0 {
			inputPaths = append(inputPaths, a)
		} else {
			inputPaths = append(inputPaths, matches...)
		}
	}
	sort.Strings(inputPaths)

	// UV vocab
	uvVocab := align.ParseUVVocab(alignUVVocab)
	if alignUVVocabPath != "" {
		raw, err := os.ReadFile(alignUVVocabPath)
		if err != nil { return err }
		for _, v := range strings.Fields(string(raw)) { uvVocab[v] = true }
	}
	if alignNoWB { uvVocab = nil }

	fmt.Printf("Loading ONNX models from: %s\n", alignFlags.OnnxDir)
	sess, err := onnx.Load(alignFlags.OnnxDir, alignFlags.Device)
	if err != nil { return err }

	customTs, err := ParseCustomTs(alignFlags.CustomTs)
	if err != nil { return err }

	params := onnx.InferParams{
		LanguageID:   sess.GetLanguageID(alignFlags.Language),
		SegThreshold: float32(alignFlags.SegThreshold),
		SegRadius:    SegRadiusFrames(alignFlags.SegRadius, sess.Config.Timestep),
		EstThreshold: float32(alignFlags.EstThreshold),
		T0:           float32(alignFlags.T0),
		NSteps:       alignFlags.NSteps,
		CustomTs:     customTs,
	}

	for _, indexPath := range inputPaths {
		if err := processAlignIndex(indexPath, sess, params, uvVocab, inputPaths, alignUVNoteCond); err != nil {
			fmt.Fprintf(os.Stderr, "[WARN] %s: %v\n", indexPath, err)
		}
	}
	return nil
}

func processAlignIndex(indexPath string, sess *onnx.Sessions, params onnx.InferParams, uvVocab map[string]bool, allPaths []string, uvNoteCond string) error {
	f, err := os.Open(indexPath)
	if err != nil { return err }
	r := csv.NewReader(f)
	records, err := r.ReadAll()
	f.Close()
	if err != nil { return err }
	if len(records) < 2 { return fmt.Errorf("empty index: %s", indexPath) }

	header := records[0]
	colIdx := func(name string) int {
		for i, h := range header { if h == name { return i } }
		return -1
	}
	nameCol  := colIdx("name")
	phSeqCol := colIdx("ph_seq")
	phDurCol := colIdx("ph_dur")
	phNumCol := colIdx("ph_num")
	if nameCol < 0 { return fmt.Errorf("missing 'name' column in %s", indexPath) }

	wavDir := filepath.Join(filepath.Dir(indexPath), "wavs")

	noteSeqCol := colIdx("note_seq")
	noteDurCol := colIdx("note_dur")

	// If output columns are missing, append them to the header and extend all data rows.
	if noteSeqCol < 0 {
		noteSeqCol = len(header)
		header = append(header, "note_seq")
		records[0] = header
		for i := 1; i < len(records); i++ {
			records[i] = append(records[i], "")
		}
	}
	if noteDurCol < 0 {
		noteDurCol = len(header)
		header = append(header, "note_dur")
		records[0] = header
		for i := 1; i < len(records); i++ {
			records[i] = append(records[i], "")
		}
	}

	for ri, row := range records[1:] {
		name := row[nameCol]

		// Parse phoneme info (if present)
		var wordDur []float64
		if phSeqCol >= 0 && phDurCol >= 0 && phNumCol >= 0 && uvVocab != nil {
			phSeq := strings.Fields(row[phSeqCol])
			phDurStrs := strings.Fields(row[phDurCol])
			phNumStrs := strings.Fields(row[phNumCol])
			phDur := make([]float64, len(phDurStrs))
			phNum := make([]int, len(phNumStrs))
			for i, s := range phDurStrs { phDur[i], _ = strconv.ParseFloat(s, 64) }
			for i, s := range phNumStrs { v, _ := strconv.Atoi(s); phNum[i] = v }
			wordDur, _ = align.ParseWords(phSeq, phDur, phNum, uvVocab, alignUVWordCond, true)
		}
		_ = wordDur // will be used for known_boundaries in future dur2bd integration

		// Load audio
		wavPath := filepath.Join(wavDir, name+".wav")
		waveform, err := audio.LoadWAV(wavPath, sess.Config.Samplerate)
		if err != nil {
			fmt.Fprintf(os.Stderr, "  [WARN] item %s: %v\n", name, err)
			continue
		}

		// Run inference
		results, err := onnx.InferBatch(sess, [][]float32{waveform}, params)
		if err != nil {
			fmt.Fprintf(os.Stderr, "  [WARN] item %s infer: %v\n", name, err)
			continue
		}
		notes := output.MergeAndSortNotes(results[0])

		// Build note_seq, note_dur
		// uv-note-cond=predict: use model output directly
		// uv-note-cond=follow: TODO: requires word-note alignment (align_notes_to_words)
		var noteSeq []string
		var noteDur []string
		for _, n := range notes {
			var pitchName string
			if uvNoteCond == "follow" {
				// follow: would mark notes for unvoiced words as "rest"
				// Requires word-note alignment which is not yet implemented.
				pitchName = midiutil.MIDIToNoteNameCents(n.Pitch)
			} else {
				// predict: use model output directly
				pitchName = midiutil.MIDIToNoteNameCents(n.Pitch)
			}
			noteSeq = append(noteSeq, pitchName)
			noteDur = append(noteDur, fmt.Sprintf("%.3f", n.Offset-n.Onset))
		}

		// Write back to record
		records[ri+1][noteSeqCol] = strings.Join(noteSeq, " ")
		records[ri+1][noteDurCol] = strings.Join(noteDur, " ")
	}

	// Determine output path
	outPath := indexPath
	if alignSavePath != "" && len(allPaths) == 1 { outPath = alignSavePath }
	if alignSaveName != "" { outPath = filepath.Join(filepath.Dir(indexPath), alignSaveName) }
	if !alignOverwrite && outPath == indexPath {
		return fmt.Errorf("use --overwrite to write in place, or provide --save-path / --save-name")
	}

	out, err := os.Create(outPath)
	if err != nil { return err }
	defer out.Close()
	w := csv.NewWriter(out)
	w.WriteAll(records)
	return w.Error()
}
