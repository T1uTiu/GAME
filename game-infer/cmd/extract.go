package cmd

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/spf13/cobra"
	"github.com/openvpi/game-infer/audio"
	"github.com/openvpi/game-infer/onnx"
	"github.com/openvpi/game-infer/output"
)

var (
	extractFlags         InferFlags
	extractOutputFormats string
	extractTempo         float64
	extractPitchFormat   string
	extractRoundPitch    bool
	extractOutputDir     string
	extractInputFormats  string
	extractGlob          string
)

var extractCmd = &cobra.Command{
	Use:   "extract <path>",
	Short: "Extract MIDI from audio file(s)",
	Args:  cobra.ExactArgs(1),
	RunE:  runExtract,
}

func init() {
	rootCmd.AddCommand(extractCmd)
	f := extractCmd.Flags()
	f.StringVarP(&extractFlags.OnnxDir,      "onnx-dir",       "d", "",     "ONNX model directory [required]")
	f.StringVarP(&extractFlags.Language,     "language",       "l", "",     "Language code (zh, en, ja, yue)")
	f.StringVar( &extractFlags.Device,       "device",              "cpu",  "cpu | coreml | cuda")
	f.IntVar(    &extractFlags.BatchSize,    "batch-size",          4,      "Chunks per batch")
	f.Float64Var(&extractFlags.SegThreshold, "seg-threshold",        0.2,   "Boundary decoding threshold")
	f.Float64Var(&extractFlags.SegRadius,    "seg-radius",           0.02,  "Boundary radius (seconds)")
	f.Float64Var(&extractFlags.T0,           "t0",                   0.0,   "D3PM start T")
	f.IntVar(    &extractFlags.NSteps,       "nsteps",               8,     "D3PM steps")
	f.StringVar( &extractFlags.CustomTs,     "ts",                   "",    "Custom T values, comma-separated")
	f.Float64Var(&extractFlags.EstThreshold, "est-threshold",        0.2,   "Note presence threshold")
	f.StringVar( &extractOutputFormats,      "output-formats",       "mid", "mid,txt,csv")
	f.Float64Var(&extractTempo,              "tempo",                120.0, "MIDI BPM")
	f.StringVar( &extractPitchFormat,        "pitch-format",         "name","number | name")
	f.BoolVar(   &extractRoundPitch,         "round-pitch",          false, "Round pitch to semitone")
	f.StringVar( &extractOutputDir,          "output-dir",           "",    "Output directory")
	f.StringVar( &extractInputFormats,       "input-formats",        "wav", "Audio extensions for directory mode")
	f.StringVar( &extractGlob,               "glob",                 "",    "Glob pattern for directory filtering")
	_ = extractCmd.MarkFlagRequired("onnx-dir")
}

func runExtract(_ *cobra.Command, args []string) error {
	inputPath := args[0]
	exts := parseExts(extractInputFormats)
	filemap, err := collectFiles(inputPath, exts, extractGlob)
	if err != nil { return err }

	outDir := extractOutputDir
	if outDir == "" {
		info, _ := os.Stat(inputPath)
		if info.IsDir() { outDir = inputPath } else { outDir = filepath.Dir(inputPath) }
	}
	if err := os.MkdirAll(outDir, 0o755); err != nil { return err }

	fmt.Printf("Loading ONNX models from: %s\n", extractFlags.OnnxDir)
	sess, err := onnx.Load(extractFlags.OnnxDir, extractFlags.Device)
	if err != nil { return fmt.Errorf("load onnx: %w", err) }
	fmt.Printf("  device=%s  samplerate=%d  loop=%v\n",
		extractFlags.Device, sess.Config.Samplerate, sess.Config.Loop)

	customTs, err := ParseCustomTs(extractFlags.CustomTs)
	if err != nil { return err }

	params := onnx.InferParams{
		LanguageID:   sess.GetLanguageID(extractFlags.Language),
		SegThreshold: float32(extractFlags.SegThreshold),
		SegRadius:    SegRadiusFrames(extractFlags.SegRadius, sess.Config.Timestep),
		EstThreshold: float32(extractFlags.EstThreshold),
		T0:           float32(extractFlags.T0),
		NSteps:       extractFlags.NSteps,
		CustomTs:     customTs,
	}

	formats := parseFormats(extractOutputFormats)

	// Sort keys for deterministic progress output
	keys := make([]string, 0, len(filemap))
	for k := range filemap { keys = append(keys, k) }
	sort.Strings(keys)

	total := len(keys)
	totalNotes := 0

	for fileIdx, key := range keys {
		fpath := filemap[key]
		fmt.Printf("[%d/%d] Processing: %s\n", fileIdx+1, total, key)

		waveform, err := audio.LoadWAV(fpath, sess.Config.Samplerate)
		if err != nil {
			fmt.Fprintf(os.Stderr, "  [WARN] skip %s: %v\n", key, err)
			continue
		}
		chunks := audio.Slice(waveform, sess.Config.Samplerate)

		var allNotes []output.NoteInfo
		for bStart := 0; bStart < len(chunks); bStart += extractFlags.BatchSize {
			bEnd := bStart + extractFlags.BatchSize
			if bEnd > len(chunks) { bEnd = len(chunks) }
			batch := chunks[bStart:bEnd]

			waveforms := make([][]float32, len(batch))
			for i, c := range batch { waveforms[i] = c.Waveform }

			results, err := onnx.InferBatch(sess, waveforms, params)
			if err != nil {
				fmt.Fprintf(os.Stderr, "  [WARN] skipped batch starting at chunk %d: %v\n", bStart, err)
				continue
			}
			for i, res := range results {
				off := batch[i].Offset
				for j := range res {
					res[j].Onset  += off
					res[j].Offset += off
				}
				allNotes = append(allNotes, res...)
			}
		}

		allNotes = output.MergeAndSortNotes(allNotes)
		totalNotes += len(allNotes)
		stem := strings.TrimSuffix(key, filepath.Ext(key))

		if formats["mid"] {
			p := filepath.Join(outDir, stem+".mid")
			if err := output.WriteMIDI(allNotes, p, extractTempo); err != nil {
				fmt.Fprintf(os.Stderr, "  [WARN] midi write: %v\n", err)
			}
		}
		if formats["txt"] {
			p := filepath.Join(outDir, stem+".txt")
			_ = output.WriteTXT(allNotes, p, extractPitchFormat, extractRoundPitch)
		}
		if formats["csv"] {
			p := filepath.Join(outDir, stem+".csv")
			_ = output.WriteCSV(allNotes, p, extractPitchFormat, extractRoundPitch)
		}
		fmt.Printf("  chunks=%d  notes=%d\n", len(chunks), len(allNotes))
	}
	fmt.Printf("Done. %d files, %d notes total.\n", total, totalNotes)
	return nil
}

func parseExts(s string) map[string]bool {
	m := make(map[string]bool)
	for _, e := range strings.Split(s, ",") {
		e = strings.TrimSpace(e)
		if !strings.HasPrefix(e, ".") { e = "." + e }
		m[strings.ToLower(e)] = true
	}
	return m
}

func parseFormats(s string) map[string]bool {
	m := make(map[string]bool)
	for _, f := range strings.Split(s, ",") { m[strings.TrimSpace(f)] = true }
	return m
}

func collectFiles(root string, exts map[string]bool, glob string) (map[string]string, error) {
	info, err := os.Stat(root)
	if err != nil { return nil, err }
	m := make(map[string]string)
	if !info.IsDir() {
		m[filepath.Base(root)] = root
		return m, nil
	}
	err = filepath.WalkDir(root, func(p string, d os.DirEntry, e error) error {
		if e != nil || d.IsDir() { return e }
		if glob != "" {
			matched, _ := filepath.Match(glob, d.Name())
			if !matched { return nil }
		} else if !exts[strings.ToLower(filepath.Ext(p))] {
			return nil
		}
		rel, _ := filepath.Rel(root, p)
		m[rel] = p
		return nil
	})
	if len(m) == 0 { return nil, fmt.Errorf("no audio files found in %s", root) }
	return m, err
}
