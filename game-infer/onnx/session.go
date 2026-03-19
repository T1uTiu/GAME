package onnx

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	ort "github.com/yalue/onnxruntime_go"
)

// Config holds values from config.json alongside the ONNX models.
type Config struct {
	Samplerate   int            `json:"samplerate"`
	Timestep     float64        `json:"timestep"`
	Languages    map[string]int `json:"languages"`
	Loop         bool           `json:"loop"`
	EmbeddingDim int            `json:"embedding_dim"`
}

// Sessions holds all five ONNX sessions.
type Sessions struct {
	Config        Config
	Encoder       *ort.DynamicAdvancedSession
	Segmenter     *ort.DynamicAdvancedSession
	Estimator     *ort.DynamicAdvancedSession
	Bd2Dur        *ort.DynamicAdvancedSession
	Dur2Bd        *ort.DynamicAdvancedSession
	SegInputNames []string // cached at load: used to conditionally include optional inputs
}

// Load reads config.json and opens all ONNX sessions from onnxDir.
// device: "cpu", "coreml", or "cuda".
func Load(onnxDir, device string) (*Sessions, error) {
	raw, err := os.ReadFile(filepath.Join(onnxDir, "config.json"))
	if err != nil {
		return nil, fmt.Errorf("config.json: %w", err)
	}
	var cfg Config
	if err := json.Unmarshal(raw, &cfg); err != nil {
		return nil, fmt.Errorf("parse config.json: %w", err)
	}

	opts, err := buildOptions(device)
	if err != nil {
		return nil, err
	}

	load := func(name string, inNames, outNames []string) (*ort.DynamicAdvancedSession, error) {
		data, err := os.ReadFile(filepath.Join(onnxDir, name+".onnx"))
		if err != nil {
			return nil, fmt.Errorf("read %s.onnx: %w", name, err)
		}
		s, err := ort.NewDynamicAdvancedSessionWithONNXData(data, inNames, outNames, opts)
		if err != nil {
			return nil, fmt.Errorf("load %s.onnx: %w", name, err)
		}
		return s, nil
	}

	s := &Sessions{Config: cfg}

	// Encoder: always the same inputs/outputs
	if s.Encoder, err = load("encoder",
		[]string{"waveform", "duration"},
		[]string{"x_seg", "x_est", "maskT"},
	); err != nil {
		return nil, err
	}

	// Segmenter: input names depend on config (loop=d3pm, use_languages)
	// We build the full name list and cache it; the runner skips unsupported ones.
	segIn := []string{"x_seg"}
	if cfg.Languages != nil {
		segIn = append(segIn, "language")
	}
	segIn = append(segIn, "known_boundaries")
	if cfg.Loop {
		segIn = append(segIn, "prev_boundaries", "t")
	}
	segIn = append(segIn, "maskT", "threshold", "radius")
	s.SegInputNames = segIn

	if s.Segmenter, err = load("segmenter", segIn, []string{"boundaries"}); err != nil {
		return nil, err
	}
	if s.Estimator, err = load("estimator",
		[]string{"x_est", "boundaries", "maskT", "maskN", "threshold"},
		[]string{"presence", "scores"},
	); err != nil {
		return nil, err
	}
	if s.Bd2Dur, err = load("bd2dur",
		[]string{"boundaries", "maskT"},
		[]string{"durations", "maskN"},
	); err != nil {
		return nil, err
	}
	if s.Dur2Bd, err = load("dur2bd",
		[]string{"durations", "maskT"},
		[]string{"boundaries"},
	); err != nil {
		return nil, err
	}

	return s, nil
}

// GetLanguageID returns the integer ID for a language code (0 = unknown/universal).
func (s *Sessions) GetLanguageID(lang string) int {
	if lang == "" || s.Config.Languages == nil {
		return 0
	}
	return s.Config.Languages[lang]
}

func buildOptions(device string) (*ort.SessionOptions, error) {
	opts, err := ort.NewSessionOptions()
	if err != nil {
		return nil, err
	}
	switch device {
	case "coreml":
		if err := opts.AppendExecutionProviderCoreML(0); err != nil {
			return nil, fmt.Errorf("coreml EP: %w", err)
		}
	case "cuda":
		cudaOpts, _ := ort.NewCUDAProviderOptions()
		if err := opts.AppendExecutionProviderCUDA(cudaOpts); err != nil {
			return nil, fmt.Errorf("cuda EP: %w", err)
		}
	}
	return opts, nil
}
