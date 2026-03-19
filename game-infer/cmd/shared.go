package cmd

import (
	"fmt"
	"math"
	"strconv"
	"strings"
)

// InferFlags holds all shared CLI inference flags.
type InferFlags struct {
	OnnxDir      string
	Language     string
	Device       string
	BatchSize    int
	SegThreshold float64
	SegRadius    float64
	T0           float64
	NSteps       int
	CustomTs     string
	EstThreshold float64
}

// ParseCustomTs parses "0.1,0.5,0.9" → []float32.
func ParseCustomTs(s string) ([]float32, error) {
	if s == "" { return nil, nil }
	parts := strings.Split(s, ",")
	out := make([]float32, len(parts))
	for i, p := range parts {
		v, err := strconv.ParseFloat(strings.TrimSpace(p), 32)
		if err != nil { return nil, fmt.Errorf("invalid T value %q: %w", p, err) }
		if v < 0 || v >= 1 { return nil, fmt.Errorf("T value %f out of [0,1)", v) }
		out[i] = float32(v)
	}
	return out, nil
}

// SegRadiusFrames converts seconds → integer frames: round(sec / timestep).
func SegRadiusFrames(radiusSec, timestep float64) int64 {
	return int64(math.Round(radiusSec / timestep))
}
