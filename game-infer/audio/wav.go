package audio

import (
	"fmt"
	"math"
	"os"

	gowav "github.com/go-audio/wav"
)

// LoadWAV loads a PCM WAV, mixes down to mono, and resamples to targetSR.
// Returns float32 samples normalised to [-1, 1].
func LoadWAV(path string, targetSR int) ([]float32, error) {
	f, err := os.Open(path)
	if err != nil { return nil, fmt.Errorf("open wav %s: %w", path, err) }
	defer f.Close()

	dec := gowav.NewDecoder(f)
	if !dec.IsValidFile() {
		return nil, fmt.Errorf("invalid or non-PCM WAV: %s", path)
	}
	if dec.WavAudioFormat != 1 {
		return nil, fmt.Errorf("non-PCM WAV (format %d): %s", dec.WavAudioFormat, path)
	}

	buf, err := dec.FullPCMBuffer()
	if err != nil { return nil, fmt.Errorf("decode %s: %w", path, err) }
	if buf == nil || buf.Format == nil {
		return nil, fmt.Errorf("empty wav: %s", path)
	}

	srcSR  := buf.Format.SampleRate
	numCh  := buf.Format.NumChannels
	nFrames := len(buf.Data) / numCh

	// Mix down to mono, normalising to [-1, 1]
	bitDepth := int(dec.BitDepth)
	var mono []float32
	if bitDepth == 8 {
		// 8-bit PCM is unsigned: 0–255, center at 128
		mono = make([]float32, nFrames)
		for i := 0; i < nFrames; i++ {
			var sum float32
			for c := 0; c < numCh; c++ {
				sum += (float32(buf.Data[i*numCh+c]) - 128.0) / 128.0
			}
			mono[i] = sum / float32(numCh)
		}
	} else {
		// 16-bit and higher: signed, center at 0
		scale := float32(1.0 / math.Pow(2, float64(bitDepth-1)))
		mono = make([]float32, nFrames)
		for i := 0; i < nFrames; i++ {
			var sum float32
			for c := 0; c < numCh; c++ {
				sum += float32(buf.Data[i*numCh+c]) * scale
			}
			mono[i] = sum / float32(numCh)
		}
	}

	if srcSR == targetSR {
		return mono, nil
	}
	return resample(mono, srcSR, targetSR), nil
}

// resample performs linear interpolation resampling.
func resample(samples []float32, srcSR, dstSR int) []float32 {
	ratio := float64(srcSR) / float64(dstSR)
	dstLen := int(math.Round(float64(len(samples)) / ratio))
	out := make([]float32, dstLen)
	for i := range out {
		pos := float64(i) * ratio
		lo := int(pos)
		frac := float32(pos - float64(lo))
		if lo+1 >= len(samples) {
			out[i] = samples[len(samples)-1]
		} else {
			out[i] = samples[lo]*(1-frac) + samples[lo+1]*frac
		}
	}
	return out
}
