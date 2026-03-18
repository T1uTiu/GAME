package audio

import "math"

// Chunk is a non-silent audio segment with its start time in the original file.
type Chunk struct {
	Waveform []float32
	Offset   float64 // seconds from start of original waveform
}

// Slice splits waveform on silence. Ported from slicer2.py with fixed parameters.
func Slice(waveform []float32, sr int) []Chunk {
	const (
		thresholdDB   = -40.0
		minLengthMS   = 1000
		minIntervalMS = 200
		maxSilKeptMS  = 100
		hopSizeMS     = 20
	)
	threshold   := float32(math.Pow(10, thresholdDB/20.0))
	hopSize     := sr * hopSizeMS / 1000
	winSize     := hopSize * 4
	minLength   := sr * minLengthMS / 1000 / hopSize
	minInterval := sr * minIntervalMS / 1000 / hopSize
	maxSilKept  := sr * maxSilKeptMS / 1000 / hopSize

	rms := computeFrameRMS(waveform, hopSize, winSize)

	isSilence := make([]bool, len(rms))
	for i, r := range rms { isSilence[i] = r < threshold }

	chunks := buildChunks(waveform, isSilence, sr, hopSize, minLength, minInterval, maxSilKept)
	if len(chunks) == 0 {
		return []Chunk{{Waveform: waveform, Offset: 0}}
	}
	return chunks
}

func computeFrameRMS(y []float32, hopSize, winSize int) []float32 {
	half := winSize / 2
	padded := make([]float32, half+len(y)+half)
	copy(padded[half:], y)
	nFrames := (len(padded)-winSize)/hopSize + 1
	rms := make([]float32, nFrames)
	for i := range rms {
		start := i * hopSize
		var sumSq float64
		for j := start; j < start+winSize && j < len(padded); j++ {
			sumSq += float64(padded[j]) * float64(padded[j])
		}
		rms[i] = float32(math.Sqrt(sumSq / float64(winSize)))
	}
	return rms
}

func silKept(silLen, maxSilKept int) int {
	if silLen/2 < maxSilKept { return silLen / 2 }
	return maxSilKept
}

func buildChunks(waveform []float32, isSilence []bool, sr, hopSize, minLength, minInterval, maxSilKept int) []Chunk {
	type rng struct{ start, end int } // in frame indices
	var ranges []rng

	silStart  := -1
	chunkStart := 0

	for i, silent := range isSilence {
		if silent {
			if silStart < 0 { silStart = i }
		} else {
			if silStart >= 0 {
				silLen := i - silStart
				if silLen >= minInterval {
					end := silStart + silKept(silLen, maxSilKept)
					if end-chunkStart >= minLength {
						ranges = append(ranges, rng{chunkStart, end})
					}
					chunkStart = i - silKept(silLen, maxSilKept)
				}
				silStart = -1
			}
		}
	}
	// Final chunk
	if len(isSilence)-chunkStart >= minLength {
		ranges = append(ranges, rng{chunkStart, len(isSilence)})
	}

	chunks := make([]Chunk, 0, len(ranges))
	for _, r := range ranges {
		startS := r.start * hopSize
		endS   := r.end   * hopSize
		if endS > len(waveform) { endS = len(waveform) }
		if startS >= endS { continue }
		wav := make([]float32, endS-startS)
		copy(wav, waveform[startS:endS])
		chunks = append(chunks, Chunk{
			Waveform: wav,
			Offset:   float64(startS) / float64(sr),
		})
	}
	return chunks
}
