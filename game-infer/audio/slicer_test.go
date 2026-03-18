package audio_test

import (
	"testing"

	gameaudio "github.com/openvpi/game-infer/audio"
)

func makeSilentWav(sr, silSamples, sigSamples int) []float32 {
	total := silSamples + sigSamples + silSamples
	w := make([]float32, total)
	for i := silSamples; i < silSamples+sigSamples; i++ {
		w[i] = 0.5
	}
	return w
}

func TestSlicer_SingleVoicedChunk(t *testing.T) {
	sr := 44100
	w := makeSilentWav(sr, sr/2, sr*3) // 0.5s sil + 3s signal + 0.5s sil
	chunks := gameaudio.Slice(w, sr)
	if len(chunks) != 1 {
		t.Fatalf("want 1 chunk, got %d", len(chunks))
	}
	if chunks[0].Offset < 0 || chunks[0].Offset > 0.6 {
		t.Errorf("chunk offset out of range [0, 0.6]: %f", chunks[0].Offset)
	}
	if len(chunks[0].Waveform) == 0 {
		t.Error("chunk waveform should not be empty")
	}
}

func TestSlicer_FallbackWholeFile(t *testing.T) {
	sr := 44100
	// 500ms all-signal — shorter than min_length (1000ms), so buildChunks
	// produces no ranges and the fallback returns the whole file as 1 chunk.
	w := make([]float32, sr/2)
	for i := range w { w[i] = 0.5 }
	chunks := gameaudio.Slice(w, sr)
	if len(chunks) != 1 {
		t.Fatalf("want fallback to 1 chunk, got %d", len(chunks))
	}
	if chunks[0].Offset != 0 {
		t.Errorf("fallback chunk offset should be 0, got %f", chunks[0].Offset)
	}
	if len(chunks[0].Waveform) != sr/2 {
		t.Errorf("fallback chunk should contain full waveform (%d samples), got %d", sr/2, len(chunks[0].Waveform))
	}
}
