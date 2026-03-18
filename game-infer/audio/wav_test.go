package audio_test

import (
	"math"
	"os"
	"testing"

	"github.com/go-audio/audio"
	gowav "github.com/go-audio/wav"
	gameaudio "github.com/openvpi/game-infer/audio"
)

func writePCMWAV(t *testing.T, path string, samples []int16, sr, channels int) {
	t.Helper()
	f, err := os.Create(path)
	if err != nil { t.Fatal(err) }
	defer f.Close()
	enc := gowav.NewEncoder(f, sr, 16, channels, 1)
	buf := &audio.IntBuffer{
		Format: &audio.Format{SampleRate: sr, NumChannels: channels},
		Data:   make([]int, len(samples)),
	}
	for i, s := range samples { buf.Data[i] = int(s) }
	if err := enc.Write(buf); err != nil { t.Fatal(err) }
	enc.Close()
}

func TestLoadWAV_Mono(t *testing.T) {
	path := t.TempDir() + "/mono.wav"
	samples := make([]int16, 100)
	for i := range samples { samples[i] = 16384 } // ~0.5 normalised
	writePCMWAV(t, path, samples, 44100, 1)

	waveform, err := gameaudio.LoadWAV(path, 44100)
	if err != nil { t.Fatal(err) }
	if len(waveform) != 100 { t.Fatalf("want 100 samples, got %d", len(waveform)) }
	if math.Abs(float64(waveform[0])-0.5) > 0.002 {
		t.Errorf("want ~0.5, got %f", waveform[0])
	}
}

func TestLoadWAV_StereoMixdown(t *testing.T) {
	path := t.TempDir() + "/stereo.wav"
	// L=16384, R=0 → mono average = 8192 → ~0.25
	samples := make([]int16, 200)
	for i := 0; i < 200; i += 2 { samples[i] = 16384 }
	writePCMWAV(t, path, samples, 44100, 2)

	waveform, err := gameaudio.LoadWAV(path, 44100)
	if err != nil { t.Fatal(err) }
	if len(waveform) != 100 { t.Fatalf("want 100 mono frames, got %d", len(waveform)) }
	if math.Abs(float64(waveform[0])-0.25) > 0.002 {
		t.Errorf("want ~0.25, got %f", waveform[0])
	}
}

func TestLoadWAV_Resample(t *testing.T) {
	path := t.TempDir() + "/22k.wav"
	samples := make([]int16, 100)
	writePCMWAV(t, path, samples, 22050, 1)

	waveform, err := gameaudio.LoadWAV(path, 44100)
	if err != nil { t.Fatal(err) }
	if len(waveform) < 195 || len(waveform) > 205 {
		t.Fatalf("want ~200 samples after 2x upsample, got %d", len(waveform))
	}
}
