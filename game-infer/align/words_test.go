package align_test

import (
	"testing"
	"github.com/openvpi/game-infer/align"
)

func TestParseWords_LeadCond(t *testing.T) {
	// "AP n i" with ph_num=[1,2], AP is unvoiced
	phSeq := []string{"AP", "n", "i"}
	phDur := []float64{0.09, 0.05, 0.07}
	phNum := []int{1, 2}
	uv    := map[string]bool{"AP": true, "SP": true}

	dur, vuv := align.ParseWords(phSeq, phDur, phNum, uv, "lead", false)
	if len(dur) != 2 { t.Fatalf("want 2 words, got %d", len(dur)) }
	if vuv[0] != 0 { t.Errorf("first word (AP) should be unvoiced") }
	if vuv[1] != 1 { t.Errorf("second word (n i) should be voiced") }
	if dur[0] < 0.089 || dur[0] > 0.091 { t.Errorf("first word dur want 0.09, got %f", dur[0]) }
}

func TestMergeConsecutiveUV(t *testing.T) {
	dur := []float64{0.09, 0.07, 0.12}
	vuv := []int{0, 0, 1}
	dur2, vuv2 := align.ParseWords(
		[]string{"AP", "SP", "n"}, dur, []int{1, 1, 1},
		map[string]bool{"AP": true, "SP": true}, "lead", true,
	)
	if len(dur2) != 2 { t.Fatalf("want 2 words after merge, got %d", len(dur2)) }
	if vuv2[0] != 0 { t.Errorf("merged UV word should be unvoiced") }
	if dur2[0] < 0.159 || dur2[0] > 0.161 { t.Errorf("merged dur want 0.16, got %f", dur2[0]) }
	_ = vuv
}
