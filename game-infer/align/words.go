package align

import "strings"

// ParseWords converts phoneme sequence to word durations and v/uv flags.
// Mirrors parse_words() in inference/utils.py.
func ParseWords(
	phSeq []string, phDur []float64, phNum []int,
	uvVocab map[string]bool, uvCond string, mergeConsecUV bool,
) (wordDur []float64, wordVUV []int) {
	idx := 0
	for _, num := range phNum {
		var sum float64
		for _, d := range phDur[idx : idx+num] { sum += d }
		wordDur = append(wordDur, sum)
		vuv := 1
		if uvVocab != nil {
			switch uvCond {
			case "lead":
				if uvVocab[phSeq[idx]] { vuv = 0 }
			case "all":
				all := true
				for _, ph := range phSeq[idx : idx+num] { if !uvVocab[ph] { all = false; break } }
				if all { vuv = 0 }
			}
		}
		wordVUV = append(wordVUV, vuv)
		idx += num
	}
	if mergeConsecUV {
		wordDur, wordVUV = mergeConsecutiveUV(wordDur, wordVUV)
	}
	return
}

func mergeConsecutiveUV(dur []float64, vuv []int) ([]float64, []int) {
	if len(dur) == 0 { return nil, nil }
	md := []float64{dur[0]}
	mv := []int{vuv[0]}
	for i := 1; i < len(dur); i++ {
		if vuv[i] == 0 && mv[len(mv)-1] == 0 {
			md[len(md)-1] += dur[i]
		} else {
			md = append(md, dur[i])
			mv = append(mv, vuv[i])
		}
	}
	return md, mv
}

// ParseUVVocab parses "AP,SP,br,sil" → map.
func ParseUVVocab(s string) map[string]bool {
	m := make(map[string]bool)
	for _, v := range strings.Split(s, ",") {
		v = strings.TrimSpace(v)
		if v != "" { m[v] = true }
	}
	return m
}
