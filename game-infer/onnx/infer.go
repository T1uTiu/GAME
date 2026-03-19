package onnx

import (
	"fmt"

	ort "github.com/yalue/onnxruntime_go"
	"github.com/openvpi/game-infer/output"
)

// InferParams holds inference hyperparameters.
type InferParams struct {
	LanguageID   int
	SegThreshold float32
	SegRadius    int64   // frames: round(seg_radius_sec / timestep)
	EstThreshold float32
	T0           float32
	NSteps       int
	CustomTs     []float32
}

// ResolveTs returns D3PM time schedule. Mirrors ValidationConfig.d3pm_sample_ts_resolved.
func ResolveTs(t0 float32, nsteps int, custom []float32) []float32 {
	if custom != nil {
		return custom
	}
	ts := make([]float32, nsteps)
	step := (1.0 - t0) / float32(nsteps)
	for i := range ts {
		ts[i] = t0 + float32(i)*step
	}
	return ts
}

// InferBatch runs encoder → segmenter → bd2dur → estimator on a batch of waveforms.
// Returns one []NoteInfo per input waveform (offsets at 0.0; caller adds chunk offset).
func InferBatch(sess *Sessions, waveforms [][]float32, params InferParams) ([][]output.NoteInfo, error) {
	B := len(waveforms)
	maxLen := 0
	for _, w := range waveforms {
		if len(w) > maxLen {
			maxLen = len(w)
		}
	}

	waveFlat := make([]float32, B*maxLen)
	durations := make([]float32, B)
	for i, w := range waveforms {
		copy(waveFlat[i*maxLen:], w)
		durations[i] = float32(len(w)) / float32(sess.Config.Samplerate)
	}

	// --- Encoder ---
	xSeg, xEst, maskT, T, err := runEncoder(sess, waveFlat, durations, B, maxLen)
	if err != nil {
		return nil, fmt.Errorf("encoder: %w", err)
	}

	// --- Segmenter (D3PM loop or single pass) ---
	boundaries, err := runSegmenter(sess, xSeg, maskT, params, B, T)
	if err != nil {
		return nil, fmt.Errorf("segmenter: %w", err)
	}

	// --- Bd2Dur ---
	noteDurs, maskN, N, err := runBd2Dur(sess, boundaries, maskT, B, T)
	if err != nil {
		return nil, fmt.Errorf("bd2dur: %w", err)
	}

	// --- Estimator ---
	presence, scores, err := runEstimator(sess, xEst, boundaries, maskT, maskN, params.EstThreshold, B, T, N)
	if err != nil {
		return nil, fmt.Errorf("estimator: %w", err)
	}

	// Unpack: each item i gets its valid notes
	results := make([][]output.NoteInfo, B)
	for i := 0; i < B; i++ {
		var durs []float32
		var pres []bool
		var scrs []float32
		for j := 0; j < N; j++ {
			if maskN[i*N+j] {
				durs = append(durs, noteDurs[i*N+j])
				pres = append(pres, presence[i*N+j])
				scrs = append(scrs, scores[i*N+j])
			}
		}
		results[i] = output.ReconstructTimestamps(durs, pres, scrs, 0)
	}
	return results, nil
}

// ── Encoder ──────────────────────────────────────────────────────────────────

func runEncoder(sess *Sessions, waveFlat, durations []float32, B, maxLen int) (
	xSeg, xEst []float32, maskT []bool, T int, err error,
) {
	wT, err := ort.NewTensor(ort.NewShape(int64(B), int64(maxLen)), waveFlat)
	if err != nil {
		return nil, nil, nil, 0, err
	}
	defer wT.Destroy()

	dT, err := ort.NewTensor(ort.NewShape(int64(B)), durations)
	if err != nil {
		return nil, nil, nil, 0, err
	}
	defer dT.Destroy()

	outs := make([]ort.Value, 3) // x_seg, x_est, maskT — allocated by ort
	if err = sess.Encoder.Run(
		[]ort.Value{wT, dT},
		outs,
	); err != nil {
		return nil, nil, nil, 0, err
	}
	defer func() {
		for _, o := range outs {
			if o != nil {
				o.Destroy()
			}
		}
	}()

	T = int(outs[0].GetShape()[1])
	xSeg = cloneF32(getF32Data(outs[0]))
	xEst = cloneF32(getF32Data(outs[1]))
	maskT = cloneBool(getBoolData(outs[2]))
	return xSeg, xEst, maskT, T, nil
}

// ── Segmenter ────────────────────────────────────────────────────────────────

func runSegmenter(sess *Sessions, xSeg []float32, maskT []bool, params InferParams, B, T int) ([]bool, error) {
	knownBounds := make([]bool, B*T) // all false — no prior knowledge in extract mode

	if !sess.Config.Loop {
		return runSegmenterOnce(sess, xSeg, knownBounds, make([]bool, B*T),
			make([]float32, B), maskT, params, B, T)
	}
	// D3PM loop: initialise boundaries = knownBounds (all-false), iterate
	boundaries := cloneBool(knownBounds)
	for _, ti := range ResolveTs(params.T0, params.NSteps, params.CustomTs) {
		tArr := make([]float32, B)
		for i := range tArr {
			tArr[i] = ti
		}
		var err error
		boundaries, err = runSegmenterOnce(sess, xSeg, knownBounds, boundaries, tArr, maskT, params, B, T)
		if err != nil {
			return nil, err
		}
	}
	return boundaries, nil
}

func runSegmenterOnce(
	sess *Sessions,
	xSeg []float32,
	knownBounds, prevBounds []bool,
	tArr []float32,
	maskT []bool,
	params InferParams,
	B, T int,
) ([]bool, error) {
	xSegT, err := ort.NewTensor(ort.NewShape(int64(B), int64(T), 128), xSeg)
	if err != nil {
		return nil, err
	}
	defer xSegT.Destroy()

	// Build inputs in the same order as sess.SegInputNames
	inputs := []ort.Value{xSegT}
	var toDestroy []ort.Value
	defer func() {
		for _, t := range toDestroy {
			t.Destroy()
		}
	}()

	addBool := func(data []bool, shape ...int64) error {
		t, e := ort.NewTensor(ort.NewShape(shape...), data)
		if e != nil {
			return e
		}
		inputs = append(inputs, t)
		toDestroy = append(toDestroy, t)
		return nil
	}
	addF32 := func(data []float32, shape ...int64) error {
		t, e := ort.NewTensor(ort.NewShape(shape...), data)
		if e != nil {
			return e
		}
		inputs = append(inputs, t)
		toDestroy = append(toDestroy, t)
		return nil
	}
	addI64 := func(data []int64, shape ...int64) error {
		t, e := ort.NewTensor(ort.NewShape(shape...), data)
		if e != nil {
			return e
		}
		inputs = append(inputs, t)
		toDestroy = append(toDestroy, t)
		return nil
	}
	addScalarF32 := func(v float32) error {
		t, e := ort.NewScalar(v)
		if e != nil {
			return e
		}
		inputs = append(inputs, t)
		toDestroy = append(toDestroy, t)
		return nil
	}
	addScalarI64 := func(v int64) error {
		t, e := ort.NewScalar(v)
		if e != nil {
			return e
		}
		inputs = append(inputs, t)
		toDestroy = append(toDestroy, t)
		return nil
	}

	for _, name := range sess.SegInputNames[1:] { // skip "x_seg" already added
		switch name {
		case "language":
			langData := make([]int64, B)
			for i := range langData {
				langData[i] = int64(params.LanguageID)
			}
			if err := addI64(langData, int64(B)); err != nil {
				return nil, err
			}
		case "known_boundaries":
			if err := addBool(knownBounds, int64(B), int64(T)); err != nil {
				return nil, err
			}
		case "prev_boundaries":
			if err := addBool(prevBounds, int64(B), int64(T)); err != nil {
				return nil, err
			}
		case "t":
			if err := addF32(tArr, int64(B)); err != nil {
				return nil, err
			}
		case "maskT":
			if err := addBool(maskT, int64(B), int64(T)); err != nil {
				return nil, err
			}
		case "threshold":
			if err := addScalarF32(params.SegThreshold); err != nil {
				return nil, err
			}
		case "radius":
			if err := addScalarI64(params.SegRadius); err != nil {
				return nil, err
			}
		}
	}

	outs := make([]ort.Value, 1)
	if err := sess.Segmenter.Run(inputs, outs); err != nil {
		return nil, err
	}
	defer outs[0].Destroy()
	return cloneBool(getBoolData(outs[0])), nil
}

// ── Bd2Dur ───────────────────────────────────────────────────────────────────

func runBd2Dur(sess *Sessions, boundaries, maskT []bool, B, T int) (
	durations []float32, maskN []bool, N int, err error,
) {
	bdT, err := ort.NewTensor(ort.NewShape(int64(B), int64(T)), boundaries)
	if err != nil {
		return nil, nil, 0, err
	}
	defer bdT.Destroy()

	mT, err := ort.NewTensor(ort.NewShape(int64(B), int64(T)), maskT)
	if err != nil {
		return nil, nil, 0, err
	}
	defer mT.Destroy()

	outs := make([]ort.Value, 2)
	if err = sess.Bd2Dur.Run([]ort.Value{bdT, mT}, outs); err != nil {
		return nil, nil, 0, err
	}
	defer func() {
		for _, o := range outs {
			if o != nil {
				o.Destroy()
			}
		}
	}()

	N = int(outs[0].GetShape()[1])
	durations = cloneF32(getF32Data(outs[0]))
	maskN = cloneBool(getBoolData(outs[1]))
	return durations, maskN, N, nil
}

// ── Estimator ────────────────────────────────────────────────────────────────

func runEstimator(sess *Sessions, xEst []float32, boundaries, maskT, maskN []bool,
	threshold float32, B, T, N int,
) (presence []bool, scores []float32, err error) {
	xEstT, err := ort.NewTensor(ort.NewShape(int64(B), int64(T), 128), xEst)
	if err != nil {
		return nil, nil, err
	}
	defer xEstT.Destroy()

	bdT, err := ort.NewTensor(ort.NewShape(int64(B), int64(T)), boundaries)
	if err != nil {
		return nil, nil, err
	}
	defer bdT.Destroy()

	mTT, err := ort.NewTensor(ort.NewShape(int64(B), int64(T)), maskT)
	if err != nil {
		return nil, nil, err
	}
	defer mTT.Destroy()

	mNT, err := ort.NewTensor(ort.NewShape(int64(B), int64(N)), maskN)
	if err != nil {
		return nil, nil, err
	}
	defer mNT.Destroy()

	thrT, err := ort.NewScalar(threshold)
	if err != nil {
		return nil, nil, err
	}
	defer thrT.Destroy()

	outs := make([]ort.Value, 2)
	if err = sess.Estimator.Run([]ort.Value{xEstT, bdT, mTT, mNT, thrT}, outs); err != nil {
		return nil, nil, err
	}
	defer func() {
		for _, o := range outs {
			if o != nil {
				o.Destroy()
			}
		}
	}()

	presence = cloneBool(getBoolData(outs[0]))
	scores = cloneF32(getF32Data(outs[1]))
	return presence, scores, nil
}

// ── Helpers ──────────────────────────────────────────────────────────────────

// getF32Data extracts []float32 from an ort.Value by type-asserting to *ort.Tensor[float32].
func getF32Data(v ort.Value) []float32 {
	return v.(*ort.Tensor[float32]).GetData()
}

// getBoolData extracts []bool from an ort.Value by type-asserting to *ort.Tensor[bool].
func getBoolData(v ort.Value) []bool {
	return v.(*ort.Tensor[bool]).GetData()
}

func cloneF32(src []float32) []float32 { dst := make([]float32, len(src)); copy(dst, src); return dst }
func cloneBool(src []bool) []bool      { dst := make([]bool, len(src)); copy(dst, src); return dst }
