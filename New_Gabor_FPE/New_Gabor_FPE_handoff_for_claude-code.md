# New Gabor/FPE junction detection — handoff

**Purpose of this doc:** hand to Claude Code together with the files below. It
records *what was diagnosed, what was verified, what was not*, and what to do
next. Read §1–§3 before touching the code — the architecture changed for a
reason, and reverting to the old one will reproduce a known-unfixable bug.

**Status:** science verified in Python. Julia notebook written but **never
executed** (no Julia in the authoring sandbox). First job is to run it and diff
against the reference numbers in §5.

---

## 1. The problem being solved

Detect **endpoints, corners, T-junctions, X-crossings** on EMNIST letters from a
Gabor lift, using *linear/filtering operations only* — no peak counting, no ring
run-length analysis, no `if npk >= 2`, no hand-tuned thresholds. The output
should be a **continuous descriptor** suitable for FPE/VSA binding, not a
symbolic label.

Two prior attempts and why they failed:

- **`feature_layer.py` (Python, works but is GOFAI).** Detects the right things,
  but via `arc_count` (sample a ring, threshold, run-length analysis), peak
  counting, multi-radius voting, and a stack of magic constants (`0.62`, `0.35`,
  `sep_bins`, …). No plausible neural reading.
- **`simplified_vsa_letter_classifier_with_display.jl` (Julia, VSA).** Gabor lift
  → FPE bundle → learned pair detectors `|⟨M, C ⊙ M⟩|²`. T-junction detection
  never worked. **Not a tuning problem — see §2.**

---

## 2. Root-cause diagnosis (the important part)

### 2a. The orientation fibre cannot represent junction type — ever

`E(x,y,θ)` is **π-periodic**. Junction type is a **2π (directional)** property:
it is about *which rays leave the point*. Orientation mod π cannot distinguish
"ray going east" from "ray going west", so it cannot count rays.

Measured at the centre of canonical figures (cosine similarity of `E(θ)`):

```
L-corner   vs T-junction : 0.9031
L-corner   vs X-crossing : 0.8868
T-junction vs X-crossing : 0.9234
```

L, T and X are **the same vector** in the orientation fibre — all bimodal with
peaks at the same orientations. No downstream FPE/attention/learning can recover
what the representation never encoded.

> This also explains why `feature_layer.py` needed the ugly ring probe: the ring
> was sampling mod-2π structure *because the fibre couldn't*. The GOFAI was a
> patch over a representational hole, not a stylistic lapse.

### 2b. Two concrete bugs in the Julia VSA notebook

**Bug A — `Θ_BASE` is not periodic.**
```julia
const Θ_BASE = cis.(Float32(2π) .* rand(rng, Float32, d))   # φₖ ∈ ℝ  ← WRONG
...
fpow(Θ_BASE, 2 * s.θ)
```
With *continuous* base phases `φₖ`, `z^θ` is **not π-periodic**: `θ=0` and `θ=π`
encode to maximally different vectors despite being the same orientation. The
`2*` in the exponent was intended to fix this and **cannot** — periodicity is a
property of the base **frequencies**, not the exponent. Requires `φₖ ∈ ℤ`.

Check: `|⟨z⁰, z²ᵖⁱ⟩|/d` should be `1.0`; with continuous phases it is ≈0.

**Bug B — `|⟨M, C ⊙ M⟩|²` is a bag of pairs.**
It pools over the **whole image**, counting relative configurations *without
binding them to a location*. Two L-corners elsewhere in the image, opening
opposite ways, are indistinguishable from one T-junction. Structurally this is a
global second-order statistic — the Gram-matrix / Portilla–Simoncelli texture
family, which is exactly the class known **not to represent shape**. Junction
type is a conjunction of relations **at one point**; global pooling destroys
precisely that.

**Minor but real:** only 4 orientations (cannot resolve profile shape);
`step = round((1-0.25)*λ)` → 18 px at λ=24 on a 56 px image, so junctions were
frequently *never sampled*.

---

## 3. The fix — ray profile + circular harmonics (= FPE with integer freqs)

### 3a. Ray profile

At each point `p`, for `φ ∈ [0, 2π)`:

```
R(p, φ) = E( p + d·u(φ),  θ = φ mod π ),      u(φ) = (cos φ, sin φ)
```

"Is there a contour at distance `d` in direction `φ`, oriented **along** `φ`?"
`R` is 2π-periodic and has **one lobe per branch**: endpoint→1, straight→2
opposite, L→2 adjacent, T→3, X→4.

### 3b. Harmonics are the type signature

```
cₙ(y,x) = (1/K) Σ_φ  E[φ mod π](y + d·sin φ, x + d·cos φ) · exp(-i n φ)
```

#### Reading the expression term by term

It is a **Fourier series coefficient of the ray profile** `R(φ)` from §3a — i.e.
just `cₙ = (1/K) Σ_φ R(φ)·e^{-inφ}` with `R` written out inline. Piece by piece:

| term | meaning |
|---|---|
| `(y,x)` | the point being described. `cₙ` is computed at **every pixel** → a dense complex-valued map per `n`. |
| `Σ_φ` | sum over `K` ray directions `φ = 0, 2π/K, …` sampling the full circle. `K=96` in the notebook. **`φ` runs 0…2π** (a *direction*), unlike `θ` which runs 0…π (an *orientation*). |
| `E[·]` | the Gabor energy stack, indexed `E[θ, y, x]` = `|Gabor_θ * image|`. Phase-invariant (modulus of the quadrature pair). |
| `φ mod π` | **the orientation channel to look in.** A ray heading in direction `φ` is locally a line segment whose *orientation* is `φ mod π`. Rays east (`φ=0`) and west (`φ=π`) are both horizontal → same channel. This is the projection that loses the direction information… |
| `(y + d·sin φ, x + d·cos φ)` | …**and this is what puts it back.** Sample not at `(y,x)` but at the point a distance `d` away *in direction `φ`*. East and west now read different pixels. The `d`-offset is doing all the work of turning a mod-π quantity into a mod-2π one. |
| `exp(-i n φ)` | the Fourier kernel: project `R(φ)` onto the `n`-th circular harmonic. |
| `(1/K)` | normalisation so `cₙ` doesn't scale with the sampling density. |

Bilinear interpolation is needed because `(y + d·sin φ, x + d·cos φ)` is generally
not an integer pixel.

#### What each `n` measures

Substituting the idealisation `R(φ) = Σⱼ δ(φ − φⱼ)` for rays at angles `φⱼ`
gives `cₙ = Σⱼ e^{-i n φⱼ}` — a sum of unit phasors, one per ray, spun `n` times
faster. So:

- **`n = 0`** → `c₀ = Σⱼ 1` = **the number of rays** (in practice: total ray
  energy). Every phasor points the same way, nothing cancels. This is why the
  junction is simply the brightest point of the `c₀` map.
- **`n = 1`** → phasors point *along* each ray, so `|c₁|` is the **vector sum of
  the ray directions** = net direction / asymmetry. Two opposite rays cancel
  exactly (`e^{i0} + e^{iπ} = 0`) ⇒ `|c₁| = 0` mid-stroke and at an X. A lone ray
  has nothing to cancel it ⇒ `|c₁|/c₀ = 1` at an endpoint.
- **`n = 2`** → phasors spin twice, so rays 180° apart now *align* rather than
  cancel. `|c₂|` is large for a straight-through line, and vanishes for a 90°
  corner (its two rays land 180° apart after doubling).
- **higher `n`** → finer angular structure. **Careful with `c₄`:** on the
  canonical set it is *degenerate* — `|c₄|/c₀ = 1.000` for **all five** configs,
  because every ray there lies on the 90° grid and `e^{-i4φⱼ} = 1` for all of
  them. It carries information only for *off-grid* ray angles (a 60° corner gives
  `|c₄|/c₀ = 0.5`), i.e. it reports **deviation from 90°-symmetry**, not junction
  order. The discrimination in the table below lives in `c₀, c₁, c₂`. Beyond
  `n≈4` it is mostly noise.

`|cₙ|` is **rotation-invariant**: rotating the image by `α` sends `φⱼ → φⱼ+α`,
hence `cₙ → cₙ·e^{-inα}` — the modulus is unchanged and the *phase* records the
rotation. (This is the same fact as "rotation = binding" in §3c.)

#### Why it is not GOFAI

Read the sum in the other order. For a **fixed** `φ`, the term
`E[φ mod π](y + d·sin φ, x + d·cos φ)` is one orientation channel **rigidly
shifted** by the vector `(−d·sin φ, −d·cos φ)`. So the whole expression is:

```
take K orientation channels → shift each by a fixed offset → weight by e^{-inφ} → add
```

**K rigid shifts and a weighted sum.** That is a linear filter over the lifted
`(x, y, θ)` field — the same operation class as the Gabor convolution that
produced `E` in the first place. There is no counting, no thresholding, no
run-length analysis, no branching. Contrast `arc_count` in `feature_layer.py`,
which sampled a ring, thresholded it, and did circular run-length analysis to get
(a crude, quantised estimate of) exactly the same quantity `c₀`.

#### Practical note

Implementing this as written (`for φ, for y, for x, for n`) is `K·H·W·nmax`
scalar operations and will be slow — see §7.2. The efficient form follows
directly from the reading above: loop over `φ` only, do **one whole-array shift**
of `E[θ(φ)]`, then `axpy!` it into each `cₙ` plane with weight `e^{-inφ}`.

Ideal signatures (`cₙ = Σⱼ exp(-i n φⱼ)` over rays at `φⱼ`) — verified analytically:

| config | c₀ | \|c₁\|/c₀ | \|c₂\|/c₀ |
|---|---|---|---|
| endpoint (1 ray) | 1 | 1.000 | 1.000 |
| straight (2 opposite) | 2 | 0.000 | 1.000 |
| L-corner (2 @ 90°) | 2 | 0.707 | 0.000 |
| **T-junction (3 rays)** | **3** | **0.333** | **0.333** |
| X-crossing (4 rays) | 4 | 0.000 | 0.000 |

Interpretation: `c₀` ≈ ray count ("how many branches leave here");
`|c₁|/c₀` ≈ asymmetry / net direction (1 = endpoint, 0 = centrally symmetric).
`(c₀, |c₁|/c₀)` alone separates all five.

### 3c. FPE **is** the harmonic expansion

```
V = Σ_φ  R(φ) · z^φ
```
Component `k` is `Σ_φ R(φ)·exp(i φₖ φ)` = the circular harmonic `c_{φₖ}[R]`.
**FPE + bundling = harmonic expansion** — do *not* add a separate harmonics
module; fix the base instead.

Requirements:
- **integer frequencies** `φₖ ∈ ℤ` (periodicity — Bug A)
- concentrate on low `|n|` (n ∈ −4…4); high harmonics are noise
- for the *orientation* variable (mod π) encode `2θ`; for the *ray* variable
  (mod 2π) encode `φ` directly — **do not double the ray angle**

Free consequence: rotation by `α` gives `R(φ) → R(φ−α)`, hence `cₙ → cₙ·e^{-inα}`,
i.e. `V → z^α ⊙ V`. **Rotation = binding.** `|cₙ|` is rotation-invariant by
construction.

---

## 4. Architecture: where bundling is / isn't right

```
image
  └─ Gabor bank (multi-θ, multi-scale)      ─┐
  └─ modulus  → E[θ, y, x]                   │  DENSE, convolutional,
  └─ ray shift+combine → cₙ[y, x]            │  NO bundling
  └─ dense maps: c₀, |c₁|/c₀, …             ─┘
        │
        │  sparse keypoint selection (top-N of c₀)
        ▼
  └─ FPE bundle over keypoints, bound to
     object-relative polar position (r, α)   ─┐  SPARSE, bundled
  └─ object descriptor ∈ ℂ^d                 ─┘
```

**Why the split is forced, not an optimisation:** bundling `N` items into `d`
dimensions has SNR ≈ `√(d/N)`. With `d = 1024` and dense `112² × 48 ≈ 600k`
tokens, readout is pure noise. A few dozen keypoints is fine.

(Incidentally this mirrors dense retinotopic maps in V1/V2/V4 vs sparse object
codes in IT. `(r, α)` object-relative polar position is the Pasupathy–Connor
angular-position code.)

---

## 5. Reference numbers — **diff against these**

Measured with real Gabors on synthetic canonical figures
(λ=10, σn=5, σt=9, ks=35, d=16, K=144, N_THETA=72; ideal in parentheses):

```
endpoint     c₀= 6.82   |c₁|/c₀=0.671 (1.000)   |c₂|/c₀=0.682 (1.000)
straight     c₀=10.47   |c₁|/c₀=0.000 (0.000)   |c₂|/c₀=0.896 (1.000)
L-corner     c₀=11.84   |c₁|/c₀=0.592 (0.707)   |c₂|/c₀=0.006 (0.000)
T-junction   c₀=16.53   |c₁|/c₀=0.326 (0.333)   |c₂|/c₀=0.282 (0.333)
X-crossing   c₀=22.37   |c₁|/c₀=0.000 (0.000)   |c₂|/c₀=0.000 (0.000)
```

Ratios are attenuated vs ideal by the finite lobe width (a form factor —
expected, not a bug). **The pattern is what matters**, and all five separate.

Orientation-convention self-test: vertical line → argmax θ = 90°; horizontal → 0°.

On a rendered letter `T` (λ=12, σn=6, σt=10, d=15, N_THETA=48):
```
T-junction    c₀=18.3   |c₁|/c₀=0.23   |c₂|/c₀=0.11
arm endpoint  c₀= 9.7   |c₁|/c₀=0.22   |c₂|/c₀=0.56
mid-stroke    c₀=12.5   |c₁|/c₀=0.01   |c₂|/c₀=0.84
```
Junctions are the **brightest points of the c₀ map** — that is the detector.

---

## 6. Files

| file | status | role |
|---|---|---|
| `ray_fpe_junctions.jl` | **written, NEVER RUN** | Pluto notebook; the target implementation. 39 cells, order footer validated. |
| `rayharm.py` | **verified working** | Python reference: `estack`, `ray_profile`, `ray_harmonics`. Ground truth for the port. |
| `gabor_orientation_demo.py` | verified working | Gabor bank, glyph rendering, orientation-field figures. `render_glyphs` used as EMNIST stand-in. |
| `feature_layer.py` | works, **superseded** | The GOFAI version. Keep only as a behavioural baseline. |
| `feature_layer_design_notes.md` | — | Why the GOFAI version needed each hack. Read §"the big one" for the scale argument. |
| `pluto2md.py` | verified working | Renders a Pluto notebook's markdown without Pluto. (Had a bug: markdown cells use `╟─`, not `╟═`. Fixed.) |
| `figA_profiles.png` | — | **The key figure.** Col 2 = E(θ): L/T/X identical. Col 3 = R(φ): 1/2/2/3/4 lobes. |
| `figB_signature.png` | — | c₀ and normalised harmonics per type. |
| `figC_letters.png` | — | Dense c₀ and \|c₁\|/c₀ maps + polar probes on `T` and `K`. |

Data note: real EMNIST was **not** reachable from the authoring sandbox; glyphs
are font-rendered + elastically warped 28×28, upsampled 4× to 112. Phenomena are
identical; swap in real EMNIST on `incca`.

---

## 7. Next steps, in order

1. **Run `ray_fpe_junctions.jl`.** Diff §5 numbers. Check the periodicity cell
   (`|⟨z⁰,z²ᵖⁱ⟩|/d` must be 1.0). Check the orientation self-test.
2. **Performance.** `ray_harmonics` is a naive triple loop (`K × H × W × nmax`)
   — it will be slow. Restructure as: for each `φ`, one whole-array shift of
   `E[θ(φ)]`, then `axpy!` into each `cₙ` plane. That is ~K array ops, not
   K·H·W scalar ops. FFT-shift or `Interpolations.jl` both fine.
3. **Swap in real EMNIST** (dataset is reachable on the lab server).
4. **Multi-scale — as an encoded axis, not a conjunction.** Compute `cₙ` per
   scale, bind `z_s^{log σ}`. A curve's signature *drifts* with scale; a
   corner's stays put. That becomes a readable property of the multiscale
   vector rather than an `if`. (Do **not** implement the "bimodality survives
   across scales" test from the old design notes — same GOFAI disease.)
5. **Replace `D_RAY` with a ray transform.** It is the one real hyperparameter
   left (must exceed stroke half-width, stay below inter-junction distance).
   Compute `cₙ(d)` over several `d` and keep the profile — still linear.
6. **Object descriptor.** `object_descriptor()` in the notebook is a sketch,
   untested. Verify keypoint selection and the `(r, α)` binding.

---

## 8. Design rules — do not violate without a reason

- **No thresholds, no counting, no `if/elif` on feature type.**
  `(c₀, |c₁|/c₀, |c₂|/c₀, …)` **is** the descriptor. Bind the continuous vector;
  let similarity do the work.
- **Do not classify into symbolic labels.** A 30° corner has signature
  ≈(0.97, 0.87) — close to an *endpoint*. That is **correct**: a sharp corner
  really is nearly a doubled-back single ray. Corner-ness is graded, exactly as
  V4 curvature tuning is graded. The discrete labels were the GOFAI residue.
- **Never bundle globally before a locality-dependent readout** (Bug B).
- **Any circular FPE variable needs integer base frequencies** (Bug A).
- **Orientation mod π ⇒ encode 2θ. Ray direction mod 2π ⇒ encode φ.**

---

## 9. Open questions worth thinking about

- Should the ray profile use the *modulus* only, or also carry phase? Phase at
  the winning orientation distinguishes bright-bar / dark-bar / edge polarity
  (even vs odd Gabor) — currently discarded. Cheap to add, may help.
- `log-Gabor` instead of Gabor: zero DC, better 1/f match, cleaner multiscale
  tiling in log-frequency. Probably worth it before the multiscale step.
- Frame/tiling: is the (θ, scale, position) sampling dense enough to be lossless?
  Diagnostic: `Σₖ |ĝₖ(ξ)|²` should be ≈ flat over the covered band. Holes there
  will show up as scale/orientation-dependent artefacts in `cₙ`.
- The harmonics are a *fixed* readout. Once it works, the natural next move is to
  **learn** filters over the lifted SE(2) field (`x, y, θ`) rather than hand-pick
  `cₙ` — corners/junctions are patterns *in the lifted space*, and this is the
  Petitot–Citti–Sarti picture. That is where the "learn mid-level filters over an
  oriented representation" lesson from ViTs actually cashes out.
