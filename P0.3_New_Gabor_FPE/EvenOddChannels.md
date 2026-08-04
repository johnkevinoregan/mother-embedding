# Even/odd channel separation — motivation, and a first negative result

*Successor to `KeyPointDiagnosticity.md`. That document ended by concluding the
detector had stopped being the bottleneck in a fixable way. This one opens a new
front — splitting oriented **energy** into its **even (line)** and **odd (edge)**
parts — explains why that looked promising, and reports the first experiments,
which **failed**. The failure has a clean structural explanation, recorded here so
it isn't re-attempted.*

---

## 1. Where the previous document left off

Very briefly, the state at the end of `KeyPointDiagnosticity.md`:

- **Global shape harmonics carry letter identity** (57.2 % with `|M1..4|`, ~61 %
  with `|M1..6|` + radial profile). **Local keypoint counts are weak** — best
  24.2 %, because a census discards *configuration*.
- The best detector is the **branch profile**:
  `B_φ(p) = min(E_θ(φ)(p), E_θ(φ)(p + d·u(φ)))`, branches = angular maxima of
  `B_φ`, type = branch count + angles. 7/7 on synthetic with zero calibration.
- Its endpoint channel is **detector B** (the 1-branch case), a *termination*
  test: `min(E_θ(p), [E_θ(p−d·u) − E_θ(p+d·u)]₊)` — "stroke here, continues
  behind, gone ahead."
- Refinements that worked: a **two-gate rule** (absolute gate types junctions,
  relative gate confirms endpoints); a **drift-aware multi-scale** stability test
  (T phantoms cut 3×); a **medial/spine gate** borrowed from the end-stopped
  literature (Heitger 1992; Rodrigues & du Buf 2006; Würtz & Lourens 2000),
  which halved the outline-tracing phantom corners and lifted combined LOO
  49.7 % → 51.9 %.
- **What remained unsolved:** the corner channel. The medial gate removed the
  *flank artifact*, leaving a *curvature residue* — and corner-vs-curve in
  handwriting is a **continuum**, which any hard-typing detector thrashes at.
- Three structural lessons survived everything: (1) **rectify first** — no linear
  filter combination can detect termination; (2) **conjunctions must be gated
  (`min`), not summed**; (3) each question needs its own gate and scale.

---

## 2. Why separate even and odd

### 2.1 The polarity question that started it

Asked: *would every detector give identical results on polarity-inverted EMNIST
(black on white)?* Measured answer:

```
max|E − E'|  INTERIOR : 1.6e-5   (relative 1.2e-6)   -> float32 round-off
max|E − E'|  WHOLE    : 23.59                        -> border artifact
```

The **oriented energy is exactly polarity-invariant** in the interior. The reason
is worth stating because it recurs below: the kernel is built with
`K .- mean(K)`, so it is **DC-free** → `Gabor*(c−I) = −(Gabor*I)` → and `abs()`
discards the sign.

The *pipeline around it* is **not** invariant, for two reasons unrelated to the
detector:

- **Zero-padding.** Inverted, the white background meets black padding → a bright
  frame around the image. It dominates `maximum(E)` (10.0 → 23.7 for O), so every
  relative threshold shifts +72 % to +136 % and detection collapses
  (O keypoints `[1,29,13,2]` → `[0,0,0,0]`). *Fix: edge-replicate padding* —
  border pixels are background in either polarity, so the margin always matches.
- **The shape descriptor.** `angular_spectrum` uses `w = img/sum(img)`, i.e.
  *"brightness = mass."* Inverted, the "mass" is the paper, not the ink: T's
  centroid drifts (41.9, 51.1) → (59.0, 57.4) and the harmonics measure the
  image's square frame (inverted `|M4|` elevated for every letter). *Fix:*
  `bg = median(border pixels); mass = abs.(img .- bg)` — a no-op at current
  polarity, exactly equivalent at the other.

### 2.2 ON/OFF cells: we were already using them, collapsed

The question "could we do what the visual system does with ON and OFF cells?"
has a surprising answer: **we already do — we just throw the channel identity
away at the last step.** The standard model (Adelson & Bergen 1985) is:
retina/LGN split into **ON** and **OFF**, each **half-wave rectified**; simple
cells sum them in oriented arrangements (even- and odd-symmetric); complex cells
square and sum a quadrature pair. Writing the complex Gabor response as `c + is`:

```
E² = c² + s²  =  [c]₊² + [−c]₊² + [s]₊² + [−s]₊²
```

— exactly the four rectified simple-cell types (even-ON = bright bar, even-OFF =
dark bar, odd-ON/odd-OFF = edges of each polarity). So `E` **marginalises over**
ON/OFF rather than skipping it.

The governing rule for what we may do with them:

```
pool ON and OFF symmetrically   ->  polarity-invariant
use one of them, or weight them unequally  ->  polarity-specific
```

(A first proposal — build on the *even-ON* channel `[c]₊` — was **rejected for
exactly this reason**: under inversion even-ON becomes even-OFF and the detector
goes blind. Half-wave rectification breaks the invariance requirement.)

### 2.3 The decomposition, and the two hypotheses

Collapse only **within** each phase class, keeping the classes apart:

```
L(p)  = |Re(Gabor_θ * I)| = |Ce|   "lineness"  (even-ON + even-OFF)
Ed(p) = |Im(Gabor_θ * I)| = |Co|   "edgeness"  (odd-ON  + odd-OFF)
                E² = L² + Ed²
```

Each absolute value pools its own ON/OFF pair, so **each is exactly
polarity-invariant** by the same DC-free argument as `E` — the invariance
requirement is preserved, no calibration needed. And nothing is lost: `E` is
recoverable. What we *gain* is knowing **which kind** of structure is present,
which `E` discards.

Two hypotheses motivated the work:

- **H1 (junctions).** An even filter peaks on a stroke's **spine**; an odd filter
  peaks on its **flanks**. Building the branch profile on `L` should make the
  outline-tracing phantom corners *structurally absent*, turning the medial gate
  from a bolted-on inhibition into a property of the front end.
- **H2 (endpoints).** At a line end there is an **end-cap** — a step across the
  stroke, i.e. **odd/edge** phase — whereas at a crossing the perpendicular
  structure is another **stroke**, i.e. **even/bar** phase. So
  `min(L_θ(p), Ed_{θ+π/2}(p))` should fire at ends and reject crossings. This
  would resurrect the original `min(Gabor(p,θ), Gabor(p+d, θ+π/2))` proposal,
  which had failed as a **crossing detector** precisely because it used *energy*
  for both terms and energy merges bar and edge.

**This document reports the test of H2. It failed.** H1 remains untested.

---

## 3. Measurement first: EMNIST stroke width

Before choosing scales, measure rather than guess (`w ≈ 2·area/perimeter`):

```
STROKE WIDTH (upsampled 112x112):  median 13.4 px   (q25 10.9, q75 15.6)
```

Two consequences, both important beyond this experiment:

1. **EMNIST letters are only ~8 stroke-widths across.** There is almost **no
   scale separation** between "stroke width" and "letter size," so any operator
   wanting several stroke-widths of context necessarily straddles the whole
   letter. This is a structural constraint on the entire local-keypoint
   programme, not a tuning issue.
2. **Our working detector is tuned well below the stroke width** (λ = 8,
   σ_N = 3 against w = 13.4). It has been responding to stroke *edges and local
   structure*, not to strokes as bars.

---

## 4. Experiment 1 — first attempt (design flawed)

Design: three scale pairs `w ∈ {9.4, 13.4, 18.7}`; `L` elongated 5:1
(σ_T = 2.5w, λ_L = 2w); `Ed` compact (σ_T = w/2) with λ_Ed = 2w; combine scales
by **max** (scale selection — deliberately *not* min, after the earlier
multi-scale-min failure).

Result: **failed on every axis**, but two flaws make it uninformative about the
idea itself:

- **The 5:1 filters did not fit.** σ_T = 2.5 × 13.4 = 33.5 needs a ~168 px
  kernel on a 112 px image. All three `L` banks hit the 101 px cap and were
  **severely truncated** — a boxcar, not the intended Gaussian.
- **The synthetic test strokes were ~8 px wide**, not the measured 13.4 — so the
  clean-data test was not representative either.

Scores (want HIGH at tips, LOW elsewhere): bar TIP 0.97 ✓, but bar mid 0.68,
plus CENTER 0.61, X CENTER 0.70, **T CENTER 0.90**, T stem end 0.52 (a *tip*,
scoring lower than every false positive). EMNIST: **10–28 endpoints per letter**
against 0–4 expected (O: 28.3 vs 0).

The apparent polarity failure in this run (`max|S−S'| = 64.8`) was **an artifact
of my test, not of the method**: with ksize = 101 the DC-free argument needs
> 50 px of margin, leaving a 12 px square on a 112 px image, and I excluded only
30 px. `L` and `Ed` remain exactly polarity-invariant.

---

## 5. Experiment 2 — corrected scale grid (the real test)

Both flaws fixed: synthetic strokes widened to 13 px; elongation reduced to
ρ ∈ {2, 3} so kernels fit untruncated (ksize 67 and 101 for `L`, 33 for `Ed`).
Also swept **λ_Ed ∈ {w/4, w/2, w}** to test the follow-up hypothesis that the
first failure was **poor orientation tuning** of `Ed` (σ_T/λ had been 0.25 —
near-isotropic, so flank edges 90° away could leak in), and two readouts:
**A** = `max_θ min(L_θ, Ed_{θ+90})`, **B** = evaluate only at the *dominant*
stroke orientation.

Margin = worst TIP score − best FALSE-POSITIVE score; > 0 means separable.

| ρ | λ_Ed | σ_T/λ | readout | tips | worst FP | margin |
|---|---|---|---|---|---|---|
| 2 | 3.4 | **2.0** | A | 0.60, 0.62, 0.60 | 0.78 | **−0.19** |
| 2 | 3.4 | 2.0 | B | 1.00, 1.00, 0.43 | 0.93 | −0.50 |
| 2 | 6.7 | 1.0 | A | 0.65, 0.68, 0.65 | 0.54 | **+0.10** ← best |
| 2 | 6.7 | 1.0 | B | 1.00, 1.00, 0.47 | 0.82 | −0.35 |
| 2 | 13.4 | 0.5 | A | 0.90, 1.00, 0.87 | 0.83 | +0.04 |
| 2 | 13.4 | 0.5 | B | 1.00, 1.00, 0.56 | 0.71 | −0.14 |
| 3 | 3.4 | 2.0 | A | 0.60, 0.62, 0.60 | 0.78 | −0.19 |
| 3 | 6.7 | 1.0 | A | 0.65, 0.67, 0.65 | 0.54 | +0.10 |
| 3 | 13.4 | 0.5 | A | 0.90, 0.94, 0.87 | 0.86 | +0.01 |

Three findings:

1. **Elongation has no leverage.** ρ = 2 and ρ = 3 agree to two decimals in
   nearly every row. Whatever limits this detector, it is not the along-line
   extent — so the "5× longer along the line" intuition, once it actually fits
   the image, simply does not bite.
2. **The orientation-tuning hypothesis was backwards.** The *best*-tuned `Ed`
   (σ_T/λ = 2.0) gave the *worst* margins (−0.19, −0.50); the best result came
   from the middle value. Flank leakage was **not** the mechanism.
3. **Best margin over the whole grid = 0.10, on clean synthetic data.** Every
   precedent in this project says a thin synthetic margin becomes nothing on
   real handwriting.

Readout B is instructive: it scores genuinely free tips **1.00, 1.00** — the
detector *can* find ends — but cannot refuse crossings (worst FP 0.71–0.93).

*(Phase 2 — EMNIST counts and the polarity check at the best config — was not
run: a syntax bug stopped the script, and the synthetic margin forecloses the
outcome. It can be run for the record if wanted.)*

---

## 6. Why H2 fails — a structural reason, not a tuning problem

Take a horizontal stroke:

- its **end-cap** is a **vertical edge**;
- a **crossing stroke's flanks** are also **vertical edges**.

To an oriented edge detector at 90°, **an end-cap and a crossing stroke's flank
are the same object.** This is not leakage to be tuned away — it is genuine,
correct signal in both cases. No choice of λ, σ, or elongation can separate them,
because they differ in **neither orientation nor phase**.

What *does* differ is what lies **beyond** that edge: past a cap there is
nothing; past a crossing flank there is more stroke. That is a **termination**
question — exactly what **detector B already measures**. So the even/odd
decomposition adds nothing to endpoint detection: *the discriminating information
is not in the phase channel.*

This also corrects the earlier diagnosis in `KeyPointDiagnosticity.md` §"Endpoint
detection", which framed the perpendicular-odd-phase end-cap signature as a
usable detector cue. It is a real signature — but it is **not specific**, because
crossings produce the identical local signature.

---

## 7. What this does and does not rule out

**Ruled out:** the end-cap (odd-phase, perpendicular) as a *discriminative*
endpoint cue. Termination remains the only signal that separates an end from a
crossing.

**Not tested:** **H1**, the `L` channel for the *junction/branch* side. That is a
different claim, and a stronger one: spine-vs-flank is a genuine **geometric**
difference (centre of a bar vs boundary of a bar) rather than a coincidence of
orientation, so the even/odd split may separate it where it could not separate
cap from flank. If it works, it would *remove* the hand-added medial gate rather
than add another gate.

**Still standing:** `L` and `Ed` are each exactly polarity-invariant, so the
decomposition itself remains compatible with the requirement that inverted EMNIST
give identical results with no algorithm change. The two pipeline fixes (§2.1)
are worth applying regardless.

**A meta-observation, recorded honestly:** three successive predictions in this
thread — the multi-scale `min`, "L/Ed rejects crossings", and "sharper λ_Ed fixes
it" — were each plausible from the geometry and each falsified by measurement.
That pattern is itself evidence for the closing argument of
`KeyPointDiagnosticity.md`: hand-designed gate stacking is not converging, and
the remaining error may be the hard-typing frame rather than any missing
component.

---

## 8. A different mechanism — classical symmetric end-stopping (Ronan), A/B'd

A parallel notebook (Ronan) obtains end-stopping the **textbook hypercomplex
way**, worth recording because it is a genuinely different construction from ours
and because the A/B produced a clean, quantitative lesson. His pipeline:

1. a recurrent Dale's-law **E/I sheet** (≈80/20) whose narrow-excite / broad-inhibit
   coupling *emerges* as a Mexican-hat on-centre/off-surround — i.e. contrast
   normalisation grown from a network rather than a fixed DoG;
2. oriented **line** simple cells (Gaussian-along × Mexican-hat-across, DC-free,
   **half-wave rectified**) — the analogue of our `L = |Ce|`, but ON-only;
3. end-stopping by **symmetric iso-orientation inhibition** along the cell's own axis:

```
C_θ(p) = [ S_θ(p) − β·( S_θ(p+Δ·u_θ) + S_θ(p−Δ·u_θ) ) ]₊ ,   readout at argmax_θ S_θ
```

A straight stroke fills both flanks and cancels; an end has one empty flank and
fires. It is **symmetric** (both ends, corners, crossings) and **subtractive**,
where our detector B is **directional** and a **min-gate**. His Stage-3 is the
same "centre − displaced, rectified" skeleton as detector B —
`min(E_θ(p), [E_θ(p−d·u) − E_θ(p+d·u)]₊)` — with three deliberate differences:
symmetric vs directional, ON-only line channel vs polarity-invariant energy, and
a contrast-normalising front end we don't have.

### What was built and tested

Added to `EndpointDiagnosticsPadded.jl` as a toggle (`endstop_sym`, β slider, an
A/B panel row) so both detectors run on the same letter. Two design choices were
forced by measurement, not taste:

- **Energy, not `|Ce|`.** Run on the line channel `|Ce|` it fired **16** points on
  a plain bar: `|Ce|` of a Gabor has oscillatory side-lobes that `abs` turns into
  phantom ridges *parallel* to the stroke, and a symmetric end-stop happily fires
  on them. Switching to the line bank's **oriented energy** `√(Ce²+Co²)` (the
  phase-invariant envelope, no side-lobes) removed them (16 → 8). Both stay exactly
  polarity-invariant (`max|Es − Es'| = 6.5e-5`).

### The finding — β is scale-matched, and Ronan's own default fails here

On a bar (sweep over `w_L`, `δ`, `β`), **β decides everything**:

| β | true end `Es@(112,142)` | what fires |
|---|---|---|
| 0.7 (Ronan's default) | **0.0** — killed | only diagonal *skirt* artifacts |
| ~0.5 | positive, and the global max **lands on an end** | ends survive, middle suppressed |
| 0.3 | high everywhere | middle not suppressed → ends aren't local maxima |

The reason β=0.7 annihilates the true ends is **structural, and it is our scale
regime, not the mechanism**: our line kernel is large (`σ_T = 1.5·w_L = 12` at the
default), so at a genuine end the *centre* energy is already on a gentle down-ramp
while the *behind* flank `E(end−Δ)` is still full strength; `E(end) − β·behind`
then goes negative and rectifies to 0. Ronan's sharply-peaked `hw≈5` Mexican-hat
cell keeps the end at its peak while the flanks fall off fast, so it tolerates
β=0.7. The broad skirt our large Gabor leaves around every stroke (dominant
orientation *diagonal* there, where the artifacts live) is small in his system
chiefly because his **filter** is small — *not*, as measured below, because of his
front end. Lowering `w_L` widens the usable β range, exactly as this explanation
predicts.

This is the §3 constraint again, from the other side: **symmetric end-stopping
wants a cell sharply peaked *relative to the stroke*, and at ~8 stroke-widths per
letter with 13 px strokes our operators are deliberately too broad for it.** It is
also a third confirmation of structural lesson (1) — *rectify first*: his
subtraction detects a termination only because it sits between rectified/energy
maps; the same subtraction on the raw (signed) response is a linear operator and
marks nothing.

At the working default (β=0.5) the symmetric detector is **markedly sparser than
our min-gate** at equal threshold — EMNIST O 23→14, T 21→11, X 18→10, L 10→5,
I 6→5. Whether that sparsity drops *spurious* fires or *real* ends is a visual,
letter-by-letter judgement the A/B panels now support; it has not been scored.

### Front-end normalisation (Ronan's Stage 1) — thins strokes, does *not* help

The natural next hope was that Ronan's Stage 1 — a recurrent E/I sheet whose
narrow-excite / broad-inhibit coupling emerges as an on-centre/off-surround
band-pass — would, by sharpening and contrast-normalising, suppress the skirt the
symmetric end-stop fires on. Tested directly (a `normalise = off / DoG /
DoG+divisive` toggle in `EndpointDiagnosticsPadded.jl`, kept **sign-preserving** —
no ON-only rectification — so polarity invariance survives, verified 1e-4):

- **It thins strokes, strongly.** A DoG (σ_E = 1, σ_I = 4) cuts the effective width
  of a 13 px bar from **11.8 → 2.6 px**. The sharpening is real and large.
- **It does not widen the β-range.** In all three modes the true ends are local-max
  keypoints **only at β = 0.5**; β = 0.7 zeroes them exactly, as with no
  normalisation. `DoG+divisive` is *worse* — it adds spurious keypoints
  (bar 4 → 12–14) by gain-amplifying the low-contrast skirt.

The reason closes the loop: the end-stop reads the **energy envelope of the `w_L`
Gabor**, whose peak width is the **filter footprint** (~30 px), *not* the input
stroke width. Thinning the input never sharpens the filter's response peak, so the
behind-flank at a true end stays full strength and the subtraction still
annihilates it. **The skirt is a filter-scale property; the lever is `w_L`, and
front-end normalisation is orthogonal to it.** This is the fourth prediction in
this programme — after multi-scale `min`, "L/Ed rejects crossings", "sharper λ_Ed
fixes it" — plausible from the geometry and falsified by measurement; here it was
*my own*, recorded as such.

**What still carries forward:** the *symmetric both-ends* subtraction as a cheap
corner/crossing channel, but only paired with a filter sharply peaked relative to
the stroke — i.e. a *smaller* `w_L`, since normalising the input does not
substitute for it. Normalisation itself may still earn a place — not for the skirt,
but for the local gain-invariance and DC / low-frequency cleanup it gives (the same
contrast-invariance ACJ reaches by gradient rank-normalisation), and *if* combined
with a smaller `w_L` the thinning could buy back the scale separation §3 says we
lack. That is a separate experiment; neither it nor normalisation touches the
corner-vs-curve hard-typing wall.

## Next

1. **Test H1** — branch profile on `L` instead of `E`; check whether O's corner
   count drops *without* the medial gate.
2. Apply the two polarity fixes (edge-replicate padding; background-relative
   mass) so the invariance is a tested property rather than a convention.
3. Take seriously the §3 finding — ~8 stroke-widths per letter — as a constraint
   on how much "local context" any keypoint operator can have at this resolution.
4. **Done (§8): front-end normalisation thins strokes (13 → 2.6 px) but does not
   fix the skirt or widen the β-range** — the skirt is a filter-scale property, so
   the lever is `w_L`. The open version is the *combined* move: DoG / divisive
   normalisation **plus a smaller `w_L`**, to test whether the thinning buys back
   the §3 scale separation for bar detection generally, not just end-stopping.
