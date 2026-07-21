# Comments on `RonanLiquid1.jl`

*A transcript of the discussion (2026-07-21) comparing Ronan's notebook
"A topographic E/I sheet for end-stop detection" (`RonanLiquid1.jl`) with the
Gabor-based keypoint detectors in `EndpointDiagnosticsPadded.jl` /
`EvenOddChannels.md`. Kept as a companion to the A/B toggles those files now
carry. Questions are Kevin's; the analysis is Claude's.*

---

## 1. How does he obtain his end-stopping, compared to what we are doing?

> *"Here's an alternative approach to endstopping suggested by my friend Ronan.
> Can you tell me how he obtains his end-stopping compared to what we are doing."*

Ronan builds a 3-stage "topographic E/I sheet," all rate-neuron / biologically-framed:

**Stage 1 — recurrent on-centre/off-surround sheet.** Instead of applying a fixed
DoG/LoG, he *grows* the centre-surround from a recurrent Dale's-law network: 784 E
cells (one/pixel) + 196 I cells (stride-2 sublattice, ~20% inhibitory), narrow
excitatory Gaussian σ_E, broad inhibitory σ_I, iterated to steady state with a
rectified-saturating transfer `φ(x)=rmax·tanh(max(0,x)/rmax)`. Net effective kernel
= Mexican hat → contrast-normalised "on-map." This is a normalisation front-end we
don't have.

**Stage 2 — oriented simple cells.** A bank of K kernels, Gaussian *along* ×
Mexican-hat *across* `(1−(u/σw)²)exp(−u²/2σw²)`, DC-removed, then **half-wave
rectified**: `S_θ = max(0, onmap ⋆ k)`. This is a purely **even/line-like** cell —
the analogue of our `Ce` channel only. There is no odd/edge channel anywhere in his
pipeline.

**Stage 3 — the end-stopping.** The classical hypercomplex mechanism:

```
C_θ(x) = [ S_θ(x) − β·( S_θ(x + Δ·dir_θ) + S_θ(x − Δ·dir_θ) ) ]₊
```

Same orientation inhibits itself, sampled **±Δ along its own preferred axis,
symmetrically both ways**, then rectify. On a straight bar both flanks are filled →
with β≳0.5 it cancels; at a stroke end one flank is empty → it escapes and fires.
Then he collapses the K channels by reading `C` at the **locally dominant
orientation** `argmax_θ S_θ(x)`.

### How that compares to ours

**The mechanism is subtractive symmetric end-inhibition; ours is a min-gate.** His
core is `[centre − β·(behind + ahead)]₊` — a *difference* of the same rectified
channel. Our best energy detector (Detector B) is the same family but **asymmetric
and gated**: `min(E_θ(p), [E_θ(p−d·u) − E_θ(p+d·u)]₊)` — "line here, present behind,
gone ahead." His fires at *both* ends (symmetric); ours is directional. And our
even/odd cap detector replaces subtraction entirely with `min(|Ce|, cap)`.

**Where the two actually agree:**

- **Rectification is the essential nonlinearity.** His subtraction sits *between*
  rectified maps and is rectified again — exactly the point we kept hitting: a
  linear matched filter can't mark a line-end; his `max(0,·)` on the simple cells +
  the final `[·]₊` is what makes it work. Same reason our detector needs
  `min`/`abs`/rectified-cap.
- **Displacement tied to filter scale.** His note "Δ should match σ_len (sample one
  bar away)" is our `δ = dratio·σ_T`.
- **Dominant-orientation readout.** His `argmax_θ S_θ` = our `thdom = argmax_θ|Ce|`
  gate. Identical idea.
- His Stage-3 is essentially *iso-orientation, along-axis* end-inhibition — the
  end-stopping limb of the du Buf/Heitger scheme (we'd focused more on the
  tangential/medial limb).

**Three differences that matter for our stated goal:**

1. **Polarity.** His simple cells are ON-only half-wave (`max(0, ⋆)` on a DC-free
   kernel), so on inverse-polarity EMNIST they'd go silent — the whole thing is
   polarity-*specific* unless he adds the OFF channel `max(0,−⋆)`. Our entire recent
   push (`E²=Ce²+Co²`, sign-referenced cap) exists precisely to be
   polarity-invariant "with no change to the algorithm." His isn't, as written.
2. **Line channel only, no edge channel.** He never uses an odd/edge (`Co`)
   response. His end-stop is done inside the line channel by along-axis inhibition.
   Our even/odd endpoint detector instead *combines* the line cell at `p` with an
   edge/end-cap cell at `p+δ`. So his design sidesteps the "an end-cap and a
   crossing-flank are the same oriented edge" problem we found — by not using edges
   at all.
3. **Recurrent normalisation front-end.** His Stage 1 is a genuine extra ingredient
   (contrast normalisation as an emergent network property) we simply don't have;
   functionally it's a DoG pre-normalisation.

**Bottom line:** his end-stopping is the textbook *symmetric subtractive along-axis
hypercomplex* cell (`[S − β(S₊+S₋)]₊`), operating on a single rectified line channel,
made robust by a recurrent contrast-normalising front-end. Ours is a *min-gated,
asymmetric, polarity-invariant even/odd* construction. The closest correspondence is
**his Stage-3 ↔ our energy Detector B** — same "centre minus displaced, rectified"
skeleton — but he chose symmetric + line-only + ON-only where we chose directional +
even/odd + polarity-invariant.

*(Follow-up: the symmetric detector was added to `EndpointDiagnosticsPadded.jl` as an
A/B toggle. Finding — on our broad Gabor energy, Ronan's own default β=0.7 kills the
true ends; β≈0.5 is the working regime. See `EvenOddChannels.md` §8.)*

---

## 2. What would his recurrent normalisation front end add to what we've done?

> *"What would Ronan's recurrent normalisation front end add to what we've done?"*

His front end is really three things at once: **(a)** DC removal / band-pass (the
Mexican-hat), **(b)** off-surround sharpening (a strong stroke drives its
neighbourhood negative), and **(c)** a saturating recurrent gain control (the `tanh`
+ broad fast inhibition).

**(a) DC / low-frequency removal — mostly redundant for us.** We already build
`K .- mean(K)`, so our filters are DC-free; that's why our energy is
polarity-invariant. Where his band-pass would still help is *low-frequency intensity
ramps* that survive our per-filter DC removal. Real but small.

**(b) Off-surround sharpening — the one real lever, and not the one I first named.**
This effectively *thins* strokes and boosts edge contrast. That matters because our
chronic constraint is §3 of `EvenOddChannels.md`: ~8 stroke-widths per letter, no
scale separation, filters deliberately as big as the strokes. If normalisation
thinned a 13 px stroke to ~6–7 px, our filters would be *large relative to the
stroke* — the regime symmetric end-stopping wants. **This is the thing worth
trying.** But it pays off only if we *also* shrink `w_L` — front end + smaller
filters is one move, not two.

**(c) Saturating gain control — modest; we approximate it.** Local contrast
invariance across faint vs bold strokes. We currently do a crude *global* version
(threshold × `maximum(E)`). His gives *local* gain control — strictly better, same
contrast-invariance goal ACJ reaches by gradient rank-normalisation — but EMNIST
strokes are fairly uniform, so the practical gain is small.

**What it would *not* fix.** My §8 note said normalising away the "broad energy
skirt" would make symmetric end-stopping behave. Looking properly, that's
**misattributed**: our skirt is the *Gabor's own response envelope* (a 61-px filter
has a ~30-px footprint), computed *after* the front end. Sharpening the *input*
doesn't shrink the *filter's* footprint. So a **pre-filter** normalisation leaves the
skirt essentially intact. The things that actually kill it are: **smaller filters**,
or **divisive normalisation of the energy maps *after* the bank** — a different
architectural slot. And it does nothing for the **corner-vs-curve hard-typing
continuum**, which is a decision problem, not a front-end problem.

### Measured (the prediction was tested, and half of it was wrong)

A `normalise = off / DoG / DoG+divisive` toggle was added to the input stage (kept
**sign-preserving** — a linear DoG / divisive gain control, *not* Ronan's ON-only
rectification — so polarity invariance survives, verified 1e-4):

- **(i) Thinning — confirmed, strongly.** DoG (σ_I=4) cuts a 13 px bar to **2.6 px**
  effective.
- **(ii) β-range widening — falsified.** In all three modes the true ends survive as
  keypoints **only at β=0.5**; β=0.7 still zeroes them. `DoG+divisive` is *worse*
  (4 → 12–14 spurious fires by gain-amplifying the skirt).

So thinning the input never sharpens the filter's energy peak — **the skirt is a
filter-scale property; the lever is `w_L`, and normalisation is orthogonal to it.**
This is the fourth prediction in this programme (after multi-scale `min`, "L/Ed
rejects crossings", "sharper λ_Ed fixes it") to be plausible from the geometry and
falsified by measurement. (See `EvenOddChannels.md` §8, corrected accordingly.)

**Distilled:** the only potentially significant win is effective **stroke-thinning
that buys back scale separation** — but *only if paired with shrinking `w_L`* — and
the cheap faithful version is **DoG + divisive normalisation**, not the recurrent
sheet (the steady state of a *linear* centre-surround network is just a fixed DoG;
recurrence only matters for the saturating nonlinearity, which a one-shot divisive
step approximates — and a dense recurrent sheet on our 224² field is 50k cells, so
you'd implement it convolutionally anyway).

---

## 3. In his code, how does he avoid the kernel going outside the image?

> *"In Ronan's code, how does he avoid the kernel going outside the image?"*

He doesn't avoid it — he lets taps fall outside and **treats anything out of bounds
as zero** (implicit zero-padding), checked per-tap at evaluation time. Three places:

**1. The oriented simple-cell convolution** — `correlate`, `RonanLiquid1.jl:283`.
Evaluated at every pixel, with a per-tap guard:
```julia
yy = y+j; xx = x+i
(1 <= yy <= H && 1 <= xx <= W) || continue   # line 290
```
Any kernel sample off the image is skipped — zero-padded on the fly.

**2. The end-stop displaced reads** — `bilinear`, `RonanLiquid1.jl:326`, called by
`end_stopped` at `y±dy, x±dx`. Returns zero off-grid:
```julia
(1 <= y <= H && 1 <= x <= W) || return 0.0   # line 328
```

**3. The Stage-1 E/I sheet** — no kernel to fall off at all. `gauss_weights`
(line 214) builds a *dense position-to-position* weight matrix over the cells that
exist, then unit-sum-normalises each row (`W ./= sum(W, dims=2)`). A border cell has
a truncated surround whose weights **renormalise to 1** — cleaner than zero-padding,
affordable because it's 784 cells, not a convolution.

**Why he can get away with zero-padding where we couldn't.** His field is 28×28 and
his kernels 11×11 (`hw=5`), so the border effect only bites within 5 px of the edge,
and EMNIST glyphs sit centred with margin. We went to the padded 224 field +
*edge-replicate* padding because our kernels are huge (up to 113 px, so zero-padding
would corrupt a large fraction of the letter) and because we wanted stroke
**end-caps near the border** to sit in genuine background rather than have their cap
smeared by zero/replicated ink — which matters precisely because our endpoint
detector lives on those caps.

---

## 4. But the on-off stage means the background isn't zero — so points 1–3 don't apply?

> *"Surely for your points 1-3, because of the on-off normalization stage, background
> is not zero, so what you say does not apply."*

You're right in general, and I was too glib — but it turns out not to bite for his
actual inputs, for a specific reason. Note first that Stage 2 convolves the
**on-map** (Stage-1 output), not the raw image, so the question is: *what is the
on-map's value in the background?* Running his Stage 1 verbatim:

| input | on-map min | far-bg | off-surround | border col |
|---|---|---|---|---|
| stroke on **0** background (EMNIST) | 0.0 | **0.0** | 0.0 | 0.0 |
| same, stroke **touching** the edge | 0.0 | 0.0 | 0.0 | **0.81** |
| stroke on **0.4 gray** background | 0.138 | **0.22** | 0.147 | 0.22 |

**Your principle is correct.** With a non-zero input background the on-map carries a
**pedestal** (~0.22 for a 0.4 background) — the on-off stage does *not* clean it to
zero. In that regime zero-padding is genuinely wrong, by a concrete mechanism: the
Stage-2 oriented kernels are DC-free (`ker .- mean(ker)`), so on a uniform pedestal
`p` they give 0 **in the interior**; but `correlate` *skips* out-of-bounds taps, so
the border kernel is **truncated and no longer sums to zero**, giving
`−p·Σ(dropped taps) ≠ 0` — a spurious edge all around the frame, proportional to the
pedestal, surviving the half-wave rectification. So "zero-pad = background, therefore
fine" was wrong as a general statement.

**Why it nonetheless doesn't hurt him:** EMNIST (and his synthetic glyphs) have a
background of *literally 0*, and the measurement shows the on-map background then
stays *exactly* 0 — not because of the padding, but because his transfer function is
**rectified** (`φ(x)=tanh(max(0,x))`): uniform-0 input → 0 drive → 0 output, and even
the inhibited off-surround ring is clamped to 0 rather than pushed negative. Zero
input → zero on-map → zero-padding happens to match. What saves him is
**rectification + a literally-zero background**, not the padding.

Two refinements to the original points:

- **Points 1–2 (the zero-fills, `correlate` and `bilinear`)**: only safe because the
  EMNIST on-map background is 0. On any pedestal they'd artifact.
- **Point 3 (Stage-1 `gauss_weights`)** is the one that's *actually* robust: it
  renormalises each cell's incoming weights to sum to 1, so a truncated border
  neighbourhood still integrates correctly regardless of background level. The row-3
  border column reads a flat 0.22 (the pedestal), not an edge spike — confirming
  Stage 1 itself has no border artifact.

And row 2 stands independently: a stroke *reaching* the edge puts 0.81 on the border,
which zero-padding truncates — he escapes it only because glyphs are centred, not
because of anything the on-off stage does.
