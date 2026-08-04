# Feature-type layer — design notes

How the mid-level layer in `feature_layer.py` turns the fixed-scale Gabor
oriented-energy stack into typed keypoints (**endpoint / corner / T / X**),
and the choices that were needed to make it behave. Written as a companion to
the code: each function named below is in `feature_layer.py`.

---

## The one idea that made it work: **propose ≠ classify**

The first attempt used the **ring/spoke count as the single authority** — sample
a circle around every strong pixel, count contour branches, and label the pixel
from that count. It failed: noisy, wildly radius-sensitive, and it mislabelled
corners as endpoints and vice-versa.

The fix is a strict division of labour:

| Stage | Who does it | What it decides |
|-------|-------------|-----------------|
| **Propose + coarse class** | the two *dense, per-pixel* profile operations | *where* a feature is, and *corner-family vs endpoint* |
| **Refine only** | the *sparse, geometric* ring count | corner → **T** → **X** (junction order) |

So the ring/spoke count is **only used to refine an already-proposed point**,
never to propose one and never to make the corner-vs-endpoint call. The reason:
"is this an orientation change or a termination?" is answered robustly and
everywhere by the local profile, whereas the ring is a fragile geometric probe
that is only trustworthy once you already know roughly where to look and just
need to count branches.

This is the meta-lesson worth carrying into round 2: **dense cheap signals for
detection and coarse typing; sparse geometric probes for refinement only.**

---

## Choice 1 — the coarse class comes from *which operation fires*

Not from thresholding one score, but from *which* nonlinearity responds
(`detect`):

- **corner-family** ⟺ the orientation profile is genuinely **bimodal**
  (two real orientation peaks) → `cornerness`
- **endpoint** ⟺ a **single** orientation whose energy **stops** → `endpointness`
  (from `end_stopping`)

Keeping these two as separate proposal streams stops them contaminating each
other (a corner is also the "end" of each of its segments, so end-stopping fires
there too — but the bimodality gate claims it for the corner stream first).

---

## Choice 2 — demand *genuine* bimodality, or every curve becomes a polygon

The naive "height of the second-highest peak" fires on smooth curves: at one
scale a curving contour has a **broad, rotating** peak that looks like two nearby
peaks. Result: an `S` sprouts a string of false corners.

`peak_maps` therefore requires all three of:

1. **angular separation** — second peak ≥ `sep_bins` away (≈ 30–45°), not a shoulder;
2. **comparable strength** — `second > 0.5 * first`;
3. **a real valley** — the profile *between* the two peaks must dip below
   `0.62 ×` the smaller peak.

The valley test (3) is the one that actually kills curves: a smooth curve is
unimodal with no dip; a true corner has two peaks with a clear gap between them.

---

## Choice 3 — endpoints: ridge gate + end-stop distance

Two failure modes with `end_stopping`:

- **Interior/background false positives** → gate endpoint candidates to sit on a
  real ridge: `M > 0.35 * M.max()`.
- **Localisation drifts inward** → end-stopping ramps up *approaching* a
  termination while the modulus falls off at the very tip, so the raw score peaks
  a few pixels inside the true end. A moderate end-stop distance
  (`end_stop_d ≈ 12`) keeps the response concentrated near the tip. (This is
  still imperfect — see limitations.)

---

## Choice 4 — the ring count: **multiple radii, confirmed twice**

No single ring radius works, because the right radius depends on the feature:

- a sharp **apex** resolves its two branches at `r ≈ 16`, but collapses to one at
  larger `r`;
- an **X-crossing** needs `r ≈ 20` before the four arms separate from the bright
  merged core;
- **too small** and the ring sits *inside* the junction blob and reads "all-on"
  (the `99` sentinel in `arc_count`).

So `classify_point`:

1. samples **several radii** `(15, 18, 21, 24)`;
2. ignores degenerate rings (all-on / empty);
3. accepts a junction order of **3 (T) or 4 (X) only if confirmed at ≥ 2 radii** —
   this rejects single-radius noise spikes that would otherwise fake a T on a
   straight stroke;
4. runs a **collinearity test**: two arcs ≈ 180° apart = a straight stroke passing
   through → rejected, not a corner.

---

## Choice 5 — NMS + cross-stream dedup

Both proposal maps can fire near the same feature. Each map gets non-maximum
suppression (`maximum_filter`), and when the two streams are merged a
min-distance check (`< 11 px`) prevents a single physical corner/endpoint being
emitted twice with different labels.

---

## Choice 6 — analysis substrate (minor but relevant)

Glyphs are upsampled 4× (28→112, cubic) and given a small Gaussian **noise floor**
(σ = 0.03) before analysis, so the flat-region argmax is honest (noise, not a
perfectly-zero background) and curvature is continuous rather than blocky.

---

## The big one: **everything here is single-scale**

All of the above runs at **one Gabor scale** (`lam ≈ 12` on the upsampled glyph).
Three consequences follow directly, and they're not bugs so much as facts about a
fixed scale:

1. **Corner vs curve is scale-relative.** A curve whose radius ≈ the filter size
   reads as a sequence of corners, because one receptive field spans two
   appreciably different tangent orientations. The bimodality gate cannot tell a
   tight curve from a polygon at that scale.
2. **Junction resolvability depends on scale vs stroke width / arm spacing.** This
   is why the `A` crossbar-to-leg meets currently under-read as **corner** rather
   than **T**: the inward crossbar stub is short relative to the ring, so only two
   branches are resolved.
3. **Endpoint localisation is tied to the end-stop distance**, which is itself a
   scale.

### → the round-2 fix
Run the layer at **≥ 2 scales** and require a corner-family point to survive as
**bimodal across scales**: a true corner stays bimodal at the finer scale, while a
curve resolves into a single *rotating* orientation and drops out. Junction order
and endpoint localisation similarly get a vote across scales. This is the change
that makes the `S` behave, and it's roughly fifteen lines around `peak_maps` and
`detect`.

---

## Map of choices → code

| Concern | Function | Key knobs |
|---|---|---|
| oriented-energy stack | `energy_stack` | `Config.lam, sigma_n, sigma_t, ksize, n_orient` |
| endpoints (termination) | `end_stopping` | `end_stop_d` |
| corners (orientation change) | `peak_maps` | `relthr, sep_bins`, valley factor `0.62` |
| junction order | `arc_count`, `classify_point` | `radii`, `relthr`, `min_run`, collinear tol |
| proposal + coarse class + merge | `detect` | ridge gate `0.35`, NMS `md=11`, dedup `11px` |

---

## Known limitations (all trace back to single-scale)

- **`A` crossbar meets labelled *corner*, not *T*** — ring resolves only 2 branches
  at this glyph/scale (Choice-4 / scale point 2).
- **`T` stem endpoint sits a few px above the true foot** — end-stopping ramp +
  modulus fall-off at the tip (Choice 3).
- **A smooth `S` reads as corners at `lam ≈ 12`** — the headline scale limitation;
  fixed by round-2 multi-scale survival.

---

## What this buys the downstream code

Each emitted keypoint is a `(type, y, x)` triple. Pairing `type` with an
**object-relative position** (angular slot around the glyph centroid) gives exactly
the role–filler structure the VSA layer wants to bind
(`corner ⊗ angular-position`, `endpoint ⊗ angular-position`) — i.e. most of a
descriptor like *"pointy corner at top; two endpoints bottom-left/right"* is
already recoverable from the Gabor field, before any learned patch dictionary.
Swapping the ring's absolute arc angles for centroid-relative ones is the other
small round-2 change.
