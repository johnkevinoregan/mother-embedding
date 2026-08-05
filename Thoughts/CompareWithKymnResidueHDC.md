# The half of the FPE architecture we never built

*On Kymn, Kleyko, Frady, Bybee, Kanerva, Sommer & Olshausen, "Computing With Residue Numbers in
High-Dimensional Representation", Neural Computation 37:1–37 (2025),
[doi:10.1162/neco_a_01723](https://doi.org/10.1162/neco_a_01723). Local copy:
`KymnOlshausenResidue.pdf` in the repo root, excluded from the repo via `.git/info/exclude`.*

*Nothing here is a new experiment. It is a comparison, one experiment it suggests, and a
confirmation of a limit this project had already reached by a different road.*

---

## Why this is the closest paper to `P0.3_New_Gabor_FPE/`

It is about the machinery that phase is built on. Their Definition 2 **is** our FPE — Plate's
fractional power encoding, a random phase vector exponentiated componentwise:

```math
z(x) \;=\; z^{x} \;=\; \big[\exp(i\varphi_1 x),\, \dots,\, \exp(i\varphi_D x)\big]
```

Their Definition 4 is what `New_Gabor_FPE_handoff_for_claude-code.md` calls **Bug A**. Both are the
same statement: `z` is periodic with period `P` exactly when every base phase is an integer multiple
of `2π/P`. We fix `P = 2π` because the variable is an angle, which makes the frequencies integers;
they fix `P = m` for integer `m`, which puts the phases on the *m*th roots of unity. Same condition,
two parameterisations, arrived at independently.

Their new step is to stop using one period. Bind several moduli, and the Chinese remainder theorem
makes the composite unique over the product of them:

```math
z(x) \;=\; \bigodot_{k=1}^{K} z_{m_k}(x), \qquad \text{unique for } 0 \le x < M = \prod_k m_k
```

Addition of encoded values is the Hadamard product, as in ours. What is new is **decoding**: a
resonator network (Frady et al. 2020) factorises the composite back into its per-modulus parts, so
recovering a value of range `M` costs `Σₖ mₖ` codebook vectors rather than `M`. For `{3,5,7}`,
fifteen vectors instead of 105. Capacity is quadratic in `D` — they report `C(4096) > 2×10⁶` — and
with the codebook budget held fixed, range grows as Landau's function, exponentially.

Their Table 1 scores plain FPE — the thing we use — as ✗ on efficient decoding and ∼ on
expressivity. Fixing that column is the entire paper.

## Their equation 5.5 is our architecture

```math
s \;=\; \sum_{j,x,y} h(x) \odot v(y) \odot d_j \cdot A_j(x,y)
```

`A_j(x,y)` are sparse feature maps, `d_j` is a random vector naming feature `j`, `h` and `v` are the
position encodings. Set that beside §4 of the handoff doc — dense oriented maps, sparse keypoint
selection, "FPE bundle over keypoints, bound to object-relative polar position". It is the same
equation.

The difference is what produces `A_j`. Theirs is **convolutional sparse coding learned on MNIST**,
justified as "mirroring the neural representation in primary visual cortex". That is exactly the
slot our front end is built to occupy, with the difference that ours is designed rather than fitted
to a dataset, and is exactly polarity-invariant, which a learned dictionary is not.

They report one thing about that slot which is directly useful: the sparse coding step "helps to
decorrelate image patterns, thus achieving higher accuracy and faster convergence for the resonator
network". **Resonator convergence speed is a figure of merit for a front end**, and it is unlike
anything currently in this repo — no labels, no readout, no train/test split.

## The half we never built

Our FPE descriptors are write-only. We bundle, and then compare by inner product; nothing ever
factorises a bundle back into its parts. The resonator is that missing half, and equation 5.7 is why
it matters:

```math
s \;=\; h(x') \odot v(y') \odot O^{(i)}
```

Placing the object at `(x', y')` **binds** a position vector onto the scene vector. Identity and
position are then separable by factorisation.

### Two papers, one hole, two different fixes

[[CompareWithOsarioShapeMatters]] identified the same gap — our `g × g` retinotopic pooling grid has
no translation or scale invariance — and the two answers are not equivalent:

| | mechanism | what happens to position |
|:--|:--|:--|
| Osório et al. | object-centre on the ink centroid, normalise by `r_max` | **discarded** |
| Kymn et al. | bind `h(x) ⊙ v(y)`, factorise with a resonator | **recovered as a factor** |
| ours today | fixed retinotopic grid | retained but not invariant |

Kymn's is strictly more informative: *what* and *where* both survive in one vector, which is the
thing the project memory says FPE is wanted for. Osório's is a great deal cheaper — a centroid and a
max, against a codebook and an iterative dynamical system.

## The limit this confirms rather than lifts

The project's position has been that FPE is the wrong tool for the early AND-conjunctions, because
binding adds exponents and so products would need log-encoded magnitudes. This paper introduces a
multiplication operator and **does not rescue that**.

Their `⊘` multiplies the encoded *integers*, `z(x₁) ⊘ z(x₂) = z(x₁ x₂)`, and it needs three things:
prime moduli, access to the individual per-modulus base vectors (or a resonator run to recover
them), and integer arguments. Section 2.5 is explicit that for rationals multiplication is not well
defined — exponentiation by a rational power is multivalued, `(−1²)^{0.5} = ±1` against
`(−1^{0.5})² = −1`, and commutativity fails.

Our `A₁ = Σₖ Eₖ E_{k+h}` is a product of continuous energies at a pixel. It stays outside the
algebra. So the dense/sparse split in the handoff diagram is **forced**, and the SNR argument given
there (`√(d/N)`, hopeless at `112² × 48 ≈ 600k` dense tokens) is only half the reason — the other
half is that the algebra cannot express the operator anyway. Worth having the citation, from the
people who built the algebra.

The same point in reverse is the interesting one: their framework is integer-valued end to end,
while every feature this project produces is a graded continuous property. That is the whole premise
of Phase 9. Subinteger decoding (§2.5, and Figure 4b's Dirac-comb-convolved-with-sinc) bridges part
of the gap and gives up multiplication to do it.

## What is transferable, concretely

**Residue encoding of image position.** A 112-px axis needs 112 codebook vectors per axis in the
standard scheme; moduli `{3,5,7}` give a range of 105 from 15 vectors, and `{3,5,11}` gives 165 from
19. Two axes: 38 vectors against 224. This is the first thing that meets the handoff doc's SNR
objection with an actual remedy rather than with a restriction to a few dozen keypoints.

**Where residues would and would not apply in what we already have.** Not to the angular variable —
`α` is genuinely 2π-periodic, one modulus is correct, and `ALPHA_FREQS = RAY_FREQS` is right as it
stands. But `RHO_FREQS` in the log-polar global descriptor (`TASK_add_global_shape.md`) is drawn
`randn`: continuous, non-periodic, unbounded range. That is exactly the case residues address.

**Hexagonal coordinates.** Their §2.4.2 projects 2D position into a nonnegative triangular
("Mercedes-Benz") frame and enforces `z([1,1,1]) = z([0,0,0])` by making the three phases sum to
zero mod 2π — so equal movement in all three directions cancels, and every path to a position gives
the same vector. A hexagonal system with modulus `m` has `3m² − 3m + 1` states from `3m` codebook
vectors, against `m²` from `2m` for a square lattice: roughly triple the range for a 50 % storage
increase, with the six-fold kernel symmetry of grid cells. Our pooling grid is square by default and
nothing in the front end forces that.

**Bits per vector** (their §5.4.4, from Frady et al. 2018) as a metric that charges for accuracy and
for the number of states distinguished at once, going to zero at chance. A cleaner currency than
accuracy for anything with a capacity–resolution trade-off.

## Where to be sceptical

**The vision demo is thin.** One object on a blank field, ten MNIST digits, two factors of variation
(horizontal and vertical position). No scale, no rotation, no second object, no occlusion. And the
"objects" are a **known codebook of ten templates** — this is a search over a fixed dictionary, not
recognition of a novel shape. The headline result is 40 codebook vectors against 220 and ~800
evaluations against ~2085, which is a real result about the *factorisation* and not evidence that
the representation supports vision. Their §3.1 framing as "the disentangling problem in vision"
is considerably larger than what is demonstrated.

**The grid-cell mapping is not clean and they say so.** The 2D kernel is hexagonal, but the
individual vector elements are phasors with *band-like* receptive fields (Krupic et al. 2012), not
hexagonal ones. They flag reconciling this as future work.

**The operations are not neuron-shaped.** Componentwise complex multiplication and modular
multiplicative inverses computed with Python's `pow`. They are candid: these "are not easily
realized in perceptron-like neurons", and the offered routes — spike-timing phasor codes, dendritic
nonlinearities — are gestures rather than models. For a project whose criterion is biological
plausibility ([[CompareWithGrueningBarth]] applies the same test to Min-Nets), this is the weak
point.

## The experiment it suggests

**Swap their convolutional sparse code for our front end in equation 5.5**, keep their resonator,
keep their MNIST disentangling task, and measure iterations to convergence.

The claim under test is theirs, not ours: that a front end which decorrelates image patterns makes
the factorisation converge faster. If a designed, polarity-invariant, multi-scale bank beats a
dictionary learned on the test distribution, that is the first evaluation of these features by
something that is not a readout trained on labels — no classifier, no train/test split, no
protocol confound of the kind the root `README.md` warns about.

It also serves the stated goal directly: composing local features into object descriptors with pose
bound in. We have the local features and the bundling. This paper has the factorisation that reads
them back out.
