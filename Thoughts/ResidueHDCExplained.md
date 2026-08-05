# Residue hyperdimensional computing, explained from scratch

*An explanation of Kymn, Kleyko, Frady, Bybee, Kanerva, Sommer & Olshausen, "Computing With Residue
Numbers in High-Dimensional Representation", Neural Computation 37:1–37 (2025). Local copy:
`KymnOlshausenResidue.pdf` in the repo root.*

**Written to be read cold.** It assumes no knowledge of hyperdimensional computing, vector symbolic
architectures, residue number systems, or resonator networks, and builds each one from nothing.
Every term is defined where it first appears and again in the glossary at the end. The
project-specific comparison is a separate note, [[CompareWithKymnResidueHDC]]; this one is just the
paper.

---

## The question the paper is answering

Suppose you want to represent a number — a position, a velocity, a colour — as a pattern of activity
spread across a thousand neurons. Not one neuron per value, but a pattern in which every neuron
participates in representing every value.

You want four things from such a scheme, and the paper's Table 1 is a scorecard of earlier schemes
against exactly these:

1. **Algebra.** Simple operations on the patterns should perform arithmetic on the numbers. Combine
   the pattern for 3 with the pattern for 5 and get the pattern for 8, without ever decoding.
2. **Expressivity.** The range of numbers you can encode should grow *fast* as you add neurons —
   faster than linearly.
3. **Efficient decoding.** Getting the number back out should be cheap. Not "compare against a
   stored list of every possible value".
4. **Robustness.** Corrupt a good fraction of the pattern and the number should still be readable.

Their claim is that no previous scheme got all four, and theirs does. The name of theirs is
**residue hyperdimensional computing (RHC)**, and it is two old ideas glued together: an ancient
piece of number theory, and a 1990s scheme for computing with random vectors.

---

# Part 1 — The substrate

## 1.1 The one fact that makes all of this work

Take two vectors of length `D`, each with random entries, and measure the angle between them.

In two dimensions they are as likely as not to point in similar directions. In three, less so. In a
thousand dimensions, they are **almost certainly nearly perpendicular** — the normalised inner
product between them is on the order of `1/√D`, so for `D = 1000` it is about 0.03, and for
`D = 10,000` about 0.01.

This is the whole foundation. It means:

- You can generate as many random vectors as you like and treat each as a **distinct symbol**, with
  essentially no interference between them. There is no need to arrange them carefully; randomness
  is enough.
- "Is this pattern the symbol `A`?" is answered by one inner product, and the answer is close to 1
  for yes and close to 0 for everything else.
- The scheme degrades gracefully. If a tenth of the components are destroyed, the inner product with
  the right symbol drops a little but stays enormously larger than the ~0.03 you get from a wrong
  one.

The field built on this observation goes by three interchangeable names: **holographic reduced
representations** (Plate, 1994), **vector symbolic architectures** (Gayler, 2003), and
**hyperdimensional computing** (Kanerva, 2009). They mean the same thing.

## 1.2 Phasor vectors

The paper's vectors are complex. Each of the `D` components is a complex number of magnitude one:

```math
z \;=\; \big[e^{i\varphi_1},\; e^{i\varphi_2},\; \dots,\; e^{i\varphi_D}\big]
```

Only the angles carry information. Picture the vector as **`D` clock faces**, each with a single
hand, each hand at its own angle. A "pattern of activity" is a set of `D` angles.

This is not as biologically odd as it looks: a phase is a natural thing for a spiking neuron to
carry, in the timing of its spike relative to an ongoing oscillation. The paper cites Frady & Sommer
(2019) for that, and is candid that the arithmetic below is otherwise not very neuron-shaped.

## 1.3 The two operations

Everything in the algebra is built from exactly two operations.

**Binding**, written `⊙`, is componentwise multiplication (the Hadamard product). Two unit complex
numbers multiply to another unit complex number, and their angles **add**:

```math
e^{i\alpha} \cdot e^{i\beta} \;=\; e^{i(\alpha + \beta)}
```

So binding two phasor vectors adds all `D` pairs of angles independently. Its properties:

- The result is **quasi-orthogonal to both inputs** — binding produces something new, not something
  in between.
- It is **reversible**: bind with the complex conjugate (all angles negated) and you get the
  original back, because `α + β − β = α`.
- It is **commutative and associative**, so you can bind several things in any order.

Binding is how you glue two things together: *this feature* **at** *this place*, *this role* **with**
*this filler*.

**Bundling** is ordinary addition. Add several vectors and the sum is similar to each of its parts —
its inner product with any ingredient is large. This is how you put several things into one vector
at once, a **superposition**.

The price of bundling is signal-to-noise. Superpose `N` items into `D` dimensions and the inner
product with any one of them stands out above the noise by roughly `√(D/N)`. Bundle a few dozen
things into a thousand dimensions and readout is clean; bundle a hundred thousand and it is
hopeless.

**That is the whole algebra.** Bind to compose, bundle to collect, inner-product to ask questions.

---

# Part 2 — Making numbers out of it

## 2.1 Why symbols are not enough

Assigning a fresh random vector to each number would satisfy nothing. The vector for 5 would be no
more similar to the vector for 6 than to the vector for 4000, so the code carries no notion of
*nearby*; and there would be no way to add two of them.

What we want is a code where **similar numbers get similar vectors**, and where **arithmetic on the
numbers is one of our two operations on the vectors**.

## 2.2 Fractional power encoding

The trick, due to Tony Plate in 1992, is beautifully simple. Pick **one** random phasor vector `z`,
call it the base. Encode the number `x` by raising the base to the power `x`, componentwise:

```math
z(x) \;=\; z^{x} \;=\; \big[e^{i\varphi_1 x},\; e^{i\varphi_2 x},\; \dots,\; e^{i\varphi_D x}\big]
```

This is **fractional power encoding**, FPE. "Fractional power" because `x` need not be a whole
number — you are allowed to raise the base vector to the power 2.5.

Return to the clock-face picture and it becomes obvious. Each component `k` is a clock whose hand
turns at its own fixed rate `φ_k`. **Encoding the number `x` means letting all `D` clocks run for
time `x`** and photographing them. The number is the elapsed time; the code is the snapshot.

Two consequences follow immediately, and they are the two things we asked for.

**Adding numbers is binding vectors.** Let the clocks run for `x₁`, then let them run `x₂` more.
Running-then-running-more is the same as running for the total, and "running more" is exactly what
multiplying by another snapshot does, since angles add:

```math
z(x_1) \odot z(x_2) \;=\; z(x_1 + x_2)
```

The most useful line in the paper. Addition of the *encoded quantity* is componentwise multiplication
of the *vectors* — a parallel, local, carry-free operation on `D` independent components.

**Similarity becomes a smooth bump.** Compare the snapshots at `x₁` and `x₂`:

```math
K(x_1, x_2) \;=\; \frac{1}{D}\,\mathrm{Re}\big\{ z(x_1)^{\mathsf T}\, \overline{z(x_2)} \big\}
\;=\; \frac{1}{D}\sum_k \cos\big(\varphi_k (x_1 - x_2)\big)
```

It depends only on the **difference** `Δ = x₁ − x₂`. At `Δ = 0` every cosine is 1 and the similarity
is 1. As `Δ` grows the clocks fan out — each at its own rate — the cosines scatter over the circle,
and the average collapses towards zero. So the code has a **similarity bump** around each value:
near numbers are near vectors, distant numbers are quasi-orthogonal.

A function that reports how similar two things are is called a **kernel**. FPE therefore *is* a
kernel machine, and the shape of the bump — how wide, how it falls off — is set entirely by the
probability distribution the rates `φ_k` were drawn from. Draw them from a Gaussian and you get a
Gaussian bump. This is the same mathematics as the "random Fourier features" trick in machine
learning (Rahimi & Recht, 2007): sample frequencies at random, and inner products of the resulting
features approximate a chosen kernel.

## 2.3 Making the code wrap around

Now the step this paper turns on.

Ask when the code **repeats** — when `z(x + P) = z(x)` for every `x`. That needs every clock to have
returned to its starting angle after time `P`, that is, `φ_k P` must be a whole number of turns for
every `k`:

```math
z(x+P) = z(x) \quad\text{for all } x
\qquad\Longleftrightarrow\qquad
\varphi_k \;=\; \frac{2\pi n_k}{P}, \;\; n_k \in \mathbb{Z}, \;\text{for every } k
```

Draw the rates at random from anywhere and this essentially never happens; the code never repeats.
Draw them from that restricted set and the code repeats **exactly**, with no approximation and
regardless of the dimension `D`.

Such a code represents not `x` but **`x` modulo `P`** — the remainder of `x` after dividing by `P`.
Values that differ by a whole number of `P` get literally the same vector.

The paper's Definition 4 is this statement with `P` an integer `m`, which makes the allowed rates
the `m`th roots of unity. Choose them uniformly among those and the similarity bump becomes a comb:
exactly 1 when the two numbers are congruent mod `m`, and ≈ 0 at every other integer.

---

# Part 3 — Residue numbers

## 3.1 The ancient idea

A **residue number system** represents an integer not by its digits but by its **remainders** after
division by several fixed numbers, called the **moduli**.

With moduli `{3, 5, 7}`, the number 20 is represented by

```
20 mod 3 = 2      20 mod 5 = 0      20 mod 7 = 6      →   (2, 0, 6)
```

The **Chinese remainder theorem** — Sunzi, around the 4th century — says that if the moduli share no
common factors, this representation is **unique** for every integer from 0 up to their product. With
`{3, 5, 7}` that product is `3 × 5 × 7 = 105`, so every integer in 0…104 has its own distinct triple
of remainders, and 105 wraps back round to `(0,0,0)`.

The gear picture: three gears with 3, 5 and 7 teeth, all turned by one shaft. Each gear's position
tells you very little. The *combination* of the three positions does not repeat until the shaft has
turned 105 times.

Note the arithmetic on offer. **Range multiplies while cost adds.** Three small counters, of sizes
3, 5 and 7 — fifteen states of storage in total — pin down a range of 105.

## 3.2 Carry-free arithmetic

The other classical virtue is that addition, subtraction and multiplication are performed **on each
remainder independently**, with no carrying between them.

```
20 + 30 = 50

(2,0,6) + (0,0,2)  =  (2 mod 3, 0 mod 5, 8 mod 7)  =  (2, 0, 1)
50 mod 3 = 2 ✓     50 mod 5 = 0 ✓     50 mod 7 = 1 ✓
```

Ordinary binary addition has to propagate carries from the low bits to the high ones, which is
inherently sequential. Residue arithmetic has no such coupling — every modulus can be handled at the
same time, in parallel, by a separate piece of machinery that knows nothing about the others. This is
why residue systems are used in fault-tolerant hardware and in signal processing.

The classical cost is that **comparing** two residue numbers is awkward — nothing in `(2,0,6)`
obviously says whether it is bigger than `(1,3,2)` — and division is not available.

## 3.3 Gluing the two ideas together

Here is the whole construction of the paper, and it is one line.

Build a periodic FPE code for each modulus `mₖ` — rates drawn from the `mₖ`th roots of unity, so the
code wraps at `mₖ` — and **bind them all together**:

```math
z(x) \;=\; z_{m_1}(x) \odot z_{m_2}(x) \odot \dots \odot z_{m_K}(x)
```

Each factor encodes `x mod mₖ`. The bound composite encodes the whole tuple of remainders, which by
the Chinese remainder theorem pins down `x` uniquely over the range `M = ∏ₖ mₖ`.

Three properties come for free:

- **Addition is still binding**, because each factor's addition is binding and binding is
  componentwise: `z(x₁) ⊙ z(x₂) = z(x₁ + x₂)`.
- **The similarity bump is the product of the individual combs.** It is 1 only where *every* modulus
  agrees — that is, when the two values differ by a multiple of `M` — and ≈ 0 elsewhere. A sharp
  spike built from several coarse combs.
- **The code is fully distributed.** In a conventional residue system each remainder lives in its
  own register. Here every one of the `D` components carries information about every modulus at
  once, which is what buys the robustness.

Distinct integers therefore behave like quasi-orthogonal symbols, but unlike symbols you can do
arithmetic on them.

---

# Part 4 — Getting the number back out

Encoding is easy. The hard half is **decoding**: given the vector `z(x)`, recover `x`.

## 4.1 The obvious method, and why it is expensive

**Codebook decoding**: precompute the vector for every possible value, take the inner product of
your vector with all of them, and report the argmax.

```math
\hat{x} \;=\; \arg\max_{x_k}\, \big\langle z(x), z(x_k) \big\rangle
```

For a range of `M` this needs `M` stored vectors and `M` inner products. Over a range of 100,000
that is 100,000 stored vectors of length `D`. Hopeless.

## 4.2 The re-framing

Residues change the shape of the problem. The composite is a **product of `K` unknown factors**,
where factor `k` is known to be one of only `mₖ` possibilities:

```math
z(x) \;=\; \underbrace{z_{m_1}(x)}_{\text{one of } m_1} \odot \dots \odot \underbrace{z_{m_K}(x)}_{\text{one of } m_K}
```

So decoding is a **factorisation** problem. If you can find the factors, each is cheap to identify
against its own tiny codebook, and the Chinese remainder theorem reassembles `x`.

The catch is that the number of *combinations* is still `M = ∏ mₖ`. Trying them all gains nothing.
You need a method that searches the combinations without enumerating them.

## 4.3 The resonator network

That method is the **resonator network** (Frady et al., 2020; Kent et al., 2020). It is an iterative
guessing procedure with one very unusual feature.

Keep a running estimate `ẑⱼ` of each factor. Then repeat:

1. **Unbind the others.** Take the target vector and bind it with the conjugates of your current
   estimates of every *other* factor. Because binding is reversible, this cancels them out and
   leaves an estimate of factor `j` — contaminated by however wrong the other estimates were.
2. **Clean up.** Project that estimate onto the legal values for factor `j`: compare it against
   every entry in factor `j`'s codebook, then rebuild it as a **weighted blend of all of them**,
   with weights given by those comparisons.
3. **Normalise**, resetting every component's magnitude to one while keeping its angle.

In symbols, with `Zⱼ` the matrix whose columns are the legal vectors for factor `j`:

```math
\hat{z}_j(t+1) \;=\; g\Big( Z_j Z_j^{\dagger} \Big[\, z \odot \bigodot_{i \neq j} \overline{\hat{z}_i(t)} \,\Big] \Big)
```

`Zⱼ†` does the comparing, `Zⱼ` does the rebuilding, `g` does the normalising. The paper updates one
factor at a time and stops when successive states agree to a cosine similarity of 0.95.

**The unusual feature is step 2.** A conventional search would commit to the single best candidate.
This one keeps a *superposition* of all candidates, weighted by how well each fits. So each factor is
simultaneously entertaining many hypotheses, and each is being pruned by the current hypotheses of
the others. The network is searching an exponential number of combinations in parallel, with the
number of *operations* per step set by the sizes of the individual codebooks rather than by their
product. Wrong combinations interfere destructively; the right one reinforces — hence "resonator".

It is not guaranteed to converge. It can stall in a wrong stable state, in which case you restart it
from a fresh random initialisation. An algorithm that is always correct when it answers but takes a
random number of attempts is called a **Las Vegas algorithm**, and the paper verifies theirs behaves
like one — the chance of success after 10 restarts matches 10 independent draws at the single-run
rate.

## 4.4 What this buys, in numbers

Decoding a range `M` with moduli `{3,5,7}`:

| | codebook vectors | 
|:--|--:|
| plain codebook decoding | `M = 105` |
| residue + resonator | `3 + 5 + 7 = 15` |

A factor of 7 here, and it grows: the storage is `Σ mₖ` while the range is `∏ mₖ`.

Their measured scaling results:

- **Capacity grows quadratically with dimension.** Defining capacity as the largest range still
  decoded at 95 % accuracy, they fit `C(D) ≈ 0.13 D² + 24 D − 3700`, giving `C(4096) > 2×10⁶`.
  Accuracy is near-perfect up to the capacity and then falls off a cliff.
- **More moduli trade accuracy for range.** With more factors the resonator is a little less
  reliable, but for a *fixed codebook budget* the reachable range explodes. If you may spend `b`
  codebook vectors in total, the best possible range is Landau's function `g(b)` — the largest
  product achievable from numbers summing to `b` — which grows like `e^{√(b ln b)}`. Roughly:
  **exponential range for linear storage.**
- **Noise barely matters.** They perturb every phase with von Mises noise (the circular analogue of
  a Gaussian, with concentration `κ`: large `κ` means tight, `κ → 0` means uniformly random) and
  find capacity degrades gradually rather than breaking.

---

# Part 5 — Three extensions

## 5.1 More than one dimension

To encode a 2D position, encode each coordinate separately and bind:

```math
z(\mathbf{x}) \;=\; z_1(x_1) \odot z_2(x_2)
```

using independent base vectors for the two axes. Binding two of these adds the positions
componentwise; the similarity bump is the product of the two 1D bumps, so it is a localised blob in
the plane.

## 5.2 Hexagons, and grid cells

**Grid cells**, in the medial entorhinal cortex (Hafting et al., 2005), fire at positions forming a
regular **triangular lattice** across an environment, so one cell has many firing fields. A
population of them encodes position with a resolution that grows *exponentially* in the number of
cells, rather than linearly as a set of ordinary place-tuned cells would — and Fiete and colleagues
(2008) pointed out that the reason is essentially the residue argument of §3.1: several modules with
different lattice spacings, each ambiguous alone, jointly unambiguous over a huge range.

This paper builds that lattice explicitly. Instead of two perpendicular axes it uses **three axes at
120°** — the "Mercedes-Benz" frame. Three coordinates for a two-dimensional space is one too many,
so the redundancy has to be removed, and they remove it by requiring

```math
z\big([1,1,1]\big) \;=\; z\big([0,0,0]\big)
```

*moving one step along all three directions gets you back where you started* — which is
geometrically true for three vectors at 120°, and which turns out to be enforced simply by making
the three phases in each component sum to zero. A pleasant consequence is that different routes to
the same place produce the same vector.

The payoff is packing efficiency, which is why hexagons appear everywhere from honeycombs to grid
cells:

| lattice | distinct states | codebook vectors |
|:--|--:|--:|
| square, modulus `m` | `m²` | `2m` |
| triangular, modulus `m` | `3m² − 3m + 1` | `3m` |

About three times the range for a 50 % storage increase, and a higher-entropy code for the same
resources. The similarity bump inherits the six-fold symmetry of a grid cell's firing map.

They are honest about the imperfection: the *population* kernel is hexagonal, but the individual
vector components have **band-like** receptive fields — stripes across the environment, not
hexagons. Real grid cells are hexagonal individually. Reconciling that is left as future work.

## 5.3 Between the integers

Everything so far assumed whole numbers. FPE itself is happy with fractions — the base vector can be
raised to the power 40.4 — and it turns out that the code for a non-integer is still a stable state
of the resonator network, **even when the codebooks contain only integers**.

What identifies the fraction is *how far short of 1* the best inner product falls: the pattern of
inner products against the integer codebook is a comb of spikes convolved with a `sinc` function, so
the shortfall at the nearest integer, and the small non-zero values at its neighbours, together pin
down the offset. Decode to the nearest integer first, then generate a fine codebook around it and
decode again.

The catch is stated plainly in the paper: for non-integers, **multiplication stops being well
defined** (see §5.4 below). Raising to a fractional power is multivalued — `(−1)²` then square-rooted
gives `±1`, while `−1` square-rooted then squared gives `−1` — so the operation stops commuting.

## 5.4 A second binding operation: multiplying the values

Binding adds encoded values. Can anything **multiply** them? Previous work in this field had only
addition (or multiplication via logarithms), and this is one of the paper's genuine novelties.

To multiply the values you must multiply the *exponents*, and exponents are not what binding
combines. Their solution: when `x` is an integer, each component of `z_m(x)` is itself an `m`th root
of unity, so its angle can be read as an integer `r` in `0…m−1`. Define an operation that takes two
such components and returns `exp(2πi·rs/m)` — multiplying the two integers directly. Apply it
componentwise.

Doing that overshoots by one extra factor of the base phase, which has to be cancelled. The
cancellation binds with an **anti-base** vector whose phases are the *modular multiplicative
inverses* of the base's — the number `v` with `uv ≡ 1 (mod m)`; for example the inverse of 3 mod 5
is 2, since `3 × 2 = 6 ≡ 1`. Such inverses exist for every non-zero value only when the modulus is
**prime**, so this operation requires prime moduli.

Read the price list, because it is what limits the whole framework:

- integer arguments only, since fractional powers do not commute;
- prime moduli, so the inverses exist;
- access to the individual per-modulus factors, or a resonator run to recover them.

---

# Part 6 — What they do with it

## 6.1 Pulling a scene apart

The **disentangling** problem: given an image, recover the separate causes that produced it — what
object, at what position, at what orientation, under what lighting. It is hard because the causes
combine multiplicatively, so the number of possible combinations explodes.

Their demonstration uses one object at a time, from ten MNIST digits, at any of 105 horizontal and
105 vertical positions: a search space of `10 × 105 × 105 = 110,250`.

**Step one, a front end.** Run **convolutional sparse coding** on the image: learn a dictionary of
small image patches and represent the image as a small number of them, each at some position. The
representation is a set of feature maps `A_j(x, y)` — "feature `j` is present, this strongly, here" —
mostly zero. This mirrors the standard account of primary visual cortex.

**Step two, one vector for the scene.** Bundle a position-bound term for every active feature:

```math
s \;=\; \sum_{j,x,y} h(x) \odot v(y) \odot d_j \cdot A_j(x,y)
```

where `h` and `v` are the residue encodings of horizontal and vertical position and `dⱼ` is a random
vector naming feature `j`. Read it as: *for each feature present, bind its name to its place, and
add them all up*.

**Step three, the key identity.** Because sparse coding is *equivariant* to translation — shift the
image, and the feature maps shift with it — moving the whole object by `(x′, y′)` multiplies every
term by the same position vector, which factors straight out:

```math
s \;=\; h(x') \odot v(y') \odot O^{(i)}
```

where `O⁽ⁱ⁾` is the object's vector in its own canonical frame. The scene vector is *literally* a
product of three things: which object, where horizontally, where vertically. **So perception here is
factorisation**, and the resonator network does it.

The result, against the alternative of encoding each position as one big 105-way codebook:

| | codebook vectors | codebook evaluations to converge |
|:--|--:|--:|
| brute force | — | 110,250 |
| standard resonator | `10 + 105 × 2 = 220` | 2085.6 |
| **residue resonator** | `10 + (3+5+7) × 2 = 40` | **792.4** |

## 6.2 Subset sum

Given a set of integers, is there a subset that adds up to a target? For example
`S = {18, 4, 5, 10, 2, 23}` with target 21 — the answer is `{4, 5, 10, 2}`. The general problem is
NP-complete, and many other hard problems reduce to it.

The encoding is almost too neat. Encode the target `T` as a vector. Give each item `Sₖ` its own
factor with a codebook of exactly **two** entries: `z(0)` if you leave it out, `z(Sₖ)` if you put it
in. Since binding adds encoded values, any combination of choices binds to the vector for its sum. So

> *find the subset summing to `T`* = *factorise `z(T)` into these two-choice factors*

and the resonator does it, searching `2^{|S|}` subsets in superposition without ever writing one
down. Memory is `O(D · |S|)` rather than `O(2^{|S|})`. On a CPU it beats an exact solver beyond about
28 items — though they note most of that time is spent building the vectors, not running the
dynamics, and that the natural home for this is neuromorphic hardware.

---

# Part 7 — What to keep in mind

**The vision demo is a small one.** A single object on a blank field, two factors of variation, and —
importantly — the ten "objects" are a **known codebook of templates**. It is a search over a fixed
dictionary, not recognition of a shape never seen. No scale, no rotation, no second object, no
occlusion. The result is a real claim about the *factorisation* being cheaper; it is not evidence
that the representation supports vision.

**The whole framework is integer-valued.** Range, moduli, arithmetic and guarantees all assume whole
numbers. The subinteger extension recovers resolution but gives up multiplication to get it.

**The operations are not obviously neural.** Componentwise complex multiplication, modular
multiplicative inverses computed with Python's `pow`. The paper says so directly: these "are not
easily realized in perceptron-like neurons", and offers spike-timing phase codes and dendritic
nonlinearities as possibilities rather than as models.

**The grid-cell story is suggestive, not settled** — population kernel hexagonal, individual
components band-like, as above.

---

## Glossary

| term | meaning |
|:--|:--|
| **hyperdimensional computing / VSA / HRR** | Three names for computing with long random vectors, using binding and bundling. |
| **quasi-orthogonal** | Nearly perpendicular. Two random vectors in `D` dimensions overlap by about `1/√D`. |
| **phasor vector** | A vector whose every component is a unit complex number: `D` angles. |
| **binding** (`⊙`) | Componentwise multiplication; adds the angles. Reversible, commutative, output unlike either input. Composes. |
| **bundling** | Ordinary addition; superposition. Output similar to every input. Collects. SNR falls as `√(D/N)`. |
| **kernel** | A similarity function between two values. Here, the inner product of their codes, which depends only on their difference. |
| **FPE** | Fractional power encoding. Encode `x` as `zˣ` — one base vector raised componentwise to the power `x`. Addition of values becomes binding of vectors. |
| **modulus / modulo** | `x mod m` is the remainder of `x` after dividing by `m`. |
| **residue number system** | Representing an integer by its remainders under several moduli. |
| **Chinese remainder theorem** | If the moduli share no common factors, those remainders identify the integer uniquely up to their product. |
| **co-prime** | Sharing no common factor other than 1. |
| **carry-free** | Arithmetic in which each digit is handled independently, with nothing propagating between them. |
| **roots of unity** | The `m` evenly spaced points on the unit circle, `e^{2πik/m}`. Drawing FPE phases from these makes the code repeat with period `m`. |
| **codebook** | The stored list of legal vectors for one factor. |
| **codebook decoding** | Identify a value by inner product against every entry in a codebook. |
| **resonator network** | An iterative algorithm that factorises a bound product, keeping each factor as a weighted superposition of its candidates rather than committing to one. |
| **Las Vegas algorithm** | Always correct when it answers; the *time* to answer is random. Restart on failure. |
| **von Mises distribution** | The Gaussian's counterpart for angles. Concentration `κ`: large is tight, zero is uniform. |
| **Landau's function** `g(b)` | The largest product obtainable from positive integers summing to `b`. Grows like `e^{√(b ln b)}`. |
| **convolutional sparse coding** | Representing an image as a few dictionary patches at particular positions, by minimising reconstruction error plus an L1 penalty. |
| **equivariance** | Shift the input, and the representation shifts the same way — as opposed to *invariance*, where it does not change at all. |
| **disentangling** | Recovering the separate causes (identity, position, pose, lighting) that jointly produced an image. |
| **grid cells** | Entorhinal neurons firing on a triangular lattice of positions; a population encodes location with exponentially growing resolution. |
| **bits per vector** | An information measure charging jointly for decoding accuracy and for how many states are being told apart; zero at chance. |

## If you read the paper itself

Definitions 2 and 4 (§2.1–2.2) are the core and are short. §2.3 is the resonator and the scaling
laws. §2.4.2 is the hexagonal construction. §3.1 is the vision application, with the details in
§5.5 — equations 5.5 and 5.7 are the ones that matter. §5.1.2 is the multiplication operator and its
conditions. Everything else can be skipped on a first pass.
