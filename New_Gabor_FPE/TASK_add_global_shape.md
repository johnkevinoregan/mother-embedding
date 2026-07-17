# TASK: add a global-shape descriptor to `ray_fpe_junctions.jl`

**For:** Claude Code, working on `ray_fpe_junctions.jl` (the Pluto notebook that
does local keypoint analysis via ray harmonics).

**Read first:** `New_Gabor_FPE_summary.md` §3 (ray profile → circular harmonics)
and §8 (design rules). This task extends the *same construction to a second
level*; do not invent a different mechanism.

**Goal:** classify the **global shape** of a character (blob / oval / hollow ring
/ complicated figure) using only linear projections, FPE-compatible, no
thresholds, no `if/elif` on shape class.

---

## 1. The idea in one line

The local analysis takes circular harmonics of the **ray profile `R(φ)` around a
point**. The global analysis takes circular harmonics of the **mass distribution
around the object centroid**. Same operation, different origin.

```
Mₙ = Σ_pixels  I(y,x) · exp(-i n α(y,x))          α = angle about the centroid
M₀(r) = radial mass profile in the same frame
```

- **angular spectrum `|Mₙ|/M₀`** → shape class (oval → n=2, triangle → n=3, …)
- **radial profile `M₀(r)`** → filled vs hollow
- both are linear in `I`. Only nonlinearity is `|·|`.

The object-centred frame comes free from moments — **no decision procedure**:
origin = centroid (1st moment), scale = RMS radius (2nd moment).

---

## 2. Functions to add

Add to the notebook, reusing existing constants/conventions (`Float32`
throughout, `IMG`, `DIM`, `RAY_FREQS`, etc.).

```julia
"""Object-centred, scale-normalised polar frame from image moments.
Returns (rn, α, w) where rn = r/r_rms, α = angle about centroid, w = I/ΣI."""
function object_frame(img::Matrix{Float32})
    H, W = size(img)
    tot = sum(img)
    cy = sum(i * img[i,j] for i in 1:H, j in 1:W) / tot     # 1st moment
    cx = sum(j * img[i,j] for i in 1:H, j in 1:W) / tot
    r = [sqrt((i-cy)^2 + (j-cx)^2) for i in 1:H, j in 1:W]
    α = [atan(i-cy, j-cx)          for i in 1:H, j in 1:W]
    rms = sqrt(sum(img .* r.^2) / tot)                      # 2nd moment
    return Float32.(r ./ rms), Float32.(α), Float32.(img ./ tot)
end

"""Angular harmonics of the mass distribution. Mₙ = Σ w·exp(-i n α)."""
function angular_spectrum(img::Matrix{Float32}; nmax=8)
    _, α, w = object_frame(img)
    return [sum(w .* cis.(-Float32(n) .* α)) for n in 0:nmax]
end

"""Radial mass profile M₀(r) — separates filled from hollow."""
function radial_profile(img::Matrix{Float32}; nbins=16, rmax=2.0f0)
    rn, _, w = object_frame(img)
    edges = range(0f0, rmax, length=nbins+1)
    return [sum(w[(rn .>= edges[i]) .& (rn .< edges[i+1])]) for i in 1:nbins], edges
end

"""Angular anisotropy A = Σ_{n≥1}|Mₙ|²/|M₀|².  A = 0 ⟺ rotationally symmetric.
This is the blob→complicated axis."""
function anisotropy(M::Vector{ComplexF32})
    s = abs.(M) ./ abs(M[1])
    return sum(s[2:end] .^ 2)
end
```

### FPE log-polar descriptor

```julia
# Angular base: INTEGER frequencies (α is mod 2π — periodic).  Reuse the
# existing RAY_FREQS convention.
const ALPHA_FREQS = RAY_FREQS
alpha_pow(u::Real) = cis.(Float32(u) .* Float32.(ALPHA_FREQS))

# Radial base: CONTINUOUS frequencies (log r is NOT periodic).
const RHO_FREQS = Float32.(randn(MersenneTwister(0xC0FFEE), DIM))
rho_pow(u::Real) = cis.(Float32(u) .* RHO_FREQS)

"""Fourier–Mellin-style global descriptor.
    D = Σ_pixels I(y,x) · z_ρ^{log r} ⊙ z_α^{α}
Rotation by β  →  D ⊙ z_α^β        (binding)
Scaling by s   →  D ⊙ z_ρ^{log s}  (binding)
The whole similarity group acts by binding."""
function global_descriptor(img::Matrix{Float32}; eps=1f-3)
    rn, α, w = object_frame(img)
    D = zeros(ComplexF32, DIM)
    for idx in eachindex(w)
        w[idx] == 0f0 && continue
        lr = log(max(rn[idx], eps))          # guard: r → 0 at the centroid
        D .+= w[idx] .* rho_pow(lr) .* alpha_pow(α[idx])
    end
    return D ./ (norm(D) + 1f-8)
end
```

---

## 3. Cells to add

Append **after** the existing §9 (`b0000037`/`b0000038`) and **before** the §10
notes cell (`b0000039`). Use fresh UUIDs `b0000040 … b0000052` following the
existing pattern.

| UUID | kind | content |
|---|---|---|
| `b0000040` | md | §11 heading + the "same construction one level up" explanation |
| `b0000041` | code | `object_frame` |
| `b0000042` | code | `angular_spectrum`, `radial_profile`, `anisotropy` |
| `b0000043` | md | test-shape section heading |
| `b0000044` | code | test shape generators (§4 below) |
| `b0000045` | code | results table over all test shapes (Markdown.parse, like `b0000023`) |
| `b0000046` | md | "angular spectrum → class; radial profile → filled/hollow" |
| `b0000047` | code | figure: 3×N grid — image / `|Mₙ|/M₀` bars / `M₀(r)` line |
| `b0000048` | md | rotation-covariance section heading |
| `b0000049` | code | rotation test (§5 below) |
| `b0000050` | md | FPE log-polar section — periodic α / non-periodic log r |
| `b0000051` | code | `ALPHA_FREQS`, `RHO_FREQS`, `global_descriptor` |
| `b0000052` | code | binding test: `global_descriptor(rot(img)) ≈ D ⊙ z_α^β` |

### ⚠️ Update the `# ╔═╡ Cell order:` footer

New cells will **not appear** unless listed. Markdown cells use `╟─`, code cells
use `╠═`:

```
# ╟─b0000040-0001-4000-8000-000000000040
# ╠═b0000041-0001-4000-8000-000000000041
...
```

Insert them between `b0000038` and `b0000039` so §11 lands before the §10 notes
— or renumber §10 to §12. (This footer bug is real: `pluto2md.py` originally
missed `╟─` and silently emitted nothing.)

---

## 4. Test shapes

```julia
disk(R=50), oval(a=60,b=28), annulus(R=50,r=32), poly(3), poly(4)
```
on a 161×161 canvas, plus letters `O`, `A`, `K` (use `letter_T`-style stand-ins
or real EMNIST if reachable on the server).

Regular `nv`-gon radius at angle: `rr = R·cos(π/nv) / cos(mod(ang+π/nv, 2π/nv) - π/nv)`.

---

## 5. Verification targets — **diff against these**

Reference values from a verified Python implementation (161×161 canvas, nmax=8):

```
shape        anisotropy A   dominant n
disk             0.0000     — (round)
annulus          0.0000     — (round)
letter O         0.0003     — (round)
square           0.0226     4
triangle         0.0952     3
oval 2:1         0.1548     2
letter A         0.2269     3  (spread 4–8)
letter K         0.3585     4  (spread 1–6)
```

Spectra `|Mₙ|/M₀`, n = 1…5:

```
disk        0.000  0.000  0.000  0.002  0.000
oval 2:1    0.000  0.367  0.000  0.131  0.000
annulus     0.000  0.000  0.000  0.001  0.000
triangle    0.000  0.000  0.281  0.001  0.000
square      0.000  0.000  0.000  0.141  0.000
letter O    0.001  0.008  0.002  0.014  0.003
letter A    0.005  0.051  0.368  0.160  0.126
letter K    0.103  0.197  0.163  0.375  0.040
```

**Radial profile must separate disk from annulus** (their angular spectra are
*identical* — both flat zero). Disk ramps up then cuts off; annulus is
concentrated in two bins near `r/rms ≈ 0.75–1.0`. Letter O behaves like the
annulus.

**Rotation covariance (must be near-exact):** rotate the triangle by 30°.
```
|M₃| before = 0.2809   after = 0.2809      (invariant to 4 dp)
arg(M₃) shift = 90.0°  = 3 × 30°           (Mₙ → Mₙ·e^{-inα})
```
If `|M₃|` is not invariant to ~3 decimals, the frame (centroid/rms) is wrong.

**Binding test:** `global_descriptor(rotate(img, β))` should match
`global_descriptor(img) ⊙ alpha_pow(β)` up to interpolation error. Report
cosine similarity; expect > 0.9. If it is ~0, `ALPHA_FREQS` are not integers.

---

## 6. Pitfalls (all hit during derivation)

1. **`log r` blows up at the centroid.** `r → 0` there. Guard with
   `max(rn, 1f-3)` as in the code above.
2. **Do NOT use spectral entropy as the complexity measure.** It was tried and is
   **wrong**: normalising a near-zero spectrum amplifies numerical noise, and the
   *disk scored higher than the oval* (0.349 vs 0.213). Use `anisotropy` (a raw,
   un-normalised energy ratio) instead.
3. **Spectral centroid `n̄` is ill-conditioned as A → 0** (the disk reports
   n̄ = 5.00, pure noise). This is an honest 0/0 — a round blob has no dominant
   angular frequency. **Do not patch it with a threshold.** Either omit `n̄`, or
   report it only alongside `A` and let the reader see it is meaningless when
   `A ≈ 0`. In the FPE version it never arises: you bind the vector, you never
   extract the scalar.
4. **α must be periodic ⇒ integer frequencies. log r is NOT periodic ⇒ continuous
   frequencies.** Mixing these up is the same class of bug as Bug A in the
   summary. Note the asymmetry: `α` is encoded directly (mod 2π), **not** doubled
   — doubling is only for orientation (mod π).
5. **Use mass, not boundary radius `r(α)`.** A letter `A` is not star-shaped, so
   `r(α)` is not single-valued. The mass formulation needs no star-shapedness.
6. `poly()` and the ellipse must be **filled**, not outlines, or the radial
   profile comparison is meaningless.

---

## 7. Design rules — do not violate

- **No thresholds, no `if/elif` on shape class.** `(A, |Mₙ|/M₀, M₀(r))` *is* the
  descriptor. "Blob" is a region of that space (`A ≈ 0`), not a symbol.
- **Do not add a separate harmonics module.** FPE with integer frequencies *is*
  the harmonic expansion (summary §3c).
- Keep `nmax = 8`. Do not trim: for non-right-angle structure the higher
  harmonics carry real information.

---

## 8. Worth noting in the markdown

- **`n = 2` is the inertia tensor.** The complex 2nd moment
  `μ₂₀ − μ₀₂ + 2i·μ₁₁` is `M₂`: modulus = elongation, `arg/2` = major-axis
  orientation. The classical moment-based "is it an oval and which way does it
  point" is just the 2nd angular harmonic — structurally identical to the
  orientation tensor `c₁` of the doubled-angle `E(θ)`.
- **Log-polar buys scale covariance for the same price as rotation** — both
  become bindings. That is why the frame is log-polar, not plain polar.
- **The global descriptor and the keypoint bundle are the same object.**
  `Σ_p I(p)·z_α^α` is mass-weighted; `Σ_p V_ray(p) ⊙ z_α^α` is feature-weighted.
  Same form, different payload — so global shape and local structure can live in
  one descriptor.
