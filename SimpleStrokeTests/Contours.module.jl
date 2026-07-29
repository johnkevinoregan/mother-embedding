# ── PLAIN MODULE — .module.jl, not a Pluto notebook ─────────────────────────
# Included by other files. Opening it in Pluto rewrites it and leaves a
# "<name> backup 1.jl" beside it. Notebooks are the plain .jl files.

"""
    Contours

One stroke on a uniform grey field, described by a **vector of graded properties** rather
than a class label.

## Why properties and not classes

Phase 8 built a classification task where co-location was supposed to be decisive, and
every representation — including plain pooled orientation energy — scored 100 %. A
categorical label can only answer *"can some network separate these?"*, and on synthetic
stimuli the answer is nearly always yes, which is why that phase produced no information.

Graded targets allow a sharper question: fit a **linear** readout and ask *how much of
each property is already explicit in the representation*. "A₁/A₂ make corner angle
linearly decodable where raw pixels do not" is a falsifiable claim about what the front end
computes, and it is the claim this project rests on. `R²(linear)` versus `R²(MLP)`, per
property and per representation, is the measurement.

It also removes a confound that classes could not. Under a class scheme only `curved` and
`broken` could be closed loops, because `straight` cannot be — so closure leaked class
identity. As an output rather than a category, closure is just another thing to predict.

## The target vector

| # | property | range | notes |
|--:|:--|:--|:--|
| 1 | `curvedness` | 0–1 | retinal: mean \\|κ\\| of the base in 1/px, squashed |
| 2 | `brokenness` | 0–1 | gap size **in units of stroke width** |
| 3 | `closedness` | 0/1 | is it a closed loop |
| 4 | `vangle` | 20–180° | angle at the vertex; **180 = passes straight through** |
| 5 | `junction` | 2 / 3 / 4 | branches meeting at a point; **2 = no junction** |
| 6 | `thickness` | px | stroke width |
| 7 | `softness` | px | edge ramp width |
| 8 | `polarity` | ±1 | light or dark stroke |

**The vertex angle is unmasked: 180° means the stroke passes straight through.** An earlier
version masked this row, on the grounds that an angle is undefined without a corner. It is
not — a smooth stroke has a vertex angle of 180°, an honest value rather than a hole in the
target vector, and none/obtuse/right/acute still falls out by binning.

**It is computed as `180 − excess turn`, not measured between the limbs.** The excess turn
is what the direction jumps *at a point*, over and above the turn the smooth curvature
already accounts for. Measured directly between the limbs over a 12 px window, a cornerless
arc of κ = 0.045 reads as a 118° corner, so `curvedness` would leak straight into this row —
and did: a third of the labels landed in the wrong band before this was fixed. Subtracting
the smooth part isolates the singular part, which is what a corner is.

**Curvedness is measured on the base curve, before the event is applied.** A kink is a
curvature impulse, so computing it afterwards would make every corner register as global
curvature and entangle rows 1 and 4. Sampling curvature and vertex angle independently
is what makes "a right angle on a strongly curved arm" an ordinary sample.

**Terminations are deliberately not a row.** Every open stroke ends twice, a gap adds two
more ends, and a closed loop has none — so termination count is a deterministic function of
`closedness` and `brokenness` and a row for it would carry no information while inflating
any average taken over rows. The `junction` row counts *branches*, and its null value is 2:
a straight line, a curve and a kink all have a contour that simply passes through.

**Rows 6–8 are controls on the front end itself**, in opposite directions. Quadrature
energy is polarity-invariant by construction, so the orientation block *should* fail to
predict row 9 — inability is the correct result — while the lowpass block, which carries
mean level, should predict it perfectly. And stroke width should be nearly explicit in the
energy ratio across the three rationally-spaced scales, so row 7 should be linearly
decodable from very few samples. Both are predictions, on record.

## What stays uncontrolled, and why it is reported

Closure implies curvature: a closed loop has turned 2π and cannot be straight. The two
are still separable because curvedness here is *local* (1/R) — a large circle is closed
with low curvature, a short tight arc is open with high curvature — and radius and turn
are sampled independently. `Preview_Contours.jl` prints the target correlation matrix so
any residual dependence stays visible.
"""
module Contours

using Random, Statistics, LinearAlgebra

export PROPS, N_PROPS, Params, sample_params, render_params, contour_batch,
       targets_of, ANGLE_BANDS, band_of, respec, stimulus, excess_turn

"Names of the target vector's entries, in order."
const PROPS = (:curvedness, :brokenness, :closedness, :vangle,
               :junction, :thickness, :softness, :polarity)
const N_PROPS = length(PROPS)

"""
Named bands for the categorical readout, as the interior angle at the vertex in degrees.

Sampling puts a third of the kinked stimuli in each of the three corner bands; `:none` is
every stimulus with no kink at all, which sits at exactly 180° and so needs no arbitrary
threshold on a continuous quantity.
"""
const ANGLE_BANDS = (acute = (20.0, 80.0), right = (80.0, 100.0), obtuse = (100.0, 170.0))

"Categorical corner readout, derived from the continuous target by binning."
band_of(a) = a >= 170 ? :none : (a >= 100 ? :obtuse : (a >= 80 ? :right : :acute))

"""
    excess_turn(base, polys) -> degrees

How much the stroke's direction **jumps at a point**, over and above the turn its smooth
curvature already accounts for: the total signed turn of the drawn figure minus that of the
base curve it was made from.

A kink is a rigid rotation of the tail, so every segment direction past the vertex shifts by
the same amount and the two sums differ by exactly that rotation — the smooth contributions
cancel term by term, at every curvature, with no window to choose. That is what makes this
target separable from `curvedness`, and it is why the earlier interior-angle target was not:
interior angle measured over a finite window necessarily mixed the corner with the arms'
own bending, which is how a third of the labels ended up in the wrong band.
"""
function excess_turn(base, polys)
    st(P) = (t = 0.0; @inbounds for i in 2:length(P)-1
                 a = atan(P[i][1]-P[i-1][1], P[i][2]-P[i-1][2])
                 b = atan(P[i+1][1]-P[i][1], P[i+1][2]-P[i][2])
                 t += atan(sin(b-a), cos(b-a))
             end; t)
    rad2deg(abs(st(polys[1]) - st(base)))
end

# ── the parameter space ─────────────────────────────────────────────────────

"""
Everything that defines one stimulus. The dataset is a sample from this space; the target
vector is computed from it by `targets_of`.

`event` is one of `:none`, `:gap`, `:kink`, `:tee`, `:cross`, drawn independently of the
base geometry, so no property of the base predicts which event is present.
"""
struct Params
    kappa::Float64      # base curvature, 1/px  (0 = straight)
    turn::Float64       # total turn of the base, radians (2π = closed)
    arclen::Float64     # base arclength, px
    aspect::Float64     # 1 = circular; < 1 flattens it into an oval
    event::Symbol
    gap::Float64        # px, for :gap
    vturn::Float64      # radians, turn at the vertex, for :kink (0 = smooth)
    branch::Float64     # radians, base-to-stem angle, for :tee / :cross
    w::Float64          # stroke width, px
    ramp::Float64       # edge ramp width, px
    amp::Float64        # contrast
    pol::Int            # +1 light stroke, -1 dark
    bg::Float64
    noise::Float64
end

"""
    sample_params(rng; kw...)

Draw one point from the parameter space. Any field can be pinned or narrowed, which is how
the extrapolation splits are built: `sample_params(rng; pol = 1)` never draws a dark
stroke, so a model trained on it must generalise to inverted contrast rather than
interpolate.

Curvature and turn are drawn independently and the arclength follows from them, which is
what keeps `curvedness` and `closedness` from collapsing onto one another.
"""
function sample_params(rng; pol=nothing, event=nothing, w=(3.0, 25.0), ramp=(0.8, 20.0),
                       amp=(0.30, 1.00), bg=(0.40, 0.60), noise=0.02,
                       kappa=(0.0, 0.045), turn=(0.0, 2π/3), aspect=(0.62, 1.0),
                       p_straight=0.20, p_closed=0.18, N=112)
    u(r) = r isa Tuple ? r[1] + (r[2]-r[1])*rand(rng) : Float64(r)

    # Stroke width and edge ramp are drawn first because they decide how much room is left
    # for the figure: a 25 px stroke with a 20 px ramp reaches 23 px beyond its centreline,
    # and a shape sized for a hairline would be clipped by the frame.
    # Log-uniform, not uniform. The range runs to a 25 px stroke and a 20 px ramp, but drawn
    # uniformly the *mean* width would be 14 px and almost every figure would be a blob with
    # its structure swallowed — a 25 px stroke on a 45 px radius leaves a 16 px hole. Sampled
    # log-uniformly the extremes are reachable while the typical stroke stays around 8 px,
    # which is what "the thickest up to 25" asks for.
    lu(r) = r isa Tuple ? exp(log(r[1]) + (log(r[2])-log(r[1]))*rand(rng)) : Float64(r)
    ww = lu(w); rr = lu(ramp)
    frame = clamp(N/2 - (ww/2 + rr/2 + 3), 16.0, 45.0)

    ev = event === nothing ? rand(rng, (:none, :none, :gap, :kink, :kink, :tee, :cross)) :
                             Symbol(event)

    # Turn is either small or complete, never in between.
    #
    # An arc that turns nearly 2π brings its own free ends close together, and that opening
    # is indistinguishable from a deliberate gap: such a figure showed *two* gaps while its
    # label described one, and a nearly-closed arc with no event at all already reads as an
    # oval with a gap in it. Restricting open arcs to 2π/3 keeps the free ends far apart,
    # and a closed loop has no free ends, so on either the gap event is unambiguous.
    #
    # A kink is never applied to a closed loop: rotating the tail of a loop opens it at the
    # seam, which is a second event the label does not describe. That makes `closedness = 0`
    # for every kinked sample — a real coupling between two target rows, reported in the
    # correlation matrix rather than hidden. Constructing a closed curve with one corner is
    # possible but needs a non-circular base, and is not worth it here.
    closed = ev !== :kink && rand(rng) < p_closed
    Δ = closed ? 2π : u(turn)

    # An exactly straight stroke needs its own atom for the same reason: sampling curvature
    # from a continuous range makes κ = 0 a measure-zero event, and the dataset then has no
    # straight lines at all — which is how the first version of this generator produced a
    # `curvedness` target that never went below 0.52.
    straight = rand(rng) < p_straight
    κ = 0.0; L = min(58 + 38rand(rng), 2frame)
    if !straight
        # The figure has to fit, but the binding constraint is the arc's *chord*, not its
        # diameter: a shallow arc of very large radius fits easily. Capping the radius at a
        # constant (an earlier bug) therefore also put a floor under the curvature and left
        # a hole in the low end of the target range.
        chord = Δ >= π ? 1.0 : max(sin(Δ/2), 0.05)
        κ = max(u(kappa), 1/min(frame/chord, 600.0))
        R = 1/κ; Δ = max(Δ, 0.12); L = clamp(Δ*R, 38.0, 2π*R); Δ = L/R
    else
        Δ = 0.0
    end

    b = rand(rng, keys(ANGLE_BANDS))                 # a third of kinks in each band
    lo, hi = getfield(ANGLE_BANDS, b)

    # The gap sweeps from a nick to a clear break, in units of stroke width, with a 1 px
    # floor because anything narrower cannot be rendered. Drawing it as `1.5w + 8·rand`
    # (the first version) put every gap at 1.5 stroke widths or more, so `brokenness` never
    # went below 0.6 and the interesting regime — a gap barely wide enough to see — was
    # absent from the dataset entirely.
    # The gap is a multiple of stroke width, but capped at half the arclength: with strokes
    # now up to 25 px wide, 3.6 widths would be 90 px and would delete the whole figure.
    Params(κ, Δ, L, u(aspect), ev,
           clamp(ww * (0.10 + 3.6rand(rng)), 1.0, 0.5L),
           deg2rad(180 - (lo + (hi-lo)*rand(rng))),
           deg2rad(30 + 60rand(rng)) * rand(rng, (-1, 1)),
           ww, rr, u(amp),
           pol === nothing ? rand(rng, (-1, 1)) : Int(pol),
           u(bg), Float64(noise))
end

# ── targets ─────────────────────────────────────────────────────────────────

"Monotone squash to [0,1] with `x0` mapping to 0.5. Keeps targets bounded without clipping."
@inline squash(x, x0) = x / (x + x0)

"""
    targets_of(p) -> (v, mask)

The target vector and a mask marking which entries are defined for this stimulus. Only
`angle` is ever masked, and only when there is no vertex.

`brokenness` is the gap **relative to stroke width**, not in pixels: a 4 px gap in a 3 px
stroke is an unmistakable break, the same gap in an 11 px stroke is barely a nick, and the
target should say so.
"""
function targets_of(p::Params, meas=(corner=0.0, kappa=p.kappa, closedness=p.turn/2π))
    v = Float64[
        squash(meas.kappa, 0.012),                                # curvedness
        p.event === :gap ? squash(p.gap / p.w, 1.0) : 0.0,        # brokenness
        clamp(meas.closedness, 0, 1),                             # closedness
        180.0 - (p.event === :kink ? meas.corner : 0.0),          # vangle (180 = straight through)
        p.event === :tee ? 3.0 : (p.event === :cross ? 4.0 : 2.0),# junction order
        p.w,                                                      # thickness
        p.ramp,                                                   # softness
        Float64(p.pol),                                           # polarity
    ]
    # nothing is masked any more: every row is defined for every stimulus
    v, trues(N_PROPS)
end

# ── geometry ────────────────────────────────────────────────────────────────

"Base curve as a polyline at ~1 px spacing, centred on its own centroid."
function base_curve(p::Params)
    if p.kappa < 1e-6
        n = max(2, round(Int, p.arclen))
        return [(0.0, -p.arclen/2 + p.arclen*(k-1)/(n-1)) for k in 1:n]
    end
    R = 1/p.kappa; n = max(6, round(Int, p.arclen))
    pts = [(R*p.aspect*sin(-p.turn/2 + p.turn*(k-1)/(n-1)),
            R*cos(-p.turn/2 + p.turn*(k-1)/(n-1))) for k in 1:n]
    cy = mean(first.(pts)); cx = mean(last.(pts))
    [(y-cy, x-cx) for (y,x) in pts]
end

"Delete `g` px of arclength from the middle, leaving two ends that continue each other."
function apply_gap(q, g, rng)
    n = length(q); i = round(Int, n*(0.30 + 0.40rand(rng))); h = max(1, round(Int, g/2))
    filter(s -> length(s) >= 2, [q[1:max(1, i-h)], q[min(n, i+h):n]])
end

"""
Rotate the tail about the vertex at index `i` by `sgn·vturn`.

The rotation *is* the target: rotating the tail by `vturn` shifts every direction past the
vertex by exactly `vturn`, so `excess_turn` recovers it exactly, at any base curvature. No
search is needed — the earlier interior-angle target had to be solved for by bisection
because the arms' own bending contaminated it.
"""
function apply_kink(q, vturn, i::Int, sgn::Int)
    n = length(q)
    turn = vturn * sgn
    cy, cx = q[i]; s, c = sin(turn), cos(turn)
    r = copy(q)
    @inbounds for j in i+1:n
        dy = q[j][1] - cy; dx = q[j][2] - cx
        r[j] = (cy + c*dy + s*dx, cx - s*dy + c*dx)
    end
    [r]
end

"""
A second stroke meeting the base: `through = true` crosses it (4 rays), `false` stops on
it (3 rays). Built by the same code with one length changed, so a tee and a crossing
differ in ray count and in nothing incidental.

The tangent is taken with **modular** indices, because on a nearly-closed base a clamped
index collapses both neighbours onto the same point, and the branch then gets laid down
*along* the contour instead of across it.

The branch carries the **same curvature as the base**. Drawing it always straight — as an
earlier version did — meant a curved base could carry a straight stem, so a tee or a
crossing mixed two stroke types in one figure and the model would have to describe them
jointly. With one curvature per figure the target vector describes both strokes at once.
Separating the two strokes' descriptions is a later problem.
"""
function add_branch(q, through::Bool, brnch, κ, rng)
    n = length(q); i = round(Int, n*(0.25 + 0.50rand(rng)))
    step = max(2, n ÷ 20)
    j = mod1(i + step, n); k = mod1(i - step, n)
    β = atan(q[j][1] - q[k][1], q[j][2] - q[k][2]) + brnch
    L = 26 + 22rand(rng); lo = through ? -L : 0.0
    cy, cx = q[i]
    m = max(3, round(Int, L - lo))
    ks = κ < 1e-6 ? 0.0 : κ * rand(rng, (-1, 1))
    br = if ks == 0.0
        [(cy + s*sin(β), cx + s*cos(β)) for s in range(lo, L; length=m)]
    else
        # circular arc from the junction with tangent β and curvature ks:
        #   x(s) = x₀ + (sin(β+ks·s) − sin β)/ks,  y(s) = y₀ − (cos(β+ks·s) − cos β)/ks
        [(cy - (cos(β + ks*s) - cos(β))/ks, cx + (sin(β + ks*s) - sin(β))/ks)
         for s in range(lo, L; length=m)]
    end
    [q, br]
end

"""
    measure_base(q) -> (curvedness_kappa, closedness)

Mean absolute curvature (1/px) and whether the base is a closed loop, **measured from
the base polyline**.

Asserted from the parameters these would be wrong whenever `aspect < 1`: an ellipse does
not have the curvature of the circle it was flattened from, and its mean is not 1/R. The
corner angle taught the same lesson — a label taken from the parameter that generated a
figure is not necessarily a label the figure carries.
"""
function measure_base(q)
    n = length(q); n < 4 && return 0.0, 0.0
    tot = 0.0; sgn = 0.0; len = 0.0
    @inbounds for i in 2:n-1
        a = atan(q[i][1]-q[i-1][1], q[i][2]-q[i-1][2])
        b = atan(q[i+1][1]-q[i][1], q[i+1][2]-q[i][2])
        d = atan(sin(b-a), cos(b-a))
        tot += abs(d); sgn += d
        len += hypot(q[i][1]-q[i-1][1], q[i][2]-q[i-1][2])
    end
    len <= 0 && return 0.0, 0.0
    # Binary: turn is either at most 2π/3 or exactly 2π, with nothing in between, so a
    # graded closedness would have had no support in its middle. "Ellipse or not."
    tot/len, abs(sgn) >= 0.95*2π ? 1.0 : 0.0
end

"Do segments a and b properly cross?"
@inline function crosses(a1, a2, b1, b2)
    o(u, v, r) = (v[1]-u[1])*(r[2]-u[2]) - (v[2]-u[2])*(r[1]-u[1])
    d1 = o(b1,b2,a1); d2 = o(b1,b2,a2); d3 = o(a1,a2,b1); d4 = o(a1,a2,b2)
    ((d1 > 0) != (d2 > 0)) && ((d3 > 0) != (d4 > 0))
end

"""
Does the figure cross itself where it should not?

An unintended crossing is a **structural event no target row describes**, so the stimulus
would not match its own target vector. Adjacent segments share an endpoint and are skipped.
"""
function self_crossing(polys)
    S = [(P[k], P[k+1]) for P in polys for k in 1:length(P)-1]
    @inbounds for i in 1:length(S)-2, j in i+2:length(S)
        crosses(S[i][1], S[i][2], S[j][1], S[j][2]) && return true
    end
    false
end

"""
    geometry(p, rng) -> (polys, measured_angle)

Base curve plus its event, and the corner angle measured back out of the result — `NaN`
when there is no kink. Unintended self-crossings are redrawn, except for `:cross` where the
crossing is the target. The retry resamples the **event placement only**: the `Params` are
fixed on entry, so no retry loop can make stroke width or polarity depend on the event.

A kink whose angle cannot be measured — too near an end of a short arm to fit the window —
is redrawn rather than kept with an asserted label.
"""
function geometry(p::Params, rng; tries::Int=24)
    q = base_curve(p)
    κm, clo = measure_base(q)
    local polys
    local ang = 0.0
    for _ in 1:tries
        if p.event === :kink
            n = length(q)
            i = round(Int, n*(0.32 + 0.36rand(rng)))
            polys = apply_kink(q, p.vturn, i, rand(rng, (-1, 1)))
            ang = excess_turn(q, polys)
            !self_crossing(polys) && break
        else
            polys = if p.event === :none;  [q]
            elseif p.event === :gap;       apply_gap(q, p.gap, rng)
            else                     add_branch(q, p.event === :cross, p.branch, p.kappa, rng)
            end
            (p.event === :cross || !self_crossing(polys)) && break
        end
    end
    polys, (corner=ang, kappa=κm, closedness=clo)
end

# ── rendering ───────────────────────────────────────────────────────────────

@inline function seg_d2(y, x, y1, x1, y2, x2)
    vy = y2-y1; vx = x2-x1; L2 = vy*vy + vx*vx
    t = L2 <= 0 ? 0.0 : clamp(((y-y1)*vy + (x-x1)*vx)/L2, 0.0, 1.0)
    dy = y - (y1 + t*vy); dx = x - (x1 + t*vx)
    dy*dy + dx*dx
end

"""
    respec(p; kw...)

Copy of `p` with the named fields replaced. Used to sweep one parameter with everything
else held fixed — for the figures, and for the extrapolation splits, where a test set
differs from its training set in exactly one nuisance.
"""
respec(p::Params; kw...) =
    Params((get(values(kw), f, getfield(p, f)) for f in fieldnames(Params))...)

"""
    render_params(p, rng; N=112, rot=nothing, at=nothing)

Rasterise one stimulus: uniform grey field, random rotation, random position subject to
fitting with a margin. Returns an `N×N` image in `[0,1]`.

`rot` and `at` pin the orientation and centre, which is what makes a parameter sweep
legible: without them, re-rendering with a different stroke width would also move and
rotate the figure and the comparison would show nothing.

The distance field is built segment by segment over each segment's own bounding box
expanded by the profile's reach, rather than testing every pixel against every segment.
"""
function render_params(p::Params, rng; N::Int=112, rot=nothing, at=nothing)
    first(stimulus(p, rng; N=N, rot=rot, at=at))
end

"""
    stimulus(p, rng; N=112, rot=nothing, at=nothing) -> (img, v, mask)

Image together with the target vector and mask **computed from the geometry that was
actually drawn**. This is the entry point the dataset uses; `render_params` is the
image-only convenience wrapper for figures.

Targets are derived here rather than from `p` alone because the corner, the curvature and
the closure are all measured from the geometry that was drawn — see `excess_turn` and
`measure_base`.
"""
function stimulus(p::Params, rng; N::Int=112, rot=nothing, at=nothing)
    polys, meas = geometry(p, rng)
    v, mask = targets_of(p, meas)
    render_geom(polys, p, rng; N=N, rot=rot, at=at), v, mask
end

function render_geom(polys, p::Params, rng; N::Int=112, rot=nothing, at=nothing)

    φ = rot === nothing ? 2π*rand(rng) : Float64(rot); s, c = sin(φ), cos(φ)
    polys = [[(c*y + s*x, -s*y + c*x) for (y,x) in P] for P in polys]

    reach = p.w/2 + p.ramp/2 + 1; m = reach + 2
    ys = [y for P in polys for (y,_) in P]; xs = [x for P in polys for (_,x) in P]
    ylo, yhi = m - minimum(ys), N + 1 - m - maximum(ys)
    xlo, xhi = m - minimum(xs), N + 1 - m - maximum(xs)
    oy = at !== nothing ? at[1] : (yhi > ylo ? ylo + (yhi-ylo)*rand(rng) : (ylo+yhi)/2)
    ox = at !== nothing ? at[2] : (xhi > xlo ? xlo + (xhi-xlo)*rand(rng) : (xlo+xhi)/2)

    D = fill(Float32(1e9), N, N)
    for P in polys, k in 1:length(P)-1
        y1 = P[k][1]+oy; x1 = P[k][2]+ox; y2 = P[k+1][1]+oy; x2 = P[k+1][2]+ox
        ylo2 = max(1, floor(Int, min(y1,y2)-reach)); yhi2 = min(N, ceil(Int, max(y1,y2)+reach))
        xlo2 = max(1, floor(Int, min(x1,x2)-reach)); xhi2 = min(N, ceil(Int, max(x1,x2)+reach))
        @inbounds for x in xlo2:xhi2, y in ylo2:yhi2
            d2 = Float32(seg_d2(y, x, y1, x1, y2, x2))
            d2 < D[y,x] && (D[y,x] = d2)
        end
    end

    I = Matrix{Float32}(undef, N, N); e = p.w/2; ρ = max(p.ramp, 1e-3)
    # Peak normalisation. Once the ramp is wider than the stroke, the profile never reaches
    # full amplitude, so widening the ramp would also *dim* the stroke and `softness` would
    # be partly a contrast manipulation. Dividing by the profile's analytic peak keeps peak
    # contrast at `amp` whatever the ramp, so softness is pure blur — which is what makes
    # the two rows of the target vector independently readable.
    t0 = clamp((e + ρ/2)/ρ, 0.0, 1.0); pk = max(t0*t0*(3 - 2t0), 1e-6)
    @inbounds for q in eachindex(D)
        t = clamp((e + ρ/2 - sqrt(D[q]))/ρ, 0.0, 1.0)
        I[q] = clamp(p.bg + p.pol*p.amp*(t*t*(3-2t))/pk + p.noise*randn(rng), 0f0, 1f0)
    end
    I
end

"""
    contour_batch(n, seed; N=112, kw...)

`n` stimuli. Returns `(imgs, Y, M, params)` where `Y` is `n × N_PROPS` of targets and `M`
the matching mask. Keyword arguments are forwarded to `sample_params`, which is how a
split restricts the nuisance range.
"""
function contour_batch(n::Int, seed::Int; N::Int=112, kw...)
    rng = MersenneTwister(seed)
    imgs = Vector{Matrix{Float32}}(undef, n)
    Y = zeros(Float64, n, N_PROPS); M = trues(n, N_PROPS); ps = Vector{Params}(undef, n)
    for i in 1:n
        p = sample_params(rng; kw...)
        imgs[i], v, mk = stimulus(p, rng; N=N)
        Y[i,:] = v; M[i,:] = mk; ps[i] = p
    end
    imgs, Y, M, ps
end

end # module
