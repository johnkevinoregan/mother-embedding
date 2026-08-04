# ── PLAIN SCRIPT, not a Pluto notebook ──────────────────────────────────────
# `julia --project=.. -t 16 ConvNextStimuli.jl`   (from the P11_ConVNextTest directory)
#
# Step 1 of 3. Generate the Phase 9 stimuli and write them where Python can read them.
#
# The whole value of this experiment is that ConvNeXt is scored on the **same images** as
# every arm in `P9_P12_SimpleStrokeTests/RESULTS.md`, so the splits, seeds and generator keywords
# here are copied from `Phase9_Readouts.jl` rather than re-chosen. Change one and the
# comparison silently stops being a comparison.
#
# Interchange format is deliberately dumb: raw little-endian Float32 and a text manifest.
# No HDF5, no .npy writer, nothing that can drift between two languages. Python reads it
# with `numpy.fromfile`.
#
# ROW ORDER. Julia is column-major, numpy is row-major, so each image is written as
# `permutedims(img)` — then `fromfile(...).reshape(n, N, N)` in Python gives images the same
# way round as our front end sees them. It would barely matter for a CNN (a global transpose
# is just a different but consistent input), but it would matter the moment anyone looked at
# a heatmap, and silent transposes have already cost this project a day on Fashion-MNIST.

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using Statistics, Printf, Random, Serialization
include(joinpath(@__DIR__, "..", "P9_P12_SimpleStrokeTests", "Contours.module.jl"))
using .Contours

const N      = 112
const NTRAIN = parse(Int, get(ENV, "CX_NTRAIN", "16000"))   # Phase 9 defaults
const NTEST  = parse(Int, get(ENV, "CX_NTEST",  "4000"))
const OUT    = joinpath(@__DIR__, "data")

# Copied from Phase9_Readouts.jl. `tkw`/`ekw` restrict the training and test draws so the
# test set contains nuisance values training never saw.
const SPLITS = [
    (:iid,       1,   NamedTuple(),          NamedTuple()),
    (:polarity,  500, (pol=1,),              (pol=-1,)),
    (:fuzziness, 500, (ramp=(0.8, 3.0),),    (ramp=(8.0, 20.0),)),
    (:thickness, 500, (w=(3.0, 6.0),),       (w=(8.0, 12.0),)),
]

"Write a vector of N×N images as consecutive row-major Float32 blocks."
function write_images(path, imgs)
    open(path, "w") do io
        for im in imgs; write(io, Float32.(permutedims(im))); end
    end
end

"Write an n×p Float64 property matrix as row-major Float32."
write_matrix(path, Y) = open(path, "w") do io
    write(io, Float32.(permutedims(Y)))       # permutedims → row-major on disk
end

function main()
    mkpath(OUT)
    @printf("Generating Phase 9 stimuli for the ConvNeXt arm — %d train / %d test per split\n",
            NTRAIN, NTEST)
    @printf("properties: %s\n\n", join(String.(PROPS), ", "))
    manifest = String[]
    push!(manifest, "# split ntrain ntest N nprops")
    for (nm, seed, tkw, ekw) in SPLITS
        t = @elapsed begin
            itr, Ytr, _, _ = contour_batch(NTRAIN, seed;      N=N, tkw...)
            ite, Yte, _, _ = contour_batch(NTEST,  seed+1000; N=N, ekw...)
        end
        write_images(joinpath(OUT, "$(nm)_train_img.f32"), itr)
        write_images(joinpath(OUT, "$(nm)_test_img.f32"),  ite)
        write_matrix(joinpath(OUT, "$(nm)_train_y.f32"), Ytr)
        write_matrix(joinpath(OUT, "$(nm)_test_y.f32"),  Yte)
        # Deliberately NOT also serialised as a .jls. The Julia readout reads these same
        # `.f32` files back, so both languages consume literally the same bytes — "the same
        # stimuli" is then a fact about the file rather than a claim about two RNG streams
        # having stayed in step. It also halves the ~8 GB this would otherwise occupy.
        push!(manifest, "$(nm) $(NTRAIN) $(NTEST) $(N) $(length(PROPS))")
        @printf("  %-11s %6d + %5d images in %5.1f s   (%s)\n", nm, NTRAIN, NTEST, t,
                isempty(tkw) ? "i.i.d." : "train $(tkw) → test $(ekw)")
        flush(stdout)
    end
    write(joinpath(OUT, "manifest.txt"), join(manifest, "\n") * "\n")
    write(joinpath(OUT, "props.txt"), join(String.(PROPS), "\n") * "\n")
    @printf("\nwrote %s\n", OUT)
end

main()
