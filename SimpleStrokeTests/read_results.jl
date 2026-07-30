# ── PLAIN SCRIPT ────────────────────────────────────────────────────────────
# Print a serialised results file as a readable table.
#
#   julia --project=.. read_results.jl                        # every file in results_canon3
#   julia --project=.. read_results.jl results_canon1         # a different directory
#   julia --project=.. read_results.jl results_canon3/iid.jls # one file
#
# `.jls` is Julia's own `Serialization` format: compact, exact, and completely unreadable
# without Julia. It is written so figures can be redrawn without re-running an experiment.
# It is *not* an archival format — it is tied to the Julia version and to the types that were
# in scope when it was written, so a file that is a few versions old may refuse to load.
# The `.log` files beside these directories are the durable, human-readable record; treat
# the `.jls` as a cache.

using Serialization, Printf, Statistics

const PROPS = ["curvedness","brokenness","closedness","vangle","arms",
               "thickness","fuzziness","polarity"]

hdr() = (@printf("%-34s", ""); for p in PROPS; @printf("%11s", p[1:min(10,end)]); end; println())
row(name, v) = (@printf("%-34s", name);
                for x in v; isnan(x) ? @printf("%11s", "—") : @printf("%11.3f", x); end;
                println())

function show_file(path)
    println("\n" * "="^122); println(path); println("="^122)
    obj = try deserialize(path) catch e
        println("  could not deserialise: ", sprint(showerror, e)); return
    end

    if obj isa NamedTuple && haskey(obj, :R)                      # iid / extrap
        hdr()
        haskey(obj, :base) && row("trivial baseline", obj.base)
        arms = haskey(obj, :arms) ? obj.arms :
               ["pixels·linear","pixels·MLP","CNN","ours·linear","ours·MLP"]
        for (k, a) in enumerate(arms); row(a, obj.R[k, :]); end

    elseif obj isa NamedTuple && haskey(obj, :battr)              # block attribution
        hdr()
        for k in ["orient","lowpass","A1+A2","rays","all·SHUFFLED",
                  "all·noRAYS","all·noA","all·noR0","all·noR1","all·noR2",
                  "rays.R0","rays.R1","rays.R2","all"]
            haskey(obj.battr, k) && row(k, obj.battr[k])
        end
        if !isempty(obj.shuf)
            row("sd over $(length(obj.shuf)) permutations",
                [std([a[j] for a in obj.shuf]) for j in 1:length(PROPS)])
        end

    elseif obj isa Dict && !isempty(obj) && first(keys(obj)) isa Integer   # curve
        hdr()
        for k in sort(collect(keys(obj))), (a, nm) in ((1,"pixels·linear"), (3,"CNN"), (4,"ours·linear"))
            all(iszero, obj[k][a, :]) && continue
            row(@sprintf("k=%-6d %s", k, nm), obj[k][a, :])
        end

    elseif obj isa Dict && !isempty(obj) && first(keys(obj)) isa AbstractString  # history
        @printf("%-26s %7s %9s %9s %9s\n", "split/arm", "epochs", "best val", "final val", "final loss")
        for k in sort(collect(keys(obj)))
            h = obj[k]; ne = size(h.val, 1)
            fin = mean(filter(!isnan, h.val[end, :]))
            @printf("%-26s %7d %9.3f %9.3f %9.4f\n", k, ne, h.best, fin, h.loss[end])
        end
        println("\n  per-epoch detail is in `.val` (epochs × properties) and `.loss`")

    else
        println("  unrecognised layout: ", typeof(obj))
    end
end

target = length(ARGS) > 0 ? ARGS[1] : "results_canon3"
path = isabspath(target) ? target : joinpath(@__DIR__, target)
if isdir(path)
    for f in sort(readdir(path)); endswith(f, ".jls") && show_file(joinpath(path, f)); end
elseif isfile(path)
    show_file(path)
else
    println("no such file or directory: ", path)
end
