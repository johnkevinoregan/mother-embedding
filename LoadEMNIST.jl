module LoadEMNIST

export read_emnist_images, read_emnist_labels, emnist_class_name,
       default_class_order, filter_class, load_emnist, to_image

const DEFAULT_DATA_DIR = joinpath(homedir(), "Julia", "DATABASES", "EMNIST")
const EMNIST_LOWER = ['a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't']

# ---- IDX file parsers (EMNIST balanced, train split) ----

function read_emnist_images(filepath::AbstractString; max_images::Int=0)
    open(filepath, "r") do f
        magic = ntoh(read(f, UInt32))
        @assert magic == 2051 "Invalid magic number for images"
        n_images = ntoh(read(f, UInt32))
        n_rows   = ntoh(read(f, UInt32))
        n_cols   = ntoh(read(f, UInt32))
        n_to_read = max_images > 0 ? min(max_images, Int(n_images)) : Int(n_images)
        pixels    = read(f, n_to_read * Int(n_rows) * Int(n_cols))
        img_array = reshape(pixels, (Int(n_cols), Int(n_rows), n_to_read))
        out = zeros(Float32, Int(n_rows), Int(n_cols), n_to_read)
        for i in 1:n_to_read
            out[:, :, i] = Float32.(img_array[:, :, i]) ./ 255f0
        end
        return out
    end
end

function read_emnist_labels(filepath::AbstractString; max_labels::Int=0)
    open(filepath, "r") do f
        magic = ntoh(read(f, UInt32))
        @assert magic == 2049 "Invalid magic number for labels"
        n_labels  = ntoh(read(f, UInt32))
        n_to_read = max_labels > 0 ? min(max_labels, Int(n_labels)) : Int(n_labels)
        labels    = read(f, n_to_read)
        return Int.(labels) .+ 1
    end
end

# ---- Class index <-> name mapping (EMNIST balanced) ----
#   1-10  : digits 0-9
#   11-36 : uppercase A-Z
#   37-47 : lowercase subset a,b,d,e,f,g,h,n,q,r,t

function emnist_class_name(idx::Int)
    if 1 <= idx <= 10
        return string(idx - 1)
    elseif 11 <= idx <= 36
        return string(Char(Int('A') + (idx - 11)))
    elseif 37 <= idx <= 47
        return string(EMNIST_LOWER[idx - 36])
    else
        return "?" * string(idx)
    end
end

# Canonical ordering: uppercase letters first, then the lowercase subset, then digits.
default_class_order() = vcat(collect(11:36), collect(37:47), collect(1:10))

# Pull all images of a given class as a Vector{Matrix{Float32}}
function filter_class(imgs_3d::Array{Float32,3}, labels::Vector{Int}, class_idx::Int)
    indices = findall(==(class_idx), labels)
    return [imgs_3d[:, :, i] for i in indices]
end

"""
    to_image(m::AbstractMatrix)

Return `m` (as returned in `images`/`class_images`) in the row order expected
by `Plots.heatmap` (or similar raster-image display) so the character appears
upright, e.g. `heatmap(to_image(img), color=:grays)`.

Raw EMNIST samples are already correctly oriented row-major (row 1 = top of
the character); `Plots.heatmap` draws row 1 at the *bottom* by default, so
without this the image renders vertically flipped. This is equivalent to
passing `yflip=true` to `heatmap`, exposed here so callers don't need to
remember a plotting-library-specific keyword.
"""
to_image(m::AbstractMatrix) = reverse(m, dims=1)

"""
    load_emnist(; data_dir=DEFAULT_DATA_DIR, n_images_to_load=20000,
                  n_classes=5, class_order=default_class_order())

Load images from the EMNIST balanced training split and bucket them by class.

Returns a named tuple `(images, labels, selected_classes, class_names, class_images)`:
- `images`, `labels`: the raw scan of the first `n_images_to_load` images/labels.
- `selected_classes`: the first `n_classes` entries of `class_order`.
- `class_names`: human-readable name for each selected class.
- `class_images`: `class_images[c]` is a `Vector{Matrix{Float32}}` of every image
  of `selected_classes[c]` found within the scanned `images`.
"""
function load_emnist(; data_dir::AbstractString=DEFAULT_DATA_DIR,
                        n_images_to_load::Int=20000,
                        n_classes::Int=5,
                        class_order::Vector{Int}=default_class_order())
    images_path = joinpath(data_dir, "emnist-balanced-train-images-idx3-ubyte")
    labels_path = joinpath(data_dir, "emnist-balanced-train-labels-idx1-ubyte")

    images = read_emnist_images(images_path; max_images=n_images_to_load)
    labels = read_emnist_labels(labels_path; max_labels=n_images_to_load)

    selected_classes = class_order[1:n_classes]
    class_names = [emnist_class_name(i) for i in selected_classes]
    class_images = [filter_class(images, labels, idx) for idx in selected_classes]

    return (; images, labels, selected_classes, class_names, class_images)
end

end # module
