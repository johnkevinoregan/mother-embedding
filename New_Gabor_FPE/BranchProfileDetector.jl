### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ b1000000-0000-0000-0000-000000000001
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
end

# ╔═╡ b1000000-0000-0000-0000-000000000002
begin
    using PlutoUI
    using Plots
    using FFTW
    using Statistics
end

# ╔═╡ b1000000-0000-0000-0000-000000000003
begin
    include(joinpath(@__DIR__, "..", "LoadEMNIST.jl"))
    using .LoadEMNIST
end

# ╔═╡ b1000000-0000-0000-0000-000000000004
md"""
# The branch-profile keypoint detector

One detector for **all** keypoint types, replacing the calibrated `c₀`-harmonic
two-channel scheme. For every 2π direction φ:

```
B_φ(p) = min( E_θ(φ)(p),  E_θ(φ)(p + d·u(φ)) )
```

— "a stroke of the *matching orientation* is here **AND** still there one step
out in direction φ". The `min` anchors the detector to the stroke (background is
vetoed by the first operand), and the d-offset makes the π-periodic oriented
energy **directional**. The angular local maxima of the profile `B_φ(p)` *are
the branches* leaving p; the branch **count and angles** give the type
structurally — no sum, no calibration:

| branches | angles | type |
|---|---|---|
| 1 | — | **endpoint** |
| 2 | ≈180° apart | continuation (stroke body / smooth curve) — *not a keypoint* |
| 2 | else | **L-corner** |
| 3 | any | **T** (or Y) |
| 4 | — | **X** |

Below: per-type **heatmaps** for a slider-chosen letter (the score shown is the
*weakest accepted branch* — the strength of the conjunction), and the
**η² / leave-one-out diagnosticity survey** of the resulting typed counts.
"""

# ╔═╡ b1000000-0000-0000-0000-000000000005
# ---- constants (the small scale validated on synthetic figures) ----
begin
    const IMG=112; const N_THETA=48; const K_PHI=48; const LAM=8f0
    const SIGMA_N=3f0; const SIGMA_T=6f0; const KSIZE=25
    const THETAS=Float32.(range(0,π,length=N_THETA+1)[1:N_THETA])
    const PHIS=Float32.(range(0,2π,length=K_PHI+1)[1:K_PHI])
    const PHIS_DEG=rad2deg.(PHIS)
    const CLASSES=["O","C","I","L","T","X","K","A","H","Y","E","F"]
    const THR0=0.35f0   # survey default: abs gate — branch present if B > THR0·max(E)
    const REL0=0.40f0   # survey default: rel gate — endpoint confirmation, B > REL0·strongest-branch
    const D0=6f0        # survey default: probe offset (≈ stroke width)
end

# ╔═╡ b1000000-0000-0000-0000-000000000006
# ---- Gabor bank + precomputed kernel spectra (padded FFT convolution) ----
begin
    function gabor_kernel(θ)
        h=KSIZE÷2; K=zeros(ComplexF32,KSIZE,KSIZE); c,s=cos(θ),sin(θ)
        for i in -h:h, j in -h:h
            y,x=Float32(i),Float32(j); xt=x*c+y*s; xn=-x*s+y*c
            K[i+h+1,j+h+1]=exp(-(xt^2/(2SIGMA_T^2)+xn^2/(2SIGMA_N^2)))*cis(2f0π*xn/LAM)
        end
        K .- mean(K)
    end
    const PH,PW = IMG+KSIZE-1, IMG+KSIZE-1
    const O1 = KSIZE÷2
    const BHAT = [begin B=zeros(ComplexF32,PH,PW); B[1:KSIZE,1:KSIZE].=gabor_kernel(θ); fft(B) end
                  for θ in THETAS]
    function energy_stack(img)
        A=zeros(ComplexF32,PH,PW); A[1:IMG,1:IMG].=img; Ah=fft(A)
        E=Array{Float32,3}(undef,N_THETA,IMG,IMG)
        for t in 1:N_THETA
            f=ifft(Ah.*BHAT[t]); E[t,:,:].=abs.(@view f[O1+1:O1+IMG,O1+1:O1+IMG])
        end
        E
    end
end

# ╔═╡ b1000000-0000-0000-0000-000000000007
# ---- helpers: bilinear sampling / shifting, orientation lookup, upsampling ----
begin
    tidx(θ)=mod(round(Int,(mod(θ,Float32(π))/Float32(π))*N_THETA),N_THETA)+1
    @inline function bilinear(M,y,x)
        H,W=size(M); (y<1||x<1||y>H||x>W)&&return 0f0
        y0,x0=floor(Int,y),floor(Int,x); y1,x1=min(y0+1,H),min(x0+1,W); fy,fx=y-y0,x-x0
        (1-fy)*(1-fx)*M[y0,x0]+fy*(1-fx)*M[y1,x0]+(1-fy)*fx*M[y0,x1]+fy*fx*M[y1,x1]
    end
    # integer shift with zero fill: out[y,x]=M[y+sy,x+sx]
    function ishift(M,sy,sx)
        H,W=size(M); out=zeros(Float32,H,W)
        ys=max(1,1-sy):min(H,H-sy); xs=max(1,1-sx):min(W,W-sx)
        @views out[ys,xs].=M[ys.+sy,xs.+sx]; out
    end
    # whole-array bilinear shift: out[y,x]=M[y+dy,x+dx]
    function bshift(M,dy,dx)
        sy0,sx0=floor(Int,dy),floor(Int,dx); fy,fx=dy-sy0,dx-sx0
        (1-fy)*(1-fx).*ishift(M,sy0,sx0) .+ fy*(1-fx).*ishift(M,sy0+1,sx0) .+
        (1-fy)*fx.*ishift(M,sy0,sx0+1) .+ fy*fx.*ishift(M,sy0+1,sx0+1)
    end
    function upsample(img)
        H,W=size(img); out=zeros(Float32,IMG,IMG)
        for i in 1:IMG, j in 1:IMG
            out[i,j]=bilinear(img,Float32(1+(i-1)*(H-1)/(IMG-1)),Float32(1+(j-1)*(W-1)/(IMG-1)))
        end
        out
    end
end

# ╔═╡ b1000000-0000-0000-0000-000000000008
# ---- the branch-profile machinery ----
begin
    # B_φ stack for all pixels at once (one bilinear whole-array shift per φ)
    function branch_stack(E; d=D0)
        Bst=Array{Float32,3}(undef,K_PHI,IMG,IMG)
        for (k,φ) in enumerate(PHIS)
            Eθ=@view E[tidx(φ),:,:]
            Bst[k,:,:] .= min.(Eθ, bshift(Eθ, d*sin(φ), d*cos(φ)))
        end
        Bst
    end
    # branches at one pixel: angular local maxima (±3 samples) above thr,
    # then greedy angular NMS (suppress within 40° of a stronger accepted branch)
    function branch_list(Bst,y,x,thr)
        cand=Tuple{Int,Float32}[]      # (φ-index, strength)
        for k in 1:K_PHI
            v=Bst[k,y,x]; v<=thr && continue
            ok=true
            for j in -3:3
                j==0 && continue
                Bst[mod1(k+j,K_PHI),y,x]>v && (ok=false; break)
            end
            ok && push!(cand,(k,v))
        end
        sort!(cand,by=b->-b[2])
        br=Tuple{Int,Float32}[]
        for c in cand
            near=any(begin Δ=abs(PHIS_DEG[c[1]]-PHIS_DEG[b[1]]); min(Δ,360-Δ)<40 end for b in br)
            near || push!(br,c)
        end
        br
    end
    # per-type score maps; score = weakest accepted branch (conjunction strength).
    # TWO GATES, each used for the question it is good at (see Notes):
    #  · ABSOLUTE gate (B > thr_frac·max E) types the junctions — it rejects weak
    #    spurious branches, but where a stroke tapers it drops the weaker along-
    #    stroke branch and would mislabel mid-stroke pixels as endpoints;
    #  · RELATIVE gate (B > rel·strongest-branch-here) confirms ENDPOINTS only —
    #    a pixel is an endpoint iff the rel-gated profile has exactly 1 branch,
    #    so a tapering stroke (2nd branch weak in absolute terms but comparable
    #    in relative terms) reads "continuation", not "endpoint".
    function type_maps(E; thr_frac=THR0, d=D0, rel=REL0)
        Bst=branch_stack(E; d=d); thr=thr_frac*maximum(E)
        m=Dict(s=>zeros(Float32,IMG,IMG) for s in (:cont,:endp,:corner,:T,:X))
        for y in 1:IMG, x in 1:IMG
            mx=0f0
            for k in 1:K_PHI; Bst[k,y,x]>mx && (mx=Bst[k,y,x]); end
            mx<=thr && continue
            br=branch_list(Bst,y,x,thr)          # abs-gated
            n=length(br); n==0 && continue
            brr=branch_list(Bst,y,x,rel*mx)      # rel-gated
            score=br[end][2]
            if length(brr)==1
                m[:endp][y,x]=brr[1][2]
            elseif n<=2
                if n==1
                    m[:cont][y,x]=score          # 1 abs-branch but >1 rel-branch: taper, not an end
                else
                    Δ=abs(PHIS_DEG[br[1][1]]-PHIS_DEG[br[2][1]]); Δ=min(Δ,360-Δ)
                    (abs(Δ-180)<30 ? m[:cont] : m[:corner])[y,x]=score
                end
            elseif n==3
                m[:T][y,x]=score
            else
                m[:X][y,x]=score
            end
        end
        m
    end
    # keypoints of one type map: local maxima (5×5) + greedy spatial NMS
    function kpts(M; r=8)
        cand=Tuple{Int,Int,Float32}[]
        for y in 3:IMG-2, x in 3:IMG-2
            v=M[y,x]; v<=0 && continue
            ok=true
            for dy in -2:2, dx in -2:2
                (dy==0&&dx==0) && continue
                M[y+dy,x+dx]>v && (ok=false; break)
            end
            ok && push!(cand,(y,x,v))
        end
        sort!(cand,by=p->-p[3])
        keep=Tuple{Int,Int,Float32}[]
        for c in cand
            any(hypot(c[1]-k[1],c[2]-k[2])<r for k in keep) || push!(keep,c)
        end
        keep
    end
end

# ╔═╡ b1000000-0000-0000-0000-000000000009
# ---- global shape harmonics (for the survey comparison) ----
begin
    function angular_spectrum(img; nmax=4)
        H,W=size(img); tot=sum(img)
        cy=sum(i*img[i,j] for i in 1:H,j in 1:W)/tot
        cx=sum(j*img[i,j] for i in 1:H,j in 1:W)/tot
        α=[atan(i-cy,j-cx) for i in 1:H,j in 1:W]; w=img./tot
        [sum(w.*cis.(-Float32(n).*α)) for n in 0:nmax]
    end
end

# ╔═╡ b1000000-0000-0000-0000-00000000000a
em = load_emnist(n_images_to_load=8000, n_classes=47);

# ╔═╡ b1000000-0000-0000-0000-00000000000b
md"""
## Per-type detector heatmaps

Pick a letter and an instance; adjust the branch threshold and the probe
offset `d`. The **continuation** map (2 branches ≈180° apart) is not a keypoint
type — it traces the stroke body and is shown as a sanity check. Dots on the
letter mark the surviving keypoints per type (spatial NMS radius 8 px).
"""

# ╔═╡ b1000000-0000-0000-0000-00000000000c
md"""
letter: $(@bind ci Slider(1:length(CLASSES), default=5, show_value=true))
instance: $(@bind inst Slider(1:30, default=1, show_value=true))
abs threshold: $(@bind thr_f Slider(0.20f0:0.01f0:0.50f0, default=THR0, show_value=true))
rel gate: $(@bind rel_f Slider(0.30f0:0.05f0:0.60f0, default=REL0, show_value=true))
d: $(@bind d_probe Slider(3f0:1f0:10f0, default=D0, show_value=true))
"""

# ╔═╡ b1000000-0000-0000-0000-00000000000d
begin
    letter = CLASSES[ci]
    img_sel = upsample(em.class_images[findfirst(==(letter), em.class_names)][inst])
    E_sel = energy_stack(img_sel)
    maps_sel = type_maps(E_sel; thr_frac=thr_f, d=d_probe, rel=rel_f)
    kp_sel = Dict(s=>kpts(maps_sel[s]) for s in (:endp,:corner,:T,:X))
    md"**letter $(letter)**, instance $(inst) — endpoints: $(length(kp_sel[:endp])), corners: $(length(kp_sel[:corner])), T: $(length(kp_sel[:T])), X: $(length(kp_sel[:X]))"
end

# ╔═╡ b1000000-0000-0000-0000-00000000000e
begin
    kwargs=(yflip=true, aspect_ratio=:equal, axis=false, ticks=false, cbar=false,
            xlims=(1,IMG), ylims=(1,IMG))
    p0=heatmap(img_sel; c=:grays, title="letter $(letter)", kwargs...)
    kpcolor=Dict(:endp=>:cyan, :corner=>:orange, :T=>:magenta, :X=>:red)
    for s in (:endp,:corner,:T,:X)
        isempty(kp_sel[s]) && continue
        scatter!(p0,[p[2] for p in kp_sel[s]],[p[1] for p in kp_sel[s]];
                 mc=kpcolor[s], msw=0, ms=5, label="")
    end
    panels=[p0]
    for (s,ttl,cm) in ((:cont,"continuation",:viridis), (:endp,"ENDPOINT",:viridis),
                       (:corner,"L-CORNER",:viridis), (:T,"T-junction",:viridis),
                       (:X,"X-crossing",:viridis))
        push!(panels, heatmap(maps_sel[s]; c=cm, title=ttl, kwargs...))
    end
    plot(panels...; layout=(2,3), size=(950,640))
end

# ╔═╡ b1000000-0000-0000-0000-00000000000f
md"""
## Diagnosticity survey

Typed counts from the branch-profile detector (`n_end, n_cor, n_T, n_X`) plus
the four shape harmonics `|M1..4|/M0`, over 12 classes × 30 instances, at the
fixed defaults (abs $(THR0), rel $(REL0), d = $(Int(D0))). Reference numbers from
`KeyPointDiagnosticity.md`: previous typed-count detectors reached **16.7–18.9 %**
local-only and **49–54 %** full; shape-only is **57.2 %**. *(This cell takes a
few minutes — it does not depend on the sliders above.)*
"""

# ╔═╡ b1000000-0000-0000-0000-000000000010
survey = let
    FEAT=["n_end","n_cor","n_T","n_X","|M1|/M0","|M2|/M0","|M3|/M0","|M4|/M0"]
    function features(img0)
        img=upsample(img0); E=energy_stack(img)
        m=type_maps(E; thr_frac=THR0, d=D0)
        M=angular_spectrum(img); g=[abs(M[n+1])/abs(M[1]) for n in 1:4]
        Float32[length(kpts(m[:endp])),length(kpts(m[:corner])),
                length(kpts(m[:T])),length(kpts(m[:X])),g...]
    end
    X=Vector{Vector{Float32}}(); y=String[]
    for c in CLASSES
        imgs=em.class_images[findfirst(==(c),em.class_names)]
        for k in 1:min(30,length(imgs)); push!(X,features(imgs[k])); push!(y,c); end
    end
    D=Matrix(reduce(hcat,X)')
    (; FEAT, D, y)
end;

# ╔═╡ b1000000-0000-0000-0000-000000000011
begin
    function eta2(col,labels)
        gm=mean(col); ss_tot=sum((col.-gm).^2); ss_b=0.0
        for c in unique(labels)
            idx=labels.==c; ss_b+=count(idx)*(mean(col[idx])-gm)^2
        end
        ss_tot==0 ? 0.0 : ss_b/ss_tot
    end
    function loo_acc(Z,y)
        n=size(Z,1); correct=0
        for i in 1:n
            best=Inf; bc=""
            for c in unique(y)
                idx=[j for j in 1:n if y[j]==c && j!=i]
                isempty(idx) && continue
                m=vec(mean(Z[idx,:],dims=1)); dd=sum((Z[i,:].-m).^2)
                dd<best && (best=dd; bc=c)
            end
            bc==y[i] && (correct+=1)
        end
        correct/n
    end
end

# ╔═╡ b1000000-0000-0000-0000-000000000012
let
    (; FEAT, D, y) = survey
    mu=vec(mean(D,dims=1)); sd=vec(std(D,dims=1)); sd[sd.==0].=1f0
    Z=(D.-mu')./sd'
    esc(s)=replace(s,"|"=>"\\|")
    rows=String[]
    for (j,f) in enumerate(FEAT)
        e=eta2(D[:,j],y)
        bar=repeat("█",round(Int,30*max(e,0.0)))
        push!(rows,"| `$(esc(f))` | $(round(e,digits=3)) | $(round(minimum(D[:,j]),digits=2)) .. $(round(maximum(D[:,j]),digits=2)) | $(bar) |")
    end
    a_loc=loo_acc(Z[:,1:4],y); a_shp=loo_acc(Z[:,5:8],y); a_all=loo_acc(Z,y)
    Markdown.parse("""
### η² per feature

| feature | η² | range | |
|---|---|---|---|
$(join(rows,"\n"))

### Leave-one-out nearest-class-mean accuracy (chance 8.3 %)

| feature set | LOO |
|---|---|
| typed counts only | **$(round(100a_loc,digits=1)) %** |
| shape harmonics only | $(round(100a_shp,digits=1)) % |
| both | $(round(100a_all,digits=1)) % |

Previous detectors (from the md): typed counts 16.7–18.9 % local-only.
""")
end

# ╔═╡ b1000000-0000-0000-0000-000000000013
md"""
## Notes

- **Why two gates.** With the absolute gate alone, real EMNIST produced *phantom
  endpoints* en masse (O averaged ~36!): wherever a stroke tapers, the weaker
  along-stroke branch falls below `0.35·max(E)` and a mid-stroke pixel reads
  "1 branch". With a relative gate alone (`B > 0.4·strongest-branch-here`) the
  endpoints became sane (O ≈ 0.7, C ≈ 1.3, Y ≈ 2.8 — right ordering) but weak
  angular noise-peaks now cleared the gate and *junctions* exploded (O: ~21
  phantom T's). Hence the split: **abs gate types junctions** (good at rejecting
  weak spurious branches), **rel gate confirms endpoints** (good at not being
  fooled by taper). Both are sliders above.
- **Score shown in the maps** = the *weakest* accepted branch at that pixel —
  the strength of the AND. A T pixel is only as strong as its stem.
- **Known fragilities** (from the synthetic validation): the T-stem branch reads
  ≈0.36 at the center (the crossbar dominates locally), just above the default
  abs threshold — lowering it recovers missed stems at the cost of noise
  branches. The rel gate must stay ≤ ≈0.43 or it drops the synthetic T-stem
  (ratio 0.36/0.84). And the 2-point `min` probe is sensitive to local gaps in
  wobbly strokes; probing 2–3 points along each ray is the likely
  robustification.
- **Corner vs continuation** is an angle judgment (±30° around 180°): a strongly
  curved stroke legitimately drifts toward "corner" — on real letters the corner
  and T channels still over-fire (curvature + wobble), and that is the open
  problem the heatmaps let you inspect.
- Scale here is the *small* validated one (λ=8, σ=3/6, probe d≈stroke width) —
  deliberately different from the `D_RAY=15` ray-harmonic scale of
  `KeyPointDiagnosticity.jl`.
"""

# ╔═╡ Cell order:
# ╠═b1000000-0000-0000-0000-000000000001
# ╠═b1000000-0000-0000-0000-000000000002
# ╠═b1000000-0000-0000-0000-000000000003
# ╟─b1000000-0000-0000-0000-000000000004
# ╠═b1000000-0000-0000-0000-000000000005
# ╠═b1000000-0000-0000-0000-000000000006
# ╠═b1000000-0000-0000-0000-000000000007
# ╠═b1000000-0000-0000-0000-000000000008
# ╠═b1000000-0000-0000-0000-000000000009
# ╠═b1000000-0000-0000-0000-00000000000a
# ╟─b1000000-0000-0000-0000-00000000000b
# ╟─b1000000-0000-0000-0000-00000000000c
# ╟─b1000000-0000-0000-0000-00000000000d
# ╠═b1000000-0000-0000-0000-00000000000e
# ╟─b1000000-0000-0000-0000-00000000000f
# ╠═b1000000-0000-0000-0000-000000000010
# ╠═b1000000-0000-0000-0000-000000000011
# ╠═b1000000-0000-0000-0000-000000000012
# ╟─b1000000-0000-0000-0000-000000000013
