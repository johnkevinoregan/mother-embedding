# P11_ConVNextTest — frozen ConvNeXt on the broken-lines task

Phase 9 found that a linear readout on our 31 hand-designed features beat a properly trained
CNN on four of five geometric properties, and transferred perfectly across contrast polarity
where the CNN reached −2.107. **The obvious objection is that the CNN only had 10,000 stroke
images to learn from.** A representation learned from more data might make curvedness,
brokenness and vertex angle linearly available without anyone designing operators for them.

This directory answers that objection with the strongest cheaply available counter-example: a
**frozen ImageNet ConvNeXt**. ConvNeXt-Base carries ~88 M parameters fitted to 1.28 M natural
images. If a large general-purpose learned representation already makes these properties
explicit, then the front end is solving a problem that scale has already solved.

## Why this is a cleaner comparison than Phase 9's CNN

Both arms are **fixed functions of the image**. Nothing is fitted to strokes except the
readout, and the readout is identical: same ridge with per-property λ chosen on validation,
same two-hidden-layer MLP, same epoch budget, same R², same splits. Phase 9's CNN arm
confounded *representation* with *training data*; this separates them.

And it is not a handicapped opponent. Frozen ConvNeXt gets **768 numbers** (stage 4) or 1,440
(all stages) where ours gets **31**, plus 28.6–88 M learned parameters against our zero.

| | ours | frozen ConvNeXt-Tiny | frozen ConvNeXt-Base |
|:--|--:|--:|--:|
| learned parameters | **0** | 28.6 M | ~88 M |
| images used to build it | 0 | 1.28 M | 1.28 M |
| features handed to the readout | 31 | 768 / 1440 | 1024 / 2048 |

## What is read out

Global average pooling of each of the four ConvNeXt stages. Stage 4's GAP **is** the standard
penultimate feature vector — torchvision's classifier is `LayerNorm2d → Flatten → Linear` on
the pooled tensor — so reading all four stages includes the conventional "frozen features"
arm and adds the earlier ones for free. Earlier stages matter because ImageNet's last stage is
tuned to object category, and none of these properties is an object category.

Global pooling rather than a spatial grid because Phase 9 found grid 1 beat grid 5 on every
property here: stroke position is randomised, so a fixed grid is pure liability. GAP is the
analogue of the configuration our own front end wins with, which if anything favours ConvNeXt.

## Predictions, recorded before the first full run

1. **ConvNeXt beats raw pixels on every property.** It is a strong general representation and
   the pixel arms are the floor.
2. **Competitive on `thickness` and `fuzziness`, worse on `vangle` and `arms`.** Scale and blur
   are things ImageNet features encode well; vertex angle and arm count are 2π ray-counting
   properties. Geirhos et al. (ICLR 2019) measured ImageNet CNNs at 22.1 % shape bias against
   humans' 95.9 %, so a texture-biased representation should be weakest where geometry is the
   whole task.
3. **It collapses on the polarity extrapolation split.** Trained on light strokes, tested on
   dark. Ours is unchanged because quadrature energy discards the sign of contrast by
   construction; ImageNet features certainly do not.
4. **Stage 3 may beat stage 4**, for the reason above.

> **Prediction 3 already looks wrong.** A 400-image smoke test had frozen ConvNeXt-Tiny
> *ahead* of ours on the polarity split (curvedness 0.904 vs 0.630, vangle 0.625 vs 0.288),
> while separately predicting polarity itself at R² 0.83–0.95 on the other splits — so it
> plainly *encodes* polarity yet its shape readouts still transfer across it. That is not what
> the invariance argument predicts and it is recorded here before the full run rather than
> after. Whether it survives 16,000 images is the main thing this experiment now asks.

## Running it

Three steps, in order. Step 1 writes ~4 GB to `data/`, step 2 writes features to `features/`,
step 3 writes tables to `results/` and caches our own features in `cache/`. All four are
gitignored.

```bash
cd ~/claude-code/mother-embedding/P11_ConVNextTest
julia --project=.. -t 16 ConvNextStimuli.jl      # generate the Phase 9 stimuli
./.venv/bin/python extract_convnext.py           # frozen ConvNeXt features (GPU)
julia --project=.. -t 16 ConvNextReadout.jl      # score every arm, same protocol
```

| variable | default | |
|:--|:--|:--|
| `CX_NTRAIN` / `CX_NTEST` | 16000 / 4000 | Phase 9's sizes |
| `CX_MODELS` | `tiny,base` | which ConvNeXt sizes |
| `CX_GRID` | 1 | pooling grid for *our* arm; 1 is Phase 9's best |
| `CX_EPOCHS` | 100 | MLP epochs, matched across arms |
| `CX_BATCH` / `CX_RES` | 128 / 224 | extraction batch and input resolution |

### The Python environment

`torch` is not in the Julia project, so there is a venv here. Two things bit during setup and
are recorded so they don't bite again:

* **Debian's `python3 -m venv` produced a venv with no `pip`** and no `ensurepip` module.
  Bootstrapped with `get-pip.py`.
* **The default PyPI wheels are built for CUDA 13**, and this machine's driver is 560.35
  (CUDA 12.6), so `torch.cuda.is_available()` was `False` with *"the NVIDIA driver on your
  system is too old"*. Fixed by installing from the **cu126** index:

```bash
python3 -m venv .venv
curl -sSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py && ./.venv/bin/python /tmp/get-pip.py
./.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

On the RTX 4090 extraction runs at **0.5 ms/image**, so all 80,000 images take under a minute
for Tiny. On CPU the same job is 30–60 minutes, and Base is hours — which is why it is worth
getting the CUDA wheel right rather than settling for the CPU build.

## Files

| file | kind | |
|:--|:--|:--|
| `ConvNextStimuli.jl` | plain script | step 1 — writes the Phase 9 stimuli as raw Float32 |
| `extract_convnext.py` | plain script | step 2 — frozen ConvNeXt stage features |
| `ConvNextReadout.jl` | plain script | step 3 — every arm, one table per split |
| `Readout.module.jl` | **plain module** | ⚠ do not open in Pluto. The Phase 9 readout, copied verbatim |

`Readout.module.jl` duplicates `Phase9_Readouts.jl` rather than including it, because that file
runs `main()` on load and including it would regenerate 20,000 stimuli and run all of Phase 9
as a side effect. **If the Phase 9 readout changes, change it here too** — otherwise the
comparison quietly stops being a comparison.

## Interchange format

Raw little-endian Float32 plus a text manifest. No HDF5, no `.npy` writer, nothing that can
drift between two languages.

Each image is written **row-major** (`permutedims` before `write`), so Python's
`fromfile(...).reshape(n, N, N)` and Julia's `permutedims(reshape(·, N, N))` both recover the
same orientation. The Julia readout reads these same `.f32` files back rather than keeping a
parallel `.jls`, so "both arms saw identical stimuli" is a fact about the file rather than a
claim about two RNG streams staying in step. A silent transpose already cost this project a
day on Fashion-MNIST.
