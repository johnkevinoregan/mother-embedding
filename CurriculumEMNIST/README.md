# Curriculum EMNIST — what a training-subset switch reveals

`Phase 13.` EMNIST balanced, 47 classes, official split (112,800 train / 18,800 test). The
training set is cut into **4 disjoint subsets of 28,200**; epochs 1–15 train on subset 1, 16–30
on subset 2, 31–45 on subset 3, 46–60 on subset 4. The optimiser state is never reset — the run
continues, only the data underneath it changes.

**Why this is worth doing.** The four subsets are a random partition of one dataset, so they come
from the *same* distribution. A learner that had extracted the structure of the task rather than
memorised particular examples should be unable to tell that the data changed. Every discontinuity
at epoch 15, 30 or 45 is therefore a direct measurement of how much of the fit was specific to the
examples in front of it — and unlike a train/test gap, it is measured *during* training, on the
same curve.

Two arms, identical readout, identical subsets, identical order:

| arm | features |
|:--|--:|
| our front end — 4-scale ladder + spatial max, grid 3 | 381 |
| frozen ImageNet ConvNeXt-base, stage 4, global average pooled | 1024 |

## Running it

```bash
julia --project=.. -t 14 Extract_Ours.jl          # ~60 min, cached
../ConVNextTest/.venv/bin/python extract_convnext_emnist.py   # ~3 min, cached
julia --project=.. Curriculum.jl                  # ~2 min
julia --project=.. Plot_Curriculum.jl
```

`CU_ARMS=convnext` runs one arm alone. `extract_convnext_emnist.py --check` prints the pixel
statistics Julia prints, so the two readers can be compared rather than assumed equal — they
agree exactly, which matters because EMNIST stores its images transposed and the two languages
correct for it differently.

## Three curves, because test accuracy alone cannot separate what is happening

* **test** — held-out accuracy on all 18,800 test images.
* **current subset** — accuracy on the subset being trained on right now. Its gap above `test`
  *is* memorisation, measured directly.
* **subset 1** — accuracy on subset 1 throughout, long after training has left it. Once training
  moves on this decays from "memorised" back toward "test", and how fast is what forgetting means
  here.

**Control:** each arm also runs 60 epochs on subset 1 alone — same gradient steps, same data per
epoch, no switches. Anything the switching run does that the control does not is caused by the
switch, not by the epoch count or the smaller training set.

**No regularisation** — no dropout, no weight decay, no early stopping. Memorisation is the thing
being measured, so it must not be suppressed. Standardisation uses subset 1's statistics only,
fixed for the whole run, so subsets 2–4 cannot influence the model before they arrive.

## A caveat on the configuration

λ = 8 px is part of the best stroke-set configuration and is included here, but EMNIST is 28×28
upsampled 4×, so **λ = 8 sits at its original Nyquist limit** and that channel largely measures
bilinear interpolation. Grid 3 rather than the stroke set's grid 1 because EMNIST characters are
centred, as Phase 10 found on Fashion-MNIST.

See `RESULTS.md` for the numbers and `figures/` for the curves.
