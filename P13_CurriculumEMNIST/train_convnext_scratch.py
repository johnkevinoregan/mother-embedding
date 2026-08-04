#!/usr/bin/env python3
"""
ConvNeXt trained END TO END from random initialisation on EMNIST, with the same
training-subset switching schedule as `Curriculum.jl`.

    ../P11_ConVNextTest/.venv/bin/python train_convnext_scratch.py

This is the arm the frozen experiment could not provide. `Curriculum.jl` compares two FROZEN
representations -- our log-Gabor front end and ImageNet ConvNeXt features -- with only a small
readout trained on top. This script instead trains all 87.6M (or 27.9M) parameters, so the
question becomes what the ARCHITECTURE achieves on this task with no ImageNet pretraining at all.

WHAT TO EXPECT, said in advance so the result is not read as a strawman. ConvNeXt's published
numbers depend on a recipe this does not use: 300 epochs, RandAugment, Mixup, CutMix, stochastic
depth, EMA, and 1.28M images. Here it gets 28,200 images per block and no augmentation --
deliberately, because every other arm in this phase is unregularised too, and the point of the
phase is to make memorisation visible rather than suppress it. A large net trained plainly on
28k images will overfit hard. That IS the answer to "what does the architecture buy without
ImageNet", but it is not ConvNeXt at its best.

RECIPE. AdamW, lr 3e-4, weight decay 0.05 (ConvNeXt's own value), cosine decay over the full 60
epochs with a 3-epoch linear warmup. lr 1e-3 -- what the frozen readouts use -- is far too high
for a from-scratch transformer-style net at this batch size. The schedule spans the whole run and
is identical for the switching and control arms, so it cannot explain a difference between them.

The subset partition is read from `cache/partition.txt`, dumped by Julia, so all arms in this
phase train on byte-identical subsets in the same order.
"""
import json, os, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, torchvision

from extract_convnext_emnist import read_images, read_labels, SRC, CACHE

OUT    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scratch_runs")
NSET, PER = 4, 15
EPOCHS = NSET * PER
NCLASS = 47
LR, WD, WARMUP = 3e-4, 0.05, 3


def loaders():
    Xtr = read_images(os.path.join(SRC, "emnist-balanced-train-images-idx3-ubyte"))
    ytr = read_labels(os.path.join(SRC, "emnist-balanced-train-labels-idx1-ubyte")) - 1
    Xte = read_images(os.path.join(SRC, "emnist-balanced-test-images-idx3-ubyte"))
    yte = read_labels(os.path.join(SRC, "emnist-balanced-test-labels-idx1-ubyte")) - 1
    perm = np.loadtxt(os.path.join(CACHE, "partition.txt"), dtype=np.int64) - 1   # Julia is 1-based
    sets = [perm[k * 28200:(k + 1) * 28200] for k in range(NSET)]
    return Xtr, ytr, Xte, yte, sets


def prep(batch_np, res, dev):
    """28x28 grey -> res x res, 3 channels, ImageNet-normalised. Same path as the frozen arm."""
    x = torch.from_numpy(batch_np).to(dev, non_blocking=True).unsqueeze(1).repeat(1, 3, 1, 1)
    x = F.interpolate(x, size=(res, res), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    return (x - mean) / std


@torch.no_grad()
def accuracy(m, X, y, res, dev, batch):
    m.eval(); ok = 0
    for s in range(0, len(y), batch):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            p = m(prep(X[s:s + batch], res, dev)).argmax(1)
        ok += (p.cpu().numpy() == y[s:s + batch]).sum()
    m.train()
    return ok / len(y)


def run(arch, res, batch, switching, Xtr, ytr, Xte, yte, sets, dev):
    tag = f"{arch}{res}_{'switch' if switching else 'set1'}"
    path = os.path.join(OUT, f"{tag}.tsv")
    if os.path.exists(path):
        print(f"{tag}: already done"); return
    torch.manual_seed(1); np.random.seed(1)
    m = getattr(torchvision.models, f"convnext_{arch}")(weights=None, num_classes=NCLASS).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=WD)
    steps = int(np.ceil(28200 / batch))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (
        (s + 1) / (WARMUP * steps) if s < WARMUP * steps else
        0.5 * (1 + np.cos(np.pi * (s - WARMUP * steps) / max(1, (EPOCHS - WARMUP) * steps)))))
    lf = nn.CrossEntropyLoss()
    rows, t0 = [], time.time()
    for ep in range(1, EPOCHS + 1):
        k = min(NSET, (ep - 1) // PER) if switching else 0
        idx = sets[k].copy(); np.random.shuffle(idx)
        for s in range(0, len(idx), batch):
            b = idx[s:s + batch]
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss = lf(m(prep(Xtr[b], res, dev)), torch.from_numpy(ytr[b]).to(dev))
            loss.backward(); opt.step(); sched.step()
        te = accuracy(m, Xte, yte, res, dev, batch)
        cu = accuracy(m, Xtr[idx], ytr[idx], res, dev, batch)
        s1 = accuracy(m, Xtr[sets[0]], ytr[sets[0]], res, dev, batch)
        rows.append((ep, k + 1, te, cu, s1))
        print(f"  {tag:<22} ep {ep:2d} set {k+1}  test {te:.4f}  current {cu:.4f}  "
              f"set1 {s1:.4f}  ({time.time()-t0:.0f} s)", flush=True)
    with open(path, "w") as fh:
        fh.write("epoch\tset\ttest\tcurrent\tset1\n")
        for r in rows:
            fh.write(f"{r[0]}\t{r[1]}\t{r[2]:.6f}\t{r[3]:.6f}\t{r[4]:.6f}\n")
    del m, opt; torch.cuda.empty_cache()


def main():
    os.makedirs(OUT, exist_ok=True)
    dev = "cuda"
    print(f"{torch.cuda.get_device_name(0)}   lr {LR}  wd {WD}  cosine + {WARMUP}ep warmup, "
          f"no augmentation", flush=True)
    Xtr, ytr, Xte, yte, sets = loaders()
    print(f"train {Xtr.shape}  test {Xte.shape}  subsets {[len(s) for s in sets]}\n", flush=True)
    # (arch, resolution, batch) -- `base` at 224 is matched exactly to the frozen arm;
    # `tiny` at 112 is the cheap check, 10x faster and less prone to overfitting 28k images
    for arch, res, batch in (("tiny", 112, 128), ("base", 224, 32)):
        for switching in (True, False):
            run(arch, res, batch, switching, Xtr, ytr, Xte, yte, sets, dev)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
