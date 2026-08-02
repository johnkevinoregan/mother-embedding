#!/usr/bin/env python3
"""
Step 1 of 2 for the few-shot test: train ConvNeXt-tiny from scratch on 37 of EMNIST's 47
classes, then dump its penultimate features for the 10 classes it has NEVER SEEN.

    ../ConVNextTest/.venv/bin/python fewshot_train.py

WHY THIS EXPERIMENT. On an i.i.d. test split, "memorised the training images and interpolated
between them" and "extracted the invariant structure" predict the same accuracy, because test
images sit near training images in any adequate representation. The two hypotheses are
indistinguishable by construction, so no amount of care with an i.i.d. split can separate them.

Holding out whole CLASSES breaks that. Remembering images of 'A' cannot help you recognise 'q'
if you have never seen a 'q'. Structure that transfers -- strokes, junctions, curvature, closure
-- can. So few-shot accuracy on unseen classes measures the part of a representation that is not
stored examples.

This is the one question where a hand-designed front end has a structural reason to win: ours
never trained on anything, so for it EMNIST's classes are ALL unseen, and its few-shot number is
not a transfer result at all -- it is just its ordinary performance.

The held-out classes are chosen at random with a fixed seed rather than picked, so they cannot
be selected to flatter any arm.
"""
import os, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, torchvision

from extract_convnext_emnist import read_images, read_labels, SRC, CACHE

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fewshot")
RES, BATCH, EPOCHS, NHELD = 112, 128, 30, 10
LR, WD, WARMUP = 3e-4, 0.05, 3


def prep(b, dev):
    x = torch.from_numpy(b).to(dev).unsqueeze(1).repeat(1, 3, 1, 1)
    x = F.interpolate(x, size=(RES, RES), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    return (x - mean) / std


def main():
    os.makedirs(OUT, exist_ok=True)
    dev = "cuda"
    rng = np.random.RandomState(1)
    held = np.sort(rng.choice(np.arange(1, 48), NHELD, replace=False))
    np.savetxt(os.path.join(OUT, "heldout.txt"), held, fmt="%d")
    print(f"held-out classes (1-based): {held.tolist()}", flush=True)

    Xtr = read_images(os.path.join(SRC, "emnist-balanced-train-images-idx3-ubyte"))
    ytr = read_labels(os.path.join(SRC, "emnist-balanced-train-images-idx3-ubyte".replace(
        "images-idx3", "labels-idx1")))
    Xte = read_images(os.path.join(SRC, "emnist-balanced-test-images-idx3-ubyte"))
    yte = read_labels(os.path.join(SRC, "emnist-balanced-test-labels-idx1-ubyte"))

    base_mask = ~np.isin(ytr, held)
    base_cls = np.setdiff1d(np.arange(1, 48), held)
    remap = {c: i for i, c in enumerate(base_cls)}
    Xb, yb = Xtr[base_mask], np.array([remap[c] for c in ytr[base_mask]])
    print(f"training on {len(base_cls)} base classes, {len(yb)} images; "
          f"{NHELD} classes withheld entirely\n", flush=True)

    torch.manual_seed(1); np.random.seed(1)
    m = torchvision.models.convnext_tiny(weights=None, num_classes=len(base_cls)).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=WD)
    steps = int(np.ceil(len(yb) / BATCH))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (
        (s + 1) / (WARMUP * steps) if s < WARMUP * steps else
        0.5 * (1 + np.cos(np.pi * (s - WARMUP * steps) / max(1, (EPOCHS - WARMUP) * steps)))))
    lf = nn.CrossEntropyLoss()
    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        idx = np.random.permutation(len(yb))
        tot = ok = 0
        for s in range(0, len(idx), BATCH):
            b = idx[s:s + BATCH]
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = m(prep(Xb[b], dev)); loss = lf(out, torch.from_numpy(yb[b]).to(dev))
            loss.backward(); opt.step(); sched.step()
            ok += (out.argmax(1).cpu().numpy() == yb[b]).sum(); tot += len(b)
        print(f"  ep {ep:2d}/{EPOCHS}  train acc {ok/tot:.4f}  ({time.time()-t0:.0f} s)",
              flush=True)

    # penultimate features = stage-4 global average pool, exactly what the frozen arm reads
    backbone = m.features
    @torch.no_grad()
    def feats(X):
        m.eval(); out = []
        for s in range(0, len(X), BATCH):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out.append(backbone(prep(X[s:s + BATCH], dev)).mean(dim=(2, 3)).float().cpu().numpy())
        return np.concatenate(out)

    for tag, X, y in (("train", Xtr, ytr), ("test", Xte, yte)):
        sel = np.isin(y, held)
        feats(X[sel]).astype(np.float32).tofile(os.path.join(OUT, f"scratch_{tag}.f32"))
        np.nonzero(sel)[0].astype(np.int64).tofile(os.path.join(OUT, f"heldidx_{tag}.i64"))
        print(f"  {tag}: {sel.sum()} held-out-class images -> features", flush=True)
    with open(os.path.join(OUT, "scratch_dim.txt"), "w") as fh:
        fh.write("768\n")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
