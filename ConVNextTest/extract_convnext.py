#!/usr/bin/env python3
"""
Step 2 of 3. Frozen ConvNeXt features for the Phase 9 stroke stimuli.

    .venv/bin/python extract_convnext.py            # tiny + base, every split
    CX_MODELS=tiny .venv/bin/python extract_convnext.py

"Frozen" means exactly that: ImageNet weights, `eval()`, `inference_mode()`, no gradient
ever taken. The network is a fixed function, used the same way our log-Gabor bank is used —
so the comparison is representation against representation, with the readout held constant.

WHAT IS READ OUT. Global average pooling of each of the four ConvNeXt stages:

    stage 1   96 channels    (tiny)   192  (base)
    stage 2  192                      384
    stage 3  384                      768
    stage 4  768                     1024

Stage 4's GAP *is* the penultimate feature — torchvision's classifier is
`LayerNorm2d → Flatten → Linear` on the pooled tensor — so reading the four stages already
includes the standard "frozen ConvNeXt features" vector, and adds the earlier ones for free.
Earlier stages are worth having because ImageNet's late layers are tuned to object category,
and the properties we are asking about are geometric.

Global pooling rather than a spatial grid because Phase 9 found grid 1 beat grid 5 on every
property in this dataset — position is randomised, so a fixed grid is pure liability. GAP is
the analogue of the configuration our own front end wins with, which if anything favours
ConvNeXt: it gets 768 numbers where ours gets 31.

PREPROCESSING, and the one thing to be suspicious of. Our stimuli are 112x112, single
channel, mid-grey background. ConvNeXt wants 224x224 RGB with ImageNet statistics. So the
image is replicated to three channels, resized bilinearly, and normalised. The replication
is not neutral in principle -- a grey-world network is being fed a perfectly achromatic
image, which is off the manifold it was trained on -- but it is the standard way to put a
greyscale image through an ImageNet model, and it is applied identically to every stimulus.
"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import torchvision

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
FEAT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "features")
MODELS = os.environ.get("CX_MODELS", "tiny,base").split(",")
BATCH = int(os.environ.get("CX_BATCH", "128"))
RES = int(os.environ.get("CX_RES", "224"))

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# torchvision's ConvNeXt `features` alternates downsample / stage, so the CNBlock stages are
# the odd indices. Asserted against the channel counts below rather than trusted.
STAGE_IDX = (1, 3, 5, 7)


def read_manifest():
    splits = []
    with open(os.path.join(DATA, "manifest.txt")) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            name, ntr, nte, n, npr = line.split()
            splits.append((name, int(ntr), int(nte), int(n), int(npr)))
    return splits


def load_images(name, kind, count, n):
    """Raw Float32 written row-major by ConvNextStimuli.jl."""
    path = os.path.join(DATA, f"{name}_{kind}_img.f32")
    a = np.fromfile(path, dtype=np.float32)
    expect = count * n * n
    if a.size != expect:
        raise SystemExit(f"{path}: got {a.size} floats, expected {expect}")
    return a.reshape(count, n, n)


def build(model_name):
    """
    `tiny` / `base` load ImageNet weights; the `rand` suffix (`tinyrand`) builds the *same
    architecture* with random initialisation and no training whatsoever.

    That pair is the control that separates the two explanations for ConvNeXt's result:
    architectural inductive bias — depthwise 7x7 convolutions, patchify stem, LayerNorm —
    versus anything actually learned from the 1.28M ImageNet photographs. Random-weight
    convnet features are known to be a surprisingly strong baseline, so this is a live
    possibility rather than a formality. Seeded so it is reproducible.
    """
    random_init = model_name.endswith("rand")
    arch = model_name[:-4] if random_init else model_name
    fn = getattr(torchvision.models, f"convnext_{arch}")
    if random_init:
        torch.manual_seed(int(os.environ.get("CX_SEED", "0")))
        model = fn(weights=None).eval()
    else:
        model = fn(weights="IMAGENET1K_V1").eval()
    # Confirm the stage indices really are the stages, so a torchvision version bump that
    # reorders `features` fails loudly instead of silently reading a downsample layer.
    chans = []
    with torch.inference_mode():
        x = torch.zeros(1, 3, RES, RES)
        for i, block in enumerate(model.features):
            x = block(x)
            if i in STAGE_IDX:
                chans.append(x.shape[1])
    if len(chans) != 4 or chans != sorted(chans) or len(set(chans)) != 4:
        raise SystemExit(f"unexpected ConvNeXt stage channels {chans}")
    return model, chans


@torch.inference_mode()
def extract(model, imgs, dev):
    """Return a list of four (n, C) arrays, one per stage, global-average-pooled."""
    out = [[] for _ in STAGE_IDX]
    mean, std = IMAGENET_MEAN.to(dev), IMAGENET_STD.to(dev)
    for s in range(0, len(imgs), BATCH):
        chunk = torch.from_numpy(imgs[s:s + BATCH]).to(dev).unsqueeze(1)   # (b,1,H,W)
        x = chunk.repeat(1, 3, 1, 1)
        x = F.interpolate(x, size=(RES, RES), mode="bilinear", align_corners=False)
        x = (x - mean) / std
        k = 0
        for i, block in enumerate(model.features):
            x = block(x)
            if i in STAGE_IDX:
                out[k].append(x.mean(dim=(2, 3)).float().cpu().numpy())
                k += 1
    return [np.concatenate(p, axis=0) for p in out]


def main():
    os.makedirs(FEAT, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}  device={dev}")
    if dev == "cuda":
        print(f"  {torch.cuda.get_device_name(0)}")
    else:
        print("  WARNING: running on CPU; this is ~50x slower.", file=sys.stderr)
    splits = read_manifest()
    print(f"splits: {[s[0] for s in splits]}\n")

    for mname in MODELS:
        model, chans = build(mname)
        nparam = sum(p.numel() for p in model.parameters())
        model = model.to(dev)
        print(f"convnext_{mname}: {nparam/1e6:.1f}M parameters, stage channels {chans}")
        with open(os.path.join(FEAT, f"{mname}_dims.txt"), "w") as fh:
            fh.write(" ".join(str(c) for c in chans) + "\n")
        for name, ntr, nte, n, _ in splits:
            for kind, count in (("train", ntr), ("test", nte)):
                imgs = load_images(name, kind, count, n)
                t0 = time.time()
                stages = extract(model, imgs, dev)
                dt = time.time() - t0
                for si, arr in enumerate(stages, start=1):
                    path = os.path.join(FEAT, f"{mname}_{name}_{kind}_s{si}.f32")
                    arr.astype(np.float32).tofile(path)
                print(f"  {name:<10} {kind:<5} {count:6d} imgs  {dt:6.1f} s "
                      f"({1000*dt/count:.1f} ms/img)")
                sys.stdout.flush()
        del model
        if dev == "cuda":
            torch.cuda.empty_cache()
    print(f"\nwrote {FEAT}")


if __name__ == "__main__":
    main()
