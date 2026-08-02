#!/usr/bin/env python3
"""
Frozen ConvNeXt features for all of EMNIST balanced.

    ../ConVNextTest/.venv/bin/python extract_convnext_emnist.py

Reads the IDX files directly rather than going through an intermediate dump, so there is one
fewer thing to get wrong -- but that puts the burden on getting the ORIENTATION right here.

EMNIST stores its images transposed relative to MNIST. `LoadEMNIST.read_emnist_images` corrects
for that implicitly: it reshapes the row-major IDX bytes as (cols, rows, n) in Julia's
column-major order, so its `out[a,b]` is the pixel at (row b, col a) -- the transpose of the
naive reading, which is the upright character. The numpy equivalent of that is an explicit
`.transpose(0, 2, 1)` after the naive reshape.

This matters because both arms must see the SAME pixels for the comparison to mean anything.
`--check` prints the statistics Julia prints, so the two can be compared directly rather than
assumed equal.
"""
import os, sys, time
import numpy as np, torch, torch.nn.functional as F, torchvision

SRC   = os.path.expanduser("~/Julia/DATABASES/EMNIST/emnist_source_files")
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
BATCH = int(os.environ.get("CX_BATCH", "256"))
RES   = int(os.environ.get("CX_RES", "224"))
MODEL = os.environ.get("CX_MODEL", "base")
STAGE_IDX = (1, 3, 5, 7)

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def read_images(path):
    with open(path, "rb") as fh:
        magic, n, rows, cols = np.frombuffer(fh.read(16), dtype=">u4")
        assert magic == 2051, f"bad magic {magic}"
        a = np.frombuffer(fh.read(), dtype=np.uint8).reshape(n, rows, cols)
    # the un-transpose; see the module docstring
    return np.ascontiguousarray(a.transpose(0, 2, 1)).astype(np.float32) / 255.0


def read_labels(path):
    with open(path, "rb") as fh:
        magic, n = np.frombuffer(fh.read(8), dtype=">u4")
        assert magic == 2049, f"bad magic {magic}"
        return np.frombuffer(fh.read(), dtype=np.uint8).astype(np.int64) + 1


def build():
    model = getattr(torchvision.models, f"convnext_{MODEL}")(weights="IMAGENET1K_V1").eval()
    chans = []
    with torch.inference_mode():
        x = torch.zeros(1, 3, RES, RES)
        for i, b in enumerate(model.features):
            x = b(x)
            if i in STAGE_IDX:
                chans.append(x.shape[1])
    if len(chans) != 4 or chans != sorted(chans) or len(set(chans)) != 4:
        raise SystemExit(f"unexpected ConvNeXt stage channels {chans}")
    return model, chans


@torch.inference_mode()
def extract(model, imgs, dev):
    out = [[] for _ in STAGE_IDX]
    mean, std = IMAGENET_MEAN.to(dev), IMAGENET_STD.to(dev)
    for s in range(0, len(imgs), BATCH):
        x = torch.from_numpy(imgs[s:s + BATCH]).to(dev).unsqueeze(1).repeat(1, 3, 1, 1)
        x = F.interpolate(x, size=(RES, RES), mode="bilinear", align_corners=False)
        x = (x - mean) / std
        k = 0
        for i, b in enumerate(model.features):
            x = b(x)
            if i in STAGE_IDX:
                out[k].append(x.mean(dim=(2, 3)).float().cpu().numpy()); k += 1
    return [np.concatenate(p, axis=0) for p in out]


def main():
    os.makedirs(CACHE, exist_ok=True)
    if "--check" in sys.argv:
        A = read_images(os.path.join(SRC, "emnist-balanced-test-images-idx3-ubyte"))
        y = read_labels(os.path.join(SRC, "emnist-balanced-test-labels-idx1-ubyte"))
        print(f"n={A.shape[0]} shape={A.shape[1:]}  label[1]={y[0]}")
        print(f"img1 mean={A[0].mean():.6f} sum={A[0].sum():.4f} "
              f"rowsums[1:5]={np.array2string(A[0].sum(1)[:5], precision=4)}")
        return
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"torch {torch.__version__}  device={dev}", flush=True)
    model, chans = build()
    print(f"convnext_{MODEL}: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params, "
          f"stages {chans}", flush=True)
    model = model.to(dev)
    with open(os.path.join(CACHE, "cx_dims.txt"), "w") as fh:
        fh.write(" ".join(map(str, chans)) + "\n")
    for kind in ("train", "test"):
        if os.path.exists(os.path.join(CACHE, f"cx_{kind}_s4.f32")):
            print(f"{kind} already cached"); continue
        A = read_images(os.path.join(SRC, f"emnist-balanced-{kind}-images-idx3-ubyte"))
        read_labels(os.path.join(SRC, f"emnist-balanced-{kind}-labels-idx1-ubyte")).astype(
            np.int64).tofile(os.path.join(CACHE, f"cx_{kind}_y.i64"))
        t0 = time.time()
        for si, arr in enumerate(extract(model, A, dev), start=1):
            arr.astype(np.float32).tofile(os.path.join(CACHE, f"cx_{kind}_s{si}.f32"))
        dt = time.time() - t0
        print(f"  {kind:<5} {len(A):6d} imgs  {dt:6.1f} s ({1000*dt/len(A):.2f} ms/img)",
              flush=True)
    print(f"\nwrote {CACHE}")


if __name__ == "__main__":
    main()
