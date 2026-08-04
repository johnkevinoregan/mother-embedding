#!/usr/bin/env python3
"""
feature_layer.py  -- mid-level feature-type layer on the Gabor oriented-energy field.

Turns the fixed-scale complex-Gabor stack (from gabor_orientation_demo.py) into
discrete, typed keypoints -- endpoint / corner / T-junction / X-crossing -- using
only operations that read off the oriented-energy field:

  * peak-count of the orientation profile   (1 orientation = simple edge, >=2 = corner/junction)
  * end-stopping along the winning orientation (fires where an oriented segment terminates)
  * a multi-radius ring "spoke count"        (# contour branches -> junction order)

These are the classic V1->V2 nonlinearities (end-stopping = hypercomplex cells;
spoke/junction counting = the corner/junction selectivity attributed to V2/V4).
The point of the demo: much of a descriptor list like
"pointy corner at top; two endpoints bottom-left/right" is already recoverable
from the Gabor field, before any learned patch dictionary.
"""
from __future__ import annotations
import numpy as np
from collections import Counter
from scipy.signal import fftconvolve
from scipy.ndimage import (zoom, shift as ndshift, map_coordinates,
                           maximum_filter, gaussian_filter1d)
from gabor_orientation_demo import Config, gabor, render_glyphs, load_emnist_letters


# --------------------------------------------------------------------------- #
def prep(a, cfg):
    b = np.clip(zoom(a, cfg.upsample, order=3), 0, 1)
    if cfg.noise_sigma > 0:
        b = np.clip(b + np.random.default_rng(cfg.seed).normal(0, cfg.noise_sigma, b.shape), 0, 1)
    return b

def energy_stack(img, thetas, cfg):
    return np.abs(np.stack([fftconvolve(img, gabor(t, cfg), mode="same") for t in thetas]))


# ---- operation 1: end-stopping along winning orientation ------------------- #
def end_stopping(S, thetas, d=15.0):
    es = np.zeros_like(S)
    for t, th in enumerate(thetas):
        sx, sy = np.cos(th) * d, np.sin(th) * d
        plus  = ndshift(S[t], (-sy, -sx), order=1)     # sample p + d*tangent
        minus = ndshift(S[t], ( sy,  sx), order=1)     # sample p - d*tangent
        es[t] = np.maximum(S[t] - 0.5 * (plus + minus), 0)
    return es

# ---- operation 2: orientation-profile peak structure ----------------------- #
def peak_maps(S, relthr=0.40, sep_bins=10):
    """Return n_peaks, first-peak height, and a *bimodality-gated* second-peak
    height.  The second peak counts only if it is well separated in orientation
    (>= sep_bins) AND there is a real valley between the two peaks -- so a smoothly
    curving contour (one broad, rotating peak) does not masquerade as a corner."""
    N = S.shape[0]; M = S.max(0)
    Ss = gaussian_filter1d(S, 1.0, axis=0, mode="wrap")           # smooth along theta
    ispk = (Ss >= np.roll(Ss, -1, 0)) & (Ss > np.roll(Ss, 1, 0)) & (Ss > relthr * np.maximum(M, 1e-9))
    npk = ispk.sum(0)
    h = np.where(ispk, Ss, 0.0); i1 = h.argmax(0)
    ii, jj = np.indices(M.shape); h2 = h.copy()
    for s in range(-sep_bins, sep_bins + 1):                      # blank window around 1st peak
        h2[(i1 + s) % N, ii, jj] = 0.0
    second = h2.max(0); i2 = h2.argmax(0)

    # valley test: the minimum of the profile on the arc between the two peaks
    # must sit well below the smaller peak (a genuine bimodal dip, not a shoulder).
    valley = np.full(M.shape, np.inf)
    for s in range(N):                                            # scan all offsets once
        idx = (i1 + s) % N
        between = (s <= (i2 - i1) % N)                            # on the short arc i1->i2
        v = Ss[idx, ii, jj]
        valley = np.where(between, np.minimum(valley, v), valley)
    real_dip = valley < 0.62 * np.minimum(np.maximum(second, 1e-9), h.max(0))
    second = second * real_dip
    return npk, h.max(0), second                                 # n_peaks, first, second

# ---- operation 3: ring spoke count ----------------------------------------- #
def arc_count(M, y, x, r, K=72, relthr=0.4, min_run=3):
    ph = np.linspace(0, 2 * np.pi, K, endpoint=False)
    g = map_coordinates(M, [y + r * np.sin(ph), x + r * np.cos(ph)], order=1, mode="constant")
    if g.max() <= 0: return 0, []
    on = g > relthr * g.max()
    if on.all():  return 99, []
    n = 0; centers = []
    for st in np.where(on & ~np.roll(on, 1))[0]:
        run = []; j = st
        while on[j % K] and (j - st) <= K:
            run.append(j % K); j += 1
        if len(run) >= min_run:
            n += 1; centers.append(np.angle(np.exp(1j * ph[run]).mean()))
    return n, centers

def classify_point(M, y, x, radii=(15, 18, 21, 24)):
    res = [arc_count(M, y, x, r) for r in radii]
    valid = [n for n, _ in res if 0 < n < 99]
    if not valid: return None
    cnt = Counter(valid)
    order = max([k for k in (4, 3) if cnt[k] >= 2] or [0])       # junction order confirmed at >=2 radii
    if order >= 3: return {3: "T", 4: "X"}[order]
    collinear = any(n == 2 and len(c) == 2 and
                    abs(abs(np.angle(np.exp(1j * (c[0] - c[1])))) - np.pi) < np.deg2rad(30)
                    for n, c in res)
    ones = sum(1 for n in valid if n == 1)
    if ones >= len(valid) / 2.0: return "endpoint"
    if 2 in valid: return "straight" if collinear else "corner"
    return "endpoint"


# --------------------------------------------------------------------------- #
def detect(img, cfg, end_stop_d=12.0):
    thetas = np.linspace(0, np.pi, cfg.n_orient, endpoint=False)
    S = energy_stack(img, thetas, cfg); M = S.max(0); Mmax = M.max()
    ES = end_stopping(S, thetas, d=end_stop_d)
    ESw = np.take_along_axis(ES, S.argmax(0)[None], 0)[0]
    npk, first, second = peak_maps(S)
    on = M > 0.20 * Mmax

    # coarse class comes from WHICH operation fires:
    #   orientation change (two genuine orientation peaks) -> corner-family
    #   segment termination (end-stopping, single orientation) -> endpoint
    cornerness   = second * (npk >= 2) * (second > 0.5 * first) * on
    endpointness = ESw * (npk <= 1) * (M > 0.35 * Mmax)          # must sit on a real ridge

    def nms(score, md, frac):
        return np.argwhere((score == maximum_filter(score, md)) & (score > frac * score.max()) & (score > 0))

    proposals = ([("corner", y, x) for y, x in nms(cornerness, 11, 0.35)] +
                 [("endpoint", y, x) for y, x in nms(endpointness, 11, 0.38)])
    kept, taken = [], []
    for src, y, x in proposals:
        if any((y - yy) ** 2 + (x - xx) ** 2 < 11 ** 2 for yy, xx in taken):
            continue
        ring = classify_point(M, y, x)                           # 'corner'/'T'/'X'/'endpoint'/'straight'/None
        if src == "corner":
            t = ring if ring in ("T", "X") else "corner"         # cornerness already asserts >=2 orientations
        else:  # endpoint proposal
            if ring in ("T", "X"):
                t = ring                                         # ring overrides if clearly a junction
            elif ring == "straight":
                continue                                         # mid-stroke false alarm
            else:
                t = "endpoint"
        kept.append((t, int(y), int(x))); taken.append((y, x))
    return dict(img=img, M=M, ori=thetas[S.argmax(0)], ESw=ESw,
                cornerness=cornerness, keypoints=kept)


STYLE = {"corner":   ("s", "#ff3b30"), "endpoint": ("o", "#00c2d1"),
         "T":        ("^", "#ffcc00"), "X":        ("*", "#ff2fd0")}

if __name__ == "__main__":
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt, matplotlib.lines as ml
    cfg = Config(n_orient=60)
    chars = ["A", "K", "X", "T"]
    src = load_emnist_letters(chars, cfg.seed) or render_glyphs(chars)
    res = {c: detect(prep(src[c], cfg), cfg) for c in chars}

    # --- fig5: labelled keypoints ---
    fig, ax = plt.subplots(1, len(chars), figsize=(2.6 * len(chars), 3.3))
    for a, c in zip(ax, chars):
        r = res[c]; a.imshow(r["img"], cmap="gray"); a.set_title(c); a.axis("off")
        for t, y, x in r["keypoints"]:
            mk, col = STYLE[t]
            a.scatter([x], [y], marker=mk, s=170, facecolors="none", edgecolors=col, linewidths=2.4)
        print(c, r["keypoints"])
    h = [ml.Line2D([], [], marker=m, ls="", mfc="none", mec=cl, mew=2, ms=11, label=l)
         for l, (m, cl) in STYLE.items()]
    fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(); plt.savefig("fig5_features.png", dpi=120, facecolor="white", bbox_inches="tight"); plt.close()

    # --- fig6: the operations feeding the classifier, on A ---
    r = res["A"]
    fig, ax = plt.subplots(1, 3, figsize=(10, 3.4))
    ax[0].imshow(r["M"], cmap="magma"); ax[0].set_title("oriented energy  (modulus)", fontsize=10)
    ax[1].imshow(r["ESw"], cmap="viridis"); ax[1].set_title("end-stopping  → endpoints", fontsize=10)
    ax[2].imshow(r["cornerness"], cmap="viridis"); ax[2].set_title("bimodality  → corners/junctions", fontsize=10)
    for a in ax: a.axis("off")
    plt.tight_layout(); plt.savefig("fig6_operations.png", dpi=120, facecolor="white", bbox_inches="tight"); plt.close()
    print("wrote fig5_features.png fig6_operations.png")
