#!/usr/bin/env python3
"""
gabor_orientation_demo.py
=========================
Fixed-scale complex-Gabor "argmax orientation" analysis on EMNIST-style characters.

At one chosen scale we convolve the image with a bank of complex (quadrature) Gabor
filters spanning orientation, take the MODULUS (oriented energy = sqrt(even^2+odd^2))
and PHASE at every pixel, and visualise:

  fig1  the argmax-orientation flow field (contour tangents; junction flips)
  fig2  raw vs modulus-masked argmax  (why the flat-region speckle must be masked)
  fig3  the full modulus(theta) profile at a stroke / a junction / the background
        (unimodal vs bimodal vs noise-floor -- the info argmax throws away)
  fig4  phase at the winning orientation -> even(line)/odd(edge) classification

ORIENTATION CONVENTION
----------------------
`theta` is the CONTOUR-TANGENT orientation. The carrier modulates along the
normal, so the stripes run along the tangent and the filter matches a line/edge
oriented at `theta`. Sanity: a vertical line -> argmax theta = 90 deg; a
horizontal line -> 0 deg. (Verified in `_selftest`.)

DATA
----
Tries to load real EMNIST letters (handles the standard transpose quirk). If the
dataset isn't installed / downloadable, falls back to rendering glyphs from a
system font with optional elastic warp, so the script is self-contained.

Author: generated for J.K. O'Regan.  Public-domain / do as you like.
----
COMMENTS from Claude

A few notes to orient you before you start tweaking:
Everything you'll want to change lives in the Config dataclass at the top. The scale is lam (carrier 
wavelength) with sigma_n/sigma_t the across- and along-contour envelope widths — set sigma_t > sigma_n for 
the elongated, simple-cell-like aspect, or make them equal for an isotropic filter. n_orient controls the 
field resolution, n_orient_prof the fineness of the modulus(θ) curves in fig3. upsample and noise_sigma 
handle the 28→112 cubic lift and the noise floor.
The orientation convention is documented in gabor() and enforced by _selftest(): theta is the contour 
tangent, so argmax gives you the local stroke/edge tangent directly (vertical line → 90°). If you'd 
rather have theta mean the gradient/normal direction, swap xt↔xn in the carrier and envelope, or just add 
90°.
The EMNIST loader tries the emnist package first and undoes the standard transpose; on your lab server 
where the dataset is reachable it'll use real samples automatically, and pick_points() will still find a 
stroke/junction/background triplet on whatever glyph you pass (the "junction" point is just the centroid, 
which lands on the crossing for X/T/A/K but not for junction-free letters — worth adjusting if you feed it 
an S or O).
Two hooks for the extensions I mentioned: analyze() currently returns only the argmax, but the full stack 
S is right there — to get the runner-up peak at junctions, keep S and take the second local maximum along 
axis 0 instead of collapsing with argmax. And for your polar convention, the cleanest spot is to measure 
orientation relative to the radial direction from center_of_mass — subtract the local polar angle from 
orientation before the HSV/needle mapping rather than touching the filter bank.


"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from scipy.signal import fftconvolve
from scipy.ndimage import zoom, center_of_mass, gaussian_filter, map_coordinates
import matplotlib
matplotlib.use("Agg")                       # drop this line for interactive use
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import hsv_to_rgb


# ----------------------------------------------------------------------------- #
#  CONFIG  -- everything you'll want to tweak lives here                         #
# ----------------------------------------------------------------------------- #
@dataclass
class Config:
    # --- Gabor scale (in pixels of the analysed, upsampled image) ---
    lam:      float = 12.0      # carrier wavelength  (THE "scale")
    sigma_n:  float = 6.0       # Gaussian sd across the contour (normal)
    sigma_t:  float = 11.0      # Gaussian sd along  the contour (tangent) -> elongation
    ksize:    int   = 41        # kernel support (odd)

    # --- orientation sampling ---
    n_orient:      int = 36     # orientations for the argmax FIELD (fig1/2/4)
    n_orient_prof: int = 180    # orientations for the point PROFILES (fig3)

    # --- image handling ---
    upsample:    int   = 4      # 28 -> 28*upsample, cubic (continuous analysis)
    noise_sigma: float = 0.03   # additive Gaussian noise floor (0 to disable)
    seed:        int   = 0

    # --- display ---
    mask_thresh: float = 0.14   # needles hidden below this fraction of max modulus
    gamma:       float = 0.60   # brightness gamma for the modulus channel
    needle_step: int   = 4      # needle grid stride (px)
    needle_len:  float = 3.2    # needle half-length (px)

    # --- which characters ---
    chars: tuple = ("X", "T", "A", "K", "S")


# ----------------------------------------------------------------------------- #
#  GABOR BANK                                                                    #
# ----------------------------------------------------------------------------- #
def gabor(theta: float, cfg: Config) -> np.ndarray:
    """Complex Gabor whose preferred (stripe) orientation is `theta`."""
    h = cfg.ksize // 2
    y, x = np.mgrid[-h:h + 1, -h:h + 1].astype(float)
    xt =  x * np.cos(theta) + y * np.sin(theta)      # along tangent
    xn = -x * np.sin(theta) + y * np.cos(theta)      # along normal
    env = np.exp(-(xt**2 / (2 * cfg.sigma_t**2) + xn**2 / (2 * cfg.sigma_n**2)))
    car = np.exp(1j * 2 * np.pi * xn / cfg.lam)      # modulate across the contour
    g = env * car
    return g - g.mean()                              # zero DC -> uniform regions give ~0


def analyze(img: np.ndarray, cfg: Config):
    """Return (orientation, modulus, winning_phase) fields via argmax over theta."""
    thetas = np.linspace(0, np.pi, cfg.n_orient, endpoint=False)
    R = np.stack([fftconvolve(img, gabor(t, cfg), mode="same") for t in thetas])
    S = np.abs(R)
    k = np.argmax(S, axis=0)
    ii, jj = np.indices(k.shape)
    orientation = thetas[k]
    modulus     = S.max(axis=0)
    win_phase   = np.angle(R)[k, ii, jj]
    return orientation, modulus, win_phase


def modulus_profile(img: np.ndarray, y: int, x: int, cfg: Config):
    """Full modulus(theta) at a single pixel (fine orientation sampling)."""
    thetas = np.linspace(0, np.pi, cfg.n_orient_prof, endpoint=False)
    vals = np.array([np.abs(fftconvolve(img, gabor(t, cfg), mode="same")[y, x])
                     for t in thetas])
    return thetas, vals


# ----------------------------------------------------------------------------- #
#  CHARACTER SOURCES                                                             #
# ----------------------------------------------------------------------------- #
def load_emnist_letters(chars, seed=0):
    """One real EMNIST sample per requested letter, or None if unavailable.

    EMNIST 'letters' split: labels 1..26 == a..z (case-merged). Images are stored
    transposed, so we undo that. Returns dict{char: 28x28 float in [0,1]}.
    """
    try:
        from emnist import extract_test_samples
        X, y = extract_test_samples("letters")
    except Exception as e:
        print(f"[emnist] unavailable ({type(e).__name__}); using rendered glyphs.")
        return None
    rng = np.random.default_rng(seed)
    out = {}
    for c in chars:
        if not c.isalpha():
            return None
        label = ord(c.lower()) - ord("a") + 1
        idx = np.where(y == label)[0]
        if len(idx) == 0:
            return None
        img = X[rng.choice(idx)].astype(float) / 255.0
        out[c] = img.T                        # undo EMNIST transpose
    return out


def _elastic(img, alpha, sigma, rng):
    h, w = img.shape
    dx = gaussian_filter(rng.random((h, w)) * 2 - 1, sigma) * alpha
    dy = gaussian_filter(rng.random((h, w)) * 2 - 1, sigma) * alpha
    y, x = np.mgrid[0:h, 0:w]
    return map_coordinates(img, [(y + dy).ravel(), (x + dx).ravel()],
                           order=1, mode="constant").reshape(h, w)


def render_glyphs(chars, warp=True, seed=7):
    """Fallback: render EMNIST-style 28x28 glyphs from a system font."""
    from PIL import Image, ImageDraw, ImageFont
    font_path = matplotlib.get_data_path() + "/fonts/ttf/DejaVuSans.ttf"
    rng = np.random.default_rng(seed)
    out = {}
    for c in chars:
        size = 200
        im = Image.new("L", (size, size), 0)
        d = ImageDraw.Draw(im)
        f = ImageFont.truetype(font_path, int(size * 0.72))
        bb = d.textbbox((0, 0), c, font=f)
        w, h = bb[2] - bb[0], bb[3] - bb[1]
        d.text(((size - w) / 2 - bb[0], (size - h) / 2 - bb[1]), c, fill=255, font=f)
        a = np.asarray(im, float) / 255.0
        if warp:
            a = _elastic(a, alpha=size * 0.10, sigma=size * 0.10, rng=rng)
        a28 = np.asarray(Image.fromarray((a * 255).astype("uint8"))
                         .resize((28, 28), Image.LANCZOS), float) / 255.0
        out[c] = a28
    return out


def get_characters(cfg: Config):
    imgs = load_emnist_letters(cfg.chars, seed=cfg.seed) or render_glyphs(cfg.chars)
    # upsample + noise floor
    prepped = {}
    for i, c in enumerate(cfg.chars):
        b = np.clip(zoom(imgs[c], cfg.upsample, order=3), 0, 1)
        if cfg.noise_sigma > 0:
            b = np.clip(b + np.random.default_rng(cfg.seed + i)
                        .normal(0, cfg.noise_sigma, b.shape), 0, 1)
        prepped[c] = b
    return prepped


# ----------------------------------------------------------------------------- #
#  VISUALISATION HELPERS                                                         #
# ----------------------------------------------------------------------------- #
def hsv_map(orientation, modulus, gamma):
    """Hue = orientation (mod pi, cyclic), Value = normalised modulus."""
    v = (modulus / modulus.max()) ** gamma
    h = orientation / np.pi
    return hsv_to_rgb(np.stack([h, np.ones_like(h), v], axis=-1))


def needle_plot(ax, orientation, modulus, cfg: Config):
    v = modulus / modulus.max()
    ax.set_facecolor("black")
    ys, xs = np.mgrid[0:orientation.shape[0]:cfg.needle_step,
                      0:orientation.shape[1]:cfg.needle_step]
    for y, x in zip(ys.ravel(), xs.ravel()):
        m = v[y, x]
        if m < cfg.mask_thresh:
            continue
        t = orientation[y, x]
        dx, dy = np.cos(t) * cfg.needle_len, np.sin(t) * cfg.needle_len
        ax.plot([x - dx, x + dx], [y - dy, y + dy],
                color=cm.hsv(t / np.pi), lw=1.1,
                alpha=min(1.0, m * 1.3), solid_capstyle="round")
    ax.set_xlim(0, orientation.shape[1]); ax.set_ylim(orientation.shape[0], 0)
    ax.set_aspect("equal"); ax.axis("off")


# ----------------------------------------------------------------------------- #
#  FIGURES                                                                       #
# ----------------------------------------------------------------------------- #
def fig_fields(imgs, res, cfg, path="fig1_fields.png"):
    n = len(cfg.chars)
    fig, ax = plt.subplots(n, 3, figsize=(9, 3 * n))
    for r, c in enumerate(cfg.chars):
        o, m, _ = res[c]
        ax[r, 0].imshow(imgs[c], cmap="gray")
        ax[r, 0].set_ylabel(c, fontsize=16, rotation=0, labelpad=18, va="center")
        ax[r, 0].set_xticks([]); ax[r, 0].set_yticks([])
        ax[r, 1].imshow(hsv_map(o, m, cfg.gamma)); ax[r, 1].axis("off")
        needle_plot(ax[r, 2], o, m, cfg)
        if r == 0:
            ax[r, 0].set_title("character", fontsize=12)
            ax[r, 1].set_title("orientation (hue=θ, bright=modulus)", fontsize=10.5)
            ax[r, 2].set_title("argmax-orientation flow (masked)", fontsize=10.5)
    plt.tight_layout(); plt.savefig(path, dpi=115, facecolor="white"); plt.close()


def fig_mask(res, cfg, char="X", path="fig2_mask.png"):
    o, m, _ = res[char]
    fig, a = plt.subplots(1, 2, figsize=(7, 3.9))
    a[0].imshow(cm.hsv(o / np.pi))
    a[0].set_title("raw argmax, UNMASKED\nflat regions → random speckle", fontsize=10)
    a[0].axis("off")
    a[1].imshow(hsv_map(o, m, cfg.gamma))
    a[1].set_title("masked by modulus\nonly contours survive", fontsize=10)
    a[1].axis("off")
    plt.tight_layout(); plt.savefig(path, dpi=120, facecolor="white"); plt.close()


def pick_points(img):
    """Return {label:(y,x)} for a clean stroke, the central junction, and background."""
    cy, cx = (int(round(v)) for v in center_of_mass(img))
    bright = np.argwhere(img > 0.5)
    rad = np.hypot(bright[:, 0] - cy, bright[:, 1] - cx)
    stroke = bright[np.argmin(np.abs(rad - 0.55 * rad.max()))]
    return {"stroke (clean edge)": (int(stroke[0]), int(stroke[1])),
            "crossing (junction)": (cy, cx),
            "background (flat)":    (10, 10)}


def fig_profiles(imgs, cfg, char="X", path="fig3_profiles.png"):
    img = imgs[char]
    pts = pick_points(img)
    fig, a = plt.subplots(1, 3, figsize=(11, 3.1))
    for ax_, (label, (y, x)) in zip(a, pts.items()):
        th, pr = modulus_profile(img, y, x, cfg)
        ax_.plot(np.degrees(th), pr, lw=2)
        ax_.axvline(np.degrees(th[np.argmax(pr)]), color="r", ls="--", lw=1)
        ax_.set_title(label, fontsize=11); ax_.set_xlabel("orientation θ (deg)")
        ax_.set_xlim(0, 180)
    a[0].set_ylabel("modulus |Gabor(θ)|")
    plt.tight_layout(); plt.savefig(path, dpi=120, facecolor="white"); plt.close()
    return pts


def fig_phase(res, cfg, chars=("X", "T"), path="fig4_phase.png"):
    fig, a = plt.subplots(1, len(chars), figsize=(3.5 * len(chars), 3.9))
    a = np.atleast_1d(a)
    yellow, teal = np.array([1, .8, .1]), np.array([.1, .7, .9])
    for ax_, c in zip(a, chars):
        o, m, ph = res[c]
        edge = np.abs(np.sin(ph))                 # 0 = even/line, 1 = odd/edge
        v = (m / m.max()) ** cfg.gamma
        rgb = ((1 - edge)[..., None] * yellow + edge[..., None] * teal) * v[..., None]
        ax_.imshow(np.clip(rgb, 0, 1))
        ax_.set_title(f"{c}: phase at winning θ", fontsize=10); ax_.axis("off")
    a[0].text(0.5, -0.08, "yellow = even/line-like    teal = odd/edge-like",
              transform=a[0].transAxes, ha="center", fontsize=9)
    plt.tight_layout(); plt.savefig(path, dpi=120, facecolor="white"); plt.close()


# ----------------------------------------------------------------------------- #
#  SELF-TEST + MAIN                                                              #
# ----------------------------------------------------------------------------- #
def _selftest(cfg: Config):
    """Confirm argmax orientation == contour tangent."""
    v = np.zeros((112, 112)); v[:, 54:58] = 1.0        # vertical line
    o, _, _ = analyze(v, cfg)
    assert abs(np.degrees(o[56, 56]) - 90) < 6, "vertical-line convention broken"
    h = np.zeros((112, 112)); h[54:58, :] = 1.0        # horizontal line
    o, _, _ = analyze(h, cfg)
    assert np.degrees(o[56, 56]) < 6 or np.degrees(o[56, 56]) > 174
    print("[selftest] orientation convention OK")


def main(cfg: Config = None):
    cfg = cfg or Config()
    _selftest(cfg)
    imgs = get_characters(cfg)
    res = {c: analyze(imgs[c], cfg) for c in cfg.chars}
    fig_fields(imgs, res, cfg)
    fig_mask(res, cfg, char=cfg.chars[0])
    fig_profiles(imgs, cfg, char=cfg.chars[0])
    fig_phase(res, cfg, chars=cfg.chars[:2])
    print("[done] wrote fig1_fields.png fig2_mask.png fig3_profiles.png fig4_phase.png")


if __name__ == "__main__":
    main()
