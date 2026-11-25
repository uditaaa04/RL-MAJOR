"""
SynthBox - synthetic photos of cardboard boxes on a conveyor belt.

Why this exists
---------------
You have no photos of your own boxes yet. To get working, trained models on
day one, we render thousands of fake top-down photos of boxes:
  * GOOD boxes    : clean cardboard, tape, label, harmless printed marks
  * DAMAGED boxes : dents, tears/holes, crushed corners, water stains,
                    torn labels / lifted tape, scratches

Each damage has a "severity" from subtle (0.25) to obvious (1.0). Subtle
damage is hard to see in a small, low-resolution image - that is exactly why
the small gateway model is less accurate than the big cloud model.

IMPORTANT: models trained only on these images are a starting point. Once
your conveyor and camera exist, take ~200 real photos and re-train with
`python -m vision.train --data data/real ...` (see docs/BUILD_GUIDE.md).

Usage
-----
    python -m vision.synth_boxes --out data/synth --train 6000 --val 1000 --test 1000
"""
from __future__ import annotations

import argparse
import csv
import io
import math
import os
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

S = 256  # output size (pixels). The real pipeline also crops/resizes to 256x256.
SS = 2   # supersampling factor for smooth edges

DEFECT_TYPES = ["dent", "tear", "crushed_corner", "stain", "torn_label", "scratch"]


@dataclass
class Meta:
    label: str
    defects: list = field(default_factory=list)  # list of (type, severity)

    @property
    def min_severity(self) -> float:
        return min((s for _, s in self.defects), default=0.0)


# ----------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------
def _smooth_noise(rng, h, w, scale, octaves=3):
    """Cheap fractal noise in [-1, 1] (upsampled random grids)."""
    out = np.zeros((h, w), np.float32)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        cells = max(2, int(scale * (2 ** o)))
        grid = rng.standard_normal((cells, cells)).astype(np.float32)
        img = Image.fromarray(grid).resize((w, h), Image.BICUBIC)
        out += amp * np.asarray(img, np.float32)
        total += amp
        amp *= 0.5
    out /= total
    m = np.abs(out).max() + 1e-6
    return out / m


def _hsv_to_rgb(h, s, v):
    i = int(h * 6) % 6
    f = h * 6 - int(h * 6)
    p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    return [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i]


def _blob_polygon(rng, cx, cy, r, n=18, jag=0.35):
    pts = []
    for k in range(n):
        a = 2 * math.pi * k / n + rng.uniform(-0.1, 0.1)
        rr = r * (1 + rng.uniform(-jag, jag))
        pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
    return pts


def _mask_from(draw_fn, size):
    m = Image.new("L", size, 0)
    d = ImageDraw.Draw(m)
    draw_fn(d)
    return m


# ----------------------------------------------------------------------------
# scene parts
# ----------------------------------------------------------------------------
def _belt(rng, W):
    """Dark rubber/PVC belt with weave texture and optional side rails."""
    base = rng.uniform(25, 75)
    tint = np.array([1.0, 1.0, 1.0], np.float32)
    r = rng.random()
    if r < 0.15:
        tint = np.array([0.80, 1.10, 0.85], np.float32)  # green PVC belt
    elif r < 0.25:
        tint = np.array([0.85, 0.92, 1.12], np.float32)  # blue belt
    tex = _smooth_noise(rng, W, W, 6, 3) * rng.uniform(4, 10)
    fine = rng.standard_normal((W, W)).astype(np.float32) * rng.uniform(2, 6)
    rows = np.arange(W, dtype=np.float32)[:, None]
    weave = np.sin(rows * rng.uniform(0.6, 1.4)) * rng.uniform(0, 3)
    g = base + tex + fine + weave
    img = np.clip(g[..., None] * tint[None, None, :], 0, 255)
    # side rails (aluminium) top/bottom
    if rng.random() < 0.7:
        rh = int(W * rng.uniform(0.03, 0.09))
        rail = rng.uniform(110, 170)
        rail_tex = rail + _smooth_noise(rng, rh, W, 4, 2) * 8
        img[:rh] = rail_tex[..., None]
        img[W - rh:] = rail_tex[::-1][..., None]
    return img.astype(np.float32)


def _box_color(rng):
    if rng.random() < 0.15:  # white mailer box
        s, v = rng.uniform(0.0, 0.08), rng.uniform(0.78, 0.95)
        h = rng.uniform(0.05, 0.12)
    else:  # brown kraft cardboard
        h = rng.uniform(22, 38) / 360.0
        s = rng.uniform(0.35, 0.65)
        v = rng.uniform(0.50, 0.85)
    return np.array(_hsv_to_rgb(h, s, v), np.float32) * 255.0


def _draw_label(rng, W, H):
    """Returns (RGBA label image, size)."""
    lw = int(W * rng.uniform(0.25, 0.42))
    lh = int(H * rng.uniform(0.20, 0.32))
    lab = Image.new("RGBA", (lw, lh), (int(rng.uniform(225, 250)),) * 3 + (255,))
    d = ImageDraw.Draw(lab)
    y = int(lh * 0.12)
    for _ in range(rng.integers(2, 5)):  # "text" lines
        L = int(lw * rng.uniform(0.3, 0.85))
        th = max(2, int(lh * 0.06))
        d.rectangle([int(lw * 0.08), y, int(lw * 0.08) + L, y + th], fill=(40, 40, 40, 255))
        y += int(th * 2.2)
    # barcode
    x = int(lw * 0.08)
    by0, by1 = int(lh * 0.62), int(lh * 0.9)
    while x < lw * 0.9:
        bw = int(rng.integers(1, 4))
        if rng.random() < 0.6:
            d.rectangle([x, by0, x + bw, by1], fill=(10, 10, 10, 255))
        x += bw + int(rng.integers(1, 3))
    return lab


def _harmless_marks(rng, d, x0, y0, x1, y1, color):
    """Printed icons / marker lines that appear on GOOD and DAMAGED boxes."""
    c = tuple(int(v) for v in (color * rng.uniform(0.35, 0.6)))
    for _ in range(rng.integers(1, 4)):
        kind = rng.integers(0, 3)
        cx, cy = rng.uniform(x0 + 20, x1 - 20), rng.uniform(y0 + 20, y1 - 20)
        r = rng.uniform(6, 16) * SS
        if kind == 0:  # "this side up" arrows
            for k in (-1, 1):
                ax = cx + k * r * 0.7
                d.polygon([(ax, cy - r), (ax - r * 0.4, cy), (ax + r * 0.4, cy)], fill=c)
                d.rectangle([ax - r * 0.12, cy, ax + r * 0.12, cy + r], fill=c)
        elif kind == 1:  # recycle-ish ring
            d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=c, width=max(2, int(r * 0.2)))
        else:  # handwritten marker note (blue/black ink, thin)
            ink = (int(rng.uniform(10, 40)), int(rng.uniform(10, 40)), int(rng.uniform(40, 120)))
            x = cx - r * 1.5
            for _ in range(int(rng.integers(3, 7))):  # a few "letters"
                pts = [(x + rng.uniform(0, r * 0.5), cy + rng.uniform(-r * 0.4, r * 0.4)) for _ in range(3)]
                d.line(pts, fill=ink, width=max(1, int(SS)))
                x += r * 0.55


# ----------------------------------------------------------------------------
# defects (drawn on the box layer, in supersampled box coordinates)
# ----------------------------------------------------------------------------
def _apply_dent(rng, arr, bx, by, bw, bh, sev, md):
    H, W = arr.shape[:2]
    r = min(bw, bh) * rng.uniform(0.10, 0.30) * (0.5 + 0.5 * sev)
    cx, cy = rng.uniform(bx + r, bx + bw - r), rng.uniform(by + r, by + bh - r)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    ang = rng.uniform(0, 2 * math.pi)
    ex, ey = r, r * rng.uniform(0.5, 1.0)
    dx, dy = xx - cx, yy - cy
    u = (dx * math.cos(ang) + dy * math.sin(ang)) / ex
    v = (-dx * math.sin(ang) + dy * math.cos(ang)) / ey
    d2 = u * u + v * v
    inside = np.exp(-d2 * 1.6)
    shade = 1 - (0.18 + 0.32 * sev) * inside * (0.6 + 0.4 * np.tanh(u * 2))  # dark side
    shade += (0.05 + 0.10 * sev) * inside * np.clip(-u, 0, 1)                 # highlight side
    arr *= shade[..., None]
    md.ellipse([cx - r * 0.9, cy - r * 0.9, cx + r * 0.9, cy + r * 0.9], fill=255)
    # crease lines
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    col = tuple(int(c) for c in (arr[int(np.clip(cy, 0, H - 1)), int(np.clip(cx, 0, W - 1))] * (0.55 - 0.15 * sev)))
    for _ in range(rng.integers(2, 5)):
        a = rng.uniform(0, 2 * math.pi)
        L = r * rng.uniform(0.6, 1.3)
        pts = [(cx, cy)]
        for k in range(1, 4):
            pts.append((cx + L * k / 3 * math.cos(a + rng.uniform(-0.3, 0.3)),
                        cy + L * k / 3 * math.sin(a + rng.uniform(-0.3, 0.3))))
        d.line(pts, fill=col, width=max(1, int(SS * (1 + 1.5 * sev))))
    arr[:] = np.asarray(img, np.float32)


def _apply_tear(rng, arr, bx, by, bw, bh, sev, md):
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    r = min(bw, bh) * rng.uniform(0.05, 0.18) * (0.45 + 0.55 * sev)
    cx, cy = rng.uniform(bx + r * 1.5, bx + bw - r * 1.5), rng.uniform(by + r * 1.5, by + bh - r * 1.5)
    outer = _blob_polygon(rng, cx, cy, r * 1.25, n=22, jag=0.45)
    inner = _blob_polygon(rng, cx, cy, r, n=22, jag=0.40)
    paper = tuple(int(v) for v in np.clip(arr[int(cy), int(cx)] * 1.25 + 25, 0, 255))
    d.polygon(outer, fill=paper)                                     # torn light fibres
    dark = tuple(int(v) for v in rng.uniform(15, 55, 3))
    d.polygon(inner, fill=dark)                                      # dark interior
    md.polygon(outer, fill=255)
    arr[:] = np.asarray(img, np.float32)


def _apply_crushed_corner(rng, arr, corners, sev, md):
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    k = int(rng.integers(0, 4))
    p = np.array(corners[k], np.float32)
    a = np.array(corners[(k + 1) % 4], np.float32)
    b = np.array(corners[(k + 3) % 4], np.float32)
    la = rng.uniform(0.10, 0.30) * (0.5 + 0.5 * sev)
    lb = rng.uniform(0.10, 0.30) * (0.5 + 0.5 * sev)
    pa = p + (a - p) * la
    pb = p + (b - p) * lb
    col = arr[int(np.clip((pa[1] + pb[1]) / 2, 0, arr.shape[0] - 1)),
              int(np.clip((pa[0] + pb[0]) / 2, 0, arr.shape[1] - 1))]
    fold = tuple(int(v) for v in col * (0.55 + 0.15 * (1 - sev)))
    inner_pt = p + ((pa + pb) / 2 - p) * 1.6
    d.polygon([tuple(pa), tuple(inner_pt), tuple(pb)], fill=fold)    # folded flap
    d.line([tuple(pa), tuple(pb)], fill=tuple(int(v) for v in col * 0.35), width=max(2, int(SS * 2)))
    md.polygon([tuple(p), tuple(pa), tuple(inner_pt), tuple(pb)], fill=255)
    arr[:] = np.asarray(img, np.float32)
    # the very tip of the corner is pushed in -> belt shows through
    return [tuple(p), tuple(pa), tuple(pb)]


def _apply_stain(rng, arr, bx, by, bw, bh, sev, md):
    H, W = arr.shape[:2]
    r = min(bw, bh) * rng.uniform(0.12, 0.32) * (0.5 + 0.5 * sev)
    cx, cy = rng.uniform(bx + r, bx + bw - r), rng.uniform(by + r, by + bh - r)
    m = _mask_from(lambda d: [d.ellipse([cx + ox - rr, cy + oy - rr, cx + ox + rr, cy + oy + rr], fill=255)
                              for ox, oy, rr in [(rng.uniform(-r * .5, r * .5), rng.uniform(-r * .5, r * .5),
                                                  r * rng.uniform(0.4, 0.8)) for _ in range(5)]], (W, H))
    m = m.filter(ImageFilter.GaussianBlur(radius=SS * 2))
    mm = np.asarray(m, np.float32) / 255.0
    edge = np.asarray(m.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.GaussianBlur(SS)), np.float32) / 255.0
    darken = (0.08 + 0.22 * sev) * mm + (0.10 + 0.25 * sev) * np.clip(edge * 3, 0, 1)
    tint = np.array([1.0, 0.97, 0.90], np.float32)
    arr *= (1 - darken)[..., None] * (1 - (1 - tint) * mm[..., None])
    md.bitmap((0, 0), m.point(lambda v: 255 if v > 60 else 0).convert("1"), fill=255)


def _apply_torn_label(rng, arr, label_box, sev, md):
    if label_box is None:
        return False
    x0, y0, x1, y1 = label_box
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    frac = 0.25 + 0.5 * sev
    cut_x = x0 + (x1 - x0) * (1 - frac)
    pts = [(cut_x, y0)]
    n = 10
    for k in range(1, n):
        pts.append((cut_x + rng.uniform(-0.08, 0.08) * (x1 - x0), y0 + (y1 - y0) * k / n))
    pts += [(cut_x, y1), (x1 + 2, y1 + 2), (x1 + 2, y0 - 2)]
    residue = tuple(int(v) for v in rng.uniform(150, 200, 3) * np.array([1.0, 0.95, 0.85]))
    d.polygon(pts, fill=residue)
    md.polygon(pts, fill=255)
    for _ in range(int(3 + 6 * sev)):  # glue/paper fibres
        fx, fy = rng.uniform(cut_x, x1), rng.uniform(y0, y1)
        d.ellipse([fx - SS * 2, fy - SS * 2, fx + SS * 2, fy + SS * 2], fill=(235, 235, 230))
    arr[:] = np.asarray(img, np.float32)
    return True


def _apply_scratch(rng, arr, bx, by, bw, bh, sev, md):
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    for _ in range(int(rng.integers(1, 3))):
        x0, y0 = rng.uniform(bx, bx + bw), rng.uniform(by, by + bh)
        a = rng.uniform(0, math.pi)
        L = min(bw, bh) * rng.uniform(0.3, 0.8)
        pts = [(x0, y0)]
        for k in range(1, 6):
            pts.append((x0 + L * k / 5 * math.cos(a) + rng.uniform(-3, 3),
                        y0 + L * k / 5 * math.sin(a) + rng.uniform(-3, 3)))
        col = tuple(int(v) for v in np.clip(arr[int(y0), int(x0)] * (1.25 + 0.2 * sev) + 20, 0, 255))
        d.line(pts, fill=col, width=max(1, int(SS * (0.8 + 1.6 * sev))))
        md.line(pts, fill=255, width=max(6, int(SS * 5)))
    arr[:] = np.asarray(img, np.float32)


# ----------------------------------------------------------------------------
# full render
# ----------------------------------------------------------------------------
def render(rng: np.random.Generator, damaged: bool) -> tuple[Image.Image, Meta]:
    W = S * SS
    bg = _belt(rng, W)

    # ---- box geometry (axis-aligned in its own layer, rotated later)
    bw = W * rng.uniform(0.45, 0.72)
    bh = W * rng.uniform(0.40, 0.66)
    layer_sz = int(W * 1.0)
    bx, by = (layer_sz - bw) / 2, (layer_sz - bh) / 2
    color = _box_color(rng)

    # base cardboard with fibre texture + lighting gradient
    box = np.ones((layer_sz, layer_sz, 3), np.float32) * color[None, None, :]
    tex = 1 + _smooth_noise(rng, layer_sz, layer_sz, 5, 3) * rng.uniform(0.03, 0.07)
    fib = 1 + rng.standard_normal((layer_sz, layer_sz)).astype(np.float32) * rng.uniform(0.01, 0.03)
    gx = np.linspace(-1, 1, layer_sz, dtype=np.float32)[None, :]
    gy = np.linspace(-1, 1, layer_sz, dtype=np.float32)[:, None]
    ga = rng.uniform(0, 2 * math.pi)
    grad = 1 + rng.uniform(0.03, 0.10) * (gx * math.cos(ga) + gy * math.sin(ga))
    box *= (tex * fib * grad)[..., None]

    img = Image.fromarray(np.clip(box, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)

    # seam + tape along the long axis
    horizontal = bw >= bh
    tape_kind = rng.integers(0, 3)
    if horizontal:
        tw = bh * rng.uniform(0.12, 0.20)
        ty = by + bh / 2 + rng.uniform(-0.04, 0.04) * bh
        tape_rect = [bx - 2, ty - tw / 2, bx + bw + 2, ty + tw / 2]
        seam = [(bx, ty), (bx + bw, ty)]
    else:
        tw = bw * rng.uniform(0.12, 0.20)
        tx = bx + bw / 2 + rng.uniform(-0.04, 0.04) * bw
        tape_rect = [tx - tw / 2, by - 2, tx + tw / 2, by + bh + 2]
        seam = [(tx, by), (tx, by + bh)]
    d.line(seam, fill=tuple(int(v) for v in color * 0.55), width=max(2, SS))
    if tape_kind == 0:      # clear tape: slight gloss
        tcol = tuple(int(min(255, v * 1.12 + 10)) for v in color)
    elif tape_kind == 1:    # brown tape
        tcol = tuple(int(v) for v in color * np.array([0.82, 0.70, 0.55]))
    else:                   # white paper tape
        tcol = (int(rng.uniform(215, 240)),) * 3
    tape_layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(tape_layer).rectangle(tape_rect, fill=tcol + (int(rng.uniform(150, 230)),))
    img = Image.alpha_composite(img.convert("RGBA"), tape_layer).convert("RGB")
    d = ImageDraw.Draw(img)

    # label
    label_box = None
    if rng.random() < 0.85:
        lab = _draw_label(rng, bw, bh)
        lx = int(rng.uniform(bx + 0.06 * bw, bx + bw - lab.width - 0.06 * bw))
        ly = int(rng.uniform(by + 0.06 * bh, by + bh - lab.height - 0.06 * bh))
        img.paste(lab, (lx, ly), lab)
        label_box = (lx, ly, lx + lab.width, ly + lab.height)
        d = ImageDraw.Draw(img)

    if rng.random() < 0.45:
        _harmless_marks(rng, d, bx, by, bx + bw, by + bh, color)

    # outline (box edges)
    d.rectangle([bx, by, bx + bw, by + bh], outline=tuple(int(v) for v in color * 0.6), width=max(2, int(SS * 1.5)))

    arr = np.asarray(img, np.float32).copy()
    meta = Meta(label="damaged" if damaged else "good")
    cut_poly = None
    dmask = Image.new("L", (layer_sz, layer_sz), 0)
    md = ImageDraw.Draw(dmask)
    if damaged:
        n_def = 1 if rng.random() < 0.7 else 2
        types = list(rng.choice(DEFECT_TYPES, size=n_def, replace=False))
        for t in types:
            sev = float(rng.uniform(0.25, 1.0))
            if t == "dent":
                _apply_dent(rng, arr, bx, by, bw, bh, sev, md)
            elif t == "tear":
                _apply_tear(rng, arr, bx, by, bw, bh, sev, md)
            elif t == "crushed_corner":
                cut_poly = _apply_crushed_corner(rng, arr, [(bx, by), (bx + bw, by), (bx + bw, by + bh), (bx, by + bh)], sev, md)
            elif t == "stain":
                _apply_stain(rng, arr, bx, by, bw, bh, sev, md)
            elif t == "torn_label":
                if not _apply_torn_label(rng, arr, label_box, sev, md):
                    t = "scratch"
                    _apply_scratch(rng, arr, bx, by, bw, bh, sev, md)
            elif t == "scratch":
                _apply_scratch(rng, arr, bx, by, bw, bh, sev, md)
            meta.defects.append((t, round(sev, 3)))

    box_img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    # alpha mask of the box (minus crushed tip)
    mask = Image.new("L", box_img.size, 0)
    md = ImageDraw.Draw(mask)
    md.rectangle([bx, by, bx + bw, by + bh], fill=255)
    if cut_poly is not None:
        md.polygon(cut_poly, fill=0)

    # rotate + place
    angle = rng.uniform(-10, 10)
    box_img = box_img.rotate(angle, resample=Image.BICUBIC, expand=False)
    mask = mask.rotate(angle, resample=Image.BICUBIC, expand=False)
    dmask = dmask.rotate(angle, resample=Image.NEAREST, expand=False)
    ox = int(rng.uniform(-0.08, 0.08) * W)
    oy = int(rng.uniform(-0.06, 0.06) * W)

    scene = Image.fromarray(np.clip(bg, 0, 255).astype(np.uint8))
    # soft shadow
    sh = Image.new("L", scene.size, 0)
    sh.paste(mask, (ox + int(W * 0.015), oy + int(W * 0.02)))
    sh = sh.filter(ImageFilter.GaussianBlur(radius=W * 0.012))
    sh_arr = np.asarray(sh, np.float32)[..., None] / 255.0
    sc = np.asarray(scene, np.float32) * (1 - 0.45 * sh_arr)
    scene = Image.fromarray(np.clip(sc, 0, 255).astype(np.uint8))
    scene.paste(box_img, (ox, oy), mask)
    full_dmask = Image.new("L", scene.size, 0)
    full_dmask.paste(dmask, (ox, oy))
    defect_mask = full_dmask.resize((S, S), Image.BILINEAR).point(lambda v: 255 if v > 64 else 0)

    # ---- camera effects
    out = scene.resize((S, S), Image.LANCZOS)
    a = np.asarray(out, np.float32)
    a = (a - 128) * rng.uniform(0.85, 1.15) + 128
    a *= rng.uniform(0.75, 1.20)
    a *= np.array([rng.uniform(0.92, 1.08), 1.0, rng.uniform(0.92, 1.08)], np.float32)
    gx = np.linspace(-1, 1, S, dtype=np.float32)[None, :, None]
    gy = np.linspace(-1, 1, S, dtype=np.float32)[:, None, None]
    g2 = rng.uniform(0, 2 * math.pi)
    a *= 1 + rng.uniform(0, 0.12) * (gx * math.cos(g2) + gy * math.sin(g2))
    out = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    if rng.random() < 0.6:  # motion blur along the belt (horizontal)
        k = int(rng.integers(2, 7))
        a = np.asarray(out, np.float32)
        pad = np.pad(a, ((0, 0), (k, k), (0, 0)), mode="edge")
        acc = np.zeros_like(a)
        for i in range(k):
            acc += pad[:, k - k // 2 + i: k - k // 2 + i + S]
        out = Image.fromarray(np.clip(acc / k, 0, 255).astype(np.uint8))
    if rng.random() < 0.5:
        out = out.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.2, 1.0)))
    a = np.asarray(out, np.float32) + rng.standard_normal((S, S, 3)).astype(np.float32) * rng.uniform(1.5, 7)
    out = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    buf = io.BytesIO()
    out.save(buf, format="JPEG", quality=int(rng.integers(60, 96)))
    buf.seek(0)
    meta.mask = defect_mask
    return Image.open(buf).convert("RGB"), meta


def generate(out_dir: str, n: int, split: str, seed: int):
    rng = np.random.default_rng(seed)
    rows = []
    for lab in ("good", "damaged"):
        os.makedirs(os.path.join(out_dir, split, lab), exist_ok=True)
    os.makedirs(os.path.join(out_dir, split, "masks"), exist_ok=True)
    for i in range(n):
        damaged = i % 2 == 1
        img, meta = render(rng, damaged)
        name = f"{split}_{i:05d}.jpg"
        img.save(os.path.join(out_dir, split, meta.label, name), quality=92)
        if damaged:   # where the damage is (white) - used as extra training signal
            meta.mask.save(os.path.join(out_dir, split, "masks", name.replace(".jpg", ".png")))
        rows.append({"split": split, "file": f"{meta.label}/{name}", "label": meta.label,
                     "defects": ";".join(f"{t}:{s}" for t, s in meta.defects),
                     "min_severity": meta.min_severity})
    with open(os.path.join(out_dir, f"{split}_meta.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return rows


def main():
    ap = argparse.ArgumentParser(description="Generate synthetic conveyor box images")
    ap.add_argument("--out", default="data/synth")
    ap.add_argument("--train", type=int, default=6000)
    ap.add_argument("--val", type=int, default=1000)
    ap.add_argument("--test", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    for split, n, off in (("train", a.train, 0), ("val", a.val, 1), ("test", a.test, 2)):
        if n > 0:
            generate(a.out, n, split, a.seed * 100 + off)
            print(f"{split}: {n} images -> {a.out}/{split}")


if __name__ == "__main__":
    main()
