"""Synthetic hatch patches for the material-hatch classifier (stoa_qa.hatch).

Drawings show a material by its hatch: concrete stipple, brick courses, insulation batts, wood grain, tile grids...
Real sheets with labelled hatches are rare, but the patterns themselves are standard: AutoCAD's pattern library
(ezdxf ships 172 of them as line families) plus a few CAD/Revit styles drawn procedurally here. Each pattern belongs to
one hatch class (CLASSES); `sample()` renders a random patch of a random class the way it shows on a sheet: any
scale and angle, thin or bold ink, grey or coloured, partly covered by walls, text and dimension lines, scanned or
JPEG-compressed. Used on Modal to train (modal_jobs/hatch_cloud.py) and locally for tests.

numpy + Pillow (+ ezdxf for the pattern library); no torch.
"""
from __future__ import annotations

import math
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

PATCH = 96          # network input (pixels)
WORK = 2            # supersampling factor for antialiased drawing

# Hatch classes. Order = class id of the trained network: keep stable (append only).
CLASSES = [
    "none",            # no hatch: blank floor, furniture, text, dimension lines, wall outlines
    "concrete",        # stipple with aggregate triangles (AR-CONC, Revit concrete)
    "diagonal",        # single parallel diagonal lines (ANSI31): the general section hatch, masonry by convention
    "brick",           # brick courses (running bond)
    "block",           # concrete masonry units / large blocks
    "stone",           # natural stone, rubble, rock
    "earth",           # soil / earth
    "gravel",          # gravel, pebbles
    "stipple",         # fine dots: sand, screed, plaster, render
    "insulation",      # batt insulation (zigzag / loops)
    "wood",            # wood grain, timber in section
    "planks",          # boards, decking, battens (parallel lines with staggered joints)
    "parquet",         # herringbone / basket-weave flooring
    "tile",            # square or diagonal tile grid
    "metal",           # steel (paired diagonals), standing seam, zinc
    "crosshatch",      # crossing diagonals (ANSI37): rigid insulation, sound insulation, lead
    "solid",           # solid grey or black fill (poché)
]
IDS = {c: i for i, c in enumerate(CLASSES)}

# What each hatch conventionally stands for, when the sheet has no legend that says otherwise.
# (family key in stoa_qa.materials.FAMILIES, how sure the convention is)
CONVENTION = {
    "concrete": ("concrete", "strong"),
    "diagonal": ("masonry", "weak"),       # ANSI31 is also the generic "cut" hatch for any material
    "brick": ("brick", "strong"),
    "block": ("block", "strong"),
    "stone": ("stone", "strong"),
    "earth": ("earth", "strong"),
    "gravel": ("gravel", "strong"),
    "stipple": ("screed", "weak"),         # sand, screed, plaster and render share it
    "insulation": ("insulation", "strong"),
    "wood": ("wood", "strong"),
    "planks": ("wood", "weak"),            # boards / decking; also siding or composite decking
    "parquet": ("parquet", "strong"),
    "tile": ("tile", "weak"),              # a grid is also a ceiling grid or a paving pattern
    "metal": ("steel", "weak"),
    "crosshatch": ("insulation", "weak"),
    "solid": (None, "none"),              # poché: concrete in some offices, any cut wall in others
}

# AutoCAD / ezdxf library patterns -> hatch class (checked by eye on a rendered contact sheet,
# _analysis/hatch_patterns.py). Decorative or ambiguous patterns are left out.
PAT_CLASS = {
    # concrete: stipple + aggregate (AR-CONC, CONCRETE1-3), diagonals with dots (SACNCR)
    "AR-CONC": "concrete", "CONCRETE1": "concrete", "CONCRETE2": "concrete", "CONCRETE3": "concrete", "SACNCR": "concrete",
    # single diagonals: ANSI31 (general / masonry), ANSI35 (fire brick), BRICK_EXISTING (same look)
    "ANSI31": "diagonal", "ANSI35": "diagonal", "BRICK_EXISTING": "diagonal",
    "AR-BRELM": "brick", "AR-BRSTD": "brick", "BRICK": "brick", "BRICK_FLBOND": "brick", "BRICK_STRBOND": "brick",
    "BRICK_PAIRS": "brick", "V_MASONRY200x60": "brick", "V_MASONRY200x75": "brick", "V_MASONRY220x80": "brick",
    "V_MASONRY300x75": "brick",
    # AR-B816 / AR-B88 are AutoCAD's 8x16 and 8x8 concrete block
    "AR-B816": "block", "AR-B816C": "block", "AR-B88": "block", "BLOCKS": "block", "V_MASONRY200x100": "block",
    "V_MASONRY300x100": "block", "V_MASONRY300x150": "block", "V_MASONRY300x200": "block", "V_MASONRY400x100": "block",
    "V_MASONRY400x200": "block",
    "GRAVEL": "gravel", "GRAVEL1": "gravel",
    "AR-SAND": "stipple", "SAND": "stipple", "DOTS": "stipple", "DOTS1": "stipple", "DOTS2": "stipple", "SCREED": "stipple",
    "WOOD1": "wood", "WOOD2": "wood", "WOOD3": "wood", "WOOD4": "wood",
    "AR-RSHKE": "planks", "BOARD": "planks", "SHAKES": "planks",
    "AR-PARQ1": "parquet", "AR-HBONE": "parquet", "HERRING_45": "parquet", "HERRING_H": "parquet",
    "HERRING_UNI": "parquet", "HERRING_V": "parquet",
    "NET": "tile", "GRID": "tile", "TILEPAT1": "tile",
    "ANSI32": "metal", "STEEL": "metal", "ANSI33": "metal",
    "ANSI37": "crosshatch", "INSULATION": "crosshatch",
    # Left out (look like another class, or carry no material): EARTH* (= parquet basket weave), ROCK / DOLMIT
    # (= planks), JIS_* / GOST_* diagonals, plain line sets (LINE, V_PARQUET, V_BATTEN_FLOOR, V_ZINC, PLAST),
    # BRICK_STBOND (= tile grid), CONC_EXISTING / CONC_PRECAST (= crosshatch), decorative patterns.
}


# ------------------------------------------------------------------ pattern library (line families)

_LIB: dict | None = None


def library() -> dict[str, list]:
    """Pattern name -> line families [(angle deg, (x0, y0), (dx, dy), [dash, ...])], ISO units (mm)."""
    global _LIB
    if _LIB is None:
        from ezdxf.tools import pattern

        _LIB = {k: v for k, v in pattern.ISO_PATTERN.items() if PAT_CLASS.get(k)}
    return _LIB


def _draw_family(d: ImageDraw.ImageDraw, size: int, fam, scale: float, rot: float, shift, width: int, ink: int) -> None:
    """One line family of a DXF hatch pattern. The offset to the next line is a world vector (DXF convention)."""
    angle, (ox, oy), (dx, dy), dashes = fam
    a = math.radians(angle + rot)
    u = np.array([math.cos(a), math.sin(a)])                 # along the line
    n = np.array([-math.sin(a), math.cos(a)])                # to the next line
    rr = math.radians(rot)
    R = np.array([[math.cos(rr), -math.sin(rr)], [math.sin(rr), math.cos(rr)]])
    o = R @ np.array([ox, oy]) * scale + shift
    v = R @ np.array([dx, dy]) * scale
    step_n, step_u = float(np.dot(v, n)), float(np.dot(v, u))
    if abs(step_n) < 1.2:                                    # lines closer than a pixel: the pattern is a grey wash
        return
    c = np.array([size / 2, size / 2])
    reach = size * 0.75
    k0 = math.floor((np.dot(c - o, n) - reach) / step_n)
    k1 = math.ceil((np.dot(c - o, n) + reach) / step_n)
    if k1 - k0 > 400:
        return
    dash = [x * scale for x in dashes] if dashes else []
    period = sum(abs(x) for x in dash)
    for k in range(min(k0, k1), max(k0, k1) + 1):
        p = o + k * (step_n * n + step_u * u)
        t0 = np.dot(c - p, u) - reach
        if not dash or period < 1e-6:
            a0, a1 = p + t0 * u, p + (t0 + 2 * reach) * u
            d.line([tuple(a0), tuple(a1)], fill=ink, width=width)
            continue
        if period < 1.5:
            continue
        t = math.floor(t0 / period) * period
        while t < t0 + 2 * reach:
            for seg in dash:
                L = abs(seg)
                if seg > 0:
                    d.line([tuple(p + t * u), tuple(p + (t + L) * u)], fill=ink, width=width)
                elif seg == 0:
                    q = p + t * u
                    d.ellipse([q[0] - width * 0.6, q[1] - width * 0.6, q[0] + width * 0.6, q[1] + width * 0.6], fill=ink)
                t += max(L, 0.0)
                if L == 0:
                    t += 0.0
            if period == 0:
                break


def render_pattern(name: str, size: int, scale: float, rot: float, width: int, ink: int, rng: random.Random) -> Image.Image:
    """One library pattern filling a size x size greyscale image (white background)."""
    im = Image.new("L", (size, size), 255)
    d = ImageDraw.Draw(im)
    shift = np.array([rng.uniform(0, size), rng.uniform(0, size)])
    for fam in library()[name]:
        _draw_family(d, size, fam, scale, rot, shift, width, ink)
    return im


# ------------------------------------------------------------------ procedural styles (CAD / Revit looks the library lacks)

def _concrete(size, rng, scale, width, ink):
    im = Image.new("L", (size, size), 255)
    d = ImageDraw.Draw(im)
    dens = rng.uniform(0.004, 0.02) / max(scale / 3, 0.3)
    for _ in range(int(size * size * dens)):
        x, y = rng.uniform(0, size), rng.uniform(0, size)
        r = rng.uniform(0.5, 1.2) * width
        d.ellipse([x - r, y - r, x + r, y + r], fill=ink)
    for _ in range(int(size * size * dens * rng.uniform(0.04, 0.2))):
        x, y, s = rng.uniform(0, size), rng.uniform(0, size), rng.uniform(2, 6) * scale ** 0.5 * WORK
        a = rng.uniform(0, 2 * math.pi)
        pts = [(x + s * math.cos(a + k * 2.094 + rng.uniform(-0.3, 0.3)), y + s * math.sin(a + k * 2.094 + rng.uniform(-0.3, 0.3)))
               for k in range(3)]
        d.polygon(pts, outline=ink, width=max(1, width // 2))
    return im


def _stipple(size, rng, scale, width, ink):
    im = Image.new("L", (size, size), 255)
    d = ImageDraw.Draw(im)
    dens = rng.uniform(0.006, 0.04)
    for _ in range(int(size * size * dens)):
        x, y = rng.uniform(0, size), rng.uniform(0, size)
        r = rng.uniform(0.4, 0.9) * width
        d.ellipse([x - r, y - r, x + r, y + r], fill=ink)
    return im


def _insulation(size, rng, scale, width, ink):
    """Batt insulation: a zigzag or a row of loops between two lines."""
    im = Image.new("L", (size, size), 255)
    d = ImageDraw.Draw(im)
    h = rng.uniform(14, 40) * WORK * scale ** 0.3
    pitch = h * rng.uniform(0.35, 0.8)
    style = rng.choice(["loops", "zigzag", "wave"])
    y = rng.uniform(-h, 0)
    while y < size + h:
        x0 = rng.uniform(-pitch, 0)
        ts = np.linspace(0, (size + 2 * pitch) / pitch * 2 * math.pi, int((size + 2 * pitch) / pitch * 24))
        if style == "loops":  # prolate cycloid: the looping line of batt insulation
            b = rng.uniform(1.3, 2.2)
            pts = [(x0 + pitch / (2 * math.pi) * (t - b * math.sin(t)), y + h / 2 - h / 2 * math.cos(t)) for t in ts]
        elif style == "wave":
            pts = [(x0 + pitch / (2 * math.pi) * t, y + h / 2 - h / 2 * math.cos(t)) for t in ts]
        else:
            n = int((size + 2 * pitch) / (pitch / 2)) + 2
            pts = [(x0 + k * pitch / 2, y if k % 2 else y + h) for k in range(n)]
        d.line(pts, fill=ink, width=width)
        if rng.random() < 0.7:
            d.line([(0, y - 2), (size, y - 2)], fill=ink, width=width)
            d.line([(0, y + h + 2), (size, y + h + 2)], fill=ink, width=width)
        y += h + rng.uniform(4, 30) * WORK
    return im.rotate(rng.choice([0, 90, 0, 0]) + rng.uniform(-3, 3), fillcolor=255)


def _wood(size, rng, scale, width, ink):
    """Wood grain: wavy roughly parallel lines, sometimes knots."""
    im = Image.new("L", (size, size), 255)
    d = ImageDraw.Draw(im)
    gap = rng.uniform(3, 9) * WORK * scale ** 0.4
    amp, freq = gap * rng.uniform(0.25, 0.9), rng.uniform(0.015, 0.05) / WORK
    y = -gap
    ph = rng.uniform(0, 6)
    while y < size + gap:
        pts = [(x, y + amp * math.sin(freq * x + ph + y * 0.03)) for x in range(0, size + 4, 4)]
        d.line(pts, fill=ink, width=max(1, width - rng.choice([0, 1])))
        y += gap * rng.uniform(0.7, 1.3)
    for _ in range(rng.choice([0, 0, 1, 2])):
        x, y, r = rng.uniform(0, size), rng.uniform(0, size), rng.uniform(4, 10) * WORK
        d.ellipse([x - r, y - r / 2, x + r, y + r / 2], outline=ink, width=width)
    return im.rotate(rng.uniform(0, 180), fillcolor=255)


def _stone(size, rng, scale, width, ink):
    """Rubble / random stone: irregular polygons."""
    im = Image.new("L", (size, size), 255)
    d = ImageDraw.Draw(im)
    s = rng.uniform(10, 30) * WORK * scale ** 0.4
    y = -s
    while y < size + s:
        x = -s * rng.random()
        while x < size + s:
            w, h = s * rng.uniform(0.6, 1.5), s * rng.uniform(0.5, 1.1)
            pts = [(x + rng.uniform(0, w * 0.2), y + rng.uniform(0, h * 0.2)), (x + w - rng.uniform(0, w * 0.2), y + rng.uniform(0, h * 0.3)),
                   (x + w - rng.uniform(0, w * 0.2), y + h - rng.uniform(0, h * 0.2)), (x + rng.uniform(0, w * 0.3), y + h - rng.uniform(0, h * 0.2))]
            d.polygon(pts, outline=ink, width=width)
            x += w
        y += s * rng.uniform(0.6, 1.0)
    return im


def _gravel(size, rng, scale, width, ink):
    im = Image.new("L", (size, size), 255)
    d = ImageDraw.Draw(im)
    for _ in range(int(size * size / (rng.uniform(150, 500) * WORK * WORK))):
        x, y = rng.uniform(0, size), rng.uniform(0, size)
        rx, ry = rng.uniform(2, 6) * WORK, rng.uniform(1.5, 4) * WORK
        d.ellipse([x - rx, y - ry, x + rx, y + ry], outline=ink, width=max(1, width - 1))
    return im


def _earth(size, rng, scale, width, ink):
    """Earth: scattered tufts of short strokes at random angles, with dots (the Revit / ArchiCAD look)."""
    im = Image.new("L", (size, size), 255)
    d = ImageDraw.Draw(im)
    g = rng.uniform(12, 26) * WORK
    for _ in range(int(size * size / (g * g) * 1.3)):
        x, y = rng.uniform(-g, size), rng.uniform(-g, size)
        a = rng.uniform(0, math.pi)
        L = g * rng.uniform(0.3, 0.7)
        u, n = (math.cos(a), math.sin(a)), (-math.sin(a), math.cos(a))
        for k in range(rng.randint(2, 4)):
            ox, oy = x + n[0] * k * g / 7, y + n[1] * k * g / 7
            d.line([(ox, oy), (ox + u[0] * L, oy + u[1] * L)], fill=ink, width=width)
    for _ in range(int(size * size / (g * g) * 3)):
        x, y = rng.uniform(0, size), rng.uniform(0, size)
        d.ellipse([x - width, y - width, x + width, y + width], fill=ink)
    return im


def _planks(size, rng, scale, width, ink):
    im = Image.new("L", (size, size), 255)
    d = ImageDraw.Draw(im)
    w = rng.uniform(6, 22) * WORK * scale ** 0.3
    L = w * rng.uniform(4, 12)
    y = -w
    while y < size + w:
        d.line([(0, y), (size, y)], fill=ink, width=width)
        x = -L * rng.random()
        while x < size:
            d.line([(x, y), (x, y + w)], fill=ink, width=max(1, width - 1))
            x += L * rng.uniform(0.6, 1.2)
        y += w
    return im.rotate(rng.choice([0, 90]) + rng.uniform(-3, 3), fillcolor=255)


def _tile(size, rng, scale, width, ink):
    im = Image.new("L", (size, size), 255)
    d = ImageDraw.Draw(im)
    s = rng.uniform(8, 40) * WORK * scale ** 0.3
    off = rng.uniform(0, s)
    for k in np.arange(-s + off, size + s, s):
        d.line([(k, 0), (k, size)], fill=ink, width=width)
        d.line([(0, k), (size, k)], fill=ink, width=width)
    return im.rotate(rng.choice([0, 90]) + rng.uniform(-3, 3), fillcolor=255)


def _solid(size, rng, scale, width, ink):
    v = rng.choice([rng.randint(0, 60), rng.randint(90, 200)])
    return Image.new("L", (size, size), v)


PROCEDURAL = {"concrete": _concrete, "stipple": _stipple, "insulation": _insulation, "wood": _wood, "stone": _stone,
              "gravel": _gravel, "earth": _earth, "planks": _planks, "tile": _tile, "solid": _solid}


# ------------------------------------------------------------------ the sheet around the hatch

def _font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        try:
            return ImageFont.truetype("arial.ttf", size)
        except OSError:
            return ImageFont.load_default()


WORDS = ["KITCHEN", "BATH", "101", "OFFICE", "3'-0\"", "CONC.", "GYP. BD.", "TYP.", "W1", "F2", "Zimmer", "Bad", "12.5 m²",
         "MH", "KH", "OH", "2.03", "A-401", "EQ", "SEE DETAIL", "STOR.", "Küche", "WC", "1:100", "N", "EXIST."]


def clutter(im: Image.Image, rng: random.Random, amount: float) -> None:
    """Things printed over and around hatches on a sheet: wall lines, dimension lines, arcs, text."""
    d = ImageDraw.Draw(im)
    s = im.size[0]
    for _ in range(int(rng.uniform(0, 4) * amount)):
        w = rng.choice([1, 2, 3, 5]) * WORK
        if rng.random() < 0.5:
            x = rng.uniform(0, s)
            d.line([(x, 0), (x, s)], fill=rng.randint(0, 80), width=w)
        else:
            y = rng.uniform(0, s)
            d.line([(0, y), (s, y)], fill=rng.randint(0, 80), width=w)
    for _ in range(int(rng.uniform(0, 2) * amount)):
        x, y, r = rng.uniform(0, s), rng.uniform(0, s), rng.uniform(20, 120) * WORK
        d.arc([x - r, y - r, x + r, y + r], rng.uniform(0, 360), rng.uniform(0, 360), fill=rng.randint(0, 100), width=WORK)
    for _ in range(int(rng.uniform(0, 3) * amount)):
        f = _font(int(rng.uniform(7, 16) * WORK))
        x, y = rng.uniform(-20, s), rng.uniform(-10, s)
        tw = d.textlength(t := rng.choice(WORDS), font=f)
        if rng.random() < 0.6:  # text is usually printed on a white mask over the hatch
            d.rectangle([x - 2, y - 2, x + tw + 2, y + f.size + 2], fill=255)
        d.text((x, y), t, fill=rng.randint(0, 90), font=f)


def _region_mask(size: int, rng: random.Random) -> Image.Image:
    """Where the hatch is: the whole patch, or a band / corner / polygon of it (the rest is another material or nothing)."""
    m = Image.new("L", (size, size), 255)
    r = rng.random()
    if r < 0.55:
        return m
    d = ImageDraw.Draw(m)
    if r < 0.8:  # a wall band: the hatch between two lines
        t = rng.uniform(0.35, 0.8) * size
        c = rng.uniform(t / 2, size - t / 2)
        m = Image.new("L", (size, size), 0)
        d = ImageDraw.Draw(m)
        d.rectangle([0, c - t / 2, size, c + t / 2], fill=255)
        return m.rotate(rng.choice([0, 90, rng.uniform(0, 180)]), fillcolor=0)
    k = rng.uniform(0.25, 0.45) * size  # a corner or edge left out
    x, y = rng.choice([(0, 0), (size - k, 0), (0, size - k), (size - k, size - k)])
    d.rectangle([x, y, x + k, y + k], fill=0)
    return m


def finish(im: Image.Image, rng: random.Random, out: int = PATCH, outline: bool = True) -> Image.Image:
    """Supersampled drawing -> patch as a sheet shows it: down-sampled, paper tone, colour/greys, blur, noise, JPEG."""
    im = im.resize((out, out), Image.LANCZOS)
    a = np.asarray(im, dtype=np.float32) / 255.0
    paper = rng.uniform(0.86, 1.0)
    ink = rng.uniform(0.0, 0.45)
    a = ink + (paper - ink) * a
    if rng.random() < 0.3:
        a = 1.0 - (1.0 - a) * rng.uniform(0.3, 0.8)  # a light grey hatch (Revit screens hatches)
    if rng.random() < 0.35:
        a = a + np.random.default_rng(rng.randint(0, 1 << 30)).normal(0, rng.uniform(0.01, 0.06), a.shape)
    a = np.clip(a, 0, 1)
    im = Image.fromarray((a * 255).astype(np.uint8))
    if rng.random() < 0.3:
        im = im.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.0)))
    if rng.random() < 0.3:
        import io

        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=rng.randint(25, 80))
        im = Image.open(io.BytesIO(buf.getvalue())).convert("L")
    return im


def _none(size: int, rng: random.Random) -> Image.Image:
    """No hatch: white paper with sheet clutter, furniture-like shapes, or a plain wall outline."""
    im = Image.new("L", (size, size), 255)
    d = ImageDraw.Draw(im)
    r = rng.random()
    if r < 0.25:
        pass
    elif r < 0.5:  # furniture symbols: rectangles, circles
        for _ in range(rng.randint(1, 4)):
            x, y, w, h = rng.uniform(0, size), rng.uniform(0, size), rng.uniform(20, 150) * WORK, rng.uniform(20, 150) * WORK
            if rng.random() < 0.5:
                d.rectangle([x, y, x + w, y + h], outline=rng.randint(0, 120), width=WORK)
            else:
                d.ellipse([x, y, x + w, y + w], outline=rng.randint(0, 120), width=WORK)
    elif r < 0.7:  # a stair: parallel lines spaced like treads, with an arrow line (not a hatch)
        g = rng.uniform(25, 45) * WORK
        for k in np.arange(0, size, g):
            d.line([(k, 0), (k, size)], fill=rng.randint(0, 90), width=WORK)
        d.line([(0, size / 2), (size, size / 2)], fill=0, width=WORK)
    else:  # a few wall outlines (double lines)
        for _ in range(rng.randint(1, 3)):
            y, t = rng.uniform(0, size), rng.uniform(8, 30) * WORK
            d.line([(0, y), (size, y)], fill=0, width=WORK * 2)
            d.line([(0, y + t), (size, y + t)], fill=0, width=WORK * 2)
    clutter(im, rng, rng.uniform(0.5, 2.0))
    return im.rotate(rng.choice([0, 90, 180, 270]), fillcolor=255)


def sample(rng: random.Random, cls: str | None = None, out: int = PATCH) -> tuple[Image.Image, int]:
    """One synthetic patch and its class id. A hatch patch that came out (nearly) empty is drawn again."""
    cls = cls or rng.choice(CLASSES)
    for _ in range(6):
        im, k = _sample(rng, cls, out)
        if cls in ("none", "solid"):
            return im, k
        a = np.asarray(im, dtype=np.float32)
        if (a < np.median(a) - 40).mean() > 0.03:
            return im, k
    return im, k


def _sample(rng: random.Random, cls: str, out: int) -> tuple[Image.Image, int]:
    size = out * WORK * 2  # drawn larger, then a random crop is the visible part (scale variety)
    if cls == "none":
        im = _none(size, rng)
    else:
        width = rng.choice([1, 1, 2, 2, 3]) * WORK // 2 + 1
        ink = rng.randint(0, 70)
        names = [n for n, c in PAT_CLASS.items() if c == cls and n in library()]
        if cls in PROCEDURAL and (not names or rng.random() < 0.45):
            im = PROCEDURAL[cls](size, rng, rng.uniform(0.5, 3.0), width, ink)
        else:
            name = rng.choice(names)
            spacing = _pattern_spacing(name)
            target = rng.uniform(5, 20) * WORK  # pixels between neighbouring lines on the patch
            im = render_pattern(name, size, target / max(spacing, 1e-3), _rotation(cls, rng), width, ink, rng)
        if cls != "solid":
            m = _region_mask(size, rng)
            im = Image.composite(im, Image.new("L", (size, size), 255), m)
            if rng.random() < 0.6:  # the region's boundary lines
                edges = m.filter(ImageFilter.FIND_EDGES)
                im = Image.composite(Image.new("L", (size, size), 0), im, edges.point(lambda v: 255 if v > 0 else 0))
        clutter(im, rng, rng.uniform(0.0, 1.3))
    c = rng.uniform(0.35, 0.6) * size
    x, y = rng.uniform(0, size - c), rng.uniform(0, size - c)
    im = im.crop((int(x), int(y), int(x + c), int(y + c)))
    return finish(im, rng, out), IDS[cls]


# Orientation is part of what tells look-alikes apart: a diagonal grid is crosshatch, an orthogonal one is tile; diagonal
# parallels are the section hatch, horizontal ones are boards. Library patterns carry their own angle (ANSI31 = 45 deg).
ORTHO = {"tile", "brick", "block", "planks", "parquet"}
DIAG = {"diagonal", "metal", "crosshatch"}


def _rotation(cls: str, rng: random.Random) -> float:
    if cls in ORTHO:
        return rng.choice([0.0, 90.0]) + rng.uniform(-3, 3)
    if cls in DIAG:
        return rng.choice([0.0, 90.0]) + rng.uniform(-12, 12)
    return rng.uniform(0, 360)


_SPACING: dict[str, float] = {}


def _pattern_spacing(name: str) -> float:
    """The pattern's typical line spacing in its own units, from its ink density (a sparse many-family pattern like
    AR-CONC has few lines per family but many families): a 1-px rendering at scale 1 has ink fraction ~ 1 / spacing."""
    if name not in _SPACING:
        fams = library()[name]
        ds = [abs(f[2][0] * -math.sin(math.radians(f[0])) + f[2][1] * math.cos(math.radians(f[0]))) for f in fams]
        ds = [x for x in ds if x > 1e-6] or [1.0]
        k = 600 / (40 * min(ds))  # scale so the densest family has ~40 px between lines on a 600 px canvas
        im = render_pattern(name, 600, k, 0.0, 1, 0, random.Random(0))
        ink = float((np.asarray(im) < 128).mean())
        _SPACING[name] = (1.0 / max(ink, 1e-4)) / k if ink > 0 else min(ds)
    return _SPACING[name]


def composite(bg: np.ndarray, rng: random.Random, cls: str | None = None) -> tuple[Image.Image, int]:
    """A synthetic hatch drawn into a real sheet window (bg: 96 x 96 uint8): the hatch fills a region covering most
    of the window, the rest stays the real sheet. Teaches the network that a hatch is the texture, not clean paper."""
    cls = cls or rng.choice([c for c in CLASSES if c != "none"])
    h, k = sample(rng, cls)
    hv = np.asarray(h, dtype=np.float32)
    b = bg.astype(np.float32)
    m = np.asarray(_region_mask(PATCH * 2, rng).resize((PATCH, PATCH)), dtype=np.float32) / 255.0
    if m.mean() < 0.55:
        m = np.ones_like(m)
    keep = b if rng.random() < 0.3 else np.full_like(b, 255.0)  # lines and text of the sheet over the hatch, sometimes
    out = m * np.minimum(hv, keep) + (1 - m) * b
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)), k


def contact_sheet(path: str, per_class: int = 8, seed: int = 0) -> None:
    """A grid of samples per class, for checking the generator by eye."""
    rng = random.Random(seed)
    W = Image.new("L", (per_class * (PATCH + 4) + 120, len(CLASSES) * (PATCH + 4)), 255)
    d = ImageDraw.Draw(W)
    for i, c in enumerate(CLASSES):
        d.text((4, i * (PATCH + 4) + PATCH // 2), c, fill=0)
        for j in range(per_class):
            im, _ = sample(rng, c)
            W.paste(im, (120 + j * (PATCH + 4), i * (PATCH + 4)))
    W.save(path)


if __name__ == "__main__":
    import sys

    contact_sheet(sys.argv[1] if len(sys.argv) > 1 else "hatch_samples.png")
