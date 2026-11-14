"""Floor-plan reconstruction in image pixels: walls, rooms and openings for the 3D view of any raster plan.

The CubiCasa5K network (blueprint3d/cubicasa_model.py) gives per-pixel class maps. Its own vectoriser builds walls from
corner heatmaps and loses most walls on drawings unlike its training data (thin CAD lines, grey renders, small images).
Here:
  1. the image is cleaned (solid margins and site fill reaching the edge turn white, as on CubiCasa's paper) and
     enlarged so its long side is ~1,250 px (small images: the network sees walls at the thickness it was trained on);
  2. walls = the per-pixel wall map, united with the vectoriser's walls and with thin drawn partitions (long straight
     ink lines that meet a wall; stairs and hatching excluded); the sheet's border frame is dropped;
  3. each detector door / window is placed in the wall it sits in and cut through it; one in no wall is furniture;
  4. rooms = the space the walls enclose (doorways shut), named by the network's majority room type.
Every choice was measured on 150 CubiCasa val plans (never test) against their SVG walls, rooms, doors and windows,
as they are and shrunk to 45% (an internet-sized image); see the numbers in Params.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
BLUEPRINT3D = REPO / "blueprint3d"
MAX_SIDE = 2600          # CPU time bound for one network pass (pixels on the long side after rescaling)
VECTOR_SECONDS = 60      # time budget for the network's own vectoriser; past it, the pixel wall map alone is used
ROOM_CLASSES = ["Background", "Outdoor", "Wall", "Kitchen", "Living Room", "Bed Room", "Bath", "Entry", "Railing",
                "Storage", "Garage", "Undefined"]


@dataclass
class Maps:
    """Network output at the working resolution (image rescaled by `factor`)."""
    factor: float
    wall: np.ndarray          # P(wall) per pixel, float32
    room: np.ndarray          # argmax room class per pixel, uint8 (ROOM_CLASSES)
    room_conf: np.ndarray     # max room-class probability, float32
    door: np.ndarray          # P(door icon)
    window: np.ndarray        # P(window icon)
    extra: dict = field(default_factory=dict)


def _model():
    if str(BLUEPRINT3D) not in sys.path:
        sys.path.insert(0, str(BLUEPRINT3D))
    import cubicasa_model as M

    return M


def read_rgb(path) -> np.ndarray:
    """Any image (PNG with alpha, palette, grey, JPEG) -> RGB uint8 on a white background."""
    from PIL import Image

    with Image.open(path) as im:
        im.load()
        if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
            im = im.convert("RGBA")
            bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
            im = Image.alpha_composite(bg, im)
        return np.asarray(im.convert("RGB")).copy()


def clean_background(rgb: np.ndarray, area: float = 0.01, depth: float = 0.02) -> np.ndarray:
    """Turn large solid areas reaching the image edge (black margins, grey site fill, coloured paper) white.

    CubiCasa plans, which the network was trained on, are drawn on white. A region is cleared only if it is big (area
    share of the image) and thick (its deepest point is `depth` x the short side from its edge): a wall that runs into
    the image edge is thin, so it stays."""
    import cv2

    h, w = rgb.shape[:2]
    img = rgb.copy()
    seen = np.zeros((h + 2, w + 2), np.uint8)
    step = max(4, min(h, w) // 60)
    seeds = [(x, 0) for x in range(0, w, step)] + [(x, h - 1) for x in range(0, w, step)] + \
            [(0, y) for y in range(0, h, step)] + [(w - 1, y) for y in range(0, h, step)]
    min_area, min_depth = area * h * w, max(6.0, depth * min(h, w))
    for x, y in seeds:
        if seen[y + 1, x + 1] or rgb[y, x].min() >= 245:
            continue
        mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(img.copy(), mask, (x, y), 0, (12, 12, 12), (12, 12, 12), 4 | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY | (1 << 8))
        seen |= mask
        region = mask[1:-1, 1:-1].astype(bool)
        if region.sum() < min_area:
            continue
        if cv2.distanceTransform(region.astype(np.uint8), cv2.DIST_L2, 3).max() < min_depth:
            continue
        img[region] = 255
    return img


def network_maps(rgb: np.ndarray, factor: float, vector: bool = True) -> Maps:
    """Run the network on `rgb` rescaled by `factor` (capped at MAX_SIDE); maps at that working resolution."""
    import cv2
    import torch
    import torch.nn.functional as F

    M = _model()
    h0, w0 = rgb.shape[:2]
    factor = min(factor, MAX_SIDE / max(h0, w0))
    img = cv2.resize(rgb, (round(w0 * factor), round(h0 * factor)), interpolation=cv2.INTER_CUBIC if factor > 1 else cv2.INTER_AREA) \
        if abs(factor - 1) > 1e-3 else rgb
    h, w = img.shape[:2]
    x = torch.from_numpy(np.moveaxis(img, -1, 0)).float().unsqueeze(0)
    x = 2 * (x / 255.0) - 1
    with torch.no_grad():
        pred = F.interpolate(M.load_model()(x), size=(h, w), mode="bilinear", align_corners=True)[0]
    rooms = torch.softmax(pred[21:33], 0)
    icons = torch.softmax(pred[33:44], 0)
    conf, lab = rooms.max(0)
    extra = {}
    if vector:  # the network's own vectoriser (corner heatmaps -> straight walls): precise on CubiCasa-like drawings
        pp = M._load_module("cc_post", "floortrans/post_prosessing.py")
        from concurrent.futures import ThreadPoolExecutor
        from concurrent.futures import TimeoutError as Late

        args = ((pred[:21].numpy().copy(), rooms.numpy().copy(), icons.numpy().copy()), 0.2, [1, 2])  # it edits its inputs
        pool = ThreadPoolExecutor(max_workers=1)
        try:
            polys, types, _, _ = pool.submit(pp.get_polygons, *args).result(timeout=VECTOR_SECONDS)
            extra["vec_walls"] = [np.asarray(p, np.float64) for p, ty in zip(polys, types) if ty["type"] == "wall"]
        except (Late, Exception):  # too slow or failed on an unusual drawing: the pixel walls still stand
            extra["vec_walls"] = []
        finally:
            pool.shutdown(wait=False)
    return Maps(factor=factor * 1.0, wall=rooms[2].numpy().astype(np.float32), room=lab.numpy().astype(np.uint8),
                room_conf=conf.numpy().astype(np.float32), door=icons[2].numpy().astype(np.float32),
                window=icons[1].numpy().astype(np.float32), extra=extra)


# ---------------------------------------------------------------- geometry helpers (working pixels)

def _odd(n: float) -> int:
    n = max(1, int(round(n)))
    return n if n % 2 else n + 1


def thickness(mask: np.ndarray) -> float:
    """Typical stroke thickness of a binary mask: median of 2 x distance-to-edge along its skeleton."""
    import cv2
    from skimage.morphology import skeletonize

    if not mask.any():
        return 0.0
    dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    sk = skeletonize(mask.astype(bool))
    v = dt[sk]
    return float(2 * np.median(v)) if v.size else 0.0


def _drop_small(mask: np.ndarray, min_area: float) -> np.ndarray:
    import cv2

    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[lab]


def _snap(poly: np.ndarray, tol_deg: float = 8.0) -> np.ndarray:
    """Make edges within tol_deg of horizontal / vertical exactly axis-aligned (vertices move, topology stays)."""
    p = poly.astype(np.float64).copy()
    n = len(p)
    if n < 3:
        return p
    t = np.tan(np.radians(tol_deg))
    for _ in range(2):
        for i in range(n):
            a, b = p[i], p[(i + 1) % n]
            dx, dy = b[0] - a[0], b[1] - a[1]
            if abs(dy) <= t * abs(dx):      # near horizontal: share the mean y
                y = (a[1] + b[1]) / 2
                p[i][1] = p[(i + 1) % n][1] = y
            elif abs(dx) <= t * abs(dy):    # near vertical: share the mean x
                x = (a[0] + b[0]) / 2
                p[i][0] = p[(i + 1) % n][0] = x
    return p


def polygons(mask: np.ndarray, eps: float, min_area: float, snap: bool = True) -> list[tuple[np.ndarray, list[np.ndarray]]]:
    """Binary mask -> [(outer, [holes])] simplified (Douglas-Peucker eps) and optionally axis-snapped."""
    import cv2

    cs, hier = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    out = []
    if hier is None:
        return out
    hier = hier[0]
    for i, c in enumerate(cs):
        if hier[i][3] != -1 or cv2.contourArea(c) < min_area:
            continue
        outer = cv2.approxPolyDP(c, eps, True)[:, 0, :].astype(np.float64)
        holes = []
        j = hier[i][2]
        while j != -1:
            if cv2.contourArea(cs[j]) >= min_area:
                holes.append(cv2.approxPolyDP(cs[j], eps, True)[:, 0, :].astype(np.float64))
            j = hier[j][0]
        if len(outer) < 3:
            continue
        if snap:
            outer, holes = _snap(outer), [_snap(h) for h in holes if len(h) >= 3]
        out.append((outer, holes))
    return out


# ---------------------------------------------------------------- reconstruction

@dataclass
class Params:
    """Tuned on CubiCasa val (see the module docstring); the defaults are the chosen values."""
    wall_thr: float = 0.5        # P(wall) above this is wall
    eps: float = 0.3             # polygon simplification, x wall thickness
    snap: bool = True            # snap near-axis edges to exact horizontal / vertical
    close: float = 0.35          # closing of small wall gaps, x wall thickness (door openings are ~4.5 x)
    min_room: float = 1.5        # smallest room, in (wall thickness)^2 units x 10 (1.5 -> ~0.06 m2 at 0.2 m walls)
    target_thick: float = 8.0    # wall thickness (px) the network sees best: CubiCasa's median
    band: float = 0.8            # half-height of an opening cut through a wall, x wall thickness
    frame: float = 0.04          # edge band (share of the short side) where a wall piece is taken for the sheet frame
    background: bool = True      # clear solid margins / site fill reaching the image edge before the network
    walls: str = "union"         # pixel (wall map) | vector (the network's own vectoriser) | union of both
    min_factor: float = 1.0      # never shrink a plan below its own size
    scale_mode: str = "long"     # long: long side to target_long px | confidence: the scale the network is surest of walls
    target_long: float = 1250.0
    max_long: float = 1800.0     # larger plans are shrunk to this: the 2019 vectoriser takes minutes on 2,500 px
    lines: bool = True           # add thin drawn partitions (long straight ink lines meeting a wall)
    line_m: float = 2.5          # shortest such line, metres
    split: float = 1.0           # split a room when two network room types each cover this share (1 = never)
    regular: bool = True         # straighten walls into bands of even thickness at the plan's own wall angles
    keep_share: float = 0.05     # building parts smaller than this share of the largest are stray marks (legend, scale bar)
    crop: bool = True            # run on the building's part of a large sheet (title block, legend, margins left out)
    door_place: str = "profile_all"  # profile_all: doors and windows in axis-aligned walls by the wall profile around the box (val: doors 0.89 vs 0.79 by band overlap); angled walls always by band


@dataclass
class Opening:
    kind: str                    # door | window
    index: int | None            # detector box index (the 3D id), None for the network's own icons
    rect: list                   # 4 corners in original pixels: the cut through the wall
    horizontal: bool             # the wall it sits in runs horizontally
    on_wall: bool                # a wall was found around it
    width: float                 # opening width along the wall, original pixels


@dataclass
class Plan:
    factor: float                # working pixels per original pixel
    thick: float                 # wall thickness, original pixels
    walls: list                  # [(outer, [holes])] original pixels
    rooms: list                  # [{"poly", "holes", "cls", "area"}] original pixels
    openings: list               # [Opening]
    slabs: list = field(default_factory=list)  # [(outer, [])] the building footprint: one continuous floor


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    import cv2

    m = mask.astype(np.uint8)
    h, w = m.shape
    flood = np.pad(m, 1).copy()
    cv2.floodFill(flood, None, (0, 0), 1)
    return (m | (1 - flood[1:-1, 1:-1])).astype(bool)


def ink_partitions(rgb: np.ndarray, W: np.ndarray, building: np.ndarray, t: float, length: float) -> np.ndarray:
    """Thin drawn partitions the network misses: long straight horizontal / vertical ink lines inside the building
    that meet a known wall at an end. `length` is the shortest line taken (working pixels, ~2.5 m); desks, chairs and
    text are shorter. Returned as a wall mask about 0.6 wall thicknesses thick."""
    import cv2

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    ink = cv2.adaptiveThreshold(gray, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, _odd(max(15, 3 * t)), 18)
    ink &= building.astype(np.uint8)
    L = max(15, int(length))
    half = max(1, int(round(0.3 * t)))
    cands = []  # (horizontal, x, y, w, h) of every lone long line
    for kern, horizontal in ((cv2.getStructuringElement(cv2.MORPH_RECT, (L, 1)), True),
                             (cv2.getStructuringElement(cv2.MORPH_RECT, (1, L)), False)):
        lines = cv2.morphologyEx(ink, cv2.MORPH_OPEN, kern)
        n, lab, st, _ = cv2.connectedComponentsWithStats(lines, 8)
        segs = [(i, *st[i][:4]) for i in range(1, n) if (st[i][3] if horizontal else st[i][2]) <= 1.5 * t]  # lines, not areas
        # stairs and hatching: many parallel lines close together. A partition stands alone.
        gap = length * 0.25                          # ~0.6 m
        def pos(s): return s[2] + s[4] / 2 if horizontal else s[1] + s[3] / 2           # noqa: E306
        def span(s): return (s[1], s[1] + s[3]) if horizontal else (s[2], s[2] + s[4])  # noqa: E306
        def ln(s): return span(s)[1] - span(s)[0]  # noqa: E306
        # neighbours of a similar length: a long wall next to short stair treads is still a wall
        crowd = {s[0] for s in segs if sum(1 for o in segs if o is not s and abs(pos(o) - pos(s)) <= gap
                                            and min(span(o)[1], span(s)[1]) - max(span(o)[0], span(s)[0]) > 0
                                            and ln(o) >= 0.6 * ln(s)) >= 2}
        cands += [(horizontal, x, y, w, h) for i, x, y, w, h in segs if i not in crowd]
    out = np.zeros(W.shape, bool)
    reach = _odd(1.5 * t)
    for _ in range(4):  # a partition meets a wall, or a partition already taken (office grids end on each other)
        near = cv2.dilate((W | out).astype(np.uint8), np.ones((reach, reach), np.uint8)).astype(bool)
        added = []
        for c in cands:
            horizontal, x, y, w, h = c
            ends = [(y + h // 2, x), (y + h // 2, x + w - 1)] if horizontal else [(y, x + w // 2), (y + h - 1, x + w // 2)]
            if not any(near[min(max(r, 0), W.shape[0] - 1), min(max(cc, 0), W.shape[1] - 1)] for r, cc in ends):
                continue
            added.append(c)
            if horizontal:
                yc = y + h // 2
                out[max(0, yc - half):yc + half + 1, x:x + w] = True
            else:
                xc = x + w // 2
                out[y:y + h, max(0, xc - half):xc + half + 1] = True
        if not added:
            break
        cands = [c for c in cands if c not in added]
    return out & ~W


def _drop_frame(W: np.ndarray, band: float, room: np.ndarray | None = None) -> np.ndarray:
    """Drop wall pieces lying mostly in a band along the image edge: the sheet's border frame, not the building."""
    import cv2

    if band <= 0:
        return W
    h, w = W.shape
    b = max(2, int(band * min(h, w)))
    edge = np.zeros_like(W)
    edge[:b], edge[-b:], edge[:, :b], edge[:, -b:] = True, True, True, True
    n, lab = cv2.connectedComponents(W.astype(np.uint8), connectivity=8)
    if n <= 1:
        return W
    tot = np.bincount(lab.ravel(), minlength=n)
    inb = np.bincount(lab[edge], minlength=n)
    frame = (inb >= 0.6 * np.maximum(tot, 1))
    frame[0] = False
    W = W & ~frame[lab]
    if room is None:
        return W
    # a frame joined to the building at a corner: a long wall line in the edge band with paper, not rooms, inside it
    inside = (room != 0) & (room != 2)
    t = max(2, int(0.006 * min(h, w)))
    out = W.copy()
    for axis in (0, 1):
        n_lines = h if axis == 0 else w
        cover = W.mean(axis=1) if axis == 0 else W.mean(axis=0)
        for i in list(range(b)) + list(range(n_lines - b, n_lines)):
            if cover[i] < 0.5:
                continue
            step = 1 if i < b else -1
            a0, a1 = sorted((i + step * 2 * t, i + step * 4 * t))
            a0, a1 = max(0, a0), min(n_lines, a1)
            strip = inside[a0:a1] if axis == 0 else inside[:, a0:a1]
            if strip.size and strip.mean() < 0.3:
                if axis == 0:
                    out[i] = False
                else:
                    out[:, i] = False
    return _drop_small(out, 4 * t * t)


def _place(W: np.ndarray, x0, y0, x1, y1, t: float, band: float) -> tuple[bool, bool, tuple[int, int, int, int]]:
    """Find the wall an opening box sits in: (on_wall, horizontal, cut rect x0, y0, x1, y1) in working pixels."""
    H, Wd = W.shape
    e = max(2, int(round(0.9 * t)))
    X0, Y0, X1, Y1 = max(0, int(x0) - e), max(0, int(y0) - e), min(Wd, int(np.ceil(x1)) + e), min(H, int(np.ceil(y1)) + e)
    R = W[Y0:Y1, X0:X1]
    if R.size == 0 or R.sum() < max(4, t * t):
        return False, (x1 - x0) >= (y1 - y0), (int(x0), int(y0), int(x1), int(y1))
    k = max(1, int(round(t)))
    rows = np.convolve(R.sum(1).astype(float), np.ones(k), "same")    # wall pixels in a horizontal band of height t
    cols = np.convolve(R.sum(0).astype(float), np.ones(k), "same")
    sh, sv = rows.max() / max(R.shape[1], 1), cols.max() / max(R.shape[0], 1)
    hb = band * t
    if sh >= sv:
        yc = Y0 + int(rows.argmax())
        return True, True, (int(x0), int(yc - hb), int(np.ceil(x1)), int(np.ceil(yc + hb)))
    xc = X0 + int(cols.argmax())
    return True, False, (int(xc - hb), int(y0), int(np.ceil(xc + hb)), int(np.ceil(y1)))


SCALES = (1.0, 1.5, 2.2, 3.0)
PROBE = 900  # working-pixel side of the crop each candidate scale is tried on


def choose_factor(rgb: np.ndarray, params: Params = Params()) -> float:
    """The enlargement at which the network is surest about walls (mean P(wall) over its wall pixels).

    Measured on CubiCasa val, plans shrunk to 45% (an internet-sized image): picking this way reaches wall F1 0.913
    against 0.920 for the best scale per plan, and 0.845 without enlarging. Each candidate runs on a central crop of
    about PROBE x PROBE working pixels, so trying four scales costs about one full pass."""
    h, w = rgb.shape[:2]
    if params.scale_mode == "long":  # small plans up to about target_long px; big ones down to max_long (CPU time)
        return float(min(np.clip(params.target_long / max(h, w), params.min_factor, 3.0), params.max_long / max(h, w)))
    cands = [c for c in SCALES if c >= params.min_factor and max(h, w) * c <= MAX_SIDE * 1.15]
    if len(cands) <= 1:
        return cands[0] if cands else min(1.0, MAX_SIDE / max(h, w))
    ink = rgb.min(axis=2) < 200
    ys, xs = np.nonzero(ink)
    cy, cx = (int(np.median(ys)), int(np.median(xs))) if len(ys) else (h // 2, w // 2)
    best, best_c = -1.0, cands[0]
    for c in cands:
        side = int(PROBE / c)
        y0, x0 = min(max(0, cy - side // 2), max(0, h - side)), min(max(0, cx - side // 2), max(0, w - side))
        m = network_maps(rgb[y0:y0 + side, x0:x0 + side], c, vector=False)
        sel = m.wall > params.wall_thr
        score = float(m.wall[sel].mean()) if sel.sum() >= 50 else 0.0
        if score > best:
            best, best_c = score, c
    return best_c


def building_box(clean: np.ndarray, params: Params = Params()) -> tuple[int, int, int, int] | None:
    """Where the building is on the sheet (x0, y0, x1, y1 in pixels, with a margin), from one quick low-resolution
    pass; None when it already fills most of the image. Sheets with a title block, legend and wide margins then give
    the network all of their resolution budget for the building itself."""
    import cv2

    h, w = clean.shape[:2]
    m = network_maps(clean, min(1.0, 1000 / max(h, w)), vector=False)
    W = _drop_frame(_drop_small(m.wall > params.wall_thr, 30), params.frame, m.room)
    t = thickness(W) or 4.0
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(4 * t),) * 2)
    fp = _fill_holes(cv2.morphologyEx((W | ((m.room != 0) & (m.room != 2))).astype(np.uint8), cv2.MORPH_CLOSE, k).astype(bool))
    n, lab, st, _ = cv2.connectedComponentsWithStats(fp.astype(np.uint8), 4)
    if n <= 1:
        return None
    areas = st[1:, cv2.CC_STAT_AREA]
    keep = [i + 1 for i in range(len(areas)) if areas[i] >= params.keep_share * areas.max()]
    xs0 = min(st[i, 0] for i in keep); ys0 = min(st[i, 1] for i in keep)
    xs1 = max(st[i, 0] + st[i, 2] for i in keep); ys1 = max(st[i, 1] + st[i, 3] for i in keep)
    mh, mw = m.wall.shape
    if (xs1 - xs0) * (ys1 - ys0) >= 0.6 * mh * mw:
        return None
    pad = 0.04 * max(mh, mw)
    f = m.factor
    return (max(0, int((xs0 - pad) / f)), max(0, int((ys0 - pad) / f)),
            min(w, int(np.ceil((xs1 + pad) / f))), min(h, int(np.ceil((ys1 + pad) / f))))


def _shift(plan: Plan, dx: float, dy: float) -> Plan:
    o = np.array([dx, dy])
    walls = [(a + o, [h + o for h in hs]) for a, hs in plan.walls]
    rooms = [{**r, "poly": r["poly"] + o, "holes": [h + o for h in r["holes"]]} for r in plan.rooms]
    ops = [Opening(x.kind, x.index, (np.asarray(x.rect) + o).tolist(), x.horizontal, x.on_wall, x.width) for x in plan.openings]
    slabs = [(a + o, [h + o for h in hs]) for a, hs in plan.slabs]
    return Plan(plan.factor, plan.thick, walls, rooms, ops, slabs)


def reconstruct(rgb: np.ndarray, boxes: list | None = None, factor: float | None = None, maps: Maps | None = None,
                params: Params = Params()) -> Plan:
    """Walls, floor slab, rooms and openings of one plan image. boxes: [(index, kind, x0, y0, x1, y1)] in original
    pixels, kind door | window (the detector boxes the answers count); without boxes the network's own door/window
    maps are used. Everything returned is in original-image pixels."""
    P = params
    clean = normalize_polarity(rgb)
    if P.background:
        clean = clean_background(clean)
    if maps is None and P.crop:
        box = building_box(clean, P)
        if box is not None:  # work on the building's part of the sheet, then put everything back where it was
            x0, y0, x1, y1 = box
            sub = [(i, k, bx0 - x0, by0 - y0, bx1 - x0, by1 - y0) for i, k, bx0, by0, bx1, by1 in boxes] if boxes is not None else None
            return _shift(_reconstruct(clean[y0:y1, x0:x1].copy(), sub, factor, None, P), x0, y0)
    return _reconstruct(clean, boxes, factor, maps, P)


def _reconstruct(clean: np.ndarray, boxes: list | None, factor: float | None, maps: Maps | None, P: Params) -> Plan:
    import cv2

    if maps is None:
        maps = network_maps(clean, factor if factor is not None else choose_factor(clean, P))
    f = maps.factor
    W = maps.wall > P.wall_thr
    t = thickness(_drop_small(W, 30)) or P.target_thick
    W = _drop_small(W, max(20.0, (1.5 * t) ** 2))
    vec = maps.extra.get("vec_walls") or []
    if P.walls in ("vector", "union") and vec:
        V = np.zeros(W.shape, np.uint8)
        for p in vec:
            if len(p) >= 3:
                cv2.fillPoly(V, [np.round(p).astype(np.int32)], 1)
        W = V.astype(bool) if P.walls == "vector" else (W | V.astype(bool))
    W = _drop_frame(W, P.frame, maps.room)
    if P.close > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (_odd(P.close * t * 2), _odd(P.close * t * 2)))
        W = cv2.morphologyEx(W.astype(np.uint8), cv2.MORPH_CLOSE, k).astype(bool)
    if P.lines and boxes is not None:
        doors = sorted(max(x1 - x0, y1 - y0) * f for _, kind, x0, y0, x1, y1 in boxes if kind == "door")
        ppm = (doors[len(doors) // 2] / 0.9) if len(doors) >= 2 else t / 0.2   # working pixels per metre
        work = cv2.resize(clean, (W.shape[1], W.shape[0]), interpolation=cv2.INTER_CUBIC) if clean.shape[:2] != W.shape else clean
        # inside the building: what the walls enclose once gaps up to ~1 m are bridged (a wing open onto a courtyard)
        k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(5 * t),) * 2)
        building0 = _fill_holes(cv2.morphologyEx((W | (maps.room != 0)).astype(np.uint8), cv2.MORPH_CLOSE, k5).astype(bool))
        W = _drop_frame(W | ink_partitions(work, W, building0, t, P.line_m * ppm), P.frame, maps.room)
    bands: list[Band] = []
    if P.regular:
        W, bands = regularize(W, t)

    # ---- openings: each detector door / window in the wall it sits in, cut through it at the wall's own angle
    if boxes is None:
        boxes = []
        for kind, prob in (("door", maps.door), ("window", maps.window)):
            n, lab, st, _ = cv2.connectedComponentsWithStats((prob > 0.5).astype(np.uint8), 8)
            for i in range(1, n):
                x, y, w, h, a = st[i]
                if a >= (0.8 * t) ** 2:
                    boxes.append((None, kind, x / f, y / f, (x + w) / f, (y + h) / f))
    openings, cuts = [], []
    for idx, kind, x0, y0, x1, y1 in boxes:
        bx = (x0 * f, y0 * f, x1 * f, y1 * f)
        # doors: the wall-profile placement (val: door F1 0.88 vs 0.79 by band overlap) unless the wall runs at an angle
        hit = place_on_band(bands, bx, t, None, kind) if bands else None
        if hit is not None and (kind == "door" or P.door_place == "profile_all") and P.door_place.startswith("profile")                 and _axis_aligned(hit[0].axis):
            hit = None
        if hit is not None:
            band, s0, s1 = hit
            ax = band.axis
            nrm = np.array([-ax[1], ax[0]]) * (band.th / 2 + 1.0)
            q0, q1 = band.p0 + ax * s0, band.p0 + ax * s1
            poly = np.array([q0 + nrm, q1 + nrm, q1 - nrm, q0 - nrm])
            on, horiz, width = True, bool(abs(ax[0]) >= abs(ax[1])), s1 - s0
        else:
            on, horiz, (cx0, cy0, cx1, cy1) = _place(W, *bx, t, P.band)
            poly = np.array([[cx0, cy0], [cx1, cy0], [cx1, cy1], [cx0, cy1]], np.float64)
            width = (cx1 - cx0) if horiz else (cy1 - cy0)
        if on:
            cuts.append(poly)
        openings.append(Opening(kind, idx, (poly / f).tolist(), horiz, on, float(width) / f))
    closed = W.astype(np.uint8)   # walls with every opening shut: what separates rooms
    cut = W.astype(np.uint8)      # walls with the openings cut out: what is extruded
    for poly in cuts:
        q = [np.round(poly).astype(np.int32)]
        cv2.fillPoly(closed, q, 1)
        cv2.fillPoly(cut, q, 0)
    closed, cut = closed.astype(bool), cut.astype(bool)

    # ---- the building: one floor slab under everything the walls enclose; stray marks outside it are dropped
    shell = _fill_holes(cv2.morphologyEx(closed.astype(np.uint8), cv2.MORPH_CLOSE,
                                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(2 * t),) * 2)).astype(bool))
    # walls with gaps up to ~1.2 m bridged define the building; the network's room map may only fill in next to that
    # (a missing wall), never spill onto the paper around it
    wide = _fill_holes(cv2.morphologyEx(closed.astype(np.uint8), cv2.MORPH_CLOSE,
                                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(6 * t),) * 2)).astype(bool))
    inside = _fill_holes((maps.room != 0) & (maps.room != 2))
    n, lab = cv2.connectedComponents(inside.astype(np.uint8), connectivity=4)
    touching = np.unique(lab[(shell | wide) & (lab > 0)])
    inside = np.isin(lab, touching[touching > 0])
    # blank paper is not building, even where the network calls it a room: keep its rooms only where something is drawn
    g = cv2.resize(cv2.cvtColor(clean, cv2.COLOR_RGB2GRAY), (W.shape[1], W.shape[0]), interpolation=cv2.INTER_AREA)
    ink = cv2.blur((g < 200).astype(np.float32), (_odd(8 * t),) * 2) > 0.01
    # the slab may be generous (its job: the floor never breaks); rooms use the stricter outline, without the wide
    # gap bridging that also fills concave corners outside the walls
    floor = _fill_holes(shell | wide | (inside & ink))  # anything the building's floor surrounds is floor: no holes
    fp = _fill_holes(shell | (inside & ink))
    fp = cv2.morphologyEx(fp.astype(np.uint8), cv2.MORPH_OPEN, np.ones((_odd(0.8 * t),) * 2, np.uint8)).astype(bool)
    n, lab, st, _ = cv2.connectedComponentsWithStats(fp.astype(np.uint8), 4)
    if n > 1:
        areas = st[1:, cv2.CC_STAT_AREA]
        keep = np.zeros(n, bool)
        keep[1:] = areas >= max(P.keep_share * areas.max(), 30 * t * t)
        fp = keep[lab]
    reach = cv2.dilate(fp.astype(np.uint8), np.ones((_odd(t),) * 2, np.uint8)).astype(bool)
    cut &= reach
    closed &= reach
    dirs = wall_directions(W) if P.regular else [0.0, 90.0]
    # the floor outline: smoothed where no wall defines it (no ragged edge from the network's room map), then straight
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(1.5 * t),) * 2)
    floor = _fill_holes((floor & cv2.dilate(fp.astype(np.uint8), np.ones((_odd(6 * t),) * 2, np.uint8)).astype(bool)) | fp)
    smooth = cv2.morphologyEx(cv2.morphologyEx(floor.astype(np.uint8), cv2.MORPH_CLOSE, k), cv2.MORPH_OPEN, k).astype(bool)
    slabs = [(snap_dirs(o, dirs) / f, []) for o, _ in polygons(smooth | fp | (floor & closed), max(1.0, 1.0 * t), 30 * t * t, False)]

    weps = 1.0 if P.regular else max(1.0, P.eps * t)
    walls = [(o / f, [h / f for h in hs]) for o, hs in polygons(cut, weps, (0.8 * t) ** 2, P.snap and not P.regular)]

    # ---- rooms: the floor the walls divide (doorways shut), named by the network's majority room type
    free = fp & ~cv2.dilate(closed.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    n, lab = cv2.connectedComponents(free.astype(np.uint8), connectivity=4)
    rooms = []
    min_area = P.min_room * 10 * t * t
    for i in range(1, n):
        m = lab == i
        a = int(m.sum())
        if a < min_area:
            continue
        cls = np.bincount(maps.room[m], minlength=len(ROOM_CLASSES))
        cls[[0, 2, 8]] = 0                      # background, wall, railing never name a room
        name = ROOM_CLASSES[int(cls.argmax())] if cls.sum() else "Undefined"
        for o, hs in polygons(m, max(1.0, 0.5 * t), min_area / 2, P.snap and not P.regular)[:1]:
            # straight room edges on the plan's own wall directions: an edge never follows pixel noise
            rooms.append({"poly": snap_dirs(o, dirs) / f, "holes": [snap_dirs(h, dirs) / f for h in hs], "cls": name,
                          "area": a / f / f})
    return Plan(f, t / f, walls, rooms, openings, slabs)


def _smooth_classes(room: np.ndarray, k: int) -> np.ndarray:
    """Room-type map with speckles removed: each pixel takes the type most present in a k x k window."""
    import cv2

    best = np.full(room.shape, -1.0, np.float32)
    out = np.zeros(room.shape, np.uint8)
    for c in range(len(ROOM_CLASSES)):
        if c in (0, 2, 8) or not (room == c).any():
            continue
        s = cv2.blur((room == c).astype(np.float32), (k, k))
        better = s > best
        out[better], best[better] = c, s[better]
    return out


# ---------------------------------------------------------------- persistence (one reconstruction per image)

def to_json(plan: Plan) -> dict:
    L = lambda a: np.asarray(a, float).round(2).tolist()  # noqa: E731
    return {"factor": plan.factor, "thick": plan.thick, "walls": [[L(o), [L(h) for h in hs]] for o, hs in plan.walls],
            "rooms": [{"poly": L(r["poly"]), "holes": [L(h) for h in r["holes"]], "cls": r["cls"], "area": r["area"]}
                      for r in plan.rooms],
            "openings": [{"kind": o.kind, "index": o.index, "rect": L(o.rect), "horizontal": o.horizontal,
                          "on_wall": o.on_wall, "width": o.width} for o in plan.openings],
            "slabs": [[L(o), [L(h) for h in hs]] for o, hs in plan.slabs]}


def from_json(d: dict) -> Plan:
    A = lambda a: np.asarray(a, float)  # noqa: E731
    return Plan(d["factor"], d["thick"], [(A(o), [A(h) for h in hs]) for o, hs in d["walls"]],
                [{"poly": A(r["poly"]), "holes": [A(h) for h in r["holes"]], "cls": r["cls"], "area": r["area"]} for r in d["rooms"]],
                [Opening(o["kind"], o["index"], o["rect"], o["horizontal"], o["on_wall"], o["width"]) for o in d["openings"]],
                [(A(o), [A(h) for h in hs]) for o, hs in d.get("slabs", [])])


# ---------------------------------------------------------------- straight walls at any angle

@dataclass
class Band:
    """One straight wall: centreline p0 -> p1 (working pixels) and thickness."""
    p0: np.ndarray
    p1: np.ndarray
    th: float

    @property
    def axis(self) -> np.ndarray:
        d = self.p1 - self.p0
        n = float(np.hypot(*d))
        return d / n if n else np.array([1.0, 0.0])

    @property
    def length(self) -> float:
        return float(np.hypot(*(self.p1 - self.p0)))

    def poly(self, extend: float = 0.0) -> np.ndarray:
        a = self.axis
        nrm = np.array([-a[1], a[0]]) * self.th / 2
        p0, p1 = self.p0 - a * extend, self.p1 + a * extend
        return np.array([p0 + nrm, p1 + nrm, p1 - nrm, p0 - nrm])


def normalize_polarity(rgb: np.ndarray) -> np.ndarray:
    """Light lines on a dark ground (blueprints, inverted scans) -> dark lines on white, in grey."""
    lum = rgb.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
    if np.median(lum) >= 110:
        return rgb
    g = (255 - lum).clip(0, 255).astype(np.uint8)
    return np.stack([g, g, g], axis=-1)


def _rot(img: np.ndarray, deg: float, nearest: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Rotate by deg (OpenCV convention) on a canvas large enough to keep everything; returns (image, 2x3 matrix)."""
    import cv2

    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    c, s = abs(M[0, 0]), abs(M[0, 1])
    W2, H2 = int(np.ceil(h * s + w * c)), int(np.ceil(h * c + w * s))
    M[0, 2] += W2 / 2 - w / 2
    M[1, 2] += H2 / 2 - h / 2
    out = cv2.warpAffine(img, M, (W2, H2), flags=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR)
    return out, M


def wall_directions(W: np.ndarray, min_rise: float = 0.1) -> list[float]:
    """Rotations (degrees, OpenCV convention) that each lay one family of walls horizontal.

    Projection sharpness: rotate the wall map through every angle and score how strongly its rows concentrate (sum of
    squared row totals); walls lying along the rows give a peak. A rectilinear plan peaks at 0 and 90, a scan a few
    degrees off at its skew, angled wings at their own angles."""
    import cv2

    k = 400 / max(W.shape)
    m = cv2.resize(W.astype(np.float32), None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else W.astype(np.float32)
    if m.sum() == 0:
        return [0.0, 90.0]

    def score(r: float) -> float:
        rows = _rot(m, r)[0].sum(axis=1)
        return float((rows ** 2).sum())
    coarse = np.arange(-90.0, 90.0, 1.0)
    sc = np.array([score(r) for r in coarse])
    base, top = np.median(sc), sc.max()
    peaks: list[float] = []
    for i in np.argsort(-sc):
        if sc[i] < base + min_rise * (top - base):
            break
        r = coarse[i]
        if sc[i] >= sc[(i - 1) % len(sc)] and sc[i] >= sc[(i + 1) % len(sc)] and                 all(min(abs(r - q), 180 - abs(r - q)) >= 12 for q in peaks):
            fine = np.arange(r - 1.0, r + 1.01, 0.2)
            peaks.append(round(float(fine[int(np.argmax([score(f) for f in fine]))]), 1))
        if len(peaks) >= 6:
            break
    # walls come in perpendicular pairs: a wing at 20 degrees has walls at 110 too, often too few to peak on their own
    out: list[float] = []
    for p in [x for pk in peaks for x in (pk, pk + 90)]:  # strongest first, each with its perpendicular
        q = ((p + 90) % 180) - 90
        if all(min(abs(q - o), 180 - abs(q - o)) >= 3 for o in out):
            out.append(round(q, 1))
    return out or [0.0, 90.0]


SUPPORT = 0.7  # share of a straightened wall that must be wall in the network's map


def extract_bands(W: np.ndarray, t: float, rots: list[float]) -> tuple[list[Band], np.ndarray]:
    """Straight walls in the wall map: for each wall direction, the horizontal runs of the map rotated to lie flat,
    each turned into a band (centreline + thickness). Pieces on one line with a small gap (< 0.8 wall) are joined;
    door openings (~4 walls wide) are kept open. Returns (bands, mask of what they cover)."""
    import cv2

    bands: list[Band] = []
    L = max(3, int(round(1.5 * t)))
    left = W.astype(np.uint8)   # wall pixels no band explains yet: a wall is taken once, at its strongest direction
    for r in rots:
        R, M = _rot(left, r, nearest=True)
        Mi = cv2.invertAffineTransform(M)
        H = cv2.morphologyEx(R, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (L, 1)))
        n, lab, st, _ = cv2.connectedComponentsWithStats(H, 8)
        found = []  # (yc, x0, x1, th) in the rotated frame
        for i in range(1, n):
            x, y, w, h, _ = st[i]
            if w < 1.2 * t:
                continue
            sub = lab[y:y + h, x:x + w] == i
            rows = sub.sum(axis=1)
            j = int(rows.argmax())
            thr = 0.5 * rows[j]
            y0 = j
            while y0 > 0 and rows[y0 - 1] >= thr:
                y0 -= 1
            y1 = j
            while y1 < h - 1 and rows[y1 + 1] >= thr:
                y1 += 1
            th = y1 - y0 + 1
            if th > 2.5 * t:          # a filled area, not a wall: left to the residual pass
                continue
            xs = np.nonzero(sub[y0:y1 + 1].any(axis=0))[0]
            found.append([y + (y0 + y1 + 1) / 2, x + xs.min(), x + xs.max() + 1, float(th)])
        found.sort(key=lambda b: (round(b[0] / max(t * 0.5, 1)), b[1]))
        merged: list[list] = []
        for b in found:                # join pieces on one line separated by a small gap
            for m in merged:
                if abs(m[0] - b[0]) <= 0.5 * t and b[1] - m[2] <= 0.8 * t and m[1] - b[2] <= 0.8 * t:
                    wa, wb = m[2] - m[1], b[2] - b[1]
                    m[0] = (m[0] * wa + b[0] * wb) / (wa + wb)
                    m[3] = (m[3] * wa + b[3] * wb) / (wa + wb)
                    m[1], m[2] = min(m[1], b[1]), max(m[2], b[2])
                    break
            else:
                merged.append(list(b))
        for yc, x0, x1, th in merged:
            pts = np.array([[x0, yc], [x1, yc]], np.float64)
            back = np.hstack([pts, np.ones((2, 1))]) @ Mi.T
            b = Band(back[0], back[1], float(np.clip(th, max(2.0, 0.3 * t), 1.8 * t)))
            area = np.zeros(W.shape, np.uint8)
            cv2.fillPoly(area, [np.round(b.poly()).astype(np.int32)], 1)
            inside = area.astype(bool)
            if inside.sum() == 0 or W[inside].mean() < SUPPORT:  # a band must be wall in the map, never a room
                continue
            bands.append(b)
            left[cv2.dilate(area, np.ones((3, 3), np.uint8)).astype(bool)] = 0
    mask = np.zeros(W.shape, np.uint8)
    for b in bands:
        _fit_ends(b, W)
        cv2.fillPoly(mask, [np.round(b.poly()).astype(np.int32)], 1)
    return bands, mask.astype(bool)


def _reach(W: np.ndarray, p: np.ndarray, d: np.ndarray, most: float) -> float:
    """How far the wall map continues from p in direction d (consecutive wall pixels), up to `most` pixels."""
    h, w = W.shape
    k = 0.0
    while k < most:
        x, y = int(round(p[0] + d[0] * (k + 1))), int(round(p[1] + d[1] * (k + 1)))
        if not (0 <= x < w and 0 <= y < h) or not W[y, x]:
            break
        k += 1.0
    return k


def _fit_ends(b: Band, W: np.ndarray) -> None:
    """Extend a band's ends only as far as the wall map continues (into the perpendicular wall at a corner or a
    junction, up to one wall thickness): corners close, free ends (door jambs, wall ends) get no stub."""
    a = b.axis
    most = 1.0 * b.th
    b.p0 = b.p0 - a * _reach(W, b.p0, -a, most)
    b.p1 = b.p1 + a * _reach(W, b.p1, a, most)


CORNER_REACH = 3.0  # walls ending this many wall thicknesses (~0.6 m) short of another wall are joined to it


def _cross(a0: np.ndarray, ad: np.ndarray, b0: np.ndarray, bd: np.ndarray) -> tuple[float, float] | None:
    """Parameters (s along a, u along b) where the lines a0 + s*ad and b0 + u*bd meet; None if near parallel."""
    den = ad[0] * bd[1] - ad[1] * bd[0]
    if abs(den) < np.sin(np.radians(20)):
        return None
    d = b0 - a0
    return float((d[0] * bd[1] - d[1] * bd[0]) / den), float((d[0] * ad[1] - d[1] * ad[0]) / den)


def close_corners(bands: list[Band], t: float, reach: float = CORNER_REACH) -> int:
    """Join walls that stop short of each other: an end within `reach` wall thicknesses of another, non-parallel wall
    is extended to where their centrelines cross (plus half that wall's thickness, so the corner is solid). L-corners
    (both stop short) and T-junctions (one runs into the other's side) close at any angle; a doorway (~4.5 walls
    wide) is never bridged. Returns how many ends moved."""
    moved = 0
    r = reach * t
    for a in bands:
        for end in (0, 1):
            e = a.p0 if end == 0 else a.p1
            out = -a.axis if end == 0 else a.axis              # pointing away from the wall, past its end
            best = None
            for b in bands:
                if b is a:
                    continue
                hit = _cross(e, out, b.p0, b.axis)
                if hit is None:
                    continue
                s_out, u = hit
                if not (-0.5 * t - b.th / 2 <= s_out <= r):  # the other wall's line is just past this end
                    continue
                if not (-r <= u <= b.length + r):            # ...and its wall (or its own short end) is there
                    continue
                if best is None or s_out < best[0]:
                    best = (s_out, b, u)
            if best is None:
                continue
            s_out, b, u = best
            grow = s_out + b.th / 2
            if grow > 0.5:
                if end == 0:
                    a.p0 = a.p0 - a.axis * grow
                else:
                    a.p1 = a.p1 + a.axis * grow
                moved += 1
            if u < 0:                                        # the other wall stops short too: an L-corner, grow it
                b.p0 = b.p0 + b.axis * (u - a.th / 2)
            elif u > b.length:
                b.p1 = b.p1 + b.axis * (u - b.length + a.th / 2)
    return moved


def regularize(W: np.ndarray, t: float, rots: list[float] | None = None) -> tuple[np.ndarray, list[Band]]:
    """The wall map as straight walls of even thickness meeting at clean corners, at whatever angles the plan uses.
    What is not a straight run (columns, curved walls, stubs) stays as drawn if it is big enough to matter."""
    import cv2

    rots = rots if rots is not None else wall_directions(W)
    bands, covered = extract_bands(W, t, rots)
    # what no band explains: junction fillers, short walls, columns, curves. Hairline slivers along band edges are
    # opened away; every real piece is kept, so straightening never opens a gap between rooms
    rest = cv2.morphologyEx((W & ~covered).astype(np.uint8), cv2.MORPH_OPEN,
                            np.ones((3, 3), np.uint8)).astype(bool)
    out = covered.copy()
    n, lab, st, _ = cv2.connectedComponentsWithStats(rest.astype(np.uint8), 8)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < 0.25 * t * t:
            continue
        comp = (lab == i).astype(np.uint8)
        cs, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        (cx, cy), (w, h), a = cv2.minAreaRect(max(cs, key=cv2.contourArea))
        if max(w, h) >= 3 * min(w, h) and max(w, h) >= 2 * t and min(w, h) <= 2.5 * t:  # a straight piece at its own angle
            ang = np.radians(a if w >= h else a + 90)
            d = np.array([np.cos(ang), np.sin(ang)]) * max(w, h) / 2
            c = np.array([cx, cy])
            b = Band(c - d, c + d, float(np.clip(min(w, h), max(2.0, 0.3 * t), 1.8 * t)))
            _fit_ends(b, W)
            bands.append(b)
            cv2.fillPoly(out.view(np.uint8), [np.round(b.poly()).astype(np.int32)], 1)
        else:
            out |= comp.astype(bool)
    if close_corners(bands, t):  # walls that stop short of each other meet, so every corner is closed
        m = out.view(np.uint8)
        for b in bands:
            cv2.fillPoly(m, [np.round(b.poly()).astype(np.int32)], 1)
    return out, bands


def snap_dirs(poly: np.ndarray, rots: list[float], tol: float = 8.0, rounds: int = 8) -> np.ndarray:
    """Turn polygon edges lying within tol degrees of one of the plan's wall directions onto that direction, at any
    angle (rots as from wall_directions: rotation r lays walls of image direction (cos r, sin r) flat). Each vertex
    moves to the mean of where its two edges put it."""
    p = np.asarray(poly, np.float64).copy()
    n = len(p)
    if n < 3 or not rots:
        return p
    dirs = [np.array([np.cos(np.radians(r)), np.sin(np.radians(r))]) for r in rots]
    cos_tol = np.cos(np.radians(tol))
    for _ in range(rounds):
        acc, cnt = np.zeros_like(p), np.zeros(n)
        for i in range(n):
            a, b = p[i], p[(i + 1) % n]
            e = b - a
            L = float(np.hypot(*e))
            if L < 1e-6:
                continue
            u = e / L
            best = max(dirs, key=lambda d: abs(float(u @ d)))
            if abs(float(u @ best)) < cos_tol:
                continue
            d = best if float(u @ best) > 0 else -best
            mid = (a + b) / 2
            acc[i] += mid - d * L / 2
            acc[(i + 1) % n] += mid + d * L / 2
            cnt[i] += 1
            cnt[(i + 1) % n] += 1
        moved = cnt > 0
        p[moved] = acc[moved] / cnt[moved][:, None]
    return p


def _axis_aligned(a: np.ndarray, tol: float = 3.0) -> bool:
    return min(abs(a[0]), abs(a[1])) <= np.sin(np.radians(tol))


def place_on_band(bands: list[Band], box: tuple, t: float, W: np.ndarray | None = None,
                  kind: str = "door") -> tuple[Band, float, float] | None:
    """The wall an opening box sits in and the opening's span along that wall's line (s0, s1 from the band's p0).

    A doorway is a gap in its wall: the wall's two pieces end at the door and barely overlap the door's box, while a
    perpendicular wall along the box's side overlaps it fully. So each candidate line is scored by its span over the
    box times how empty the wall map is along that span (the gap). Windows are drawn inside the wall, so for them the
    span along the wall itself counts."""
    x0, y0, x1, y1 = box
    corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float64)
    best, best_score = None, 0.0
    for b in bands:
        a = b.axis
        nrm = np.array([-a[1], a[0]])
        d = (corners - b.p0) @ nrm                     # signed distance of the box corners from the wall's centreline
        reach = b.th / 2 + 0.5 * t
        if d.min() > reach or d.max() < -reach:        # the box does not touch the wall's strip
            continue
        s = (corners - b.p0) @ a
        s0, s1 = float(s.min()), float(s.max())
        if s1 < -1.5 * t or s0 > b.length + 1.5 * t:  # beyond the wall's ends (a doorway gap is ~1 m, ~4.5 walls)
            continue
        span = s1 - s0
        if span < 0.8 * t:
            continue
        if kind == "door" and W is not None:
            k = np.linspace(s0, s1, 24)
            pts = np.round(b.p0 + np.outer(k, a)).astype(int)
            ok = (pts[:, 0] >= 0) & (pts[:, 0] < W.shape[1]) & (pts[:, 1] >= 0) & (pts[:, 1] < W.shape[0])
            cover = float(W[pts[ok, 1], pts[ok, 0]].mean()) if ok.any() else 1.0
            score = span * (1.2 - cover)
        else:
            inside = max(0.0, min(s1, b.length) - max(s0, 0.0))
            score = inside
        if score > best_score:
            best, best_score = (b, max(s0, -1.5 * t), min(s1, b.length + 1.5 * t)), score
    return best
