"""Materials drawn as hatches: read from CAD, or recognised on the sheet image by the trained hatch classifier.

CAD (Swiss DXF plans): every HATCH entity carries its pattern name (ANSI31, SOLID...), read exactly by
stoa_qa.swiss.read_dxf; the material it stands for follows drafting convention unless the sheet says otherwise.

Images (PDF sheets rendered, scans, CubiCasa plans, uploads): a ResNet-18 trained on synthetic hatches
(stoa_qa.hatch_synth, modal_jobs/hatch_cloud.py, weights third_party/weights/hatch-<run>.pt) classifies 96 px windows
of the sheet at two scales into 17 hatch classes. Windows of one class join into regions (connected cells); each region
is a piece of evidence boxed on the sheet. What material a region is:
  * the sheet's own legend, when it has one: legend rows whose swatch the classifier puts in the same class and whose
    text names a material (PDF text layer; stoa_qa.legend finds the legend blocks)          -> 'legend'
  * otherwise drafting convention (hatch_synth.CONVENTION), stated as such                   -> 'convention'
Both are 'inferred' on the confidence heatmap: a person can see the region and decide.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import sheets
from .core import Answer, Evidence
from .hatch_synth import CLASSES, CONVENTION, PATCH
from .materials import LABEL, Query, family_set, matches

REPO = Path(__file__).resolve().parent.parent
WEIGHTS = REPO / "third_party" / "weights"
CACHE = REPO / "eval_out" / "hatch"
RUN = "hatch-h17-al"
STRIDE = 48              # window step at scale 1 (pixels of the sheet image)
SCALES = (1.0, 0.5)      # 0.5: windows of 192 px seen at 96 (coarse hatches, big tiles)
MIN_PROB = 0.80          # a window joins a region only above this class probability
MIN_CELLS = 3            # a region needs at least this many cells
MAX_SIDE = 3000          # sheet images are scanned at most this size (long side)

# what a hatch class reads as, in words, for answers
HATCH_WORDS = {"concrete": "concrete stipple", "diagonal": "diagonal section hatch", "brick": "brick-course hatch",
               "block": "concrete-block hatch", "stone": "stone hatch", "earth": "earth hatch", "gravel": "gravel hatch",
               "stipple": "fine stipple (sand / screed / plaster)", "insulation": "batt-insulation hatch", "wood": "wood-grain hatch",
               "planks": "board / plank pattern", "parquet": "parquet pattern", "tile": "tile grid", "metal": "steel hatch",
               "crosshatch": "crosshatch", "solid": "solid fill (poché)"}
# CAD pattern names (DXF) -> hatch class
CAD_PATTERN = {"ANSI31": "diagonal", "ANSI32": "metal", "ANSI33": "metal", "ANSI37": "crosshatch", "AR-CONC": "concrete",
               "SOLID": "solid", "AR-SAND": "stipple", "AR-B816": "block", "AR-B88": "block", "AR-BRSTD": "brick", "BRICK": "brick",
               "INSUL": "insulation", "EARTH": "earth", "GRAVEL": "gravel", "AR-PARQ1": "parquet", "AR-HBONE": "parquet",
               "NET": "tile", "GRID": "tile", "WOOD1": "wood", "STEEL": "metal"}


@dataclass
class Region:
    cls: str
    prob: float
    box: list[float]             # x0, y0, x1, y1 in sheet-image pixels
    cells: int
    material: str = ""           # what it stands for (legend text or convention)
    family: str | None = None
    how: str = "convention"      # legend | convention
    where: str = ""              # the room / label it sits in, when the sheet names one


# ------------------------------------------------------------------ the classifier

@lru_cache(maxsize=1)
def _model():
    import torch
    import torch.nn as nn
    import torchvision

    path = WEIGHTS / f"{RUN}.pt"
    if not path.exists():
        return None
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = torchvision.models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, len(ck["classes"]))
    m.load_state_dict(ck["state"])
    m.eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return m.to(dev), dev


def available() -> bool:
    try:
        return _model() is not None
    except Exception:  # torch / torchvision missing
        return False


def classify(patches: np.ndarray, with_features: bool = False):
    """patches: N x 96 x 96 floats in [0, 1] -> class probabilities N x 17 (and 512-d features when asked)."""
    import torch

    m, dev = _model()
    out, feats = [], []
    with torch.no_grad():
        for i in range(0, len(patches), 256):
            x = torch.from_numpy(patches[i:i + 256]).float()[:, None].to(dev)
            x = ((x - 0.5) / 0.25).repeat(1, 3, 1, 1)
            if with_features:
                h = torch.nn.Sequential(*list(m.children())[:-1])(x).flatten(1)
                feats.append(torch.nn.functional.normalize(h, dim=1).cpu().numpy())
                out.append(torch.softmax(m.fc(h), 1).cpu().numpy())
            else:
                out.append(torch.softmax(m(x), 1).cpu().numpy())
    probs = np.concatenate(out) if out else np.zeros((0, len(CLASSES)))
    return (probs, np.concatenate(feats) if feats else np.zeros((0, 512))) if with_features else probs


def _gray(img_path: Path) -> tuple[np.ndarray, float]:
    """The sheet image as greyscale floats, shrunk to MAX_SIDE; returns (image, factor image->original)."""
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    im = Image.open(img_path).convert("L")
    f = 1.0
    if max(im.size) > MAX_SIDE:
        f = max(im.size) / MAX_SIDE
        im = im.resize((round(im.width / f), round(im.height / f)), Image.LANCZOS)
    return np.asarray(im, dtype=np.float32) / 255.0, f


def _windows(g: np.ndarray, scale: float):
    """96 px windows of the image at one scale, skipping blank paper: (patches, boxes in image px)."""
    from PIL import Image

    if scale != 1.0:
        im = Image.fromarray((g * 255).astype(np.uint8)).resize((max(1, round(g.shape[1] * scale)), max(1, round(g.shape[0] * scale))),
                                                                  Image.LANCZOS)
        a = np.asarray(im, dtype=np.float32) / 255.0
    else:
        a = g
    H, W = a.shape
    ps, boxes = [], []
    for y in range(0, max(1, H - PATCH + 1), STRIDE):
        for x in range(0, max(1, W - PATCH + 1), STRIDE):
            p = a[y:y + PATCH, x:x + PATCH]
            if p.shape != (PATCH, PATCH):
                continue
            ink = (p < 0.75).mean()
            if ink < 0.02:  # blank paper: nothing to read (the "none" class, without asking the network)
                continue
            ps.append(p)
            boxes.append([x / scale, y / scale, (x + PATCH) / scale, (y + PATCH) / scale])
    return (np.stack(ps) if ps else np.zeros((0, PATCH, PATCH), np.float32)), boxes


def scan(img_path: Path, key: str | None = None) -> list[Region]:
    """Hatched regions of a sheet image (cached per image and model run)."""
    key = key or hashlib.sha1(str(img_path).encode()).hexdigest()[:16]
    cache = CACHE / RUN / f"{key}.json"
    if cache.exists():
        return [Region(**r) for r in json.loads(cache.read_text())]
    g, f = _gray(img_path)
    H, W = g.shape
    gh, gw = H // STRIDE + 1, W // STRIDE + 1
    best = np.zeros((gh, gw, len(CLASSES)), np.float32)   # per coarse cell: best probability per class over windows
    for s in SCALES:
        ps, boxes = _windows(g, s)
        if not len(ps):
            continue
        probs = classify(ps)
        for (x0, y0, x1, y1), p in zip(boxes, probs):
            # a window votes for the cells it fully covers (its centre cells at scale 1)
            cx0, cy0 = int(x0 // STRIDE) + (1 if s == 1.0 else 1), int(y0 // STRIDE) + (1 if s == 1.0 else 1)
            cx1, cy1 = max(cx0 + 1, int(x1 // STRIDE) - (0 if s == 1.0 else 1)), max(cy0 + 1, int(y1 // STRIDE) - (0 if s == 1.0 else 1))
            sl = best[cy0 - 1:cy1, cx0 - 1:cx1]
            np.maximum(sl, p[None, None, :], out=sl)
    regions = _regions(best, f)
    CACHE.joinpath(RUN).mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps([r.__dict__ for r in regions]))
    return regions


def _regions(best: np.ndarray, f: float) -> list[Region]:
    from scipy import ndimage

    cls_map = best.argmax(2)
    conf = best.max(2)
    out = []
    for k, c in enumerate(CLASSES):
        if c == "none":
            continue
        mask = (cls_map == k) & (conf >= MIN_PROB)
        if mask.sum() < MIN_CELLS:
            continue
        lab, n = ndimage.label(mask)
        for i in range(1, n + 1):
            cells = lab == i
            if cells.sum() < MIN_CELLS:
                continue
            ys, xs = np.nonzero(cells)
            box = [float(xs.min() * STRIDE * f), float(ys.min() * STRIDE * f), float((xs.max() + 1) * STRIDE * f),
                   float((ys.max() + 1) * STRIDE * f)]
            out.append(Region(c, round(float(conf[cells].mean()), 3), [round(v, 1) for v in box], int(cells.sum())))
    out.sort(key=lambda r: -r.cells)
    return out


# ------------------------------------------------------------------ reading regions as materials

def _legend_swatches(pdf: str, page: int, img_size: tuple[int, int]) -> list[tuple[str, int, np.ndarray]]:
    """(legend row text, hatch class, feature) for legend rows that name a material and have a hatched swatch
    beside them (left of the text, as architects draw material legends)."""
    from PIL import Image

    from . import legend

    lgs = legend.find(pdf, page)
    if not lgs:
        return []
    img = sheets.raster_path("pdf", pdf, page)
    im = Image.open(img).convert("L")
    sx = im.width / legend.RENDER_SIDE * max(im.width, im.height) / max(im.width, im.height)
    out = []
    crops, rows = [], []
    for lg in lgs:
        for r in lg["rows"]:
            if not family_set(r["text"], abbreviations=True):
                continue
            x0, y0, x1, y1 = r["bbox"]
            h = max(8.0, y1 - y0)
            box = (max(0, x0 - 5 * h), y0 - 1.2 * h, max(1, x0 - 0.4 * h), y1 + 1.2 * h)
            c = im.crop(tuple(int(v * sx) for v in box))
            if c.width < 8 or c.height < 8:
                continue
            a = np.asarray(c.resize((PATCH, PATCH)), dtype=np.float32) / 255.0
            if (a < 0.75).mean() < 0.04:
                continue
            crops.append(a)
            rows.append(r["text"])
    if not crops:
        return []
    probs, feats = classify(np.stack(crops), with_features=True)
    for t, p, ft in zip(rows, probs, feats):
        k = int(p.argmax())
        if CLASSES[k] != "none" and p[k] >= 0.6:
            out.append((t, k, ft))
    return out


def _name_regions(regions: list[Region], rows: list[dict] | None, k: float) -> None:
    """Each region's 'where': the printed room name / number inside it (PDF text rows; `k` = rows -> image px)."""
    if not rows:
        return
    import re

    for r in regions:
        x0, y0, x1, y1 = r.box
        inside = [row["text"] for row in rows
                  if x0 <= (row["bbox"][0] + row["bbox"][2]) / 2 * k <= x1 and y0 <= (row["bbox"][1] + row["bbox"][3]) / 2 * k <= y1
                  and re.search(r"[A-Za-zÄÖÜäöü]{3,}", row["text"]) and len(row["text"]) < 40]
        if inside:
            r.where = " / ".join(inside[:2])


def interpret(regions: list[Region], swatches: list[tuple[str, int, np.ndarray]]) -> None:
    by_class: dict[int, list[str]] = {}
    for t, k, _ in swatches:
        by_class.setdefault(k, []).append(t)
    for r in regions:
        k = CLASSES.index(r.cls)
        if k in by_class:  # the sheet's legend shows this hatch: its words say what it is
            r.material, r.how = by_class[k][0], "legend"
            fams = family_set(r.material, abbreviations=True)
            r.family = sorted(fams)[0] if fams else None
            continue
        fam, strength = CONVENTION.get(r.cls, (None, "none"))
        r.family = fam
        r.how = "convention"
        r.material = HATCH_WORDS[r.cls] + (f" — by convention {LABEL[fam]}" if fam else " — material not stated")
        if strength == "weak" and fam:
            r.material += " (one of several materials drawn this way)"


# ------------------------------------------------------------------ answers

def _swiss(d, q: Query) -> Answer | None:
    from . import swiss

    stem = Path(d.ask_path).stem
    try:
        fl = swiss.load_floor(stem)
    except Exception:
        return None
    if not fl.hatches:
        return None
    groups: dict[str, list] = {}
    for h, layer, pat, poly in fl.hatches:
        groups.setdefault(pat, []).append((h, layer, poly))
    lines, ev, fams_all = [], [], set()
    for pat, items in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        cls = CAD_PATTERN.get(pat.upper())
        fam, strength = CONVENTION.get(cls, (None, "none")) if cls else (None, "none")
        if q.families and not (fam and matches({fam}, q.families)):
            continue
        layer = items[0][1].lower()
        what = HATCH_WORDS.get(cls, f"{pat} hatch") if cls else f"{pat} hatch"
        meaning = (f"by convention {LABEL[fam]}" + (" (the same hatch is used for other cut materials)" if strength == "weak" else "")
                   if fam else "the drawing does not say which material")
        lines.append(f"{len(items)} {layer} areas drawn with {pat} ({what}): {meaning}")
        if fam:
            fams_all.add(fam)
        for h, _, poly in items[:200]:
            ev.append(Evidence("polygon", h, f"{pat} hatch ({layer})", None, [[round(x, 4), round(y, 4)] for x, y in poly.exterior.coords],
                               f"hatch:{h}"))
    if not lines:
        return Answer(None, None, "no_material", 0.0,
                      f"hatches: the CAD plan's hatches ({', '.join(sorted(groups))}) do not stand for "
                      f"{', '.join(LABEL[f] for f in q.families)}")
    text = ("The plan names no material; its only material information is the wall hatching:\n" + "\n".join(lines)
            + "\nThere is no legend on the sheet, so this is drafting convention, not a specification.")
    return Answer(text, None, "materials_hatch", 0.45, "CAD hatch patterns read from the DXF (exact); their meaning is "
                  "drafting convention because the sheet has no legend", ev)


def _sheet_image(d, sheet) -> tuple[str, str, int] | None:
    from .pipeline import primary_raster

    if sheet and sheet[0].lower().endswith(".pdf"):
        return "pdf", sheet[0], sheet[1]
    return primary_raster(d)


def answer(d, q: Query, sheet: tuple[str, int] | None = None) -> Answer | None:
    if d.source == "swiss_dwellings":
        return _swiss(d, q)
    if q.ask in ("count", "quantity", "thickness"):
        return Answer(None, None, "no_material", 0.0, "hatches: a hatch shows where a material is, not how many or how much")
    if set(q.elements) & {"door", "window", "column", "beam", "pipe", "duct", "conduit", "stair", "railing"}:
        return Answer(None, None, "no_material", 0.0, "hatches: doors, windows and members are not shown by area hatches")
    raster = _sheet_image(d, sheet)
    if raster is None or not available():
        return Answer(None, None, "no_material", 0.0, "hatches: no sheet image, or the hatch classifier is not installed")
    kind, path, page = raster
    img = sheets.raster_path(kind, path, page)
    key = hashlib.sha1(f"{path}|{page}".encode()).hexdigest()[:16]
    regions = scan(img, key)
    if not regions:
        return Answer(None, None, "no_material", 0.0, "hatches: no hatched area recognised on the sheet")
    w, h = sheets.image_size(kind, path, page)
    swatches, rows, k = [], None, 1.0
    if kind == "pdf":
        try:
            swatches = _legend_swatches(path, page, (w, h))
        except Exception:
            swatches = []
        from . import docqa

        try:
            rows, pw, _ = docqa.layout(path, page)
            k = w / pw
        except Exception:
            rows = None
    interpret(regions, swatches)
    _name_regions(regions, rows, k)
    keep = [r for r in regions if (not q.families or (r.family and matches({r.family}, q.families)))]
    if not keep:
        seen = sorted({HATCH_WORDS[r.cls] for r in regions})
        return Answer(None, None, "no_material", 0.0,
                      f"hatches: {len(regions)} hatched area(s) found ({', '.join(seen[:6])}), none standing for "
                      f"{', '.join(LABEL[f] for f in q.families)}")
    groups: dict[str, list[Region]] = {}
    for r in keep:
        groups.setdefault(r.material, []).append(r)
    lines = []
    for mat, rs in sorted(groups.items(), key=lambda kv: -sum(r.cells for r in kv[1])):
        places = sorted({r.where for r in rs if r.where})
        lines.append(f"{mat}: {len(rs)} area{'s' if len(rs) != 1 else ''}" + (f" (in {', '.join(places[:5])})" if places else ""))
    legend_used = any(r.how == "legend" for r in keep)
    ev = [Evidence("box", f"hatch-{i}", f"{r.material[:60]} ({r.prob:.0%})", None, [[r.box[0], r.box[1]], [r.box[2], r.box[3]]])
          for i, r in enumerate(keep[:60])]
    text = ("Read from the hatching drawn on the sheet:\n" + "\n".join(lines)
            + ("" if legend_used else "\nThe sheet has no legend for these hatches: their meaning is drafting convention."))
    return Answer(text, None, "materials_hatch", 0.55 if legend_used else 0.4,
                  f"Hatched areas recognised by the hatch classifier ({RUN}) and read through "
                  f"{'the sheet legend' if legend_used else 'drafting convention'}; each area is boxed for checking", ev)
