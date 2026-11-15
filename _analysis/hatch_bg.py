"""Real-sheet windows for training the hatch classifier (modal_jobs/hatch_cloud.py, second round).

The first model, trained on synthetic patches only, called nearly every real window a hatch: its "none" examples were
clean paper. This cuts 96 px windows from real sheets the same way stoa_qa.hatch scans them, from sheets that are NOT in
the hand-labelled val / test set (_analysis/hatch_real.py), and labels them:

  * Swiss plans: the DXF says exactly where the wall hatches are (stoa_qa.swiss.read_dxf -> Floor.hatches, mapped to the
    PNG by stoa_qa.sheets.swiss_to_px): windows mostly inside ANSI31 hatches -> "diagonal", inside SOLID -> "solid",
    touching no hatch -> "none". Real positives, exact labels.
  * everything else (opening_design PDF pages, CubiCasa train-split plans, archive scans): label -1 = "real background".
    Mostly no hatch (about 5% of hand-labelled windows were hatched), used as "none" and as the canvas synthetic hatches
    are drawn onto.

    python _analysis/hatch_bg.py            -> eval_out/hatch_bg/bg.npz  (x: N x 96 x 96 uint8, y: N int8, src: N str)
    modal volume put stoa-cubicasa eval_out/hatch_bg/bg.npz hatch/bg.npz
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from stoa_qa import hatch, sheets, swiss
from stoa_qa.core import DATASET
from stoa_qa.hatch_synth import IDS, PATCH

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "eval_out" / "hatch_bg"
PER_SHEET = 60


def held_out() -> set[str]:
    man = REPO / "eval_out" / "hatch_real" / "manifest.json"
    return {v["source"] for v in json.loads(man.read_text()).values()} if man.exists() else set()


def cubicasa_train(n: int, rng) -> list[str]:
    split = REPO / "third_party" / "cubicasa_split" / "train.txt"
    names = []
    if split.exists():
        names = [ln.strip() for ln in split.read_text().splitlines() if ln.strip()]
    files = sorted((DATASET / "data" / "cubicasa5k").glob("*.png"))
    if names:
        keep = {Path(x).name for x in names} | {Path(x).stem for x in names}
        f2 = [f for f in files if f.name in keep or f.stem in keep]
        files = f2 or files
    rng.shuffle(files)
    return [str(f.relative_to(DATASET)).replace("\\", "/") for f in files[:n]]


def windows(img: Path):
    g, f = hatch._gray(img)
    out = []
    for s in hatch.SCALES:
        ps, boxes = hatch._windows(g, s)
        out += [(p, [v * f for v in b]) for p, b in zip(ps, boxes)]
    return out


def swiss_masks(stem: str, size: tuple[int, int]):
    """Two masks in PNG pixels: ANSI31-hatched walls and SOLID walls."""
    fl = swiss.load_floor(stem)
    masks = {"diagonal": Image.new("L", size, 0), "solid": Image.new("L", size, 0)}
    for _, _, pat, poly in fl.hatches:
        cls = "solid" if pat.upper() == "SOLID" else "diagonal" if pat.upper() == "ANSI31" else None
        if cls is None:
            continue
        px = sheets.swiss_to_px(stem, [[x, y] for x, y in poly.exterior.coords])
        if px and len(px) >= 3:
            ImageDraw.Draw(masks[cls]).polygon([tuple(p) for p in px], fill=255)
    return {k: np.asarray(v, dtype=np.uint8) > 0 for k, v in masks.items()}


def main() -> None:
    rng = random.Random(5)
    skip = held_out()
    data = DATASET / "data"
    srcs: list[tuple[str, str]] = []
    pdfs = sorted(str(p.relative_to(DATASET)).replace("\\", "/") for p in (data / "opening_design").rglob("*.pdf"))
    rng.shuffle(pdfs)
    srcs += [("pdf", p) for p in pdfs if p not in skip][:260]
    srcs += [("image", p) for p in cubicasa_train(140, rng)]
    for src, n in (("habs", 40), ("finna", 30), ("europeana", 20), ("wikimedia", 25)):
        files = sorted(str(p.relative_to(DATASET)).replace("\\", "/") for p in (data / src).rglob("*")
                       if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        rng.shuffle(files)
        srcs += [("image", f) for f in files if f not in skip][:n]
    swiss_files = sorted(str(p.relative_to(DATASET)).replace("\\", "/") for p in (data / "swiss_dwellings").glob("*.png"))
    rng.shuffle(swiss_files)
    swiss_files = [f for f in swiss_files if f not in skip][:160]

    xs, ys, ss = [], [], []
    for kind, rel in srcs:
        try:
            img = sheets.raster_path(kind, rel, 0)
            ws = windows(img)
        except Exception as e:
            print("skip", rel[:80], type(e).__name__)
            continue
        rng.shuffle(ws)
        for p, _ in ws[:PER_SHEET]:
            xs.append((p * 255).astype(np.uint8))
            ys.append(-1)
            ss.append(rel)
        print(kind, rel[:80], min(len(ws), PER_SHEET), flush=True)
    for rel in swiss_files:
        stem = Path(rel).stem
        try:
            img = sheets.raster_path("image", rel, 0)
            W, H = Image.open(img).size
            masks = swiss_masks(stem, (W, H))
            ws = windows(img)
        except Exception as e:
            print("skip", rel, type(e).__name__, e)
            continue
        rng.shuffle(ws)
        got = {"diagonal": 0, "solid": 0, "none": 0}
        for p, (x0, y0, x1, y1) in ws:
            sl = (slice(int(y0), int(y1)), slice(int(x0), int(x1)))
            fr = {k: float(m[sl].mean()) if m[sl].size else 0.0 for k, m in masks.items()}
            if fr["diagonal"] >= 0.35 and fr["solid"] < 0.05:
                lab = "diagonal"
            elif fr["solid"] >= 0.35 and fr["diagonal"] < 0.05:
                lab = "solid"
            elif fr["diagonal"] + fr["solid"] < 0.01:
                lab = "none"
            else:
                continue
            if got[lab] >= {"diagonal": 25, "solid": 15, "none": 20}[lab]:
                continue
            got[lab] += 1
            xs.append((p * 255).astype(np.uint8))
            ys.append(IDS[lab])
            ss.append(rel)
        print("swiss", stem, got, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "bg.npz", x=np.stack(xs), y=np.array(ys, np.int8), src=np.array(ss))
    print("windows", len(xs), {k: int((np.array(ys) == v).sum()) for k, v in (("bg", -1), ("diagonal", IDS["diagonal"]),
                                                                                ("solid", IDS["solid"]), ("none", 0))})


if __name__ == "__main__":
    main()
