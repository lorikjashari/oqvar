"""Real-drawing patches for checking the hatch classifier (stoa_qa.hatch) against what sheets actually look like.

    python _analysis/hatch_real.py pool          # cut candidate windows from real sheets -> eval_out/hatch_real/pool/
    python _analysis/hatch_real.py sheets        # contact sheets (numbered) to label by eye -> eval_out/hatch_real/sheets/
    python _analysis/hatch_real.py split         # labels.json -> eval_out/hatch_real/{val,test}/<class>/<id>.png
    python _analysis/hatch_real.py score [run]   # accuracy of a local weights file on val / test

Windows are cut exactly as stoa_qa.hatch scans a sheet (image shrunk to 3000 px, 96 px windows, scales 1 and 0.5, blank
paper skipped). Per sheet, half the windows are the ones the model calls hatched (so rare classes appear at all) and half
are random inked windows (so the "none" class and the model's misses are represented). Labels are given by a person
looking at the patch (labels.json: {"<id>": "<class>" | "skip"}). Val / test are split by SOURCE SHEET (hash of the path),
so the two never share a drawing; only val may be used to choose anything.
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from stoa_qa import hatch, sheets
from stoa_qa.core import DATASET
from stoa_qa.hatch_synth import CLASSES, PATCH

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "eval_out" / "hatch_real"
PER_SHEET = 14


def sources(seed: int = 7) -> list[tuple[str, str, int]]:
    """(kind, dataset-relative path, page) of the sheets to cut from."""
    rng = random.Random(seed)
    data = DATASET / "data"
    out = []
    survey = OUT / "od_survey.json"
    pdfs = []
    if survey.exists():
        pdfs = [s["path"] for s in json.loads(survey.read_text()) if s["fills"] > 150 and s["words"] > 40]
    if not pdfs:
        pdfs = [str(p.relative_to(data / "opening_design")) for p in (data / "opening_design").rglob("*.pdf")]
    pdfs = sorted(set(pdfs))
    rng.shuffle(pdfs)
    out += [("pdf", f"data/opening_design/{p}".replace("\\", "/"), 0) for p in pdfs[:40]]
    for src, n in (("habs", 12), ("finna", 6), ("europeana", 6), ("wikimedia", 6), ("cubicasa5k", 8), ("swiss_dwellings", 4)):
        files = sorted(p for p in (data / src).rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        rng.shuffle(files)
        out += [("image", str(p.relative_to(DATASET)).replace("\\", "/"), 0) for p in files[:n]]
    return out


def pool() -> None:
    rng = random.Random(11)
    (OUT / "pool").mkdir(parents=True, exist_ok=True)
    manifest = {}
    for kind, rel, page in sources():
        try:
            img = sheets.raster_path(kind, rel, page)
            g, f = hatch._gray(img)
        except Exception as e:
            print("skip", rel, e)
            continue
        cands = []
        for s in hatch.SCALES:
            ps, boxes = hatch._windows(g, s)
            if len(ps):
                cands += [(p, b, s) for p, b in zip(ps, boxes)]
        if not cands:
            continue
        probs = hatch.classify(np.stack([c[0] for c in cands]))
        pred = probs.argmax(1)
        hatched = [i for i in range(len(cands)) if CLASSES[pred[i]] != "none" and probs[i, pred[i]] > 0.5]
        rng.shuffle(hatched)
        # one per predicted class first, so rare classes appear
        picked, seen = [], set()
        for i in hatched:
            if pred[i] not in seen:
                picked.append(i)
                seen.add(pred[i])
        picked += [i for i in hatched if i not in picked][: max(0, PER_SHEET // 2 - len(picked))]
        rest = [i for i in range(len(cands)) if i not in picked]
        rng.shuffle(rest)
        picked += rest[: max(0, PER_SHEET - len(picked))]
        sheet_id = hashlib.sha1(rel.encode()).hexdigest()[:8]
        for i in picked:
            p, b, s = cands[i]
            pid = f"{sheet_id}-{i}"
            Image.fromarray((p * 255).astype(np.uint8)).save(OUT / "pool" / f"{pid}.png")
            manifest[pid] = {"source": rel, "page": page, "box": [round(v * f, 1) for v in b], "scale": s,
                             "pred": CLASSES[pred[i]], "p": round(float(probs[i, pred[i]]), 3)}
        print(rel[:90], len(picked), flush=True)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=0))
    print("pool", len(manifest))


def contact_sheets(per: int = 40, cols: int = 8) -> None:
    man = json.loads((OUT / "manifest.json").read_text())
    ids = sorted(man)
    (OUT / "sheets").mkdir(parents=True, exist_ok=True)
    S = PATCH * 2
    for k in range(0, len(ids), per):
        chunk = ids[k:k + per]
        rows = (len(chunk) + cols - 1) // cols
        W = Image.new("RGB", (cols * (S + 8), rows * (S + 22)), "white")
        d = ImageDraw.Draw(W)
        for j, pid in enumerate(chunk):
            im = Image.open(OUT / "pool" / f"{pid}.png").resize((S, S), Image.NEAREST).convert("RGB")
            x, y = (j % cols) * (S + 8), (j // cols) * (S + 22)
            W.paste(im, (x, y + 18))
            d.rectangle([x, y + 18, x + S - 1, y + 18 + S - 1], outline=(200, 0, 0))
            d.text((x + 2, y + 2), f"{k + j}", fill=(0, 0, 0))
        W.save(OUT / "sheets" / f"sheet_{k // per:02d}.png")
    (OUT / "sheets" / "index.json").write_text(json.dumps(ids))
    print(len(ids), "patches on", (len(ids) + per - 1) // per, "sheets")


def split() -> None:
    man = json.loads((OUT / "manifest.json").read_text())
    ids = json.loads((OUT / "sheets" / "index.json").read_text())
    labels = json.loads((OUT / "labels.json").read_text())
    n = {"val": 0, "test": 0}
    for key, cls in labels.items():
        pid = ids[int(key)] if key.isdigit() else key
        if cls == "skip" or cls not in CLASSES:
            continue
        src = man[pid]["source"]
        part = "val" if int(hashlib.sha1(src.encode()).hexdigest(), 16) % 2 == 0 else "test"
        d = OUT / part / cls
        d.mkdir(parents=True, exist_ok=True)
        Image.open(OUT / "pool" / f"{pid}.png").save(d / f"{pid}.png")
        n[part] += 1
    print(n)


def score(run: str = hatch.RUN) -> None:
    hatch.RUN = run
    hatch._model.cache_clear()
    for part in ("val", "test"):
        xs, ys = [], []
        for d in sorted((OUT / part).glob("*")):
            for f in d.glob("*.png"):
                xs.append(np.asarray(Image.open(f).convert("L"), dtype=np.float32) / 255.0)
                ys.append(CLASSES.index(d.name))
        if not xs:
            continue
        p = hatch.classify(np.stack(xs)).argmax(1)
        y = np.array(ys)
        acc = float((p == y).mean())
        hatched = y != 0
        det = float(((p != 0) == hatched).mean())
        print(f"{part}: n={len(y)} accuracy={acc:.3f}  hatched-vs-not={det:.3f}  "
              f"hatch-class accuracy={float((p[hatched] == y[hatched]).mean()) if hatched.any() else float('nan'):.3f}")
        conf = {}
        for t, q in zip(y, p):
            if t != q:
                conf[(CLASSES[t], CLASSES[q])] = conf.get((CLASSES[t], CLASSES[q]), 0) + 1
        print("  confusions:", sorted(conf.items(), key=lambda kv: -kv[1])[:12])


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "pool":
        pool()
    elif cmd == "sheets":
        contact_sheets()
    elif cmd == "split":
        split()
    elif cmd == "score":
        score(*(sys.argv[2:3] or []))
