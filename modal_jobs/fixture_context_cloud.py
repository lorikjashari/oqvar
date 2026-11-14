"""Train a joint fixture and wet-room detector, including look-alike symbols."""
from __future__ import annotations

import gzip
import json
import random
import shutil
import zipfile
from collections import Counter
from pathlib import Path

import modal

VOL = Path("/vol")
vol = modal.Volume.from_name("stoa-cubicasa")
app = modal.App("stoa-fixture-context")
image = (modal.Image.debian_slim(python_version="3.12")
         .apt_install("libgl1", "libglib2.0-0")
         .pip_install("ultralytics==8.4.212", "pillow")
         .add_local_python_source("stoa_qa", ignore=["**/__pycache__/**"]))


def distractor(label):
    if label.startswith("FixedFurniture ShowerScreen"):
        return "shower_screen"
    if label == "FixedFurniture ElectricalAppliance WashingMachine":
        return "washing_machine"
    if label.startswith(("FixedFurniture ElectricalAppliance IntegratedStove",
                         "FixedFurniture ElectricalAppliance Stove")):
        return "stove"
    return None


def prepare(dest, preset, edition="fixture-layout-v1"):
    from PIL import Image

    from stoa_qa.cubicasa_rules import CLASSES as ORIGINAL
    from stoa_qa.cubicasa_userplans import from_yolo_text, write_labels
    from stoa_qa.fixture_context import IDS

    rng = random.Random(31)
    with gzip.open(VOL / "objects.json.gz", "rt") as f:
        metadata = {plan["name"]: plan for plan in json.load(f)}
    limits = {"train": (950, 300), "val": (130, 56), "test": (160, 65)}
    if preset == "broad":
        limits["train"] = (1800, 540)
    manifest = {}
    for split, sizes in limits.items():
        images, labels = dest / "images" / split, dest / "labels" / split
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        manifest[split] = {}
        for source, limit in zip(("yolo", "swiss"), sizes):
            root = VOL / source
            pool = sorted((root / "images" / split).glob("*.png"))
            if split == "train":
                rng.shuffle(pool)
                # Retain the rare tubs and provide labelled look-alikes to the joint model.
                pool.sort(key=lambda p: not any(o["c"].startswith("FixedFurniture Bathtub")
                                                for o in metadata.get(p.stem, {}).get("objs", ())))
                selected = pool[:limit]
            else:
                selected = random.Random(104 + int(split == "test")).sample(pool, min(limit, len(pool)))
            counts = Counter()
            for file in selected:
                name = f"{source}-{file.stem}"
                if split == "train":
                    shutil.copyfile(file, images / f"{name}.png")
                else:
                    # Keep the original held-out sheets available for the inference evaluation.
                    (images / f"{name}.png").symlink_to(file)
                with Image.open(file) as im:
                    data = (root / "labels" / split / f"{file.stem}.txt").read_text()
                    labeled = from_yolo_text(data, *im.size)
                    boxes = [(IDS[ORIGINAL[c]], *xyxy) for c, _, *xyxy in labeled if ORIGINAL[c] in IDS]
                    meta = metadata.get(file.stem) if source == "yolo" else None
                    if meta:
                        sx, sy = im.width / meta["w"], im.height / meta["h"]
                        boxes += [(IDS[kind], a*sx, b*sy, c*sx, d*sy) for o in meta["objs"] if o.get("b")
                                  and (kind := distractor(o["c"])) for a,b,c,d in [o["b"]]]
                    write_labels(labels / f"{name}.txt", boxes, *im.size)
                    counts["sheets"] += 1
                    if split != "train":
                        continue
                    # Train on original sheets and dense crops. Center some crops on look-alikes.
                    anchors = [b for b in boxes if b[0] in (IDS["bathroom"], IDS["wc_room"], IDS["bathtub"],
                                                                       IDS["washing_machine"], IDS["stove"])]
                    if not anchors:
                        anchors = boxes
                    for i, (_, a, b, c, d) in enumerate(rng.sample(anchors, min(2, len(anchors)))):
                        side = min(max(im.size), max(160, round(max(c-a, d-b) * rng.uniform(3.2, 8))))
                        x = max(0, min(max(0, im.width-side), round((a+c-side)/2 + rng.uniform(-.12,.12)*side)))
                        y = max(0, min(max(0, im.height-side), round((b+d-side)/2 + rng.uniform(-.12,.12)*side)))
                        x1, y1 = min(im.width,x+side), min(im.height,y+side)
                        tiles = [(k, l-x,t-y,r-x,bot-y) for k,l,t,r,bot in boxes
                                 if min(r,x1)-max(l,x)>2 and min(bot,y1)-max(t,y)>2]
                        im.crop((x,y,x1,y1)).convert("RGB").save(images / f"{name}-tile{i}.jpg", quality=93)
                        write_labels(labels / f"{name}-tile{i}.txt", tiles, x1-x, y1-y)
                        counts["tiles"] += 1
            manifest[split][source] = dict(counts)
            print(f"{edition} {split} {source}: {dict(counts)}", flush=True)
    return manifest


@app.function(image=image, volumes={str(VOL): vol}, gpu=["H100", "A100-80GB"], cpu=12, memory=32768, timeout=7200)
def fit(run: str, epochs: int, preset: str, imgsz: int, adaptation: str = "", resume: bool = False) -> dict:
    from ultralytics import YOLO

    from stoa_qa.fixture_context import CLASSES

    dest = Path("/tmp/fixture-context")
    target = VOL / "runs" / run
    if target.exists() and not resume:
        raise FileExistsError(target)
    manifest = prepare(dest, preset, run)
    if adaptation:
        archive = VOL / "datasets" / Path(adaptation).name
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)
        manifest["adaptation"] = archive.name
    data = dest / "data.yaml"
    data.write_text(f"path: {dest}\ntrain: images/train\nval: images/val\ntest: images/test\nnames: {list(CLASSES)!r}\n")
    base = VOL / "runs/mixed-m13/weights/best.pt"
    model = YOLO(str(target / "weights/last.pt") if resume else str(base))
    model.add_callback("on_model_save", lambda _: vol.commit())
    batch = 16 if imgsz == 1024 else 12
    if resume:
        model.train(resume=True, device=0, workers=8)
    else:
        model.train(data=str(data), imgsz=imgsz, epochs=epochs, device=0, batch=batch, workers=8, seed=33,
                optimizer="AdamW", lr0=0.001, mosaic=0.05, close_mosaic=4, fliplr=0.5, flipud=0.35,
                degrees=5, hsv_h=0.01, hsv_s=0.35, patience=8, project=str(target.parent),
                name=run, plots=False, max_det=1200)
    vol.commit()
    model = YOLO(str(target / "weights/best.pt"))
    result = model.val(data=str(data), split="test", imgsz=imgsz, batch=batch, device=0, plots=False, max_det=1200)
    m = result.box
    report = {"dataset": manifest, "imgsz": imgsz, "ap50": float(m.map50), "classes": {
        CLASSES[i]: {"ap50": float(ap), "precision": float(p), "recall": float(r)}
        for i, ap, p, r in zip(m.ap_class_index, m.ap50, m.p, m.r)}}
    (target / "metrics.json").write_text(json.dumps(report, indent=2))
    vol.commit()
    return report


@app.local_entrypoint()
def main(run: str = "fixture-context-v1", epochs: int = 12, preset: str = "pilot", imgsz: int = 1024,
         adaptation: str = "", resume: bool = False):
    print(json.dumps(fit.remote(run, epochs, preset, imgsz, adaptation, resume), indent=2))


@app.function(image=image, volumes={str(VOL): vol}, gpu=["L40S", "A100-80GB"], cpu=8, memory=24576, timeout=5400)
def fit_verifier(run: str, epochs: int) -> dict:
    """Train a symbol-level second opinion close to real detector box scale."""
    from PIL import Image
    from ultralytics import YOLO

    from stoa_qa.cubicasa_rules import label_of
    from stoa_qa.fixture_context import FIXTURES

    rng = random.Random(41)
    with gzip.open(VOL / "objects.json.gz", "rt") as f:
        metadata = json.load(f)
    groups = ("other", *FIXTURES)
    dataset = Path("/tmp/fixture-verify")
    distribution = {}
    for split, limit in (("train", 1100), ("val", 220), ("test", 250)):
        by_kind = {kind: [] for kind in groups}
        for p in metadata:
            if p["split"] != split:
                continue
            for o in p["objs"]:
                if not o.get("b"):
                    continue
                kind = label_of(o["c"])
                if kind in FIXTURES:
                    by_kind[kind].append((p, o))
                elif distractor(o["c"]) or o["c"].startswith(("FixedFurniture Closet", "FixedFurniture Fireplace")):
                    by_kind["other"].append((p, o))
        distribution[split] = {}
        for kind, items in by_kind.items():
            folder = dataset / split / kind
            folder.mkdir(parents=True)
            picked = rng.sample(items, min(len(items), limit))
            saved = 0
            for i, (p, o) in enumerate(picked):
                path = VOL / "yolo" / "images" / split / f"{p['name']}.png"
                with Image.open(path) as im:
                    sx, sy = im.width / p["w"], im.height / p["h"]
                    a,b,c,d = o["b"]
                    crop = verifier_crop(im, [a*sx,b*sy,c*sx,d*sy], rng.uniform(1.45,2.1) if split == "train" else 1.7)
                    if crop is not None:
                        crop.convert("RGB").save(folder / f"{p['name']}-{i}.jpg", quality=93)
                        saved += 1
            distribution[split][kind] = saved
        print(f"{split}: {distribution[split]}", flush=True)

    target = VOL / "runs" / run
    if target.exists():
        raise FileExistsError(target)
    model = YOLO("yolo11n-cls.pt")
    model.add_callback("on_model_save", lambda _: vol.commit())
    model.train(data=str(dataset), epochs=epochs, imgsz=160, batch=96, device=0, workers=6, project=str(target.parent),
                name=run, seed=41, degrees=180, fliplr=.5, flipud=.5, patience=8, plots=False)
    model = YOLO(str(target / "weights/best.pt"))
    confusion = {kind: Counter() for kind in groups}
    files = sorted((dataset / "test").glob("*/*.jpg"))
    preds = model.predict([str(path) for path in files], stream=True, imgsz=160, batch=48, verbose=False)
    for file, pred in zip(files, preds):
        confusion[file.parent.name][model.names[pred.probs.top1]] += 1
    report = {"data": distribution, "confusion": confusion,
              "accuracy": sum(confusion[k][k] for k in groups) / len(files)}
    (target / "metrics.json").write_text(json.dumps(report, indent=2))
    vol.commit()
    return report


def verifier_crop(im, box, padding):
    from stoa_qa.fixture_context import fixture_crop

    return fixture_crop(im, box, padding)


@app.local_entrypoint()
def verify(run: str = "fixture-verifier-v1", epochs: int = 18):
    print(json.dumps(fit_verifier.remote(run, epochs), indent=2))


def score_sheet(predictions, truth, kinds):
    """Location-aware metrics. Count equality alone can hide offsetting misses and false positives."""
    from stoa_qa.cubicasa_rules import _iou
    from stoa_qa.fixture_context import CLASSES

    stats = {kind: Counter() for kind in kinds}
    for pred, gt in zip(predictions, truth):
        for kind in kinds:
            idx = CLASSES.index(kind)
            detected = [b for b in pred if b[0] == idx]
            remaining = [b[1:5] for b in gt if b[0] == idx]
            row = stats[kind]
            row["predicted"] += len(detected)
            row["expected"] += len(remaining)
            row["exact_plans"] += len(detected) == len(remaining)
            row["plans"] += 1
            for det in sorted(detected, key=lambda b: -b[1]):
                overlaps = [_iou(det[2:6], target) for target in remaining]
                best = max(overlaps, default=0.0)
                if best >= (0.35 if kind in ("bathroom", "wc_room") else 0.45):
                    row["tp"] += 1
                    remaining.pop(overlaps.index(best))
    return {k: {**counts, "precision": counts["tp"] / max(1, counts["predicted"]),
                "recall": counts["tp"] / max(1, counts["expected"]),
                "f1": 2 * counts["tp"] / max(1, counts["predicted"] + counts["expected"])}
            for k, counts in stats.items()}


def contextual_rows(rows, threshold, room_min):
    from stoa_qa.fixture_context import FIXTURES, IDS, fixture_groups, room_context, selected_fixtures

    limits = dict.fromkeys(FIXTURES, threshold)
    fixtures = selected_fixtures(rows, limits)
    rooms = room_context(rows, limits, room_min)
    return fixtures + [[IDS[r.kind], *r.box[1:]] for r in rooms + fixture_groups(fixtures, rooms)]


@app.function(image=image, volumes={str(VOL): vol}, gpu=["H100", "A100-80GB"], cpu=8, memory=24576, timeout=7200)
def evaluate_remote(run: str, imgsz: int, dorm: bytes, annotations: str):
    from PIL import Image
    from ultralytics import YOLO

    from stoa_qa import cubicasa_live, fixture_context as FC
    from stoa_qa.cubicasa_rules import CLASSES as OLD
    from stoa_qa.cubicasa_userplans import from_yolo_text

    wanted = (*FC.FIXTURES, *FC.ROOMS)
    baseline_run = VOL / "runs/mixed-m13/weights/best.pt"
    weights = VOL / "runs" / run / "weights/best.pt"
    model, baseline = YOLO(str(weights)), YOLO(str(baseline_run))
    root = Path("/tmp/fixture-eval")
    root.mkdir()
    dorm_path = root / "dorm.png"
    dorm_path.write_bytes(dorm)
    samples = {}
    for split, seed in (("val", 58), ("test", 82)):
        samples[split] = []
        for source,num in (("yolo", 70), ("swiss", 30)):
            source_root = VOL / source
            paths = sorted((source_root / "images" / split).glob("*.png"))
            paths = random.Random(seed).sample(paths, min(num, len(paths)))
            for path in paths:
                label = source_root / "labels" / split / f"{path.stem}.txt"
                with Image.open(path) as im:
                    truth = [[FC.IDS[OLD[c]], *b] for c, _, *b in from_yolo_text(label.read_text(), *im.size)
                             if OLD[c] in wanted]
                samples[split].append((path, truth))

    val_rows = [FC.infer(model, p, imgsz, 960) for p,_ in samples["val"]]
    val_truth = [t for _,t in samples["val"]]
    def score(rows, truth, threshold, room_min):
        return score_sheet([contextual_rows(r,threshold,room_min) for r in rows], truth, wanted)
    choices = [(threshold, room_min, score(val_rows,val_truth,threshold,room_min)) for threshold in (.25,.35,.45,.55,.65)
               for room_min in (.28,.40,.52)]
    def objective(item):
        metrics = item[2]
        return sum(metrics[k]["f1"] for k in ("toilet","sink","shower","bathroom"))/4
    threshold, room_min, val = max(choices,key=objective)

    def base_rows(path):
        cfg = cubicasa_live.live_config()
        r = baseline.predict(str(path), imgsz=cfg["imgsz"], conf=.05, verbose=False)[0].boxes
        rows = [[int(c), round(float(s),4), *b] for c,s,b in zip(r.cls.tolist(), r.conf.tolist(), r.xyxy.tolist())]
        with Image.open(path) as im:
            tile = cfg["tile_rule"]["tile"]
            if max(im.size) > tile * 1.25:
                tiled = cubicasa_live.predict_tiled(baseline, path, cfg["imgsz"], tile, full=rows)
                nf, nt = cubicasa_live._n_objects(rows,cfg), cubicasa_live._n_objects(tiled,cfg)
                if nt >= cfg["tile_rule"]["times"]*max(nf,1) and nt-nf >= cfg["tile_rule"]["plus"]:
                    rows = tiled
        picked = []
        from stoa_qa.cubicasa_rules import select_boxes
        for q in ("toilets","sinks","baths","bathrooms","wc_rooms"):
            for c,s,*b in select_boxes(rows,q,cfg["yolo_thresholds"][q],cfg["yolo_iou"][q]):
                picked.append([FC.IDS[OLD[c]],s,*b])
        return picked

    test_rows = [FC.infer(model, p, imgsz, 960) for p,_ in samples["test"]]
    test_truth = [t for _,t in samples["test"]]
    summary = {"threshold": threshold, "room_threshold": room_min, "validation": val,
               "test": score(test_rows,test_truth,threshold,room_min),
               "baseline_test": score_sheet([base_rows(p) for p,_ in samples["test"]],test_truth,wanted)}
    regions = json.loads(annotations)["regions"]
    all_truth = [[FC.IDS[k],*b] for group in regions.values() for k,*b in group]
    report = {}
    old_dorm = base_rows(dorm_path)
    report["baseline_detector"] = score_sheet([old_dorm], [all_truth], ("bathroom","toilet","sink","shower"))
    for tile in (480,720,960):
        raw = FC.infer(model, dorm_path, imgsz, tile)
        pred = contextual_rows(raw,threshold,room_min)
        wings = {"north": (1408,474,1732,1290), "west": (808,1820,1608,2144), "east": (1926,1820,2728,2144)}
        report[str(tile)] = {}
        for wing, (x0,y0,x1,y1) in wings.items():
            region_pred = [r for r in pred if x0 <= (r[2]+r[4])/2 <= x1 and y0 <= (r[3]+r[5])/2 <= y1]
            region_true = [[FC.IDS[k],*b] for k,*b in regions[wing]]
            report[str(tile)][wing] = score_sheet([region_pred], [region_true], ("bathroom","toilet","sink","shower"))
        report[str(tile)]["sheet"] = score_sheet([pred],[all_truth],("bathroom","toilet","sink","shower"))
        report[str(tile)]["raw_boxes"] = raw
    summary["dormitory"] = report
    dest = weights.parent.parent / "evaluation.json"
    dest.write_text(json.dumps(summary,indent=2))
    vol.commit()
    return {**{k:v for k,v in summary.items() if k!="dormitory"},
            "dormitory": {key:{k:v for k,v in rows.items() if k!="raw_boxes"} for key,rows in report.items()}}


@app.local_entrypoint()
def evaluate(run: str, imgsz: int = 1024, dorm: str = "", annotations: str = ""):
    print(json.dumps(evaluate_remote.remote(run, imgsz, Path(dorm).read_bytes(), Path(annotations).read_text()), indent=2))
