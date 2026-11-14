"""Teach the raster detector YOUR floor plans on Modal (no local GPU or weights needed). Same steps and files as
modal_jobs/cubicasa_userplans.py; the CubiCasa data and the trained runs are on the Volume `stoa-cubicasa`
(built by cubicasa_detector.py / cubicasa_cloud.py).

    modal run modal_jobs/cubicasa_userplans_cloud.py::label --plans myplans                 # -> myplans/labels_auto/
    #   correct them in tools/plan_labeler.html (saves myplans/labels/), then, locally (no GPU needed):
    python modal_jobs/cubicasa_userplans.py split myplans
    modal run --detach modal_jobs/cubicasa_userplans_cloud.py::finetune --plans myplans --name ft-mine
    python modal_jobs/cubicasa_userplans.py evaluate myplans --before m13-1280 --after ft-mine

`finetune` uploads your corrected plans, mixes them with a CubiCasa train replay sample, fine-tunes runs/<base>/weights/
best.pt into runs/<name>/ on the Volume, then predicts your plans and CubiCasa val + test with both the base and the new
model and writes the boxes locally (myplans/predictions/, eval_out/yolo/<run>/), ready for `evaluate`.
Your plan images are sent to Modal for this; only the weights stay on the Volume.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent
VOL = "/vol"
RUNS = f"{VOL}/runs"
CC = f"{VOL}/yolo"  # images/<split>/<plan>.png, labels/<split>/<plan>.txt

vol = modal.Volume.from_name("stoa-cubicasa", create_if_missing=True)
app = modal.App("stoa-cubicasa-userplans")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("ultralytics>=8.3", "pillow")
    .add_local_python_source("stoa_qa", ignore=["**/__pycache__/**"])
)


def _predict(weights: list[str], files: list[str], imgsz: int, batch: int = 16) -> dict:
    """{plan: [[cls, score, x0, y0, x1, y1]]} at conf 0.05; two runs are fused like stoa_qa.cubicasa_ensemble."""
    from ultralytics import YOLO

    from stoa_qa.cubicasa_ensemble import fuse

    runs = []
    for w in weights:
        m = YOLO(w)
        out = {}
        for i in range(0, len(files), batch):
            for f, r in zip(files[i:i + batch], m.predict(files[i:i + batch], imgsz=imgsz, conf=0.05, verbose=False)):
                b = r.boxes
                out[Path(f).stem] = [[int(c), round(float(s), 4), *[round(float(v), 1) for v in xyxy]]
                                     for c, s, xyxy in zip(b.cls, b.conf, b.xyxy)]
        runs.append(out)
    return runs[0] if len(runs) == 1 else {k: fuse(runs[0][k], runs[1].get(k, [])) for k in runs[0]}


def _write_user(plans: dict, root: Path, with_labels: bool) -> list[str]:
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "labels").mkdir(parents=True, exist_ok=True)
    files = []
    for name, p in plans.items():
        f = root / "images" / f"{name}{p['ext']}"
        f.write_bytes(p["image"])
        files.append(str(f))
        if with_labels:
            (root / "labels" / f"{name}.txt").write_text(p["label"])
    return sorted(files)


def _weights(runs: list[str]) -> list[str]:
    out = [f"{RUNS}/{r}/weights/best.pt" for r in runs]
    for w in out:
        if not Path(w).exists():
            raise FileNotFoundError(f"{w} is not on the Volume (train it with cubicasa_cloud.py::fit_full first)")
    return out


@app.function(image=image, volumes={VOL: vol}, gpu="L40S", cpu=8, memory=32768, timeout=3600)
def predict_plans(runs: list[str], plans: dict, imgsz: int) -> dict:
    files = _write_user(plans, Path("/tmp/user"), with_labels=False)
    return _predict(_weights(runs), files, imgsz)


@app.function(image=image, volumes={VOL: vol}, gpu="H100", cpu=16, memory=98304, timeout=6 * 3600)
def finetune_remote(name: str, base: str, plans: dict, splits: dict, hp: dict, mix: dict, with_base: bool) -> dict:
    from ultralytics import YOLO

    from stoa_qa import cubicasa_userplans as U

    user = Path("/tmp/user")
    user_files = _write_user(plans, user, with_labels=True)
    cc = {s: sorted(p.stem for p in Path(f"{CC}/labels/{s}").glob("*.txt")) for s in ("train", "val")}
    entries = U.mix_plan(splits, cc, **mix)
    data = U.materialize(entries, user, Path(CC), Path("/tmp/data"))  # copies the sampled CubiCasa plans to local disk
    base_w = _weights([base])[0]
    m = YOLO(base_w)
    m.add_callback("on_fit_epoch_end", lambda trainer: vol.commit())
    m.train(data=str(data), project=RUNS, name=name, exist_ok=True, device=0, workers=12, **U.finetune_args(**hp))
    vol.commit()
    imgsz = hp.get("imgsz", 1280)
    cc_files = {s: sorted(str(p) for p in Path(f"{CC}/images/{s}").glob("*.png")) for s in ("val", "test")}
    out = {"mix": {f"{src}/{s}": sum(1 for e in entries if e[0] == src and e[2] == s)
                   for src in ("user", "cubicasa") for s in ("train", "val", "test")},
           "label_truth": {s: {Path(f).stem: U.label_counts(Path(f"{CC}/labels/{s}/{Path(f).stem}.txt").read_text())
                               for f in cc_files[s]} for s in ("val", "test")},
           "preds": {}}
    for tag, w in ([(name, f"{RUNS}/{name}/weights/best.pt")] + ([(base, base_w)] if with_base else [])):
        out["preds"][tag] = {"user": _predict([w], user_files, imgsz),
                             **{s: _predict([w], cc_files[s], imgsz) for s in ("val", "test")}}
    return out


def _read_plans(root: Path, with_labels: bool) -> dict:
    sys.path.insert(0, str(REPO))
    from stoa_qa import cubicasa_userplans as U

    imgs = U.list_images(root)
    out = {}
    for name, f in imgs.items():
        lbl = root / "labels" / f"{name}.txt"
        if with_labels and not lbl.exists():
            continue
        out[name] = {"image": f.read_bytes(), "ext": f.suffix.lower(), "label": lbl.read_text() if with_labels else ""}
    return out


@app.local_entrypoint()
def label(plans: str, runs: str = "m13-1280", imgsz: int = 1280, overwrite: bool = False, picks: str = ""):
    """Pseudo-label a folder of plans with the run(s) on the Volume (comma-separated runs are fused)."""
    sys.path.insert(0, str(REPO / "modal_jobs"))
    import cubicasa_userplans as L  # the local job: same writers, same file layout

    from stoa_qa import cubicasa_userplans as U

    root = Path(plans)
    run_list = runs.split(",")
    payload = _read_plans(root, with_labels=False)
    if not payload:
        raise SystemExit(f"no plan images in {U.images_dir(root)}")
    preds = predict_plans.remote(run_list, payload, imgsz)
    tag = run_list[0] if len(run_list) == 1 else "ens-" + "-".join(run_list)
    f = L.save_preds(root, tag, preds)
    thr, iou = U.load_picks(Path(picks) if picks else L.PICKS)
    n, boxes = L.write_pseudo_labels(root, preds, thr, iou, overwrite)
    print(f"{len(preds)} plans: raw boxes -> {f}; {n} pseudo-label files ({boxes} boxes) -> {root / 'labels_auto'}")
    print("next: open tools/plan_labeler.html, choose the folder, correct and save each plan (-> labels/)")


@app.local_entrypoint()
def finetune(plans: str, name: str, base: str = "m13-1280", epochs: int = 15, lr0: float = 2e-4, freeze: int = 10,
             imgsz: int = 1280, batch: int = 16, patience: int = 8, repeat: int = 3, replay_ratio: float = 2.0,
             replay_val: int = 100, seed: int = 0, with_base: bool = True):
    root = Path(plans)
    sf = root / "splits.json"
    if not sf.exists():
        raise SystemExit(f"{sf} missing: run  python modal_jobs/cubicasa_userplans.py split {root}  first")
    splits = json.loads(sf.read_text())["plans"]
    payload = _read_plans(root, with_labels=True)
    missing = sorted(set(splits) - set(payload))
    if missing:
        raise SystemExit(f"plans in splits.json without image or corrected label: {missing[:5]}")
    payload = {k: payload[k] for k in splits}
    hp = dict(epochs=epochs, lr0=lr0, freeze=freeze, imgsz=imgsz, batch=batch, patience=patience)
    mix = dict(repeat=repeat, replay_ratio=replay_ratio, replay_val=replay_val, seed=seed)
    print(f"uploading {len(payload)} corrected plans; fine-tuning runs/{base} -> runs/{name}: {hp} {mix}")
    out = finetune_remote.remote(name, base, payload, splits, hp, mix, with_base)
    print("dataset:", out["mix"])
    for tag, p in out["preds"].items():
        (root / "predictions").mkdir(exist_ok=True)
        (root / "predictions" / f"{tag}.json").write_text(json.dumps(p["user"]))
        dest = REPO / "eval_out/yolo" / tag
        dest.mkdir(parents=True, exist_ok=True)
        for s in ("val", "test"):
            f = dest / f"{s}.json"
            if tag == name or not f.exists():  # never overwrite an existing base run's boxes
                f.write_text(json.dumps(p[s]))
        print(f"{tag}: {len(p['user'])} user plans, {len(p['val'])} val / {len(p['test'])} test CubiCasa plans -> {dest}")
    (REPO / "eval_out/yolo" / name / "label_truth.json").write_text(json.dumps(out["label_truth"]))
    print(f"next: python modal_jobs/cubicasa_userplans.py evaluate {root} --before {base} --after {name}")


def _copy_replay(source, split, limit, target, rng):
    from stoa_qa.cubicasa_userplans import _link

    root = Path(VOL) / source
    images = sorted(root.glob(f"images/{split}/*.png"))
    picked = rng.sample(images, min(limit, len(images)))
    if not picked:
        raise ValueError(f"No {source} {split} images on the volume")
    for image_path in sorted(picked):
        name = f"{source}_{image_path.stem}"
        _link(image_path, target / "images" / split / f"{name}.png")
        _link(root / "labels" / split / f"{image_path.stem}.txt", target / "labels" / split / f"{name}.txt")
    return picked


@app.function(image=image, volumes={VOL: vol}, gpu="H100", cpu=8, memory=32768, timeout=5400)
def fit_adaptation(name: str, plans: dict, epochs: int) -> dict:
    from ultralytics import YOLO

    from stoa_qa import cubicasa_userplans as U
    from stoa_qa.cubicasa_live import live_config

    rng = random.Random(17)
    root = Path("/tmp/adaptation")
    user = Path("/tmp/reviewed")
    user_files = _write_user(plans, user, with_labels=True)
    entries = U.mix_plan(dict.fromkeys(plans, "train"), {}, repeat=8)
    data = U.materialize(entries, user, None, root)
    test = {}
    for source in ("yolo", "swiss"):
        for split, limit in (("train", 160), ("val", 48), ("test", 64)):
            files = _copy_replay(source, split, limit, root, rng)
            if split == "test":
                test[source] = [str(root / "images/test" / f"{source}_{f.name}") for f in sorted(files)]
    cfg = live_config()
    base = _weights([cfg["run"]])[0]
    imgsz = cfg["imgsz"]
    model = YOLO(base)
    model.add_callback("on_model_save", lambda trainer: vol.commit())
    model.train(data=str(data), project=RUNS, name=name, exist_ok=True, device=0, workers=6, seed=17,
                mosaic=0.0, **U.finetune_args(epochs=epochs, imgsz=imgsz, batch=12))
    vol.commit()

    metrics = {}
    for tag, checkpoint in (("baseline", base), ("candidate", _weights([name])[0])):
        out = metrics[tag] = {}
        for source, files in {"reviewed_training_crops": user_files, **test}.items():
            label_dir = user / "labels" if source == "reviewed_training_crops" else root / "labels/test"
            truth = {Path(f).stem: U.label_counts((label_dir / f"{Path(f).stem}.txt").read_text()) for f in files}
            predictions = _predict([checkpoint], files, imgsz)
            out[source] = U.count_accuracy(predictions, truth, cfg["yolo_thresholds"], cfg["yolo_iou"])
            print(f"{tag} {source}: {out[source]}", flush=True)
    (Path(RUNS) / name / "count_metrics.json").write_text(json.dumps(metrics, indent=2))
    vol.commit()
    return metrics


@app.local_entrypoint()
def adapt(plans: str, name: str, epochs: int = 8):
    payload = _read_plans(Path(plans), with_labels=True)
    if not payload:
        raise ValueError("Expected reviewed images and YOLO labels")
    print(f"Fine-tuning {name} with {len(payload)} reviewed crops")
    print(json.dumps(fit_adaptation.remote(name, payload, epochs), indent=2))


@app.function(image=image, volumes={VOL: vol}, gpu="H100", cpu=8, memory=32768, timeout=5400)
def fit_sliding_remote(name: str, epochs: int, tiles: bool) -> dict:
    """Preserve the sliding-door labels from the source SVG annotations."""
    import gzip

    from PIL import Image
    from ultralytics import YOLO

    from stoa_qa.cubicasa_live import tile_grid
    from stoa_qa import cubicasa_userplans as U

    def door_class(obj):
        parts = obj["c"].split()
        if len(parts) < 2 or parts[0] != "Door" or parts[1] == "None" or not obj.get("b"):
            return None
        return int(parts[1] in {"Slide", "ParallelSlide"})

    with gzip.open(f"{VOL}/objects.json.gz", "rt") as f:
        records = json.load(f)
    rng = random.Random(17)
    dest = Path("/tmp/sliding")
    counts = {}
    tile_size, imgsz = 640, 1024
    for split, positive_limit, negative_limit in (("train", 300, 200), ("val", 45, 45), ("test", 60, 60)):
        groups = {True: [], False: []}
        for plan in records:
            if plan["split"] != split:
                continue
            doors = [(cls, *obj["b"]) for obj in plan["objs"] if (cls := door_class(obj)) is not None]
            groups[any(b[0] == 1 for b in doors)].append((plan, doors))
        counts[split] = {}
        for has_slide, limit in ((True, positive_limit), (False, negative_limit)):
            group = sorted(groups[has_slide], key=lambda row: row[0]["name"])
            picked = rng.sample(group, min(limit, len(group)))
            counts[split]["with_sliding" if has_slide else "without_sliding"] = len(picked)
            for plan, doors in picked:
                stem = plan["name"]
                img = Path(CC) / "images" / split / f"{stem}.png"
                images = dest / "images" / split
                labels = dest / "labels" / split
                U._link(img, images / img.name)
                U.write_labels(labels / f"{stem}.txt", doors, plan["w"], plan["h"])
                if tiles and split == "train":
                    with Image.open(img) as im:
                        sx, sy = im.width / plan["w"], im.height / plan["h"]
                        scaled = [(c, a * sx, b * sy, d * sx, e * sy) for c, a, b, d, e in doors]
                        for i, (x0, y0, x1, y1) in enumerate(tile_grid(*im.size, tile_size)):
                            if (x0, y0, x1, y1) == (0, 0, *im.size):
                                continue
                            visible = [(c, a - x0, b - y0, d - x0, e - y0) for c, a, b, d, e in scaled
                                       if min(d, x1) - max(a, x0) > 2 and min(e, y1) - max(b, y0) > 2]
                            if any(b[0] == 1 for b in visible) or (visible and rng.random() < 0.25):
                                tile = f"{stem}_tile{i}"
                                im.crop((x0, y0, x1, y1)).save(images / f"{tile}.png")
                                U.write_labels(labels / f"{tile}.txt", visible, x1 - x0, y1 - y0)
    data = dest / "data.yaml"
    data.write_text(f"path: {dest}\ntrain: images/train\nval: images/val\ntest: images/test\n"
                    "names: [other_door, sliding_door]\n")
    print(f"Source plans by split: {counts}", flush=True)
    model = YOLO(_weights(["mixed-m13"])[0])
    model.add_callback("on_model_save", lambda trainer: vol.commit())
    model.train(data=str(data), project=RUNS, name=name, exist_ok=True, device=0, workers=6, seed=17,
                mosaic=0.0, **U.finetune_args(epochs=epochs, lr0=0.001, imgsz=imgsz, batch=12))
    best = YOLO(_weights([name])[0])
    result = best.val(data=str(data), split="test", imgsz=1024, batch=12, device=0, plots=False)
    metrics = {"plans": counts, "sliding_door_ap50": float(result.box.ap50[list(result.box.ap_class_index).index(1)])}
    (Path(RUNS) / name / "count_metrics.json").write_text(json.dumps(metrics, indent=2))
    vol.commit()
    return metrics


@app.local_entrypoint()
def fit_sliding(name: str = "sliding-v1", epochs: int = 12, tiles: bool = False):
    print(json.dumps(fit_sliding_remote.remote(name, epochs, tiles), indent=2))


@app.function(image=image, volumes={VOL: vol}, gpu="L40S", cpu=4, memory=16384, timeout=2400)
def fit_equipment_remote(name: str, plans: dict, epochs: int) -> dict:
    """Check that YOLO can learn the reviewed pump and valve symbols on the example sheet."""
    from ultralytics import YOLO

    from stoa_qa import cubicasa_userplans as U

    user = Path("/tmp/equipment-user")
    files = _write_user(plans, user, with_labels=True)
    dest = Path("/tmp/equipment")
    # Reuse the same sheet for an overfit check. This run has no independent validation data.
    entries = [("user", stem, split, f"{stem}_{i}")
               for stem in plans for split, repeat in (("train", 12), ("val", 1)) for i in range(repeat)]
    data = U.materialize(entries, user, None, dest)
    data.write_text(f"path: {dest}\ntrain: images/train\nval: images/val\nnames: [check_valve, fuel_pump]\n")
    model = YOLO(_weights(["mixed-m13"])[0])
    model.add_callback("on_model_save", lambda trainer: vol.commit())
    model.train(data=str(data), project=RUNS, name=name, exist_ok=True, device=0, workers=4, seed=17,
                mosaic=0.0, **U.finetune_args(epochs=epochs, lr0=0.001, imgsz=640, batch=8))
    preds = _predict(_weights([name]), files, 640)
    metrics = {"training_sheet_only": True, "counts_at_confidence_0_5": {
        stem: {c: sum(b[0] == i and b[1] >= 0.5 for b in boxes)
               for i, c in enumerate(("check_valve", "fuel_pump"))}
        for stem, boxes in preds.items()}}
    (Path(RUNS) / name / "count_metrics.json").write_text(json.dumps(metrics, indent=2))
    vol.commit()
    return metrics


@app.local_entrypoint()
def fit_equipment(plans: str, name: str = "equipment-pilot-v1", epochs: int = 25):
    payload = _read_plans(Path(plans), with_labels=True)
    if not payload:
        raise ValueError("Expected reviewed images and YOLO labels")
    print(json.dumps(fit_equipment_remote.remote(name, payload, epochs), indent=2))


@app.function(image=image, volumes={VOL: vol}, gpu="L40S", cpu=4, memory=16384, timeout=3600)
def sliding_threshold_remote(run: str) -> dict:
    """Choose an operating point on validation sheets and report its test count accuracy."""
    import gzip

    from PIL import Image

    from stoa_qa.cubicasa_rules import _iou

    with gzip.open(f"{VOL}/objects.json.gz", "rt") as f:
        records = json.load(f)
    truth = {}
    predictions = {}
    for split in ("val", "test"):
        plans = [p for p in records if p["split"] == split]
        truth[split], paths = {}, []
        for p in plans:
            path = f"{CC}/images/{split}/{p['name']}.png"
            with Image.open(path) as im:
                sx, sy = im.width / p["w"], im.height / p["h"]
            truth[split][p["name"]] = [[a * sx, b * sy, c * sx, d * sy] for o in p["objs"] if o.get("b")
                                       and o["c"].split()[:2] in (["Door", "Slide"], ["Door", "ParallelSlide"])
                                       for a, b, c, d in [o["b"]]]
            paths.append(path)
        predictions[split] = _predict(_weights([run]), paths, 1024)

    def score(split: str, threshold: float) -> dict:
        tp = predicted = expected = exact = 0
        for name, gt in truth[split].items():
            boxes = [b for b in predictions[split][name] if b[0] == 1 and b[1] >= threshold]
            remaining = list(gt)
            predicted += len(boxes)
            expected += len(gt)
            exact += len(boxes) == len(gt)
            for box in sorted(boxes, key=lambda b: -b[1]):
                hits = [_iou(box[2:6], other) for other in remaining]
                best = max(hits, default=0.0)
                if best >= 0.5:
                    remaining.pop(hits.index(best))
                    tp += 1
        return {"threshold": threshold, "precision": tp / max(1, predicted), "recall": tp / max(1, expected),
                "f1": (2 * tp) / max(1, predicted + expected), "exact_count": exact, "sheets": len(truth[split])}

    chosen = max((score("val", n / 20) for n in range(1, 20)), key=lambda s: (s["f1"], s["precision"]))
    out = {"validation": chosen, "test": score("test", chosen["threshold"])}
    (Path(RUNS) / run / "threshold_metrics.json").write_text(json.dumps(out, indent=2))
    vol.commit()
    return out


@app.local_entrypoint()
def sliding_threshold(run: str = "sliding-v1"):
    print(json.dumps(sliding_threshold_remote.remote(run), indent=2))


@app.function(image=image, volumes={VOL: vol}, gpu="L40S", cpu=8, memory=32768, timeout=5400)
def fit_wet_rooms_remote(name: str, extra_baths: dict, epochs: int) -> dict:
    import gzip

    from PIL import Image
    from ultralytics import YOLO

    from stoa_qa.cubicasa_rules import room_of
    from stoa_qa.wet_room import context

    with gzip.open(f"{VOL}/objects.json.gz", "rt") as f:
        records = json.load(f)
    by_name = {p["name"]: p for p in records}
    root = Path("/tmp/wet-rooms")
    rng = random.Random(21)
    names = ("bathroom", "wc_room")
    totals = {}
    for split, limit in (("train", 1100), ("val", 175), ("test", 200)):
        samples = {k: [] for k in names}
        for plan in records:
            if plan["split"] != split:
                continue
            rooms = [(k, obj["b"]) for obj in plan["objs"]
                     if obj.get("b") and (k := room_of(obj["c"], obj.get("t", ""))) in names]
            for i, obj in enumerate(plan["objs"]):
                if obj["c"] != "FixedFurniture Toilet" or not obj.get("b"):
                    continue
                x0, y0, x1, y1 = obj["b"]
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                match = {k for k, (a, b, c, d) in rooms if a <= cx <= c and b <= cy <= d}
                if len(match) == 1:
                    samples[match.pop()].append((plan["name"], i, obj["b"]))
        totals[split] = {}
        for kind, cases in samples.items():
            folder = root / split / kind
            folder.mkdir(parents=True, exist_ok=True)
            picked = rng.sample(cases, min(limit, len(cases)))
            totals[split][kind] = 0
            for stem, i, box in picked:
                with Image.open(Path(CC) / "images" / split / f"{stem}.png") as im:
                    sx, sy = im.width / by_name[stem]["w"], im.height / by_name[stem]["h"]
                    a, b, c, d = box
                    crop = context(im.convert("RGB"), [a * sx, b * sy, c * sx, d * sy])
                    if crop is not None:
                        crop.save(folder / f"{stem}_{i}.jpg", quality=94)
                        totals[split][kind] += 1
    for stem, payload in extra_baths.items():
        for i in range(12):
            (root / "train/bathroom" / f"extra_{stem}_{i}.png").write_bytes(payload["image"])
    print(f"Source fixture crops: {totals}, reviewed bathroom crops: {len(extra_baths)}", flush=True)

    model = YOLO("yolo11n-cls.pt")
    model.add_callback("on_model_save", lambda trainer: vol.commit())
    model.train(data=str(root), project=RUNS, name=name, epochs=epochs, imgsz=224, batch=64, device=0,
                exist_ok=True, seed=21, workers=6, fliplr=0.5, flipud=0.5, degrees=180, plots=False)
    model = YOLO(_weights([name])[0])
    scores = {}
    for split in ("val", "test"):
        cases = sorted(p for k in names for p in (root / split / k).glob("*.jpg"))
        predictions = model.predict([str(p) for p in cases], stream=True, imgsz=224, batch=32, verbose=False)
        scores[split] = [(case.parent.name == "bathroom", float(result.probs.data[0])) for case, result in zip(cases, predictions)]
    def measure(rows, threshold):
        correct = sum(expected and score >= threshold for expected, score in rows)
        predicted = sum(score >= threshold for _, score in rows)
        total = sum(expected for expected, _ in rows)
        return {"threshold": threshold, "precision": correct / max(predicted, 1),
                "recall": correct / max(total, 1), "f1": 2 * correct / max(predicted + total, 1),
                "samples": len(rows)}
    ranked = [measure(scores["val"], n / 20) for n in range(10, 20)]
    chosen = max((m for m in ranked if m["precision"] >= 0.85),
                 key=lambda m: m["f1"], default=max(ranked, key=lambda m: m["f1"]))
    metrics = {"dataset": totals, "val": chosen, "test": measure(scores["test"], chosen["threshold"])}
    (Path(RUNS) / name / "count_metrics.json").write_text(json.dumps(metrics, indent=2))
    vol.commit()
    return metrics


@app.local_entrypoint()
def fit_wet_rooms(plans: str, name: str = "wet-room-v1", epochs: int = 15):
    print(json.dumps(fit_wet_rooms_remote.remote(name, _read_plans(Path(plans), with_labels=False), epochs), indent=2))
