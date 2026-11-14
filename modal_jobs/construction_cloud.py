"""Train and score specialist construction drawing models on Modal."""
from __future__ import annotations

import json
import random
import shutil
import zipfile
from pathlib import Path

import modal

VOL = Path("/vol")
vol = modal.Volume.from_name("stoa-cubicasa")
app = modal.App("stoa-construction-specialists")
image = (modal.Image.debian_slim(python_version="3.12")
         .apt_install("libgl1", "libglib2.0-0")
         .pip_install("ultralytics==8.4.212", "pillow")
         .add_local_python_source("stoa_qa", ignore=["**/__pycache__/**"]))

WET_CLASSES = ("wc_room", "bathroom")
MEP_CLASSES = ("water_pipe", "other_pipe", "valve", "pipe_fitting", "pump")
PID_CLASSES = ("valve", "pump", "instrumentation")


def _data(root, names):
    path = root / "data.yaml"
    path.write_text(f"path: {root}\ntrain: images/train\nval: images/val\ntest: images/test\nnames: {list(names)!r}\n")
    return path


def _fit(root, names, run, epochs, mode, imgsz, batch, base):
    from ultralytics import YOLO

    data = _data(root, names)
    destination = VOL / "runs" / run
    if destination.exists():
        raise FileExistsError(destination)
    model = YOLO(base)
    model.add_callback("on_model_save", lambda _: vol.commit())
    model.train(data=str(data), epochs=epochs, imgsz=imgsz, batch=batch, workers=6, device=0, seed=123,
                project=str(destination.parent), name=run, mosaic=0.1, close_mosaic=5, flipud=0.25, plots=False,
                save_period=4)
    model = YOLO(str(destination / "weights/best.pt"))
    result = model.val(data=str(data), split="test", imgsz=imgsz, batch=batch, device=0, plots=False)
    metric = result.seg if mode == "segment" else result.box
    scores = {"test_images": len(list((root / "images/test").iterdir())), "ap50": float(metric.map50),
              "ap50_95": float(metric.map), "classes": {
                  names[i]: {"ap50": float(ap), "precision": float(p), "recall": float(r)}
                  for i, ap, p, r in zip(metric.ap_class_index, metric.ap50, metric.p, metric.r)}}
    manifest = root / "manifest.json"
    if manifest.exists():
        scores["dataset"] = json.loads(manifest.read_text())
    (destination / "metrics.json").write_text(json.dumps(scores, indent=2))
    vol.commit()
    return scores


@app.function(image=image, volumes={str(VOL): vol}, gpu="H100", cpu=10, memory=32768, timeout=6000)
def train_bathroom_boxes(run: str, epochs: int):
    from PIL import Image

    from stoa_qa.cubicasa_rules import CLASSES
    from stoa_qa.cubicasa_userplans import from_yolo_text, write_labels

    rng = random.Random(23)
    dest = Path("/tmp/wet-boxes")
    limits = {"train": (720, 340), "val": (150, 64), "test": (300, 85)}
    wet_indices = [CLASSES.index(name) for name in WET_CLASSES]
    stats = {}
    for split, sizes in limits.items():
        images, labels = dest / "images" / split, dest / "labels" / split
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        stats[split] = {}
        for group, limit in zip(("yolo", "swiss"), sizes):
            source = VOL / group
            pool = sorted((source / "images" / split).glob("*.png"))
            selected = rng.sample(pool, min(limit, len(pool)))
            tile_count = 0
            for file in selected:
                name = f"{group}-{file.stem}"
                label = (source / "labels" / split / f"{file.stem}.txt").read_text()
                rows = [line.split() for line in label.splitlines() if line and int(line.split()[0]) in wet_indices]
                raw = "".join(f"{wet_indices.index(int(cls))} {' '.join(coords)}\n" for cls, *coords in rows)
                shutil.copyfile(file, images / f"{name}.png")
                (labels / f"{name}.txt").write_text(raw)
                if split != "train" or not rows:
                    continue
                with Image.open(file) as im:
                    boxes = from_yolo_text(raw, *im.size)
                    for i, (_, _, a, b, c, d) in enumerate(rng.sample(boxes, min(2, len(boxes)))):
                        side = max(96, round(max(c - a, d - b) * rng.uniform(2.8, 5.5)))
                        x = max(0, min(max(0, im.width - side), round((a + c - side) / 2)))
                        y = max(0, min(max(0, im.height - side), round((b + d - side) / 2)))
                        x1, y1 = min(x + side, im.width), min(y + side, im.height)
                        tile = im.crop((x, y, x1, y1))
                        visible = [(kind, l - x, t - y, r - x, bot - y) for kind, _, l, t, r, bot in boxes
                                   if min(r, x1) - max(l, x) > 2 and min(bot, y1) - max(t, y) > 2]
                        tile.save(images / f"{name}-{i}.png")
                        write_labels(labels / f"{name}-{i}.txt", visible, *tile.size)
                        tile_count += 1
            stats[split][group] = {"sheets": len(selected), "additional_tiles": tile_count}
            print(f"Prepared {split} {group}: {len(selected)} sheets, {tile_count} tiles", flush=True)
    (dest / "manifest.json").write_text(json.dumps(stats))
    return _fit(dest, WET_CLASSES, run, epochs, "detect", 1024, 16, str(VOL / "runs/mixed-m13/weights/best.pt"))


@app.function(image=image, volumes={str(VOL): vol}, gpu="H100", cpu=10, memory=32768, timeout=6000)
def train_mep(run: str, archive: str, epochs: int):
    dest = Path("/tmp/mep-data")
    with zipfile.ZipFile(VOL / "datasets" / Path(archive).name) as zf:
        zf.extractall(dest)
    return _fit(dest, MEP_CLASSES, run, epochs, "segment", 640, 16, "yolo11s-seg.pt")


@app.function(image=image, volumes={str(VOL): vol}, gpu="L40S", cpu=8, memory=32768, timeout=6000)
def train_pid(run: str, archive: str, epochs: int):
    dest = Path("/tmp/pid-data")
    with zipfile.ZipFile(VOL / "datasets" / Path(archive).name) as zf:
        zf.extractall(dest)
    return _fit(dest, PID_CLASSES, run, epochs, "detect", 960, 16, "yolo11s.pt")


@app.local_entrypoint()
def bathrooms(run: str = "bathroom-boxes-v1", epochs: int = 16):
    print(json.dumps(train_bathroom_boxes.remote(run, epochs), indent=2))


@app.local_entrypoint()
def mep(run: str = "plumbing-ifc-v1", archive: str = "plumbing-ifc-v1.zip", epochs: int = 24):
    print(json.dumps(train_mep.remote(run, archive, epochs), indent=2))


@app.local_entrypoint()
def pid(run: str = "pid2graph-real-v1", archive: str = "pid2graph-real.zip", epochs: int = 24):
    print(json.dumps(train_pid.remote(run, archive, epochs), indent=2))


@app.function(image=image, volumes={str(VOL): vol}, gpu="H100", cpu=8, memory=32768, timeout=4800)
def train_pipe_lines(run: str, archive: str, epochs: int):
    """Pixel segmentation with overlapping labels preserved, for pipe linework review."""
    import numpy as np
    import torch
    from PIL import Image, ImageDraw
    from torch import nn
    from torch.utils.data import DataLoader, Dataset

    root = Path("/tmp/pipe-lines")
    with zipfile.ZipFile(VOL / "datasets" / Path(archive).name) as zf:
        zf.extractall(root)
    dest = VOL / "runs" / run
    dest.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(17)
    side = 384

    class Patches(Dataset):
        def __init__(self, split):
            self.split = split
            self.files = sorted((root / "images" / split).glob("*.png"))

        def __len__(self):
            return len(self.files)

        def __getitem__(self, index):
            path = self.files[index]
            with Image.open(path) as im:
                pixels = np.array(im.convert("L").resize((side, side)), dtype=np.float32) / 255
            masks = [Image.new("L", (side, side)) for _ in range(2)]
            brushes = [ImageDraw.Draw(m) for m in masks]
            for label in (root / "labels" / self.split / f"{path.stem}.txt").read_text().splitlines():
                cls, *flat = label.split()
                if cls not in ("0", "1"):
                    continue
                coords = [max(0, min(side-1, round(float(v) * side))) for v in flat]
                polygon = list(zip(coords[::2], coords[1::2]))
                brushes[0].polygon(polygon, fill=1)
                if cls == "0":
                    brushes[1].polygon(polygon, fill=1)
            return torch.from_numpy(pixels[None]), torch.from_numpy(np.stack(masks).astype(np.float32))

    def block(n, m):
        return nn.Sequential(nn.Conv2d(n, m, 3, padding=1, bias=False), nn.BatchNorm2d(m), nn.SiLU(),
                             nn.Conv2d(m, m, 3, padding=1, bias=False), nn.BatchNorm2d(m), nn.SiLU())

    class PipeNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.low = block(1, 20)
            self.mid = block(20, 40)
            self.deep = block(40, 80)
            self.up1 = block(120, 40)
            self.up2 = block(60, 20)
            self.output = nn.Conv2d(20, 2, 1)

        def forward(self, x):
            pool = nn.functional.max_pool2d
            resize = nn.functional.interpolate
            low = self.low(x)
            mid = self.mid(pool(low, 2))
            deep = self.deep(pool(mid, 2))
            up = self.up1(torch.cat((mid, resize(deep, scale_factor=2, mode="nearest")), dim=1))
            return self.output(self.up2(torch.cat((low, resize(up, scale_factor=2, mode="nearest")), dim=1)))

    loaders = {s: DataLoader(Patches(s), batch_size=24, shuffle=s == "train", num_workers=4,
                             pin_memory=True, persistent_workers=True) for s in ("train", "val", "test")}
    model = PipeNet().cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=0.002)

    @torch.no_grad()
    def measure(split):
        model.eval()
        intersections = torch.zeros(2, device="cuda")
        unions = torch.zeros_like(intersections)
        for xs, ys in loaders[split]:
            ys = ys.cuda(non_blocking=True).bool()
            pred = model(xs.cuda(non_blocking=True)) > 0
            intersections += (ys & pred).sum((0, 2, 3))
            unions += (ys | pred).sum((0, 2, 3))
        return (intersections / unions.clamp_min(1)).tolist()

    best = -1.0
    for epoch in range(epochs):
        model.train()
        for xs, ys in loaders["train"]:
            xs, ys = xs.cuda(non_blocking=True), ys.cuda(non_blocking=True)
            out = model(xs)
            probs = out.sigmoid()
            dice = (2 * (probs * ys).sum((0, 2, 3)) + 1) / ((probs + ys).sum((0, 2, 3)) + 1)
            loss = nn.functional.binary_cross_entropy_with_logits(out, ys) + (1 - dice.mean())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        val = measure("val")
        print(f"Epoch {epoch + 1}/{epochs} validation IoU: pipes={val[0]:.3f}, water={val[1]:.3f}", flush=True)
        if val[0] > best:
            best = val[0]
            model.eval()
            torch.jit.trace(model, torch.zeros((1, 1, side, side), device="cuda")).save(str(dest / "best.torchscript"))
            vol.commit()
    model = torch.jit.load(str(dest / "best.torchscript"), map_location="cuda")
    iou = measure("test")
    metrics = {"test_images": len(loaders["test"].dataset), "any_pipe_pixel_iou": iou[0], "water_pipe_pixel_iou": iou[1]}
    (dest / "metrics.json").write_text(json.dumps(metrics, indent=2))
    vol.commit()
    return metrics


@app.local_entrypoint()
def pipe_lines(run: str = "pipe-lines-v1", archive: str = "plumbing-ifc-v1.zip", epochs: int = 10):
    print(json.dumps(train_pipe_lines.remote(run, archive, epochs), indent=2))
