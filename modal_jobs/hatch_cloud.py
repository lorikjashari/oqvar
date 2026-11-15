"""Train the material-hatch classifier on Modal (stoa_qa.hatch uses it on drawings).

Every training patch is synthetic (stoa_qa.hatch_synth: AutoCAD library patterns + procedural CAD styles, rendered at any
scale / angle / ink with sheet clutter), generated on the fly by the data-loader workers, so no dataset is uploaded.
A ResNet-18 (ImageNet weights, greyscale repeated to 3 channels) learns the 17 hatch classes on 96 px patches.

Held-out checks, both written to runs/<name>/metrics.json on the volume:
  * a fixed synthetic val set (seed 777, never used for training);
  * the hand-labelled real patches in /vol/hatch/real/<split>/ (cropped from real sheets by _analysis/hatch_real.py),
    when uploaded: `modal volume put stoa-cubicasa eval_out/hatch_real hatch/real`. Only split "val" picks the
    checkpoint; split "test" is reported, never used for a choice.

Run (keep the client alive; see CLAUDE.local.md on --detach):
    .venv/Scripts/modal run modal_jobs/hatch_cloud.py::train --name h17-r18 --steps 6000
    modal volume get stoa-cubicasa hatch/runs/h17-r18/best.pt third_party/weights/hatch-h17-r18.pt
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import modal

VOL = Path("/vol")
vol = modal.Volume.from_name("stoa-cubicasa")
app = modal.App("stoa-hatch")
image = (modal.Image.debian_slim(python_version="3.12")
         .apt_install("fonts-dejavu-core")
         .pip_install("torch==2.6.0", "torchvision==0.21.0", "pillow", "numpy", "ezdxf==1.4.2")
         .add_local_python_source("stoa_qa", ignore=["**/__pycache__/**"]))


def _real(split: str):
    """Hand-labelled real patches: /vol/hatch/real/<split>/<class>/*.png -> (tensor N x 1 x 96 x 96, labels)."""
    import numpy as np
    import torch
    from PIL import Image

    from stoa_qa.hatch_synth import IDS, PATCH

    root = VOL / "hatch" / "real" / split
    xs, ys = [], []
    if not root.is_dir():
        return None
    for d in sorted(root.iterdir()):
        if d.name not in IDS:
            continue
        for f in sorted(d.glob("*.png")):
            im = Image.open(f).convert("L").resize((PATCH, PATCH), Image.LANCZOS)
            xs.append(np.asarray(im, dtype=np.float32) / 255.0)
            ys.append(IDS[d.name])
    if not xs:
        return None
    return torch.tensor(np.stack(xs))[:, None], torch.tensor(ys)


def _evaluate(model, x, y, device) -> dict:
    import torch

    from stoa_qa.hatch_synth import CLASSES

    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(x), 512):
            b = x[i:i + 512].to(device)
            preds.append(model(_norm(b)).argmax(1).cpu())
    p = torch.cat(preds)
    acc = float((p == y).float().mean())
    per = {}
    for k, c in enumerate(CLASSES):
        m = y == k
        if m.any():
            per[c] = round(float((p[m] == k).float().mean()), 3)
    conf = torch.zeros(len(CLASSES), len(CLASSES), dtype=torch.int64)
    for t, q in zip(y.tolist(), p.tolist()):
        conf[t, q] += 1
    return {"acc": round(acc, 4), "n": len(y), "per_class": per, "confusion": conf.tolist()}


def _norm(b):
    return ((b - 0.5) / 0.25).repeat(1, 3, 1, 1)


def build_model(n_classes: int, pretrained: bool = True):
    import torch.nn as nn
    import torchvision

    m = torchvision.models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
    m.fc = nn.Linear(m.fc.in_features, n_classes)
    return m


def _real_score(model, data, device) -> tuple[float, dict]:
    """Checkpoint score on real patches: hatched-vs-not balanced accuracy and the class accuracy on hatched ones,
    averaged (plain accuracy would be dominated by the many plain windows)."""
    import torch

    x, y = data
    model.eval()
    with torch.no_grad():
        p = torch.cat([model(_norm(x[i:i + 512].to(device))).argmax(1).cpu() for i in range(0, len(x), 512)])
    hatched = y != 0
    rec_none = float((p[~hatched] == 0).float().mean()) if (~hatched).any() else 0.0
    rec_hatch = float((p[hatched] != 0).float().mean()) if hatched.any() else 0.0
    cls_acc = float((p[hatched] == y[hatched]).float().mean()) if hatched.any() else 0.0
    score = 0.5 * (rec_none + rec_hatch) / 2 + 0.5 * cls_acc
    return score, {"none_recall": round(rec_none, 3), "hatch_recall": round(rec_hatch, 3), "hatch_class_acc": round(cls_acc, 3),
                   "acc": round(float((p == y).float().mean()), 3)}


@app.function(image=image, volumes={VOL: vol}, gpu="L40S", cpu=24, memory=98304, timeout=4 * 3600)
def train(name: str = "h17-r18", steps: int = 6000, batch: int = 256, lr: float = 2e-3, workers: int = 22,
          real_mix: bool = False) -> dict:
    """real_mix: use /vol/hatch/bg.npz (real sheet windows, _analysis/hatch_bg.py): real windows as "none", real Swiss
    wall hatches as positives, and synthetic hatches composited onto real windows."""
    import random

    import numpy as np
    import torch
    import torch.nn as nn

    from stoa_qa import hatch_synth as hs

    bg_x = bg_y = None
    if real_mix:
        d = np.load(VOL / "hatch" / "bg.npz")
        bg_x, bg_y = d["x"], d["y"]
        print("real windows", len(bg_x), {int(k): int((bg_y == k).sum()) for k in np.unique(bg_y)}, flush=True)
    bg_idx = np.nonzero(bg_y == -1)[0] if real_mix else None
    # labelled real windows (Swiss DXF-labelled walls + hand-labelled active-learning picks), drawn class-balanced
    by_class = {int(k): np.nonzero(bg_y == k)[0] for k in np.unique(bg_y) if k >= 0} if real_mix else {}

    class Synth(torch.utils.data.IterableDataset):
        def __iter__(self):
            info = torch.utils.data.get_worker_info()
            rng = random.Random((info.id if info else 0) * 100003 + int(time.time() * 1000) % 100000)
            while True:
                r = rng.random()
                if real_mix and r < 0.30:      # a real window: no hatch (mostly; ~5% label noise)
                    im, k = hs.Image.fromarray(bg_x[rng.choice(bg_idx)]), 0
                elif real_mix and r < 0.42:    # a labelled real window (DXF wall hatch or hand label), class-balanced
                    j = rng.choice(by_class[rng.choice(list(by_class))])
                    im, k = hs.Image.fromarray(bg_x[j]), int(bg_y[j])
                elif real_mix and r < 0.72:    # a synthetic hatch drawn into a real window
                    im, k = hs.composite(bg_x[rng.choice(bg_idx)], rng)
                else:
                    im, k = hs.sample(rng)
                if rng.random() < 0.5:
                    im = im.transpose(0)  # mirror: hatches have no handedness that matters
                if rng.random() < 0.5:
                    im = im.transpose(5)  # transpose: a rotated sheet
                yield torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0)[None], k

    out = VOL / "hatch" / "runs" / name
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda"
    rng = random.Random(777)
    vx, vy = zip(*[hs.sample(rng, c) for c in hs.CLASSES for _ in range(200)])
    val_x = torch.tensor(np.stack([np.asarray(v, dtype=np.float32) / 255.0 for v in vx]))[:, None]
    val_y = torch.tensor(vy)
    real_val, real_test = _real("val"), _real("test")

    model = build_model(len(hs.CLASSES)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    loader = torch.utils.data.DataLoader(Synth(), batch_size=batch, num_workers=workers, persistent_workers=True,
                                         prefetch_factor=4)
    best, history, t0 = -1.0, [], time.time()
    it = iter(loader)
    for step in range(1, steps + 1):
        model.train()
        x, y = next(it)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = loss_fn(model(_norm(x)), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 500 == 0 or step == steps:
            rec = {"step": step, "loss": round(float(loss), 4), "sec": round(time.time() - t0),
                   "synth_val": _evaluate(model, val_x, val_y, device)["acc"]}
            if real_val is not None:
                s, detail = _real_score(model, real_val, device)
                rec["real_val"] = round(s, 4)
                rec["real_val_detail"] = detail
            score = rec.get("real_val", rec["synth_val"]) + 0.001 * rec["synth_val"]
            if score > best:
                best = score
                torch.save({"state": model.state_dict(), "classes": hs.CLASSES, "step": step, "arch": "resnet18"},
                           out / "best.pt")
            torch.save({"state": model.state_dict(), "classes": hs.CLASSES, "step": step, "arch": "resnet18"}, out / "last.pt")
            history.append(rec)
            print(json.dumps(rec), flush=True)
            (out / "history.json").write_text(json.dumps(history, indent=1))
            vol.commit()

    ck = torch.load(out / "best.pt", map_location=device)
    model.load_state_dict(ck["state"])
    metrics = {"best_step": ck["step"], "synth_val": _evaluate(model, val_x, val_y, device)}
    if real_val is not None:
        metrics["real_val"] = {**_evaluate(model, *real_val, device), **_real_score(model, real_val, device)[1]}
    if real_test is not None:
        metrics["real_test"] = {**_evaluate(model, *real_test, device), **_real_score(model, real_test, device)[1]}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    vol.commit()
    return {k: ({kk: vv for kk, vv in v.items() if kk not in ("confusion", "per_class")} if isinstance(v, dict) else v)
            for k, v in metrics.items()}


@app.function(image=image, volumes={VOL: vol}, gpu="L40S", cpu=4, memory=16384, timeout=1800)
def evaluate(name: str) -> dict:
    """Re-score a run's best checkpoint on the real val / test patches currently on the volume."""
    import torch

    from stoa_qa import hatch_synth as hs

    ck = torch.load(VOL / "hatch" / "runs" / name / "best.pt", map_location="cuda")
    model = build_model(len(hs.CLASSES), pretrained=False).cuda()
    model.load_state_dict(ck["state"])
    out = {}
    for split in ("val", "test"):
        r = _real(split)
        if r is not None:
            out[split] = _evaluate(model, *r, "cuda")
    return out


@app.local_entrypoint()
def main(name: str = "h17-r18", steps: int = 6000, real_mix: bool = False):
    print(train.remote(name=name, steps=steps, real_mix=real_mix))
