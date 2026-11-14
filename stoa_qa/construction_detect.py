"""Candidate overlays from the specialist raster detectors."""
from __future__ import annotations

import re
from pathlib import Path

from . import cubicasa_live as live
from .core import Answer, Evidence

PID_CLASSES = ["valve", "pump", "instrumentation"]
PID_ASK = re.compile(
    r"(?:(?:highlight|mark|show me|count)(?: all| the)?|how many) "
    r"(?P<item>valves|instruments|instrumentation)(?: are (?:shown|there))?"
    r"(?: (?:in|on) (?:this|the) (?:plan|drawing|sheet|piping plan|p&id))?[?.]*", re.I,
)
PIPE_TITLE = re.compile(r"pipingplan|pipingandinstrumentationdiagram|p&id", re.I)
WET_CLASSES = ["wc_room", "bathroom"]


def boxes(path: Path, cfg: dict, classes: list[str]) -> list | None:
    run, size = cfg["run"], cfg["imgsz"]
    if not live.available(run):
        return None

    def predict():
        model = live._load(run, classes)
        with live._lock:
            if cfg.get("tile"):
                return live.predict_tiled(model, path, size, cfg["tile"], conf=0.15,
                                          tile_classes=range(len(classes)))
            result = model.predict(str(path), imgsz=size, conf=0.15, verbose=False)[0].boxes
        return [[int(c), float(p), *xy]
                for c, p, xy in zip(result.cls.tolist(), result.conf.tolist(), result.xyxy.tolist())]

    try:
        return live._cached(run, live.image_hash(path),
                            f".special{size}.t{cfg.get('tile', 0)}.{live.TILE_CACHE_VERSION}", predict)
    except Exception:
        live.LOG.exception("Specialist detector %s could not read this drawing", run)
        return None


def pid_answer(image: Path, question, rows) -> Answer | None:
    cfg = (live.live_config() or {}).get("pid_symbols")
    if not cfg or not rows or not any(PIPE_TITLE.fullmatch(re.sub(r"[\s-]+", "", row["text"])) for row in rows):
        return None
    item = question["item"].lower()
    kind, singular = ("valve", "valve") if item == "valves" else ("instrumentation", "instrument")
    found = boxes(image, cfg, PID_CLASSES)
    selected = [row for row in found or () if row[0] == PID_CLASSES.index(kind) and row[1] >= cfg["threshold"]]
    if not selected:
        return None
    ev = [Evidence("box", f"pid-{i}", f"Possible {singular} {i + 1}", row[1], [row[2:4], row[4:6]])
          for i, row in enumerate(selected)]
    return Answer(f"{len(ev)} possible {singular}{'s' if len(ev) != 1 else ''}", None, "pid_candidates", 0.45,
                  "Candidate symbols from the P&ID detector. Review the marked symbols on the sheet.", ev)


def wet_room_boxes(image: Path, kind: str) -> list[list] | None:
    cfg = (live.live_config() or {}).get("wet_room_boxes")
    if not cfg:
        return None
    found = boxes(image, cfg, WET_CLASSES)
    cls = WET_CLASSES.index(kind)
    return [b for b in found or () if b[0] == cls and b[1] >= cfg["threshold"]]
