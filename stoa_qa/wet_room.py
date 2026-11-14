"""Read the local drawing around a toilet to distinguish bathrooms from separate WCs."""
from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path


def context(image, box: list[float]):
    x0, y0, x1, y1 = box
    if x1 <= x0 or y1 <= y0 or x1 <= 0 or y1 <= 0 or x0 >= image.width or y0 >= image.height:
        return None
    size = max(x1 - x0, y1 - y0) * 4.5
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    bounds = (
        max(0, math.floor(cx - size / 2)),
        max(0, math.floor(cy - size / 2)),
        min(image.width, math.ceil(cx + size / 2)),
        min(image.height, math.ceil(cy + size / 2)),
    )
    return image.crop(bounds)


@lru_cache(maxsize=2)
def _classifier(weights, version):
    from ultralytics import YOLO

    model = YOLO(str(weights))
    if model.task != "classify" or model.names != {0: "bathroom", 1: "wc_room"}:
        raise ValueError(f"{weights.name} has unexpected wet-room classes")
    return model


def bath_scores(image: Path, boxes: list[list[float]], run: str) -> list[float] | None:
    """One score per toilet, using the same scaled crop as in training."""
    from PIL import Image

    from . import cubicasa_live as live

    checkpoint = live.weight_file(run)
    if checkpoint is None or not live.available(run):
        return None
    try:
        with Image.open(image) as im, live._lock:
            model = _classifier(checkpoint, live._weight_version(checkpoint))
            scores = []
            for i in range(0, len(boxes), 32):
                crops = [context(im, box) for box in boxes[i:i + 32]]
                valid = [crop for crop in crops if crop is not None]
                results = iter(model.predict(valid, imgsz=224, verbose=False)) if valid else iter(())
                scores.extend(float(next(results).probs.data[0]) if crop is not None else 0.0 for crop in crops)
            return scores
    except Exception:
        live.LOG.exception("Wet-room classifier could not read this plan")
        return None
