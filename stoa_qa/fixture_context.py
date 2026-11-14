"""Join bathroom candidates with the plumbing fixtures inside their boundaries."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import hypot

from .cubicasa_rules import _inside, _iou

CLASSES = ["toilet", "sink", "shower", "bathtub", "washing_machine", "stove", "shower_screen", "bathroom", "wc_room"]
FIXTURES = tuple(CLASSES[:4])
ROOMS = ("bathroom", "wc_room")
IDS = {name: index for index, name in enumerate(CLASSES)}


def unique(boxes: list[list], threshold: float = 0.45) -> list[list]:
    kept = []
    for b in sorted(boxes, key=lambda r: -r[1]):
        if all(_iou(b[2:6], k[2:6]) <= threshold and not (
            _inside(b[2:6], k[2:6], 0.88) or _inside(k[2:6], b[2:6], 0.88)
        ) for k in kept):
            kept.append(b)
    return kept


@dataclass
class WetRoom:
    box: list
    kind: str
    fixtures: list[list]

    @property
    def fixture_types(self) -> set[str]:
        return {CLASSES[b[0]] for b in self.fixtures}


def room_context(boxes: list[list], thresholds: dict[str, float], room_threshold: float) -> list[WetRoom]:
    """Assign each fixture to its best room. Surface candidate rooms only with corroborating fixtures."""
    rooms = unique([b for b in boxes if CLASSES[b[0]] in ROOMS and b[1] >= room_threshold])
    # A toilet and a sink establish a WC. A shower or tub and at least one other fixture corroborate a bathroom.
    # A separately drawn shower screen does not establish a shower.
    fixtures = selected_fixtures(boxes, thresholds)
    assigned: list[list] = [[] for _ in rooms]
    for fixture in fixtures:
        fitting = [i for i, room in enumerate(rooms) if _inside(fixture[2:6], room[2:6], 0.72)]
        if not fitting:
            continue
        best = max(fitting, key=lambda i: (rooms[i][1], -(rooms[i][4] - rooms[i][2]) * (rooms[i][5] - rooms[i][3])))
        assigned[best].append(fixture)
    matches = []
    for room, nearby in zip(rooms, assigned):
        kinds = {CLASSES[b[0]] for b in nearby}
        if kinds & {"shower", "bathtub"} and kinds & {"toilet", "sink"}:
            matches.append(WetRoom(room, "bathroom", nearby))
        elif kinds >= {"toilet", "sink"} and CLASSES[room[0]] == "wc_room":
            matches.append(WetRoom(room, "wc_room", nearby))
    return matches


def fixture_groups(fixtures: list[list], existing: list[WetRoom]) -> list[WetRoom]:
    """Find compact three-fixture groups where room outlines could not be read."""
    def center(box):
        x0, y0, x1, y1 = box[2:6]
        return ((x0 + x1) / 2, (y0 + y1) / 2)

    def scale(box):
        return max(box[4] - box[2], box[5] - box[3])

    def distance(a, b):
        x, y = center(a)
        u, v = center(b)
        return hypot(x - u, y - v)

    used = {id(b) for room in existing for b in room.fixtures}
    by_kind = {k: [b for b in fixtures if b[0] == IDS[k] and id(b) not in used] for k in FIXTURES}
    links = []
    for toilet in by_kind["toilet"]:
        size = scale(toilet)
        basins = [b for b in by_kind["sink"] if distance(toilet, b) < size * 2.25]
        bath = [b for b in by_kind["shower"] + by_kind["bathtub"] if distance(toilet, b) < size * 3.3]
        for basin in basins:
            for wash in bath:
                if distance(basin, wash) < size * 4.0:
                    links.append(((distance(toilet, basin) + distance(toilet, wash)) / size,
                                  [toilet, basin, wash]))
    result = []
    for _, group in sorted(links, key=lambda pair: pair[0]):
        if any(id(b) in used for b in group):
            continue
        pad = scale(group[0]) * 0.45
        bounds = [min(b[2] for b in group) - pad, min(b[3] for b in group) - pad,
                  max(b[4] for b in group) + pad, max(b[5] for b in group) + pad]
        if any(_iou(bounds, r.box[2:6]) > 0.35 for r in existing + result):
            continue
        score = min(b[1] for b in group)
        result.append(WetRoom([IDS["bathroom"], score, *bounds], "bathroom", group))
        used.update(id(b) for b in group)
    return result


def fixture_crop(image, box, padding=1.7):
    """Square crop centered on a fixture, with its immediate surroundings."""
    x0, y0, x1, y1 = box
    if x1 <= x0 or y1 <= y0 or x1 <= 0 or y1 <= 0 or x0 >= image.width or y0 >= image.height:
        return None
    from PIL import Image

    side = max(8, round(max(x1-x0, y1-y0) * padding))
    x, y = round((x0+x1-side)/2), round((y0+y1-side)/2)
    clipped = (max(0,x), max(0,y), min(image.width,x+side), min(image.height,y+side))
    canvas = Image.new("RGB", (side, side), "white")
    canvas.paste(image.crop(clipped), (clipped[0]-x, clipped[1]-y))
    return canvas


QUESTION_CLASSES = {"toilets": ("toilet",), "sinks": ("sink",), "baths": ("shower", "bathtub"),
                    "showers": ("shower",), "bathtubs": ("bathtub",), "bathrooms": ("bathroom",), "wc_rooms": ("wc_room",)}


def selected_fixtures(boxes, thresholds):
    """Resolve competing fixture classes over the same mark before counting."""
    candidates = unique([b for b in boxes if CLASSES[b[0]] not in ROOMS])
    return [b for b in candidates if (kind := CLASSES[b[0]]) in FIXTURES and b[1] >= thresholds[kind]]


def infer(model, image, imgsz, tile):
    """Use both full-sheet and tiled views to preserve small fixtures and larger room boundaries."""
    from PIL import Image

    from . import cubicasa_live as live

    with Image.open(image) as im:
        full = model.predict(str(image), imgsz=imgsz, conf=0.12, verbose=False, max_det=1200)[0].boxes
        rows = [[int(c), float(p), *b] for c,p,b in zip(full.cls.tolist(), full.conf.tolist(), full.xyxy.tolist())]
        if max(im.size) > tile * 1.5:
            rows += live.predict_tiled(model, image, imgsz, tile, conf=0.12, full=rows,
                                       tile_classes=range(len(CLASSES)))
    return live._nms(rows, 0.45)


def detect(image, config):
    from . import cubicasa_live as live

    run, size, tile = (config[key] for key in ("run", "imgsz", "tile"))
    if not live.available(run):
        return None

    def predict():
        detector = live._load(run, CLASSES)
        with live._lock:
            return infer(detector, image, size, tile)

    try:
        return live._cached(run, live.image_hash(image), f".fixture-context-{size}-{tile}-v1", predict)
    except Exception:
        live.LOG.exception("Fixture context detector %s could not read this drawing", run)
        return None


@lru_cache(maxsize=2)
def _verifier(checkpoint, version):
    from ultralytics import YOLO

    model = YOLO(str(checkpoint))
    if model.task != "classify" or set(model.names.values()) != {*FIXTURES, "other"}:
        raise ValueError(f"{checkpoint.name} has unexpected verifier classes")
    return model


def verify_fixtures(image, boxes, run, confidence):
    """Reject a detection when a context crop strongly supports a conflicting class."""
    from PIL import Image

    from . import cubicasa_live as live

    if not live.available(run):
        return boxes
    weights = live.weight_file(run)
    try:
        with Image.open(image) as im, live._lock:
            model = _verifier(weights, live._weight_version(weights))
            candidates = [b for b in boxes if CLASSES[b[0]] in FIXTURES]
            rejected = set()
            for i in range(0, len(candidates), 32):
                pairs = [(b, fixture_crop(im, b[2:6])) for b in candidates[i:i+32]]
                pairs = [(b, crop) for b, crop in pairs if crop is not None]
                if not pairs:
                    continue
                predictions = model.predict([crop for _, crop in pairs], imgsz=160, verbose=False)
                for (b, _), result in zip(pairs, predictions):
                    likely = model.names[result.probs.top1]
                    if likely != CLASSES[b[0]] and float(result.probs.top1conf) >= confidence:
                        rejected.add(id(b))
            return [b for b in boxes if id(b) not in rejected]
    except Exception:
        live.LOG.exception("Fixture verifier %s could not read this drawing", run)
        return boxes


def answer(image, path, key, *, candidates_only=False):
    from PIL import Image

    from . import cubicasa_live as live
    from .core import Answer, Evidence

    configured = live.live_config() or {}
    if key not in QUESTION_CLASSES:
        return None
    with Image.open(image) as plan:
        width, height = plan.size
    config = configured.get("fixture_context")
    if not config or max(width, height) < config.get("min_side", 0):
        config = configured.get("fixture_basics") if candidates_only else None
    if not config:
        return None
    boxes = detect(image, config)
    if boxes is None:
        return None
    if verifier := config.get("verifier"):
        boxes = verify_fixtures(image, boxes, verifier["run"], verifier["confidence"])
    requested = QUESTION_CLASSES[key]
    all_rooms = room_context(boxes, config["thresholds"], config["room_threshold"])
    if len(all_rooms) < config.get("min_rooms", 0) and not candidates_only:
        return None
    if key in ("bathrooms", "wc_rooms"):
        rooms = [r for r in all_rooms if r.kind in requested]
        if key == "bathrooms" and (config.get("fixture_groups") or candidates_only):
            extra = fixture_groups(selected_fixtures(boxes, config["thresholds"]), all_rooms)
            rooms += extra
        else:
            extra = []
        if not rooms:
            return None
        name = "Bathroom" if key == "bathrooms" else "WC room"
        evidence = []
        for i, r in enumerate(rooms, 1):
            title = f"Possible {name.lower()}" if candidates_only or r in extra else name
            fixtures = ", ".join(sorted(r.fixture_types))
            x0, y0, x1, y1 = r.box[2:]
            evidence.append(Evidence("box", f"fixture-room-{i}", f"{title} {i} · {fixtures}", r.box[1],
                                     [[max(x0, 0), max(y0, 0)], [min(x1, width), min(y1, height)]]))
        provisional = candidates_only or bool(extra)
        value = f"{len(evidence)} bathroom candidates" if provisional else len(evidence)
        return Answer(value, None if provisional else "count", f"context_{key}", 0.5 if provisional else 0.56,
                      ("Found groups of toilets, sinks, and showers or bathtubs. Review marks where the room outline is unclear."
                       if extra else "Estimated from room outlines and distinct plumbing fixtures detected within them."),
                      evidence, path)

    fixtures = [b for b in selected_fixtures(boxes, config["thresholds"]) if CLASSES[b[0]] in requested]
    if not fixtures:
        return Answer(None, "count", "unreadable", 0, "The fixture detector could not locate a verified symbol of this type.",
                      source_file=path)
    evidence = [Evidence("box", f"fixture-{i}", f"{CLASSES[b[0]]} {i + 1}", b[1], [b[2:4], b[4:6]])
                for i,b in enumerate(fixtures)]
    return Answer(len(evidence), "count", f"context_{key}", .56,
                  "Counted the fixture symbols after checking competing classes and removing overlapping detections.",
                  evidence, path)
