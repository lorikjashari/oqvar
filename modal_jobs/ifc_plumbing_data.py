"""Generate annotated plan-view crops from distinct Stoa plumbing IFC projects."""
from __future__ import annotations

import argparse
import json
import random
import re
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.element as ifc_element
import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import MultiPoint, box

PROJECTS = (
    ("train", "ifc_models/bsc-clinic_clinic-plumbing.ifc"),
    ("train", "ifc_bench/sixty5_plumbing.ifc"),
    ("val", "ifc_bench/digital_hub_plumbing.ifc"),
    ("test", "ifc_models/bsc-duplex_duplex-plumbing-20121113.ifc"),
)
CLASSES = ("water_pipe", "other_pipe", "valve", "pipe_fitting", "pump")
WATER = re.compile(r"hot.water|cold.water|potable|domestic.water|drinkwater|tapwater|trinkwasser|\bpw[chr]\b|s_pw[chr]", re.I)
VALVE = re.compile(r"valve|ventil|kogelkraan|druckminderer|backflow.preventer|terugstroombeveiliging", re.I)


def classify(el):
    name = el.Name or ""
    if el.is_a("IfcFlowSegment"):
        psets = ifc_element.get_psets(el, should_inherit=False)
        system = " ".join(str(value) for pset in psets.values() for key, value in pset.items()
                          if "system" in key.lower())
        return "water_pipe" if WATER.search(f"{name} {system}") else "other_pipe"
    if el.is_a("IfcValve") or (el.is_a("IfcFlowController") and VALVE.search(name)):
        return "valve"
    if el.is_a("IfcFlowFitting") and not re.search(r"drain|afvoerput", name, re.I):
        return "pipe_fitting"
    if el.is_a("IfcPump") or (el.is_a("IfcFlowMovingDevice") and "pump" in name.lower()):
        return "pump"
    return None


def geometries(path):
    model = ifcopenshell.open(str(path))
    architectural = ("IfcWall", "IfcDoor", "IfcWindow", "IfcColumn", "IfcSanitaryTerminal", "IfcFlowTerminal")
    labels = {}
    for el in model.by_type("IfcElement"):
        label = classify(el)
        if label or any(el.is_a(kind) for kind in architectural):
            storey = ifc_element.get_container(el, ifc_class="IfcBuildingStorey")
            labels[el.id()] = (label, storey.Name if storey else None)
    options = ifcopenshell.geom.settings()
    options.set("use-world-coords", True)
    iterator = ifcopenshell.geom.iterator(options, model, 4, include=[model.by_id(k) for k in labels])
    result = defaultdict(list)
    if not iterator.initialize():
        return result
    while True:
        obj = iterator.get()
        vertices = np.asarray(obj.geometry.verts).reshape((-1, 3))
        if vertices.size:
            label, storey = labels[obj.id]
            level = storey or f"z{round(float(np.median(vertices[:, 2])) / 3)}"
            poly = MultiPoint(vertices[:, :2]).convex_hull
            if poly.geom_type == "Polygon" and poly.area > 1e-7 and poly.length < 500:
                result[level].append((label, poly))
        if not iterator.next():
            return result


def render(project):
    dataset, out, split, relative = project
    stem = Path(relative).stem
    rng = random.Random(stem)
    labels = Counter()
    seen = set()
    samples = []
    for level, objects in geometries(dataset / relative).items():
        target = [(kind, poly) for kind, poly in objects if kind]
        if not target:
            continue
        backgrounds = [poly for kind, poly in objects if kind is None]
        # Oversample rare equipment. The split is assigned before generating any overlapping views.
        rare = [(kind, poly) for kind, poly in target if kind in ("valve", "pump")]
        rng.shuffle(rare)
        rng.shuffle(target)
        for label, shape in rare[:120] + target[:160]:
            size = rng.uniform(2.8, 7.5)
            x, y = shape.centroid.coords[0]
            x += rng.uniform(-size * 0.2, size * 0.2)
            y += rng.uniform(-size * 0.2, size * 0.2)
            key = (level, round(x * 2), round(y * 2), round(size))
            if key in seen:
                continue
            seen.add(key)
            region = box(x-size/2, y-size/2, x+size/2, y+size/2)
            left, bottom, right, top = region.bounds
            side = 640
            image = Image.new("RGB", (side, side), (250, 250, 248) if rng.random() < .3 else "white")
            draw = ImageDraw.Draw(image)
            rows = []

            def points(poly):
                return [((a-left)/size*side, (top-b)/size*side) for a, b in poly.exterior.coords]

            for poly in backgrounds:
                if poly.intersects(region):
                    clip = poly.intersection(region)
                    if clip.geom_type == "Polygon":
                        draw.polygon(points(clip), outline=(164, 164, 164), width=1)
            hits = [(kind, poly.intersection(region)) for kind, poly in target if poly.intersects(region)]
            hits.sort(key=lambda item: CLASSES.index(item[0]))
            local_counts = Counter()
            for kind, poly in hits:
                if poly.geom_type != "Polygon" or poly.area / size**2 * side**2 < 16:
                    continue
                pt = points(poly)
                tone = rng.randint(10, 65)
                draw.polygon(pt, outline=(tone,)*3, width=1 if rng.random() < .65 else 2)
                if kind in ("valve", "pump"):
                    draw.line([pt[0], pt[len(pt)//2]], fill=(tone,)*3, width=1)
                rows.append(f"{CLASSES.index(kind)} " + " ".join(f"{max(0,min(1,v/side)):.5f}" for p in pt[:-1] for v in p))
                local_counts[kind] += 1
            if not rows:
                continue
            name = f"{stem}-{len(samples):04d}"
            image.save(out / "images" / split / f"{name}.png")
            (out / "labels" / split / f"{name}.txt").write_text("\n".join(rows) + "\n")
            labels.update(local_counts)
            samples.append(name)
            if len(samples) >= 450:
                break
        if len(samples) >= 450:
            break
    return stem, {"split": split, "tiles": len(samples), "instances": dict(labels)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Stoa dataset directory containing data/")
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    out = args.archive.with_suffix("")
    for split in ("train", "val", "test"):
        for kind in ("images", "labels"):
            (out / kind / split).mkdir(parents=True)
    jobs = [(args.dataset / "data", out, split, name) for split, name in PROJECTS]
    with ProcessPoolExecutor(max_workers=3) as pool:
        manifest = dict(pool.map(render, jobs))
    if not all(item["tiles"] for item in manifest.values()):
        raise RuntimeError(f"Project produced no labelled views: {manifest}")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    with zipfile.ZipFile(args.archive, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for f in sorted(out.rglob("*")):
            if f.is_file():
                archive.write(f, f.relative_to(out))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
