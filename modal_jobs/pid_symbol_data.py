"""Convert the PID2Graph Dataset PID patches into a sheet-disjoint symbol dataset."""
from __future__ import annotations

import argparse
import io
import json
import random
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

NAMES = ("valve", "pump", "instrumentation")
NS = {"g": "http://graphml.graphdrawing.org/xmlns"}
SOURCE = "https://zenodo.org/records/14803338"


def rows_from_graph(raw, width, height):
    tree = ET.fromstring(raw)
    fields = {e.attrib["id"]: e.attrib["attr.name"] for e in tree.findall("g:key", NS)
              if e.attrib["for"] == "node"}
    for node in tree.findall("g:graph/g:node", NS):
        data = {fields[d.attrib["key"]]: d.text for d in node.findall("g:data", NS)}
        kind = data["label"].lower()
        if kind not in NAMES:
            continue
        x0, y0, x1, y1 = (float(data[k]) for k in ("xmin", "ymin", "xmax", "ymax"))
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(width, x1), min(height, y1)
        if x1 - x0 >= 3 and y1 - y0 >= 3:
            vals = ((x0+x1)/2/width, (y0+y1)/2/height, (x1-x0)/width, (y1-y0)/height)
            yield kind, f"{NAMES.index(kind)} " + " ".join(f"{v:.6f}" for v in vals)


def convert(source, output):
    group = defaultdict(dict)
    with zipfile.ZipFile(source) as src, zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as dst:
        for entry in src.infolist():
            parts = entry.filename.split("/")
            if len(parts) == 5 and parts[1:3] == ["Patched", "Dataset PID"]:
                file = Path(parts[4])
                if file.suffix in (".png", ".graphml"):
                    group[parts[3]].setdefault(file.stem, {})[file.suffix] = entry
        keys = sorted(group, key=lambda s: min(e.header_offset for patch in group[s].values() for e in patch.values()))[:44]
        if len(keys) < 44:
            raise ValueError("Expected the complete PID2Graph release and at least 44 Dataset PID sheets")
        random.Random(17).shuffle(keys)
        manifest = {}
        for i, sheet in enumerate(keys):
            split = "train" if i < 28 else "val" if i < 35 else "test"
            counts = Counter()
            for patch in sorted(group[sheet], key=int)[:14]:
                files = group[sheet][patch]
                if len(files) != 2:
                    continue
                image = Image.open(io.BytesIO(src.read(files[".png"]))).convert("RGB")
                labels = list(rows_from_graph(src.read(files[".graphml"]), *image.size))
                if not labels:
                    continue
                image.thumbnail((1024, 1024))
                encoded = io.BytesIO()
                image.save(encoded, format="JPEG", quality=94)
                stem = f"{sheet}-{patch}"
                dst.writestr(f"images/{split}/{stem}.jpg", encoded.getvalue())
                dst.writestr(f"labels/{split}/{stem}.txt", "\n".join(row for _, row in labels) + "\n")
                counts.update(k for k, _ in labels)
                counts["tiles"] += 1
            manifest[sheet] = {"split": split, **counts}
        dst.writestr("manifest.json", json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, epilog=SOURCE)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    convert(args.source, args.output)
