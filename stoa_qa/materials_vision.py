"""Last resort for material questions: Gemma reads the sheet image. It must name each material together with where it
read it (a box on the sheet: a note, a label, a hatched area, a legend row); a material without a box is dropped, and
the answer is flagged for a person's review. No numbers come from here: counts and quantities are never asked of it.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import llm, sheets
from .core import Answer, Evidence
from .materials import FAMILIES, LABEL, Query, family_set

REPO = Path(__file__).resolve().parent.parent
CACHE = REPO / "eval_out" / "vision_cache"

SYSTEM = ("You read architectural drawings. You only report what the drawing itself shows: printed words, legends, "
          "keynotes, hatch patterns explained by a legend, or symbols. Reply with one JSON object and nothing else.")
PROMPT = """Question about this drawing: {question}

List every MATERIAL that the drawing shows (by a printed note, label, tag, schedule entry, legend, or a hatch pattern
whose meaning the drawing states or that follows the standard drafting convention), relevant to the question.
Return JSON: {{"materials": [{{"material": <name as shown, e.g. "CONCRETE SLAB", "ceramic tile", "brick (hatch)">,
  "where": <what it is on: room / wall / floor area / element>,
  "how": "text" | "legend" | "hatch",
  "box_2d": [ymin, xmin, ymax, xmax]}}],
 "note": <one short sentence>}}
Coordinates are normalised to 0-1000. If the drawing shows no material, return {{"materials": [], "note": "<why>"}}.
Never guess a material from what a room usually has."""


def answer(d, q: Query, sheet: tuple[str, int] | None = None) -> Answer | None:
    from .pipeline import primary_raster

    if not llm.available():
        return None
    raster = primary_raster(d)
    if sheet and sheet[0].lower().endswith(".pdf"):
        raster = ("pdf", sheet[0], sheet[1])
    if raster is None:
        return None
    kind, path, page = raster
    img = sheets.raster_path(kind, path, page)
    w, h = sheets.image_size(kind, path, page)
    key = hashlib.sha1(f"{path}|{page}|{q.text}|{PROMPT}".encode()).hexdigest()[:20]
    CACHE.mkdir(parents=True, exist_ok=True)
    cached = CACHE / f"mat-{key}.json"
    if cached.exists():
        out = json.loads(cached.read_text())
    else:
        try:
            out, _ = llm.chat_json(SYSTEM, PROMPT.format(question=q.text), image=img, max_tokens=900, timeout=90)
        except llm.LLMError as e:
            return Answer(None, None, "error", 0.0, f"vision: model unavailable ({e})")
        cached.write_text(json.dumps(out))
    items, ev = [], []
    for i, it in enumerate(out.get("materials") or []):
        if not isinstance(it, dict):
            continue
        b = it.get("box_2d")
        if not (isinstance(b, list) and len(b) == 4 and all(isinstance(v, (int, float)) and 0 <= v <= 1000 for v in b)):
            continue
        y0, x0, y1, x1 = b
        if not (x0 < x1 and y0 < y1):
            continue
        name = str(it.get("material") or "").strip()
        fams = family_set(name, abbreviations=True)
        if not name or (q.families and not (fams & q.families)):
            continue
        where = str(it.get("where") or "").strip()
        items.append(f"{name}" + (f" — {where}" if where else "") + f" ({it.get('how') or 'seen'})")
        ev.append(Evidence("box", f"vision-{i}", name[:80], None,
                           [[round(x0 / 1000 * w, 1), round(y0 / 1000 * h, 1)], [round(x1 / 1000 * w, 1), round(y1 / 1000 * h, 1)]]))
    if not items:
        return Answer(None, None, "no_material", 0.0, f"vision: {str(out.get('note') or 'no material shown')[:160]}")
    return Answer("\n".join(items[:12]), None, "materials_vision", 0.4,
                  "AI vision reading of the sheet (Gemma): each material is boxed where it was read; needs a person's check",
                  ev)
