"""Materials from a sheet's printed words: schedules, finish keys, notes and labels — on this sheet and across its set.

Floor plans rarely spell materials out in a table of their own. They show them as:
  * room tags (101, 113, U107) whose finishes live in a FINISH SCHEDULE on another sheet of the set, coded (F1, B2,
    W1, C2) and decoded by a FINISH KEY ("F4 | RESILIENT FLOORING");
  * door marks whose door / frame material lives in a DOOR SCHEDULE (WD-1, AL-2/GL-2, HM);
  * notes and labels drawn on the plan itself ("REPLACE WITH NEW TILE / ALT. BID: EXPOSED CONCRETE", "PARKETTI",
    "4" CONC. SLAB", "GYP. BD.") and legends / keynotes.

This module reads all of those without an LLM: printed rows with positions (stoa_qa.docqa.layout for PDFs, the
full-resolution OCR rows for raster sheets), schedule tables rebuilt from the column positions of their header, codes
decoded through the finish key, material words matched by stoa_qa.materials' vocabulary. A room found in a schedule
is then located on the plan being looked at (its printed room number), so the answer is drawn where the room is.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from . import docqa, sheets
from .core import Answer, Evidence
from .materials import LABEL, Query, families_in, family_set, fold, matches

CODE = re.compile(r"^(?:[A-Z]{1,3}-?\d{1,2}[A-Z]?|[A-Z]{2,4})$")          # F1, B2, W1, WD-1, AL-2, HM, CPT
CODE_KEY = re.compile(r"^[A-Z]{1,3}-?\d{1,2}[A-Z]?$")                         # a key entry needs a number: F4, WD-1
CODE_IN = re.compile(r"\b[A-Z]{1,3}-?\d{1,2}[A-Z]?\b")
HEADING = re.compile(r"\b(?:schedule|key|legend|keynotes?|types|notes|abbreviations)\b", re.I)
HEAD_WORDS = re.compile(r"\b(?:room|number|no\.?|name|floor|base|wall|walls|north|south|east|west|ceiling|clg|finish(?:es)?|"
                        r"area|comments?|door|width|height|matl|material|frame|type|mark|hardware|rating|count|level)\b", re.I)
PHRASES = ["door material", "frame material", "door matl", "frame matl", "door type", "frame type", "fire rating",
           "type comments", "type mark", "room name", "floor finish", "base finish", "ceiling finish", "north wall",
           "south wall", "east wall", "west wall"]
FINISH_FIELDS = [("wall_n", r"north"), ("wall_s", r"south"), ("wall_e", r"east"), ("wall_w", r"west"), ("ceiling", r"ceil|clg"),
                 ("floor", r"floor|flr"), ("base", r"base"), ("walls", r"wall"), ("number", r"number|\bno\b|#"),
                 ("name", r"name"), ("area", r"area|\bsf\b"), ("comments", r"comment|remark|note")]
DOOR_FIELDS = [("door_matl", r"door\s*(?:matl|material)|^matl$"), ("frame_matl", r"frame\s*(?:matl|material)"),
               ("frame_type", r"frame\s*type"), ("door_type", r"door\s*type|^type$"), ("count", r"count"),
               ("number", r"number|mark|\bno\b"), ("room", r"room|name"), ("level", r"level"), ("width", r"width"),
               ("height", r"height"), ("rating", r"rating"), ("comments", r"comment|note")]
SURFACE_FIELDS = {"floor": ["floor"], "base": ["base"], "ceiling": ["ceiling"], "wall": ["wall_n", "wall_s", "wall_e", "wall_w", "walls"]}
# printed abbreviations decoded when a schedule uses them without a key on the same sheet
ABBR_TEXT = {"HM": "hollow metal", "WD": "wood", "AL": "aluminium", "GL": "glass", "FG": "fiberglass", "STL": "steel",
             "MTL": "metal", "CPT": "carpet", "VCT": "vinyl composition tile", "LVT": "luxury vinyl tile", "CT": "ceramic tile",
             "GWB": "gypsum wall board", "GYP": "gypsum board", "PT": "paint", "EPX": "epoxy", "CONC": "concrete", "RB": "rubber base"}


@dataclass
class Table:
    kind: str                                  # finish | door
    rel: str
    page: int
    fields: list[str]                          # one per column
    rows: list[dict] = field(default_factory=list)   # field -> text, plus "_row" (the printed row) and "_bbox"
    title: str = ""


# ------------------------------------------------------------------ reading a sheet

def cells(r: dict) -> list[dict]:
    """A printed row's words grouped into table cells: words closer than 2.5 character widths belong together (the
    same rule stoa_qa.docqa.rows_of uses for its ' | ' separators)."""
    ps = r.get("parts") or []
    if not ps:
        return []
    char = sum(p["bbox"][2] - p["bbox"][0] for p in ps) / max(1, sum(len(p["text"]) for p in ps))
    out = [{"text": ps[0]["text"], "bbox": list(ps[0]["bbox"])}]
    for a, b in zip(ps, ps[1:]):
        if b["bbox"][0] - a["bbox"][2] > 2.5 * char:
            out.append({"text": b["text"], "bbox": list(b["bbox"])})
        else:
            c = out[-1]
            c["text"] += " " + b["text"]
            c["bbox"] = [min(c["bbox"][0], b["bbox"][0]), min(c["bbox"][1], b["bbox"][1]),
                         max(c["bbox"][2], b["bbox"][2]), max(c["bbox"][3], b["bbox"][3])]
    return out


def _split_part(p: dict) -> list[dict]:
    """A header cell that holds two labels ("Frame Type Frame Material") split into two, x by character share."""
    t = fold(p["text"])
    hits = sorted((t.find(ph), ph) for ph in PHRASES if ph in t)
    if len(hits) < 2:
        return [p]
    x0, y0, x1, y1 = p["bbox"]
    n = max(1, len(t))
    out = []
    starts = [h[0] for h in hits] + [n]
    for (s, _), e in zip(hits, starts[1:]):
        out.append({"text": p["text"][s:e].strip(), "bbox": [x0 + (x1 - x0) * s / n, y0, x0 + (x1 - x0) * e / n, y1]})
    return out


def _label(fields_rx, label: str) -> str | None:
    f = fold(label)
    for name, rx in fields_rx:
        if re.search(rx, f):
            return name
    return None


def _runs(cs: list[dict], gap: float) -> list[list[dict]]:
    """Cells of one printed row split into blocks wherever the gap between them is wider than `gap` (a table and the
    notes or title block beside it)."""
    out: list[list[dict]] = []
    for c in sorted(cs, key=lambda c: c["bbox"][0]):
        if out and c["bbox"][0] - out[-1][-1]["bbox"][2] <= gap:
            out[-1].append(c)
        else:
            out.append([c])
    return out


def _tables(rows: list[dict], rel: str, page: int) -> list[Table]:
    out = []
    heights = sorted(r["bbox"][3] - r["bbox"][1] for r in rows) or [10]
    h = heights[len(heights) // 2]
    W = max((r["bbox"][2] for r in rows), default=1000.0)
    gap = max(12 * h, 0.05 * W)
    row_cells = [cells(r) for r in rows]
    seen_heads: set[tuple] = set()
    for i, r in enumerate(rows):
        for run in _runs(row_cells[i], gap):
            text = " ".join(c["text"] for c in run)
            if len(HEAD_WORDS.findall(text)) < 4:
                continue
            low = fold(text)
            kind = ("finish" if re.search(r"floor", low) and re.search(r"ceil|base|wall", low) else
                    "door" if re.search(r"door|matl|frame", low) and re.search(r"width|height|type|matl|material", low) else None)
            if kind is None:
                continue
            t = _table(rows, row_cells, i, run, kind, h, gap, rel, page)
            if t is not None and (t.rows[0]["_row"], kind) not in seen_heads:
                seen_heads.add((t.rows[0]["_row"], kind))
                out.append(t)
    return out


def _table(rows, row_cells, i, run, kind, h, gap, rel, page) -> Table | None:
        r = {"bbox": [min(c["bbox"][0] for c in run), min(c["bbox"][1] for c in run), max(c["bbox"][2] for c in run),
                      max(c["bbox"][3] for c in run)]}
        parts = [q for p in run for q in _split_part(p)]
        x0, x1 = r["bbox"][0], r["bbox"][2]
        # an upper header line (Door / Room / Frame over Number / Name / Matl): prefix it to the columns below it
        upper = [c for cs in row_cells for c in cs if 0 <= r["bbox"][1] - c["bbox"][3] < 2.2 * h
                 and c["bbox"][2] > x0 and c["bbox"][0] < x1]
        cols = []
        for p in parts:
            px0, _, px1, _ = p["bbox"]
            pre = " ".join(u["text"] for u in upper if u["bbox"][0] < px1 + 4 and u["bbox"][2] > px0 - 4)
            cols.append((f"{pre} {p['text']}".strip(), (px0 + px1) / 2, px1 - px0))
        fields_rx = FINISH_FIELDS if kind == "finish" else DOOR_FIELDS
        names, used = [], set()
        for lab, _, _ in cols:
            n = _label(fields_rx, lab)
            # the same field twice (two "Number" columns: door and room): the second keeps its own label
            names.append(n if n and n not in used else (fold(lab)[:20] or "col"))
            if n:
                used.add(n)
        if kind == "finish" and "floor" not in names:
            return None
        above = [c for cs in row_cells[max(0, i - 12):i] for c in cs if re.search(r"schedule", c["text"], re.I)
                 and c["bbox"][2] > x0 - gap and c["bbox"][0] < x1]
        t = Table(kind, rel, page, names, title=above[-1]["text"] if above else "")
        last_y = r["bbox"][3]
        for j, d in enumerate(rows[i + 1:], start=i + 1):
            inside = [c for c in row_cells[j] if x0 - 30 <= (c["bbox"][0] + c["bbox"][2]) / 2 <= x1 + 30]
            if not inside:
                continue                       # a block elsewhere on the sheet at this height
            y0 = min(c["bbox"][1] for c in inside)
            y1 = max(c["bbox"][3] for c in inside)
            if y0 < r["bbox"][3]:
                continue
            if y0 - last_y > 5 * h:
                break
            text = " ".join(c["text"] for c in inside)
            if HEADING.search(text) and len(text) < 40:
                break
            if len(HEAD_WORDS.findall(text)) >= 4:
                break
            vals: dict = {}
            for p in inside:
                cx = (p["bbox"][0] + p["bbox"][2]) / 2
                k = min(range(len(cols)), key=lambda kk: abs(cols[kk][1] - cx))
                vals[names[k]] = (vals.get(names[k], "") + " " + p["text"]).strip()
            box = [min(c["bbox"][0] for c in inside), y0, max(c["bbox"][2] for c in inside), y1]
            if len(vals) >= 2 or (len(vals) == 1 and t.rows):
                key_field = "number" if kind == "finish" else ("number" if "number" in names else "mark")
                prev = t.rows[-1] if t.rows else None
                if (prev is not None and key_field not in vals and y0 - prev["_bbox"][3] < 1.5 * h
                        and not set(vals) & {k for k in prev if not k.startswith("_")}):
                    prev.update(vals)          # a schedule row printed on two lines: one room / door
                    prev["_bbox"] = [min(prev["_bbox"][0], box[0]), prev["_bbox"][1], max(prev["_bbox"][2], box[2]), y1]
                elif len(vals) >= 2:
                    vals["_row"], vals["_bbox"] = d["id"], box
                    t.rows.append(vals)
                last_y = y1
        return t if t.rows else None


def _key(rows: list[dict], skip: set[str]) -> dict[str, tuple[str, dict]]:
    """Finish / material keys on a sheet: 'F4 | RESILIENT FLOORING (T.B.D.)' -> {'F4': ('RESILIENT FLOORING (T.B.D.)', row)}.
    Rows that belong to a schedule table (`skip`) are not keys, even when a code sits next to words in them."""
    out = {}
    for r in rows:
        if r["id"] in skip:
            continue
        ps = cells(r)
        for a, b in zip(ps, ps[1:]):
            code, desc = a["text"].strip(), b["text"].strip()
            if (CODE_KEY.match(code) and re.search(r"[A-Za-z]{3,}", desc) and not CODE.match(desc) and len(desc) > 3
                    and not CODE_IN.match(desc)):
                out.setdefault(code, (desc, r))
        m = re.match(r"^\s*([A-Z]{1,3}-?\d{1,2}[A-Z]?)\s*[-–:=]\s*([A-Za-z].{3,})$", r["text"])
        if m:
            out.setdefault(m.group(1), (m.group(2).strip(), r))
    return out


def _full_rows(rel: str, page: int) -> list[dict]:
    """Printed rows across the whole sheet width (not split at wide gaps): a schedule's columns stay one row."""
    import pymupdf

    pg = pymupdf.open(sheets.safe_path(rel))[page]
    words = []
    for w in pg.get_text("words"):
        if w[4].strip():
            r = (pymupdf.Rect(w[:4]) * pg.rotation_matrix).normalize()
            words.append((r.x0, r.y0, r.x1, r.y1, w[4]))
    return docqa.rows_of(words, 1e9) if words else []


@lru_cache(maxsize=256)
def _read(rel: str, page: int = 0):
    rows, pw, _ = docqa.layout(rel, page)
    k = sheets.image_size("pdf", rel, page)[0] / pw if pw else 1.0
    full = _full_rows(rel, page)
    tables = _tables(full, rel, page)
    in_tables = {row["_row"] for t in tables for row in t.rows}
    in_tables_boxes = [row["_bbox"] for t in tables for row in t.rows]
    skip = {r["id"] for r in rows if any(r["bbox"][1] >= b[1] - 1 and r["bbox"][3] <= b[3] + 1 and r["bbox"][0] >= b[0] - 1
                                         and r["bbox"][2] <= b[2] + 1 for b in in_tables_boxes)}
    return rows, k, tables, _key(full, in_tables) | {c: v for c, v in _key(rows, skip).items()}


SCHEDULE_NAME = re.compile(r"sched|finish|door|window|legend|key|material|spec|a-?[678]\d\d", re.I)


def _set_sheets(rel: str) -> list[str]:
    """The other PDF sheets of the set (same folder): sheets named like schedules / finishes / legends first."""
    from .core import DATASET

    folder = (DATASET / rel).parent
    try:
        pdfs = sorted(str(p.relative_to(DATASET)).replace("\\", "/") for p in folder.glob("*.pdf") if p.name != Path(rel).name)
    except OSError:
        return []
    return sorted(pdfs, key=lambda p: (not SCHEDULE_NAME.search(Path(p).stem), p))[:120]


@lru_cache(maxsize=64)
def set_schedules(rel: str) -> tuple[list[Table], dict]:
    """Schedule tables and keys of the drawing set: this sheet first, then the PDFs beside it (same folder), the
    schedule-named sheets first; the rest only while no schedule has been found."""
    tables, key = [], {}
    for r in [rel, *_set_sheets(rel)]:
        if tables and not SCHEDULE_NAME.search(Path(r).stem):
            break
        try:
            _, _, ts, k = _read(r, 0)
        except Exception:
            continue
        tables += ts
        for c, v in k.items():
            key.setdefault(c, (*v, r))
    return tables, key


# ------------------------------------------------------------------ answering

def _codes(text: str) -> list[str]:
    return [c for c in CODE_IN.findall(text or "")]


def _decode(code_text: str, key: dict) -> str:
    """'F2 & F3' -> 'F2 - ARTIFICIAL TURF; F3 - RESILIENT BASKETBALL FLOORING (T.B.D.)'."""
    cs = _codes(code_text)
    if not cs:
        return code_text
    parts = []
    for c in cs:
        if c in key:
            parts.append(f"{c} - {key[c][0]}")
        else:
            base = re.sub(r"-?\d+[A-Z]?$", "", c)
            parts.append(f"{c} ({ABBR_TEXT[base]})" if base in ABBR_TEXT else c)
    return "; ".join(parts)


def _families_of(code_text: str, key: dict) -> set[str]:
    out = family_set(code_text, abbreviations=True, strict=False)
    for c in _codes(code_text):
        if c in key:
            out |= family_set(key[c][0], abbreviations=True)
    return out


def _room_from_question(t: Table, text: str) -> dict | None:
    nums = re.findall(r"\b(?:room|rm\.?|raum|local|stanza|kamer|huone)\s*#?\s*([A-Z]?\d{2,4}[A-Z]?)\b", text, re.I)
    nums += re.findall(r"\(\s*(?:room\s*)?([A-Z]?\d{2,4}[A-Z]?)\s*\)", text, re.I)
    for row in t.rows:
        if nums and row.get("number", "").strip() in nums:
            return row
    names = sorted({row.get("name", "").strip() for row in t.rows if row.get("name")}, key=len, reverse=True)
    for n in names:
        if len(n) >= 4 and re.search(r"(?<!\w)" + re.escape(n) + r"(?!\w)", text, re.I):
            hits = [row for row in t.rows if row.get("name", "").strip() == n]
            if len(hits) == 1:
                return hits[0]
    return None


def _surfaces(q: Query, text: str) -> list[str]:
    t = fold(text)
    for d, f in (("north", "wall_n"), ("south", "wall_s"), ("east", "wall_e"), ("west", "wall_w")):
        if re.search(rf"\b{d}\b", t):
            return [f]
    if q.surface:
        return SURFACE_FIELDS[q.surface]
    out = []
    for s, fs in SURFACE_FIELDS.items():
        if re.search({"floor": r"\bfloor", "base": r"\bbase\b", "ceiling": r"\bceiling", "wall": r"\bwall"}[s], t):
            out += fs
    return out or ["floor", "base", "wall_n", "wall_s", "wall_e", "wall_w", "walls", "ceiling"]


SURFACE_WORD = {"floor": "floor", "base": "base", "ceiling": "ceiling", "wall_n": "north wall", "wall_s": "south wall",
                "wall_e": "east wall", "wall_w": "west wall", "walls": "walls"}


def _area(row: dict) -> float | None:
    m = re.search(r"([\d,]+(?:\.\d+)?)", row.get("area", ""))
    return float(m.group(1).replace(",", "")) if m else None


def _room_label(row: dict) -> str:
    """'113 AEROBICS & YOGA' for a room; '115.A (F. VEST)' / 'U01 UNIT DOOR' for a door or door type."""
    if "name" in row or "room" not in row and "mark" not in row and "count" not in row:
        return " ".join(x for x in (row.get("number", ""), row.get("name", "")) if x).strip()
    ident = (row.get("mark") or row.get("number") or row.get("count") or "").strip()
    room = (row.get("room") or row.get("door_type") or "").strip()
    return f"{ident} ({room})" if ident and room else (ident or room)


def _locate(rows: list[dict], k: float, token: str) -> list[float] | None:
    """Where a printed token (a room number) sits on the sheet, in sheet-image pixels."""
    rx = re.compile(r"(?<![\w.])" + re.escape(token) + r"(?![\w.])")
    for r in rows:
        for p in r["parts"]:
            if len(p["text"]) <= 16 and rx.search(p["text"]):
                x0, y0, x1, y1 = p["bbox"]
                return [x0 * k - 6, y0 * k - 6, x1 * k + 6, y1 * k + 6]
    return None


def _evidence_for_rows(found: list[dict], here_rows, here_k, t: Table, here_rel: str) -> tuple[list[Evidence], str]:
    """Each room / door found: boxed at its tag on the sheet being shown, or at its schedule row when the shown sheet is
    the schedule."""
    ev, where = [], ""
    if t.rel == here_rel:
        k = _read(t.rel, t.page)[1]
        for row in found:
            b = row["_bbox"]
            ev.append(Evidence("box", row["_row"], _room_label(row)[:60], None, [[b[0] * k - 4, b[1] * k - 3], [b[2] * k + 4, b[3] * k + 3]]))
        return ev, f"rows of {t.title or 'the schedule'} on this sheet"
    if here_rows is not None:
        for row in found:
            # a door is tagged on the plan by its mark (115.A); a room by its number
            tok = ((row.get("mark") or row.get("number") or "") if t.kind == "door" else row.get("number", "")).strip()
            tok = tok.split()[0] if tok else ""
            box = _locate(here_rows, here_k, tok) if tok and len(tok) >= 3 else None
            if box:
                ev.append(Evidence("box", f"tag-{tok}", (tok if t.kind == "door" else _room_label(row))[:60], None, [box[:2], box[2:]]))
        if ev:
            where = f"{'door marks' if t.kind == 'door' else 'room tags'} on this plan ({len(ev)} of {len(found)} found)"
    return ev, where


def _schedule_answer(q: Query, here: tuple[str, int] | None) -> Answer | None:
    if here is None:
        return None
    rel, page = here
    tables, key = set_schedules(rel)
    if not tables:
        return None
    try:
        here_rows, here_k = _read(rel, page)[:2]
    except Exception:
        here_rows, here_k = None, 1.0
    text = q.text
    sheet_of = lambda t: Path(t.rel).stem.split(" - ")[0]

    finish = [t for t in tables if t.kind == "finish"]
    door = [t for t in tables if t.kind == "door"]
    door_q = bool(re.search(r"\bdoors?\b|\bframes?\b", text, re.I))

    # 1. one room's finishes: "what is the floor finish of room 113?"
    if finish and not door_q:
        for t in finish:
            row = _room_from_question(t, text)
            if row is None:
                continue
            surf = _surfaces(q, text)
            vals = [(SURFACE_WORD[f], row[f]) for f in surf if row.get(f)]
            if not vals:
                continue
            decoded = "; ".join(f"{w}: {_decode(v, key)}" for w, v in vals)
            ev, where = _evidence_for_rows([row], here_rows, here_k, t, rel)
            value = _decode(vals[0][1], key) if len(vals) == 1 else decoded
            return Answer(value, None, "materials_schedule", 0.9,
                          f"Room {_room_label(row)} in {t.title or 'the finish schedule'} on sheet {sheet_of(t)}, codes decoded "
                          f"with the finish key ({decoded})" + (f"; shown at the {where}" if where else ""), ev)

    # 1b. one door's (or door type's) material: "what door material is specified for door 115.A?"
    codes = [c for c in _codes(text) if c in key or any(c in str(v) for t in tables for row in t.rows for v in row.values())]
    marks = re.findall(r"\b(?:door|type|mark)\s+\(?([A-Z]?\d{1,4}(?:\.[A-Z0-9]+)?|[A-Z]\d{1,3}(?:-[A-Z]{1,2})?)\)?", text, re.I)
    if door and door_q and marks and not re.search(r"\bhow many\b", text, re.I):
        want = "frame_matl" if re.search(r"\bframe", text, re.I) else "door_matl"
        for t in door:
            for row in t.rows:
                ids = {row.get(f, "").strip() for f in ("number", "mark")} | set(_codes(" ".join(str(v) for v in row.values()
                                                                                                 if isinstance(v, str))[:40]))
                if any(m in ids for m in marks) and row.get(want):
                    val = row[want].split()[0] if want == "frame_matl" else row[want]
                    ev, where = _evidence_for_rows([row], here_rows, here_k, t, rel)
                    return Answer(_decode(val, key), None, "materials_schedule", 0.9,
                                  f"{'Frame' if want == 'frame_matl' else 'Door'} material of {', '.join(marks)} in "
                                  f"{t.title or 'the door schedule'} on sheet {sheet_of(t)}" + (f"; shown at the {where}" if where else ""), ev)

    # 2. a code or a material across the schedule: "how many rooms have floor finish F0?", "rooms with carpet"
    tabs = door if door_q else finish
    if not tabs or not (codes or q.families):
        return None
    t = max(tabs, key=lambda t: len(t.rows))
    if door_q and codes:  # the table that actually holds the codes asked about (door numbers vs door types)
        holding = [x for x in tabs if any(c in _codes(" ".join(v for v in r.values() if isinstance(v, str))) for r in x.rows for c in codes)]
        t = max(holding, key=lambda x: len(x.rows)) if holding else t
    elif door_q and len(tabs) > 1:
        counting = re.search(r"\bhow many\b", text, re.I) is not None
        # a count reads the door TYPE schedule (its Count column); "which doors" reads the per-door schedule, whose
        # marks can be found on the plan
        t = max(tabs, key=lambda x: (("count" in x.fields) == counting, len(x.rows)))
    if t.kind == "finish":
        surf = _surfaces(q, text)
        every = re.search(r"\ball (?:four|4|the) walls\b", text, re.I)

        def hit(row) -> bool:
            vals = [row.get(f, "") for f in surf]
            if every and codes:
                walls = [row.get(f, "") for f in ("wall_n", "wall_s", "wall_e", "wall_w")]
                return all(codes[0] in _codes(v) and len(_codes(v)) == 1 for v in walls)
            if codes:
                return any(c in _codes(v) for v in vals for c in codes)
            return any(matches(_families_of(v, key), q.families) for v in vals)
    else:
        fields_m = [f for f in ("door_matl", "frame_matl") if f in t.fields] or t.fields
        if re.search(r"\bframe", text, re.I) and "frame_matl" in t.fields:
            fields_m = ["frame_matl"]

        words = _discriminating(text, t)
        # the column the asked codes live in ("frames (AL-1 or AL-2)" is the frame column, not the door's AL-2/GL-2)
        code_field = None
        if codes:
            tally = {f: sum(1 for row in t.rows if any(c in _codes(row.get(f, "")) for c in codes)) for f in t.fields}
            if re.search(r"\bframes?\b", text, re.I) and tally.get("frame_matl"):
                code_field = "frame_matl"
            elif max(tally.values(), default=0) > 0:
                code_field = max(tally, key=tally.get)

        def hit(row) -> bool:
            whole = " ".join(v for k, v in row.items() if isinstance(v, str))
            if words and not all(re.search(rf"(?<!\w){re.escape(w)}(?!\w)", whole, re.I) for w in words):
                return False
            if codes:
                return any(c in _codes(row.get(code_field, "")) for c in codes) if code_field else False
            vals = [row.get(f, "") for f in fields_m]
            return any(matches(_families_of(v, key) | {f for tok in re.findall(r"[A-Z]{2,4}", v)
                                                       for f in family_set(ABBR_TEXT.get(tok, ""))}, q.families) for v in vals)
    found = [row for row in t.rows if hit(row)]
    what = ", ".join(codes) if codes else ", ".join(LABEL[f] for f in sorted(q.families))
    if codes:
        what += " (" + "; ".join(_decode(c, key) for c in codes[:3]) + ")"
    noun = "rooms" if t.kind == "finish" else "doors"
    ev, where = _evidence_for_rows(found, here_rows, here_k, t, rel)
    method = (f"{len(found)} {noun} of {len(t.rows)} in {t.title or 'the schedule'} on sheet {sheet_of(t)} match {what}"
              + (f"; shown at the {where}" if where else ""))
    if not found:
        return Answer(None, None, "no_material", 0.0, f"sheet text: no {noun} in {t.title or 'the schedule'} match {what}")
    if q.ask == "quantity" or re.search(r"\b(?:total area|area of)\b", text, re.I):
        areas = [a for a in (_area(r) for r in found) if a is not None]
        if areas:
            total = round(sum(areas), 2)
            if q.ask == "count" or COUNT_AND_AREA.search(text):
                return Answer(f"{len(found)} {noun}, {total:g} SF", None, "materials_schedule", 0.9, method, ev)
            return Answer(total, "ft2", "materials_schedule", 0.9, method + f"; areas summed ({len(areas)})", ev)
    if q.ask == "count" or re.search(r"\bhow many\b", text, re.I):
        counts = [re.match(r"\s*(\d+)", r.get("count", "")) for r in found]
        if t.kind == "door" and "count" in t.fields and all(counts):
            n = sum(int(c.group(1)) for c in counts)   # a door TYPE schedule: each row says how many doors of that type
            return Answer(n, "count", "materials_schedule", 0.9, method + f"; their Count column summed ({n})", ev)
        return Answer(len(found), "count", "materials_schedule", 0.9, method, ev)
    listed = ", ".join(_room_label(r) for r in found[:20]) + (" ..." if len(found) > 20 else "")
    return Answer(f"{what}: {len(found)} {noun} — {listed}", None, "materials_schedule", 0.9, method, ev)


COUNT_AND_AREA = re.compile(r"how many .* and .*(?:total )?area", re.I)
STOP = set("""the a an of on in at to for and or with are is be by from this that these those how many much what which
scheduled schedule schedules door doors frame frames type types sheet drawing drawings room rooms number material materials
matl there have has their its total count entry unit units""".split())


def _discriminating(text: str, t: Table) -> list[str]:
    """Words of the question that pick out rows of this table ('VISTAGRANDE', '1ST'): printed in some rows, not all,
    and not material words (those are matched through the material columns)."""
    out = []
    blob = [" ".join(v for v in r.values() if isinstance(v, str)).upper() for r in t.rows]
    for w in re.findall(r"[A-Za-z0-9]{3,}", text):
        u = w.upper()
        if w.lower() in STOP or family_set(w) or CODE_IN.fullmatch(u):
            continue
        n = sum(1 for b in blob if re.search(rf"(?<![A-Z0-9]){re.escape(u)}(?![A-Z0-9])", b))
        if 0 < n < len(blob):
            out.append(u)
    return out


ROOM_HINT = re.compile(
    r"\b(?:office|kitchen|bath(?:room)?|bed(?:room)?|living|dining|corridor|hall(?:way)?|stor(?:age|\.)|closet|clos\.|lobby|"
    r"restroom|toilet|wc|mech\.?|elec\.?|laundry|garage|porch|deck|entry|vest\.?|vestibule|foyer|stair|cafe|seating|conf\.?|"
    r"lounge|reception|clinic|fitness|locker|shower|utility|pantry|study|den|family|great room|mudroom|basement|attic|"
    r"k(?:ü|u)che|zimmer|bad|wohnen|essen|flur|korridor|abstellraum|cuisine|chambre|salle|séjour|cucina|camera|soggiorno|"
    r"keittiö|keittio|makuuhuone|olohuone|eteinen|kylpyhuone|sauna|terassi|parveke|vh|mh|kh|oh|et|k)\b", re.I)


@lru_cache(maxsize=128)
def _blocks(rel: str, page: int) -> list[dict]:
    """The sheet's text blocks (a note, a label, a legend line), whatever direction they are printed in: PyMuPDF keeps
    a rotated note's lines together, where row reconstruction would scatter its words."""
    import pymupdf

    pg = pymupdf.open(sheets.safe_path(rel))[page]
    out = []
    for b in pg.get_text("dict")["blocks"]:
        lines = [" ".join(s["text"] for s in ln["spans"]).strip() for ln in b.get("lines", [])]
        lines = [x for x in lines if x]
        if not lines:
            continue
        r = (pymupdf.Rect(b["bbox"]) * pg.rotation_matrix).normalize()
        out.append({"id": f"note-{len(out)}", "text": " / ".join(lines), "bbox": [r.x0, r.y0, r.x1, r.y1]})
    return out


def _mentions(q: Query, rows: list[dict], k: float, rel: str) -> Answer | None:
    """Material words printed on the sheet itself (notes, labels, legends), each with the room label nearest to it."""
    labels = []
    for r in rows:
        t = r["text"].strip()
        if 2 <= len(t) <= 30 and ROOM_HINT.search(t) and not family_set(t):
            labels.append(r)
    found = []
    for r in rows:
        hs = [h for h in families_in(r["text"], abbreviations=True, strict=True)]
        if not hs:
            continue
        fams = {h.family for h in hs}
        if q.families and not matches(fams, q.families):
            continue
        if len(r["text"]) > 320:   # a page of general notes is not a note about one place
            continue
        cx, cy = (r["bbox"][0] + r["bbox"][2]) / 2, (r["bbox"][1] + r["bbox"][3]) / 2
        near = min(labels, key=lambda L: (((L["bbox"][0] + L["bbox"][2]) / 2 - cx) ** 2 + ((L["bbox"][1] + L["bbox"][3]) / 2 - cy) ** 2),
                   default=None)
        dist = ((((near["bbox"][0] + near["bbox"][2]) / 2 - cx) ** 2 + ((near["bbox"][1] + near["bbox"][3]) / 2 - cy) ** 2) ** 0.5
                if near is not None else 1e9)
        found.append((r, fams, near["text"] if near is not None and dist < 60 * max(1.0, (r["bbox"][3] - r["bbox"][1]) / 8) else ""))
    if not found:
        return None
    lines, ev = [], []
    for r, fams, near in found[:25]:
        b = r["bbox"]
        lines.append(f"\"{r['text'][:90]}\"" + (f" (near {near})" if near else ""))
        ev.append(Evidence("box", r["id"], r["text"][:80], None, [[b[0] * k - 4, b[1] * k - 3], [b[2] * k + 4, b[3] * k + 3]]))
    fams_all = sorted({f for _, fs, _ in found for f in fs})
    head = (f"{len(found)} note(s) / label(s) on this sheet name "
            + (", ".join(LABEL[f] for f in sorted(q.families)) if q.families else ", ".join(LABEL[f] for f in fams_all[:8])))
    return Answer(head + ":\n" + "\n".join(lines) + (f"\n... {len(found) - 25} more" if len(found) > 25 else ""), None,
                  "materials_notes", 0.8, "Material words printed on the sheet (notes, labels, legends), matched with the "
                  "multilingual material vocabulary; each is boxed where it is printed", ev)


def _raster_rows(d) -> list[dict] | None:
    """Printed rows of a raster sheet (uploads, scans, CubiCasa plans) from the full-resolution OCR, cached."""
    from .pipeline import primary_raster

    raster = primary_raster(d)
    if raster is None or raster[0] != "image":
        return None
    try:
        from . import cubicasa_live

        img = sheets.safe_path(raster[1])
        return docqa.raster_layout(img, cubicasa_live.image_hash(img))
    except Exception:
        return None


def answer(d, q: Query, sheet: tuple[str, int] | None = None) -> Answer | None:
    pdf = next((s for s in d.sheets if s.kind == "pdf"), None)
    here = (sheet[0], sheet[1]) if sheet and sheet[0].lower().endswith(".pdf") else ((pdf.path, pdf.page) if pdf else None)
    tried = []
    if here is not None and d.source == "opening_design":
        try:
            a = _schedule_answer(q, here)
        except Exception as e:  # a malformed sheet must not break the answer
            a = None
            tried.append(f"schedules unreadable ({type(e).__name__})")
        if a is not None and a.value is not None:
            return a
        if a is not None:
            tried.append(a.method)
    # one named door / room / type the schedules do not have: the notes on the sheet are not its answer
    specific = re.search(r"\b(?:door|room|type|mark|raum|porte|porta|ovi|huone)\s+\(?[A-Z]?\d", q.text, re.I) is not None
    if here is not None:
        try:
            rows, k = _read(*here)[:2]
        except Exception:
            rows, k = [], 1.0
        if q.ask in ("list", "where") and not (specific and not q.families):
            try:
                blocks = _blocks(*here)
            except Exception:
                blocks = rows
            a = _mentions(q, blocks, k, here[0])
            if a is not None:
                return a
        tried.append(f"no material word on the sheet's {len(rows)} printed rows")
    elif d.source != "swiss_dwellings":
        rows = _raster_rows(d)
        if rows and q.ask in ("list", "where"):
            a = _mentions(q, rows, 1.0, d.ask_path)
            if a is not None:
                a.method = a.method.replace("printed on the sheet", "read on the sheet by OCR")
                return a
        tried.append("no material word in the sheet's OCR text" if rows else "no readable text on the sheet (OCR)")
    else:
        tried.append("the CAD plan's texts are room names and areas only")
    return Answer(None, None, "no_material", 0.0, "sheet text: " + "; ".join(tried))
