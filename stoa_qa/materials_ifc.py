"""Materials from a BIM model: IfcRelAssociatesMaterial (layer sets with thicknesses, constituents, profiles, lists),
resolved through element types as IfcOpenShell does. When an element links no material, the model's own property
values (Pset 'Material', 'Structural Material') are used, and last its type name ("Basiswand:STB 250" = reinforced
concrete, "Gips 100" = gypsum) — flagged as read from a name.

Every element found is evidence (its GlobalId), so the walls, slabs or pipes made of a material light up in the 3D
model and on the storey plans. Quantities: volumes and areas from the model's quantity sets (or geometry), a layer's
share of an element's volume from its thickness share, a constituent's from its Fraction.
"""
from __future__ import annotations

import collections
import re
import threading
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from . import catalog, ifc
from .core import DATASET, Answer, Evidence
from .materials import ELEMENT_WORDS, LABEL, Query, family_set, fold, matches

ELEMENT_RX = {k: r"(?<![a-z])(?:" + v + r")(?![a-z])" for k, v in ELEMENT_WORDS.items()}

MAX_EVIDENCE = 400
UNSIDED = "<side unknown>"

# element words of a question -> what counts (IFC classes; predefined types for slabs and coverings)
CATEGORY_CLASSES = {
    "wall": ("IfcWall",), "door": ("IfcDoor",), "window": ("IfcWindow",), "column": ("IfcColumn",), "beam": ("IfcBeam",),
    "stair": ("IfcStair", "IfcStairFlight"), "railing": ("IfcRailing",), "member": ("IfcMember",), "plate": ("IfcPlate",),
    "pipe": ("IfcPipeSegment", "IfcPipeFitting"), "duct": ("IfcDuctSegment", "IfcDuctFitting"),
    "conduit": ("IfcCableCarrierSegment", "IfcCableSegment", "IfcCableCarrierFitting", "IfcCableFitting"),
    "covering": ("IfcCovering",), "curtain wall": ("IfcCurtainWall",), "footing": ("IfcFooting", "IfcPile"),
    "furniture": ("IfcFurnishingElement",), "proxy": ("IfcBuildingElementProxy",),
}
PLURAL = {"wall": "walls", "slab": "slabs", "floor": "floor slabs / floorings", "ceiling": "ceilings", "roof": "roof elements",
          "door": "doors", "window": "windows", "column": "columns", "beam": "beams", "stair": "stairs", "railing": "railings",
          "foundation": "foundation elements", "covering": "coverings", "pipe": "pipe segments / fittings",
          "duct": "duct segments / fittings", "conduit": "cable carriers", "member": "members", "plate": "plates",
          "curtain wall": "curtain walls", "furniture": "furniture", "proxy": "other elements", "footing": "footings",
          "structure": "structural elements", "facade": "facade elements"}
SKIP = ("IfcOpeningElement", "IfcVirtualElement", "IfcSpace", "IfcAnnotation", "IfcFeatureElement", "IfcDistributionPort",
        "IfcGrid", "IfcSite", "IfcBuilding", "IfcBuildingStorey", "IfcBuildingElementPart")  # parts roll up into their element
# IFC2X3 exports MEP as generic flow elements; their TYPE says what they are (IfcPipeSegmentType...)
TYPE_CATEGORY = {"IfcPipeSegmentType": "pipe", "IfcPipeFittingType": "pipe", "IfcDuctSegmentType": "duct",
                 "IfcDuctFittingType": "duct", "IfcCableCarrierSegmentType": "conduit", "IfcCableCarrierFittingType": "conduit",
                 "IfcCableSegmentType": "conduit"}
# material names that are exporter placeholders, not materials
PLACEHOLDER = re.compile(r"^\s*(?:<?unnamed>?|unbenannt|standard|default(?: \w+)?|solid(?: \d+)?|radial gradient fill.*|"
                         r"by category|<by category>|none|keine|material|generic|\?|-)\s*(?:\d{5,})?\s*$", re.I)
EXT_WORDS = re.compile(r"exterior|external|ext\b|outside|fassade|facade|au(?:ss|ß)en|aw\b|buiten|ulko|ydre|yttervegg|"
                       r"utvendig|v(?:ä|a)lis|curtain|envelope|cladding|parapet", re.I)
INT_WORDS = re.compile(r"interior|internal|int\b|innen|iw\b|partition|binnen|sis(?:ä|a)|indre|innervegg|trennwand|"
                       r"st(?:ä|a)nderwand|trockenbau|drywall|gk ", re.I)


@dataclass
class Rec:
    el: object
    guid: str
    eid: int
    cls: str                       # IFC class
    cats: set[str]                 # wall / slab / floor / roof / ceiling / ...
    type_name: str
    storey: str
    external: bool | None
    bearing: bool | None
    layers: list[tuple[str, float | None, float | None]]   # (material name, thickness m, fraction)
    source: str                    # ifc | property | type-name | none
    fams: set[str] = field(default_factory=set)
    side_hint: bool | None = None  # exterior / interior read from a layer or function property ("Ebene: Außenwände")
    volume_prop: float | None = None  # a 'Volume' property value (m3) when the model has no quantity set

    def label(self) -> str:
        return self.type_name or self.cls.removeprefix("Ifc")


def has_model(d) -> str | None:
    """'own' when the drawing is (or was rendered from) an IFC model; 'project' when its drawing set ships one
    (opening_design, catalog.PROJECT_MODELS); None otherwise."""
    if d.source in ("ifc_models", "ifc_bench") or d.ask_path.lower().endswith(".ifc"):
        return "own" if ifc.ifc_path_for(d.ask_path).exists() else None
    if d.source == "opening_design" and d.model and d.model_kind == "ifc" and (DATASET / d.model).exists():
        return "project"
    return None


def model_path(d) -> Path:
    if has_model(d) == "project":
        return DATASET / d.model
    return ifc.ifc_path_for(d.ask_path)


def _categories(e, t=None) -> set[str]:
    out = set()
    for cat, classes in CATEGORY_CLASSES.items():
        if any(e.is_a(c) for c in classes):
            out.add(cat)
    if t is not None:
        for tc, cat in TYPE_CATEGORY.items():
            if t.is_a(tc):
                out.add(cat)
    pre = str(getattr(e, "PredefinedType", "") or "").upper()
    if e.is_a("IfcSlab"):
        out.add("slab")
        if pre in ("ROOF",):
            out.add("roof")
        elif pre in ("BASESLAB",):
            out |= {"floor", "foundation"}
        elif pre != "LANDING":
            out.add("floor")
    if e.is_a("IfcRoof"):
        out.add("roof")
    if e.is_a("IfcCovering"):
        out |= {"CEILING": {"ceiling"}, "FLOORING": {"floor"}, "ROOFING": {"roof"}, "CLADDING": {"facade", "wall"},
                "INSULATION": {"insulation_el"}}.get(pre, set())
    if e.is_a("IfcFooting") or e.is_a("IfcPile"):
        out.add("foundation")
    if e.is_a("IfcCurtainWall"):
        out.add("facade")
    if out & {"column", "beam", "slab", "foundation", "footing"}:
        out.add("structure")
    return out


def _layers(m, e, sc: float) -> tuple[list[tuple[str, float | None, float | None]], str]:
    import ifcopenshell.util.element as uel

    try:
        mat = uel.get_material(e, should_skip_usage=True)
    except Exception:
        mat = None
    out: list[tuple[str, float | None, float | None]] = []
    if mat is not None:
        if mat.is_a("IfcMaterial"):
            out.append((mat.Name or "?", None, None))
        elif mat.is_a("IfcMaterialLayerSet"):
            for lay in mat.MaterialLayers or []:
                t = float(lay.LayerThickness) * sc if lay.LayerThickness else None
                out.append(((lay.Material.Name if lay.Material else None) or lay.Name or "?", t, None))
        elif mat.is_a("IfcMaterialConstituentSet"):
            for c in mat.MaterialConstituents or []:
                if c.Material:
                    out.append((c.Material.Name or c.Name or "?", None, float(c.Fraction) if c.Fraction else None))
        elif mat.is_a("IfcMaterialProfileSet"):
            out += [(p.Material.Name or "?", None, None) for p in mat.MaterialProfiles or [] if p.Material]
        elif mat.is_a("IfcMaterialList"):
            out += [(x.Name or "?", None, None) for x in mat.Materials or []]
    out = [x for x in out if not PLACEHOLDER.match(x[0] or "")]
    if out:
        return out, "ifc"
    # an element split into parts (Revit "parts": IfcBuildingElementPart) carries its materials on the parts
    parts = [k for r in (getattr(e, "IsDecomposedBy", None) or []) for k in r.RelatedObjects]
    for k in parts:
        try:
            out += [(x.Name or "?", None, None) for x in uel.get_materials(k) if not PLACEHOLDER.match(x.Name or "")]
        except Exception:
            pass
    if out:
        return out, "ifc"
    try:
        psets = uel.get_psets(e)
    except Exception:
        psets = {}
    for d in psets.values():
        for k in ("Material", "Structural Material", "Materials", "Werkstoff", "Baustoff"):
            v = d.get(k)
            if isinstance(v, str) and v.strip() and not PLACEHOLDER.match(v):
                return [(v.strip(), None, None)], "property"
    return [], "none"


SIDE_KEYS = re.compile(r"^(?:ebene|layer|layer name|function|funktion|position|lage|category|kategorie|usage|nutzung|"
                       r"wall function|wandfunktion|type comments|description|beschreibung)$", re.I)
FOUNDATION_WORDS = re.compile(r"foundation|footing|fundament|fondation|fondazion|funder|perustu|cimentaci|bodenplatte|"
                              r"slab on grade|strip footing|\bpiles?\b", re.I)


def _side_hint(psets: dict) -> bool | None:
    """Exterior (True) / interior (False) from a layer, function or category property ('Ebene: Außenwände')."""
    for d in psets.values():
        for k, v in d.items():
            if isinstance(v, str) and len(v) < 60 and SIDE_KEYS.match(k.strip()):
                if EXT_WORDS.search(v):
                    return True
                if INT_WORDS.search(v) or re.search(r"innen|inner|intern", v, re.I):
                    return False
    return None


def _volume_prop(psets: dict, scale: float) -> float | None:
    for d in psets.values():
        v = d.get("Volume")
        if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0:
            return float(v) * scale
    return None


def _flag(psets: dict, *names) -> bool | None:
    for d in psets.values():
        for n in names:
            v = d.get(n)
            if isinstance(v, bool):
                return v
    return None


_LOCK = threading.Lock()


def index(path: str) -> tuple[Rec, ...]:
    """Every placed element of the model with its materials (cached per model file). One model is opened at a time:
    the big models (hundreds of MB) must not be parsed twice in parallel."""
    with _LOCK:
        return _index(path)


@lru_cache(maxsize=8)
def _index(path: str) -> tuple[Rec, ...]:
    import ifcopenshell.util.element as uel

    from .ifc_props import _unit_scales

    m = ifc.load_model(path)
    sc = m.length_scale
    vol_scale = _unit_scales(m.f, sc)["m3"]
    out = []
    for e in m.f.by_type("IfcElement"):
        if any(e.is_a(c) for c in SKIP):
            continue
        parent = e.Decomposes[0].RelatingObject if e.Decomposes else None
        if parent is not None and parent.is_a(e.is_a()):
            continue
        t = uel.get_type(e)
        cats = _categories(e, t)
        if parent is not None and (parent.is_a("IfcRoof") or parent.is_a("IfcStair") or parent.is_a("IfcCurtainWall")):
            cats |= _categories(parent)  # a roof's slabs are roof elements; a curtain wall's plates are facade
        tname = (t.Name if t is not None else None) or getattr(e, "ObjectType", None) or e.Name or ""
        try:
            psets = uel.get_psets(e)
        except Exception:
            psets = {}
        layers, source = _layers(m, e, sc)
        fams = set()
        for name, _, _ in layers:
            fams |= family_set(name, abbreviations=True)
        if not fams:
            tf = family_set(tname, abbreviations=True)
            if tf:
                fams, source = tf, ("type-name" if source == "none" else source + "+type-name")
        if (FOUNDATION_WORDS.search(tname) or FOUNDATION_WORDS.search(e.Name or "")) and not e.is_a("IfcWall"):
            cats = cats | {"foundation", "structure"}   # footings modelled as slabs / proxies; foundation WALLS stay walls
        s = m.storey_of(e)
        out.append(Rec(e, e.GlobalId, e.id(), e.is_a(), cats, tname, (s.Name if s is not None else "") or "",
                       _flag(psets, "IsExternal"), _flag(psets, "LoadBearing"), layers, source, fams,
                       _side_hint(psets), _volume_prop(psets, vol_scale)))
    return tuple(out)


def _side(r: Rec) -> bool | None:
    if r.external is not None:
        return r.external
    if r.side_hint is not None:
        return r.side_hint
    if EXT_WORDS.search(r.type_name):
        return True
    if INT_WORDS.search(r.type_name):
        return False
    return None


def _scoped(recs, q: Query) -> tuple[list[Rec], list[str]]:
    """Records in the question's element / side / bearing scope, and notes on how the scope was applied."""
    notes = []
    els = [c for c in q.elements if c not in ("structure", "insulation_el")]
    if q.surface == "ceiling" and "ceiling" not in els:
        els.append("ceiling")
    if q.surface == "floor" and "floor" not in els:
        els.append("floor")
    if "covering" in els and q.surface:
        els = [e for e in els if e != "covering"] + ([q.surface] if q.surface not in els else [])
    out = list(recs)
    if els:
        out = [r for r in out if r.cats & set(els)]
    elif "structure" in q.elements or q.bearing:
        out = [r for r in out if "structure" in r.cats or ("wall" in r.cats and r.bearing)]
    if "roof" in els and "slab" in els:
        out = [r for r in out if "roof" in r.cats]   # "roof slabs": the slabs of the roof, not every slab
    if q.side:
        want = q.side == "exterior"
        known = [r for r in out if _side(r) is not None]
        if known:
            out = [r for r in known if _side(r) == want]
        else:
            notes.append(UNSIDED)
    if q.bearing is not None and els and "wall" in els:
        known = [r for r in out if r.bearing is not None]
        if known:
            out = [r for r in known if r.bearing == q.bearing]
        else:
            notes.append("the model does not mark load-bearing walls; all are shown")
    return out, notes


def _rooms(m, recs, question: str) -> tuple[list[Rec] | None, str]:
    """Elements of a room the question names: elements bounding it (IfcRelSpaceBoundary), coverings that cover it
    (IfcRelCoversSpaces) and elements contained in it. (None, '') when no room is named; ([], why) when one is named
    but nothing can be tied to it."""
    scoped = ifc._room_scope(m, question, None, None, None)
    if scoped is None:
        return None, ""
    spaces, how = scoped
    if not spaces:
        return [], f"no room matching {how} in the model"
    ids = set()
    sp_ids = {sp.id() for sp in spaces}
    for b in m.f.by_type("IfcRelSpaceBoundary"):
        if b.RelatingSpace is not None and b.RelatingSpace.id() in sp_ids and b.RelatedBuildingElement is not None:
            ids.add(b.RelatedBuildingElement.id())
    for rel in m.f.by_type("IfcRelCoversSpaces"):
        sp = getattr(rel, "RelatingSpace", None) or getattr(rel, "RelatedSpace", None)
        if sp is not None and sp.id() in sp_ids:
            ids |= {c.id() for c in (rel.RelatedCoverings or [])}
    for rel in m.f.by_type("IfcRelContainedInSpatialStructure"):
        if rel.RelatingStructure is not None and rel.RelatingStructure.id() in sp_ids:
            ids |= {x.id() for x in rel.RelatedElements or []}
    names = ", ".join(sorted({" ".join(x for x in (sp.Name, sp.LongName) if x) for sp in spaces})[:4])
    if not ids:
        return [], f"room {names} found, but the model links no walls, coverings or elements to it"
    return [r for r in recs if r.eid in ids], f"room {names}"


def _mm(t: float | None) -> str:
    return f" {t * 1000:.0f} mm" if t else ""


def _composition(r: Rec) -> str:
    if not r.layers:
        return "no material in the model" + (f" (type name says {', '.join(LABEL[f] for f in sorted(r.fams))})" if r.fams else "")
    return ", ".join(f"{n}{_mm(t)}" + (f" ({fr:.0%})" if fr else "") for n, t, fr in r.layers)


def _plural(cat: str, n: int) -> str:
    word = PLURAL.get(cat, cat + "s")
    return word if n != 1 else word.rstrip("s")


def _by_type(recs: list[Rec]) -> list[tuple[str, list[Rec]]]:
    g: dict[str, list[Rec]] = collections.defaultdict(list)
    for r in recs:
        g[r.label()].append(r)
    return sorted(g.items(), key=lambda kv: (-len(kv[1]), kv[0]))


def _ev(recs: list[Rec], label=None) -> list[Evidence]:
    return [Evidence("ifc", r.guid, (label(r) if label else f"{r.label()}: {_composition(r)}")[:120], viewer_id=r.eid)
            for r in recs[:MAX_EVIDENCE]]


def _main_cat(r: Rec, wanted: list[str]) -> str:
    for c in wanted:
        if c in r.cats:
            return c
    for c in ("wall", "slab", "roof", "ceiling", "floor", "column", "beam", "door", "window", "stair", "railing", "covering",
              "pipe", "duct", "conduit", "footing", "member", "plate", "curtain wall", "furniture", "proxy"):
        if c in r.cats:
            return c
    return r.cls.removeprefix("Ifc").lower()


def _listing(recs: list[Rec], wanted: list[str], per_cat: int = 12) -> str:
    """Grouped by element kind, then type: '<kind> (n): \n • <type> (k): layer 1 t, layer 2 t'."""
    cats: dict[str, list[Rec]] = collections.defaultdict(list)
    for r in recs:
        cats[_main_cat(r, wanted)].append(r)
    lines = []
    for cat, rs in sorted(cats.items(), key=lambda kv: -len(kv[1])):
        types = _by_type(rs)
        lines.append(f"{_plural(cat, len(rs)).capitalize()} ({len(rs)}, {len(types)} type{'s' if len(types) != 1 else ''}):")
        for name, trs in types[:per_cat]:
            lines.append(f"• {name} ({len(trs)}): {_composition(trs[0])}")
        if len(types) > per_cat:
            lines.append(f"• ... {len(types) - per_cat} more types")
    return "\n".join(lines)


def _material_totals(recs: list[Rec]) -> list[tuple[str, int]]:
    c: collections.Counter = collections.Counter()
    for r in recs:
        for n in {n for n, _, _ in r.layers}:
            c[n] += 1
    return c.most_common()


def _share(r: Rec, wanted: set[str]) -> float:
    """The part of an element made of the wanted families: thickness share of matching layers, constituent fraction, or 1."""
    if not r.layers:
        return 1.0 if matches(r.fams, wanted) else 0.0
    hits = [(n, t, fr) for n, t, fr in r.layers if matches(family_set(n, abbreviations=True), wanted)]
    if not hits:
        return 1.0 if matches(r.fams, wanted) and r.source.endswith("type-name") else 0.0
    if all(t for _, t, _ in r.layers):
        return sum(t for _, t, _ in hits) / sum(t for _, t, _ in r.layers)
    if all(fr for _, _, fr in hits):
        return min(1.0, sum(fr for _, _, fr in hits))
    return 1.0 if len(r.layers) == 1 else len(hits) / len(r.layers)


def _quantity(m, r: Rec, kind: str, scales: dict) -> float | None:
    if kind == "volume":
        v = m.qto(r.el, "NetVolume", "GrossVolume")
        if v:
            return v * scales["m3"]
        if r.volume_prop:
            return r.volume_prop
        g = m.shape(r.el)
        if g is None:
            return None
        import ifcopenshell.util.shape as ushape

        vol = ushape.get_volume(g)
        xyz = ushape.get_vertices(g)
        box = float((xyz.max(0) - xyz.min(0)).prod()) if len(xyz) else 0.0
        return vol if 0 < vol <= box * 1.02 else None   # an open or doubled mesh gives a volume bigger than its box
    if kind == "area":
        v = m.qto(r.el, "NetArea", "GrossArea", "NetSideArea", "GrossSideArea", "NetFootprintArea", "GrossFootprintArea")
        if v:
            return v * scales["m2"]
        g = m.shape(r.el)
        if g is None:
            return None
        import ifcopenshell.util.shape as ushape

        return ushape.get_footprint_area(g) if not ("wall" in r.cats) else None
    if kind == "length":
        v = m.qto(r.el, "Length", "NetLength", "GrossLength")
        return v * scales["m"] if v else None
    return None


BREAKDOWN = re.compile(r"\b(?:breakdown|broken down|break down|by (?:element|type|kind|category)|per (?:element|type)|each type|"
                       r"distribut\w*|split|aufgeteilt|aufschl\w*|r(?:é|e)partition|ripartizion\w*|verdeling)\b", re.I)


def _counts_material(q: Query) -> bool:
    """'How many vapor barriers ...': the thing counted is the material itself, named right after 'how many'."""
    m = re.match(r"^\s*(?:how many|wie viele|combien de|quanti|quante|hoeveel|hvor mange|kuinka monta)\s+(.{3,40}?)\s+"
                 r"(?:are|is|do|does|in|on|sind|gibt|y a|ci sono|zijn|er|on|ovat)\b", q.text, re.I)
    return bool(m and family_set(m.group(1)) and not any(re.search(ELEMENT_RX[e], m.group(1), re.I) for e in ELEMENT_RX))


def _breakdown(m, scoped: list[Rec], q: Query, scope_txt: str, where_model: str, caveat: str, note_txt: str, src: str) -> Answer:
    """'Total area of each floor finish': the quantity split by material (a layer's share by thickness)."""
    from .ifc_props import _unit_scales

    scales = _unit_scales(m.f, m.length_scale)
    per: dict[str, float] = collections.defaultdict(float)
    used = []
    for r in scoped:
        if not r.layers:
            continue
        v = _quantity(m, r, q.quantity, scales)
        if v is None:
            continue
        used.append(r)
        if q.quantity == "volume" and all(t for _, t, _ in r.layers):
            tot = sum(t for _, t, _ in r.layers)
            for n, t, _ in r.layers:
                per[n] += v * t / tot
        else:  # an area or a length belongs to every layer of the element
            for n in {n for n, _, _ in r.layers}:
                per[n] += v
    unit = {"volume": "m³", "area": "m²", "length": "m"}[q.quantity]
    if not used:
        return Answer(None, None, "no_material", 0.0,
                      f"model: no {scope_txt} with both a material and a {q.quantity} in {where_model}", source_file=src)
    rows = sorted(per.items(), key=lambda kv: -kv[1])
    text = (f"{q.quantity.capitalize()} by material over {len(used)} {scope_txt}:\n"
            + "\n".join(f"• {n}: {v:,.2f} {unit}" for n, v in rows[:20]) + (f"\n• ... {len(rows) - 20} more" if len(rows) > 20 else ""))
    return Answer(text, None, "materials_breakdown", 0.85,
                  f"{q.quantity} from quantity sets (else geometry) of {where_model}, split by material layer{caveat}{note_txt}",
                  _ev(used), source_file=src)


def answer(d, q: Query) -> Answer | None:
    path = model_path(d)
    if not path.exists():
        return None
    from .ifc_props import _unit_scales

    recs = index(str(path))          # opens the model under the lock
    m = ifc.load_model(str(path))
    src = path.relative_to(DATASET).as_posix()
    project = has_model(d) == "project"
    where_model = f"the project's model {path.name}" if project else "the model"
    caveat = " (the project's building model; the sheet itself may differ)" if project else ""

    rooms, room_note = _rooms(m, recs, q.text)
    if rooms is not None and not rooms:
        return Answer(None, None, "no_material", 0.0, f"model: {room_note}", source_file=src)
    pool = rooms if rooms is not None else recs
    scoped, notes = _scoped(pool, q)
    unsided = UNSIDED in notes
    notes = [n for n in notes if n != UNSIDED]
    kinds = " / ".join(PLURAL.get(e, e) for e in q.elements if e in PLURAL) or (PLURAL[q.surface] if q.surface in PLURAL else "elements")
    if unsided:
        notes.append(f"the model does not mark which {kinds} are {q.side}, so all of them are listed")
    scope_txt = " ".join(x for x in [q.side if q.side and not unsided else "", kinds] if x)
    if rooms is not None:
        scope_txt += f" of {room_note}"
    lead = ""
    if not scoped and (q.surface == "ceiling" or "ceiling" in q.elements) and q.ask in ("list", "where"):
        # no ceiling elements: say so (the slabs above are not a ceiling material); name the slabs only as context
        slabs = [r for r in pool if r.cats & {"floor", "roof", "slab"} and "foundation" not in r.cats and r.layers]
        ctx = "; ".join(f"{n} ({k})" for n, k in _material_totals(slabs)[:6])
        return Answer("Not in the model: it has no ceiling elements (no IfcCovering of type CEILING), so no ceiling "
                      "material is stored.", None, "materials_absent", 0.85,
                      "Checked every IfcCovering in the model" + (f"; the slabs above the rooms (seen from below) are made of "
                                                                 f"{ctx} — those are slab materials, not ceiling finishes" if ctx else ""),
                      _ev(slabs[:50]), source_file=src)
    if not scoped:
        return Answer(None, None, "no_material", 0.0, f"model: no {scope_txt} in {where_model}", source_file=src)
    with_mat = [r for r in scoped if r.layers or r.fams]
    note_txt = ("; " + "; ".join(notes)) if notes else ""

    if q.ask == "list":
        if not with_mat:
            return Answer(None, None, "no_material", 0.0,
                          f"model: {len(scoped)} {scope_txt} found, but {where_model} links no material to them and their "
                          f"type names name none", source_file=src)
        totals = _material_totals(with_mat)
        head = (f"{len(with_mat)} {scope_txt} with materials in {where_model}"
                + (f" ({len(scoped) - len(with_mat)} more have none)" if len(scoped) > len(with_mat) else ""))
        mats = "; ".join(f"{n} ({k})" for n, k in totals[:12]) + (f"; ... {len(totals) - 12} more" if len(totals) > 12 else "")
        text = f"{lead}{head}.\nMaterials: {mats or 'named only by type'}.\n{_listing(with_mat, q.elements)}"
        if notes:
            text += "\n(" + "; ".join(notes) + ")"
        inferred = all(r.source == "type-name" for r in with_mat)
        return Answer(text, None, "materials_list", 0.6 if inferred else 0.9,
                      f"Read from {'the element type names' if inferred else 'the material layers linked to each element'} "
                      f"in {where_model}{caveat}{note_txt}", _ev(with_mat), source_file=src)

    if q.ask == "quantity" and not q.families:
        return _breakdown(m, scoped, q, scope_txt, where_model, caveat, note_txt, src)
    hit = [r for r in scoped if matches(r.fams, q.families)]
    names = ", ".join(LABEL[f] for f in sorted(q.families))
    if q.ask == "count":
        if rooms is not None and not hit and not any(r.cats & set(q.elements) for r in pool):
            return Answer(None, None, "no_material", 0.0, f"model: nothing of {scope_txt} is tied to {room_note}", source_file=src)
        if _counts_material(q):
            # "how many vapor barriers are in the roof": the layers of that material, not the elements holding them
            layers = [(r, n) for r in hit for n, _, _ in r.layers if matches(family_set(n, abbreviations=True), q.families)]
            if layers:
                names_l = sorted({n for _, n in layers})
                return Answer(len(layers), "count", "materials_count", 0.85,
                              f"{len(layers)} {names} layer(s) in {len(hit)} {scope_txt} of {where_model}: {', '.join(names_l[:6])}"
                              + caveat + note_txt, _ev(hit), source_file=src)
        types = "; ".join(f"{n} ({len(rs)})" for n, rs in _by_type(hit)[:10])
        return Answer(len(hit), "count", "materials_count", 0.9,
                      f"{len(hit)} of {len(scoped)} {scope_txt} are {names} in {where_model}" + (f": {types}" if types else "")
                      + caveat + note_txt, _ev(hit), source_file=src)
    if not hit:
        known = sorted({LABEL.get(f, f) for r in with_mat for f in r.fams})
        return Answer(None, None, "no_material", 0.0,
                      f"model: none of the {len(scoped)} {scope_txt} is {names}"
                      + (f" (materials there: {', '.join(known[:10])})" if known else ""), source_file=src)

    if q.ask == "quantity":
        scales = _unit_scales(m.f, m.length_scale)
        total, whole, used, missing = 0.0, 0.0, [], 0
        per_kind: dict[str, float] = collections.defaultdict(float)
        per_type: dict[str, float] = collections.defaultdict(float)
        for r in hit:
            v = _quantity(m, r, q.quantity, scales)
            if v is None:
                missing += 1
                continue
            share = _share(r, q.families)
            total += v * share
            whole += v
            per_kind[_main_cat(r, q.elements)] += v * share
            per_type[r.label()] += v * share
            used.append(r)
        if not used:
            return Answer(None, None, "no_material", 0.0, f"model: {len(hit)} {names} {scope_txt} found, but none has a {q.quantity}",
                          source_file=src)
        unit = {"volume": "m3", "area": "m2", "length": "m"}[q.quantity]
        u = {"m3": "m³", "m2": "m²", "m": "m"}[unit]
        partial = whole - total > 0.005 * max(whole, 1e-9)
        method = (f"{q.quantity} of {names} in {len(used)} {scope_txt} of {where_model} (quantity sets, else geometry)"
                  + ("; for layered elements only the share of the matching layers (by thickness) is counted" if partial else "")
                  + (f"; {missing} element(s) without a {q.quantity} left out" if missing else "") + caveat + note_txt)
        if BREAKDOWN.search(q.text) or partial:
            by = per_type if re.search(r"\b(?:by|per|each) (?:\w+ )?type\b", q.text, re.I) else per_kind
            lines = [f"Total: {total:,.2f} {u}" + (f" in the {names} layers ({whole:,.2f} {u} counting the whole elements)" if partial else "")]
            if len(by) > 1 or BREAKDOWN.search(q.text):
                lines += [f"• {_plural(k, 2).capitalize() if by is per_kind else k}: {v:,.2f} {u}"
                          for k, v in sorted(by.items(), key=lambda kv: -kv[1])[:15]]
            return Answer("\n".join(lines), None, "materials_quantity", 0.85, method, _ev(used), source_file=src)
        return Answer(round(total, 2), unit, "materials_quantity", 0.85, method, _ev(used), source_file=src)

    if q.ask == "thickness":
        found = collections.OrderedDict()
        for r in hit:
            for n, t, _ in r.layers:
                if t and matches(family_set(n, abbreviations=True), q.families):
                    found.setdefault((n, round(t * 1000)), []).append(r)
        if not found:
            return Answer(None, None, "no_material", 0.0,
                          f"model: {len(hit)} {scope_txt} contain {names}, but no layer thickness is stored", source_file=src)
        rows = sorted(found.items(), key=lambda kv: -len(kv[1]))
        values = {mm for (_, mm) in found}
        text = "; ".join(f"{mm} mm {n} in {', '.join(sorted({r.label() for r in rs})[:3])} ({len(rs)})" for (n, mm), rs in rows[:8])
        value = float(next(iter(values))) if len(values) == 1 else text
        return Answer(value, "mm" if len(values) == 1 else None, "materials_thickness", 0.9,
                      f"Layer thickness from the material layer sets in {where_model}: {text}{caveat}{note_txt}",
                      _ev(sorted({id(r): r for rs in found.values() for r in rs}.values(), key=lambda r: r.eid)), source_file=src)

    # where
    per_storey = collections.Counter(r.storey or "no storey" for r in hit)
    named = sorted({n for r in hit for n, _, _ in r.layers if matches(family_set(n, abbreviations=True), q.families)})
    text = (f"{names.capitalize()} is used in {len(hit)} {scope_txt if q.elements or q.side else 'elements'}"
            + (f" (named in {where_model}: {', '.join(named[:8])}{' ...' if len(named) > 8 else ''})" if named else "") + ".\n"
            + _listing(hit, q.elements) + "\nBy storey: " + ", ".join(f"{s} {n}" for s, n in per_storey.most_common(12)))
    inferred = all(r.source == "type-name" for r in hit)
    return Answer(text, None, "materials_where", 0.6 if inferred else 0.9,
                  f"Elements whose linked materials{' (or type names)' if any(r.source.endswith('type-name') for r in hit) else ''} "
                  f"are {names}, read from {where_model}{caveat}{note_txt}", _ev(hit), source_file=src)
