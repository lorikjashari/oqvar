"""Materials specialist (stoa_qa.materials*): vocabulary, question parsing, schedules + finish keys across a sheet set,
notes on a plan, CAD wall hatches, IFC material layers. Synthetic drawings only, no LLM.

    Test_Mat set   A-101  floor plan: room tags 101 / 102 / 103 and a note "4\" CONC. SLAB ON GRADE"
                   A-801  finish schedule (rooms 101-103) + finish key (F1 carpet tile, F2 ceramic tile, W1 gyp. board...)
    test_mat.ifc   one exterior wall (brick 90 / insulation 50 / CMU 190 mm), one interior wall (gypsum), a concrete slab
"""
import pymupdf
import pytest

from tests.conftest import ROOT, STEM

PROJECT = "data/opening_design/Test_Mat/Sheets"
PLAN = f"{PROJECT}/A-101 - Floor Plan.pdf"
SCHED = f"{PROJECT}/A-801 - Finish Schedule.pdf"
IFC = "data/ifc_bench/test_mat.ifc"


def _pdf(path, texts):
    doc = pymupdf.open()
    page = doc.new_page(width=1224, height=792)
    for (x, y), text, size in texts:
        page.insert_text((x, y), text, fontsize=size)
    out = ROOT / path
    out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(out)


def _row(y, cells, xs, size=8):
    return [((x, y), c, size) for x, c in zip(xs, cells) if c]


def _ifc():
    import ifcopenshell
    import ifcopenshell.api as api

    f = ifcopenshell.file(schema="IFC4")
    proj = api.run("root.create_entity", f, ifc_class="IfcProject", name="P")
    api.run("unit.assign_unit", f)
    ctx = api.run("context.add_context", f, context_type="Model")
    site = api.run("root.create_entity", f, ifc_class="IfcSite", name="S")
    bldg = api.run("root.create_entity", f, ifc_class="IfcBuilding", name="B")
    st = api.run("root.create_entity", f, ifc_class="IfcBuildingStorey", name="EG")
    api.run("aggregate.assign_object", f, relating_object=proj, products=[site])
    api.run("aggregate.assign_object", f, relating_object=site, products=[bldg])
    api.run("aggregate.assign_object", f, relating_object=bldg, products=[st])

    def layered(name, layers):
        ls = api.run("material.add_material_set", f, name=name, set_type="IfcMaterialLayerSet")
        for mname, t in layers:
            m = api.run("material.add_material", f, name=mname)
            lay = api.run("material.add_layer", f, layer_set=ls, material=m)
            api.run("material.edit_layer", f, layer=lay, attributes={"LayerThickness": t})
        return ls

    def element(cls, name, mat, external=None, volume=None):
        e = api.run("root.create_entity", f, ifc_class=cls, name=name)
        api.run("spatial.assign_container", f, relating_structure=st, products=[e])
        api.run("material.assign_material", f, products=[e], type="IfcMaterialLayerSet", material=mat)
        if not e.HasAssociations:  # older / newer API spellings: link the layer set directly
            f.createIfcRelAssociatesMaterial(ifcopenshell.guid.new(), None, None, None, [e], mat)
        if external is not None:
            ps = api.run("pset.add_pset", f, product=e, name="Pset_WallCommon")
            api.run("pset.edit_pset", f, pset=ps, properties={"IsExternal": external})
        if volume is not None:
            q = api.run("pset.add_qto", f, product=e, name="Qto_BaseQuantities")
            api.run("pset.edit_qto", f, qto=q, properties={"NetVolume": volume})
        return e

    ext = layered("Exterior - Brick on CMU", [("Masonry - Brick", 90.0), ("Insulation - Rigid", 50.0), ("Masonry - Concrete Block", 190.0)])
    inner = layered("Interior - Partition", [("Gypsum Wall Board", 13.0), ("Metal - Stud Layer", 92.0), ("Gypsum Wall Board", 13.0)])
    slab = layered("Floor - Concrete 200", [("Concrete - Cast-in-Place", 200.0)])
    element("IfcWall", "W1", ext, external=True, volume=3.3)
    element("IfcWall", "W2", inner, external=False, volume=1.2)
    element("IfcSlab", "S1", slab, volume=10.0)
    out = ROOT / IFC
    out.parent.mkdir(parents=True, exist_ok=True)
    f.write(str(out))


@pytest.fixture(scope="module", autouse=True)
def drawings():
    xs = [100, 160, 240, 330, 380, 430, 480, 530, 580, 630]
    head = ["Area", "Number", "Room Name", "Floor Finish", "Base Finish", "North Wall", "South Wall", "East Wall", "West Wall", "Ceiling Finish"]
    rows = [("250 SF", "101", "OFFICE", "F1", "B1", "W1", "W1", "W1", "W1", "C1"),
            ("80 SF", "102", "RESTROOM", "F2", "B2", "W2", "W1", "W1", "W1", "C1"),
            ("120 SF", "103", "STORAGE", "F3", "B1", "W1", "W1", "W1", "W1", "C1")]
    texts = [((100, 80), "FINISH ROOM SCHEDULE", 12)] + _row(110, head, xs, 7)
    for i, r in enumerate(rows):
        texts += _row(130 + 16 * i, r, xs)
    key = [("F1", "CARPET TILE"), ("F2", "CERAMIC TILE"), ("F3", "SEALED CONCRETE"), ("B1", "RUBBER BASE"), ("B2", "CERAMIC TILE BASE"),
           ("W1", "PAINTED GYP. BD"), ("W2", "CERAMIC TILE WAINSCOT"), ("C1", "ACOUSTIC CEILING TILE")]
    texts += [((100, 300), "FINISH KEY", 10)] + [t for i, (c, dsc) in enumerate(key) for t in _row(320 + 14 * i, (c, dsc), (100, 140))]
    _pdf(SCHED, texts)
    _pdf(PLAN, [((200, 200), "OFFICE", 9), ((205, 212), "101", 9), ((500, 200), "RESTROOM", 9), ((510, 212), "102", 9),
                ((800, 200), "STORAGE", 9), ((805, 212), "103", 9), ((300, 500), '4" CONC. SLAB ON GRADE', 8),
                ((100, 700), "FIRST FLOOR PLAN", 10)])
    _ifc()
    from stoa_qa import materials_sheet

    materials_sheet._read.cache_clear()
    materials_sheet.set_schedules.cache_clear()
    yield


def _drawing(path, source):
    from stoa_qa.catalog import Drawing, Sheet

    return Drawing(id=path, source=source, title="t", ask_path=path,
                   sheets=[Sheet("pdf", path, "sheet", 0)] if path.endswith(".pdf") else [])


# ------------------------------------------------------------------ vocabulary and questions

def test_vocabulary_multilingual_and_abbreviations():
    from stoa_qa.materials import family_set

    assert family_set("Stahlbeton 250") == {"concrete"}
    assert family_set("Kalksandstein") == {"masonry"}
    assert family_set("Trockenbau - Rigips") == {"gypsum"}
    assert "parquet" in family_set("PARKETTI")
    assert family_set("PAINTED GYP. BD", abbreviations=True) == {"paint", "gypsum"}
    assert family_set("WD-1 | STL-1", abbreviations=True) == {"wood", "steel"}
    # look-alikes that are not materials
    assert family_set("installation without werden Korkeus Zeichen") == set()
    assert family_set("AL", abbreviations=True) == set()          # a bare short abbreviation needs a tag number


def test_intent_parsing():
    from stoa_qa.materials import intent_of

    q = intent_of("What materials are used in the exterior walls?")
    assert q.ask == "list" and q.side == "exterior" and q.elements == ["wall"]
    q = intent_of("What is the total volume of concrete in the slabs?")
    assert q.ask == "quantity" and q.quantity == "volume" and q.families == {"concrete"}
    assert intent_of("Welche Materialien haben die Außenwände?").elements == ["wall"]
    assert intent_of("What is the thickness of the insulation of the external walls?").ask == "thickness"
    for not_material in ("How many rooms are there?", "What is the area of the kitchen?",
                         "How many rooms are listed in the finish room schedule?", "What is the expected lifespan of the roofing material?"):
        assert intent_of(not_material) is None, not_material


# ------------------------------------------------------------------ schedules, keys, plan notes

def test_room_finish_decoded_from_the_schedule_sheet():
    from stoa_qa import materials

    a, _ = materials.answer(_drawing(PLAN, "opening_design"), "What is the floor finish of room 102?", use_llm=False)
    assert a.value == "F2 - CERAMIC TILE"
    assert "A-801" in a.method
    # the room is shown where its tag is printed on the plan being looked at
    assert a.evidence and a.evidence[0].ref == "tag-102"


def test_rooms_with_a_material_are_found_through_the_finish_key():
    from stoa_qa import materials

    d = _drawing(PLAN, "opening_design")
    a, _ = materials.answer(d, "How many rooms have floor finish F1?", use_llm=False)
    assert a.value == 1
    a, _ = materials.answer(d, "Which rooms have a carpet floor?", use_llm=False)
    assert "101 OFFICE" in a.value and "102" not in a.value
    a, _ = materials.answer(d, "Where is ceramic tile used?", use_llm=False)
    assert "102 RESTROOM" in a.value                       # floor F2 and base B2 / wall W2


def test_notes_on_the_plan_are_material_evidence():
    from stoa_qa import materials

    a, _ = materials.answer(_drawing(PLAN, "opening_design"), "Where is concrete used on this plan?", use_llm=False)
    # the finish schedule's F3 "SEALED CONCRETE" (room 103) or the printed slab note: both are on the drawing set
    assert a.value is not None and ("103" in str(a.value) or "CONC" in str(a.value))


# ------------------------------------------------------------------ CAD hatches (Swiss plan)

def test_swiss_wall_hatches_are_read_and_drawn():
    from stoa_qa import materials

    d = _drawing(f"data/swiss_dwellings/{STEM}.pdf", "swiss_dwellings")
    a, tried = materials.answer(d, "What are the walls made of?", use_llm=False)
    assert a.value is not None and "ANSI31" in a.value and "masonry" in a.value.lower()
    assert a.confidence < 0.6                              # convention, not a specification
    assert any(e.kind == "polygon" and e.coords for e in a.evidence)


# ------------------------------------------------------------------ BIM model

def test_ifc_layers_side_and_quantities():
    from stoa_qa import materials

    d = _drawing(IFC, "ifc_bench")
    a, _ = materials.answer(d, "Which materials are used in the exterior walls?", use_llm=False)
    assert "Masonry - Brick" in a.value and "Gypsum" not in a.value
    a, _ = materials.answer(d, "What is the thickness of the insulation of the external walls?", use_llm=False)
    assert a.value == 50.0 and a.unit == "mm"
    a, _ = materials.answer(d, "How many walls contain brick?", use_llm=False)
    assert a.value == 1
    a, _ = materials.answer(d, "What is the total volume of concrete in the slabs?", use_llm=False)
    assert a.value == 10.0 and a.unit == "m3"
    a, _ = materials.answer(d, "Which ceiling materials are used?", use_llm=False)
    assert a.value.startswith("Not in the model")         # no ceiling elements: say so, don't list slabs as ceilings


def test_nothing_found_says_what_was_searched():
    from stoa_qa import materials

    a, tried = materials.answer(_drawing(PLAN, "opening_design"), "Where is marble used?", use_llm=False)
    assert a.value is None and a.intent == "no_material"
    assert {t["source"] for t in tried} >= {"sheet text", "hatches"}


# ------------------------------------------------------------------ the hatch generator used for training

def test_hatch_samples_render_for_every_class():
    import random

    from stoa_qa import hatch_synth as hs

    rng = random.Random(0)
    for c in hs.CLASSES:
        im, k = hs.sample(rng, c)
        assert im.size == (hs.PATCH, hs.PATCH) and hs.CLASSES[k] == c
