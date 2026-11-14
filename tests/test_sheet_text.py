"""Raster sheets answered from their own text: OCR rows -> document clerk (no OCR engine or LLM needed here)."""
from stoa_qa import construction_detect, docqa, llm, sheet_ocr

LINES = [  # what sheet_ocr.read returns for a piping plan: tags on one printed line, a note, a vertical title-block line
    {"box": [100, 100, 220, 112], "text": "CV-1", "conf": 0.9}, {"box": [300, 100, 420, 112], "text": "CV-2", "conf": 0.9},
    {"box": [500, 100, 700, 112], "text": "CV-3 CV-4", "conf": 0.9},
    {"box": [100, 300, 400, 312], "text": "16'SUCTIONHEADER", "conf": 0.9},
    {"box": [900, 500, 912, 700], "text": "PIPING PLAN", "conf": 0.9, "vertical": True},
] + [{"box": [100, 400 + 20 * i, 300, 412 + 20 * i], "text": f"NOTE {i}", "conf": 0.9} for i in range(6)]


def test_rows_keep_their_printed_parts_and_vertical_lines(monkeypatch):
    monkeypatch.setattr(sheet_ocr, "read", lambda image, key=None: LINES)
    rows = docqa.raster_layout("sheet.png", "k")
    tags = next(r for r in rows if "CV-1" in r["text"])
    assert [p["text"] for p in tags["parts"]] == ["CV-1", "CV-2", "CV-3 CV-4"]
    assert any(r["text"] == "PIPING PLAN" and r["bbox"] == [900, 500, 912, 700] for r in rows)


def test_a_count_is_the_listed_items_verified_on_the_sheet(monkeypatch):
    monkeypatch.setattr(sheet_ocr, "read", lambda image, key=None: LINES)
    rows = docqa.raster_layout("sheet.png", "k")
    tag_row = next(r for r in rows if "CV-1" in r["text"])
    reply = {"kind": "count", "rows": [tag_row["id"]], "items": ["CV-1", "CV-2", "CV-3", "CV-4", "CV-4", "CV-9"],
             "answer": None, "unit": None, "note": ""}
    monkeypatch.setattr(llm, "chat_json", lambda *a, **k: (reply, None))
    a, _ = docqa.ask("uploads/x.png", "How many check valves?", rows=rows)
    # CV-4 listed twice counts once; CV-9 is not printed anywhere, so it is not counted
    assert a.value == 4 and "1 listed item(s) not found" in a.method
    cv4 = next(e for e in a.evidence if e.label == "CV-4")
    assert 597 <= cv4.coords[0][0] < cv4.coords[1][0] <= 703   # its own share of the "CV-3 CV-4" line (3 px margin)


def test_a_value_that_carries_its_unit_is_not_given_it_twice(monkeypatch):
    rows = [{"id": "L1", "text": "TOTAL SECOND FLOOR = | 26,562 SF", "bbox": [0, 0, 10, 10], "parts": []}]
    monkeypatch.setattr(llm, "chat_json", lambda *a, **k: ({"kind": "lookup", "rows": ["L1"], "answer": "26,562 SF",
                                                            "unit": "SF", "note": ""}, None))
    a, _ = docqa.ask("uploads/x.png", "Total second floor area?", rows=rows)
    assert a.value == "26,562 SF" and a.unit is None


def test_cjk_misreads_are_stripped():
    assert sheet_ocr.clean("FP-4岁") == "FP-4" and sheet_ocr.clean("FP-3 T") == "FP-3 T"


def test_a_named_tag_prefix_counts_only_that_prefix(monkeypatch):
    rows = [{"id": "L1", "text": "CV-1 | CV-2 | FSCV-1", "bbox": [0, 0, 300, 10],
             "parts": [{"text": t, "bbox": [100 * i, 0, 100 * i + 50, 10]} for i, t in enumerate(["CV-1", "CV-2", "FSCV-1"])]}]
    monkeypatch.setattr(llm, "chat_json", lambda *a, **k: ({"kind": "count", "rows": ["L1"], "items": ["CV-1", "CV-2", "FSCV-1"],
                                                            "answer": None, "unit": None, "note": ""}, None))
    a, _ = docqa.ask("uploads/x.png", "How many check valves (CV) are there?", rows=rows)
    assert a.value == 2 and [e.label for e in a.evidence] == ["CV-1", "CV-2"]


def test_exact_tags_do_not_match_longer_tags_or_repeated_notes():
    words = [(100 * i, 0, 100 * i + 60, 10, t)
             for i, t in enumerate(["FSCV-1", "CV-10", "CV-1", "CV-1", "SEE CV-99"])]
    rows = docqa.rows_of(words, 500)
    a = docqa.tagged_count(docqa.tag_query("Highlight distinct check valve tags (CV)"), rows)
    assert a.value == 2
    assert {e.label for e in a.evidence} == {"CV-1", "CV-10"}
    assert next(e.coords for e in a.evidence if e.label == "CV-1") == [[200, 0], [260, 10]]
    assert docqa.tag_query("How many check valves in the west wing (CV)?") is None


def test_ordinary_fixture_type_marks_are_not_unique_equipment_ids():
    rows = docqa.rows_of([(0, 0, 50, 10, "WC-1"), (0, 30, 50, 40, "WC-1")], 300)
    assert docqa.tagged_count(docqa.tag_query("How many toilets (WC)?"), rows) is None


def test_piping_symbol_candidates_are_labelled_and_need_a_matching_sheet(monkeypatch):
    monkeypatch.setattr(construction_detect.live, "live_config", lambda: {"pid_symbols": {"threshold": 0.5}})
    monkeypatch.setattr(construction_detect, "boxes", lambda *args: [[0, 0.8, 1, 2, 10, 20], [2, 0.9, 3, 4, 20, 30]])
    query = construction_detect.PID_ASK.fullmatch("Highlight all valves")
    assert construction_detect.pid_answer("sheet.png", query, [{"text": "Water distribution"}]) is None
    a = construction_detect.pid_answer("sheet.png", query, [{"text": "PIPING PLAN"}])
    assert a.value == "1 possible valve" and len(a.evidence) == 1
    assert a.evidence[0].coords == [[1, 2], [10, 20]]


def test_tile_fragments_inside_a_longer_reading_are_dropped():
    lines = [{"box": [100, 10, 500, 30], "text": "TOTALSECONDFLOOR=", "conf": 0.8},
             {"box": [300, 11, 500, 29], "text": "ONDFLOOR=", "conf": 0.95},
             {"box": [600, 10, 700, 30], "text": "26,562 SF", "conf": 0.9}]
    assert [ln["text"] for ln in sheet_ocr._merge(lines)] == ["TOTALSECONDFLOOR=", "26,562 SF"]


def test_the_same_label_printed_twice_counts_twice(monkeypatch):
    rows = [{"id": "L1", "text": "Bedroom", "bbox": [0, 0, 50, 10], "parts": [{"text": "Bedroom", "bbox": [0, 0, 50, 10]}]},
            {"id": "L2", "text": "Bedroom", "bbox": [300, 0, 350, 10], "parts": [{"text": "Bedroom", "bbox": [300, 0, 350, 10]}]}]
    monkeypatch.setattr(llm, "chat_json", lambda *a, **k: ({"kind": "count", "rows": ["L1", "L2"], "items": ["Bedroom", "Bedroom"],
                                                            "answer": None, "unit": None, "note": ""}, None))
    a, _ = docqa.ask("uploads/x.png", "How many bedrooms are there?", rows=rows)
    assert a.value == 2 and [e.coords[0][0] for e in a.evidence] == [-3, 297]
