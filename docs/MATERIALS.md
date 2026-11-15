# Materials: what things are made of, and where

The **Materials specialist** (`stoa_qa/materials.py`, first in the crew for material wording) answers "which materials are in the
exterior walls?", "where is concrete used?", "floor finish of room 113?", "how many doors are wood (WD-1)?", "volume of concrete in
the slabs?" in ~13 languages. It reads the drawing in this order and only gives up after all of them:

1. **BIM model** (`materials_ifc.py`): IfcRelAssociatesMaterial layers / constituents / profiles (through types and element parts),
   layer thicknesses, exterior / load-bearing flags (also from layer properties like ArchiCAD "Ebene"), volumes with a layer's
   thickness share, room scope via space boundaries. Placeholders ("Unnamed", "Standard") are not materials; "no ceiling elements"
   is said, not papered over with slab materials.
2. **The sheet's words** (`materials_sheet.py`): finish and door schedules rebuilt from their column positions anywhere in the
   drawing set, codes decoded through the finish key (F4 → RESILIENT FLOORING), the rooms / doors found located at their tags on
   the plan being looked at; material notes on the plan (also rotated ones), OCR text on raster sheets.
3. **Hatches** (`hatch.py`): CAD hatch names in DXF (Swiss walls: ANSI31 / SOLID) and hatch textures recognised on sheet images by
   a ResNet-18 trained on Modal (`modal_jobs/hatch_cloud.py`) — synthetic patterns from the AutoCAD library (`hatch_synth.py`)
   composited onto 30k real sheet windows, DXF-labelled Swiss walls and 351 hand-labelled active-learning picks. Read through the
   sheet's legend when it has one, else drafting convention — always flagged as inferred. Weights:
   `third_party/weights/hatch-h17-al.pt` (gitignored; `modal volume get stoa-cubicasa hatch/runs/h17-al/best.pt ...`).
4. **Vision** (`materials_vision.py`): Gemma names materials only with a box on the sheet; flagged for review.

## Measured

On Stoa's material questions (`_analysis/materials_eval.py`, Gemma-graded text like `stoa_qa.eval_text`):

| Questions | Before | Now |
| --- | --- | --- |
| ifc_bench free text about materials (92) | 9 correct, 14 partial | **39 correct, 15 partial** |
| opening_design schedule / finish material questions (19) | 3 correct, 4 partial (text) | **19 / 19 correct** |

Hatch classifier on 943 hand-labelled real sheet patches (`_analysis/hatch_real.py`), split by source sheet (val picks, test
reported): **test: plain windows kept plain 97.9%, hatches found 29%, hatch class right 18%** — useful as boxed, flagged hints,
not as a measurement. Template evals unchanged (Swiss 595/600, IFC 364/364 on the same samples before and after).

## Reproduce

```bash
uv run pytest tests/test_materials.py
uv run python _analysis/materials_eval.py --only ifc_bench --out ifc        # needs FEATHERLESS_API_KEY for grading
uv run python _analysis/materials_eval.py --only opening_design --out od
uv run python _analysis/hatch_real.py score hatch-h17-al                    # hatch classifier on the real val / test patches
modal run modal_jobs/hatch_cloud.py --name h17-al --steps 9000 --real-mix   # retrain (real windows: _analysis/hatch_bg.py)
```
