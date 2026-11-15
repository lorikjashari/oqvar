"""Survey: which opening_design PDF pages are floor plans, and what material wording / hatch fills they carry."""
import collections, json, re, sys
from pathlib import Path
import pymupdf

ROOT = Path(sys.argv[1])
MAT = re.compile(r"\b(CONC(?:RETE)?|CMU|BRICK|MASONRY|GYP(?:SUM)?\.?\s*B(?:OAR)?D|GWB|PLYWOOD|OSB|WOOD|WD|STEEL|STL|ALUM(?:INUM)?|GLASS|GL|"
                 r"TILE|CT|VCT|LVT|CPT|CARPET|VINYL|EPOXY|TERRAZZO|HARDWOOD|LAMINATE|INSUL(?:ATION|ATED)?|STONE|STUCCO|SIDING|"
                 r"FINISH|F\d|W\d|C\d|B\d|MATERIAL)\b", re.I)
out = []
pdfs = sorted(ROOT.rglob("*.pdf"))
for p in pdfs:
    try:
        doc = pymupdf.open(p)
    except Exception:
        continue
    for i, pg in enumerate(doc):
        if i > 0:
            break
        t = pg.get_text()
        up = t.upper()
        plan = bool(re.search(r"FLOOR PLAN|LEVEL \d+ PLAN|BASEMENT PLAN|FIRST FLOOR|SECOND FLOOR|MAIN LEVEL|PLAN\b", up))
        mats = collections.Counter(m.group(1).upper() for m in MAT.finditer(t))
        try:
            dr = pg.get_drawings()
        except Exception:
            dr = []
        fills = sum(1 for d in dr if d.get("fill") is not None)
        out.append({"path": str(p.relative_to(ROOT)), "words": len(t.split()), "plan": plan, "fills": fills, "paths": len(dr),
                    "mat": dict(mats.most_common(12))})
json.dump(out, open(sys.argv[2], "w"), indent=0)
print(len(out))
