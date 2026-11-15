"""Material questions of Stoa's dataset: run the materials specialist (or the whole pipeline) and grade the answers.

Rows = every dataset question whose wording stoa_qa.materials.intent_of recognises (ifc_bench text, opening_design schedule
look-ups and counts). Text answers are graded by the same Gemma grader as stoa_qa.eval_text; counts / areas exactly
(2%). The earlier full-pipeline grades (eval_out/text_eval*/predictions.jsonl) are shown as the baseline.

    python _analysis/materials_eval.py [--pipeline] [--limit N] [--no-grade] [--only ifc_bench|opening_design] [--out NAME]
Writes eval_out/materials_eval/<out>.jsonl and prints a summary.
"""
from __future__ import annotations

import argparse
import collections
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from stoa_qa import catalog, materials, pipeline
from stoa_qa.core import DATASET, source_of
from stoa_qa.eval_text import grade

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "eval_out" / "materials_eval"


def baseline() -> dict:
    best = {}
    for f in ("text_eval", "text_eval_redo", "text_eval_redo2"):
        p = REPO / "eval_out" / f / "predictions.jsonl"
        if p.exists():
            for line in open(p, encoding="utf-8"):
                r = json.loads(line)
                best[(r["path"], r["question"])] = r.get("grade")
    return best


def rows(only: str | None) -> list[dict]:
    out = []
    for line in open(DATASET / "questions.jsonl", encoding="utf-8"):
        r = json.loads(line)
        src = source_of(r["path"])
        if src not in ("ifc_bench", "opening_design", "ifc_models") or (only and src != only):
            continue
        if materials.intent_of(r["question"]) is not None:
            out.append(r)
    return out


def drawing_for(path: str):
    cat = catalog.catalog()
    for d in cat.values():
        if path in d.files or d.ask_path == path or d.id == path:
            return d
    return None


def run(r: dict, use_pipeline: bool, do_grade: bool) -> dict:
    d = drawing_for(r["path"])
    t0 = time.time()
    tried = []
    if d is None:
        text, value, engine = "(no drawing)", None, "none"
    elif use_pipeline:
        res = pipeline.solve(d, r["question"])
        value, text, engine = res.answer.value, pipeline.answer_text(res.answer), res.engine
    else:
        try:
            a, tried = materials.answer(d, r["question"], use_llm=False)
        except Exception as e:  # one broken model must not stop the run
            a, tried = None, [{"source": "error", "note": f"{type(e).__name__}: {e}"[:200]}]
        value = a.value if a else None
        text = (pipeline.answer_text(a) if a and a.value is not None else f"(declined) {a.method if a else tried}")
        engine = "materials"
    out = {"path": r["path"], "question": r["question"], "type": r["type"], "reference": r["answer"], "answer": text,
           "value": value, "engine": engine, "tried": tried, "seconds": round(time.time() - t0, 1)}
    if r["type"] in ("count", "area", "length", "volume"):
        ok = pipeline.is_correct(r["answer"], value, r["type"])
        out["grade"] = "correct" if ok else "incorrect"
    elif do_grade:
        g, _ = grade(r["question"], str(r["answer"]), text)
        out["grade"], out["grade_reason"] = g["verdict"], g["reason"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-grade", action="store_true")
    ap.add_argument("--only")
    ap.add_argument("--out", default="run")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    rs = sorted(rows(a.only), key=lambda r: r["path"])  # one model after another: each is parsed once
    if a.limit:
        rs = rs[: a.limit]
    base = baseline()
    OUT.mkdir(parents=True, exist_ok=True)
    res = []
    with ThreadPoolExecutor(a.workers) as ex:
        for o in ex.map(lambda r: run(r, a.pipeline, not a.no_grade), rs):
            o["baseline"] = base.get((o["path"], o["question"]))
            res.append(o)
            print(json.dumps({k: o.get(k) for k in ("question", "grade", "baseline", "engine")}, ensure_ascii=False)[:300], flush=True)
    with open(OUT / f"{a.out}.jsonl", "w", encoding="utf-8") as f:
        for o in res:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    print("now:", dict(collections.Counter(o.get("grade") for o in res)))
    print("baseline:", dict(collections.Counter(o.get("baseline") for o in res)))
    print("by source now:", dict(collections.Counter((source_of(o["path"]), o.get("grade")) for o in res)))


if __name__ == "__main__":
    main()
