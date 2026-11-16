"""Phase 5 - baseline vs fine-tuned comparison + error analysis.

Reads two evaluate.py outputs for the SAME split and writes
    results/comparison.json      overall table
    results/error_analysis.csv   one row per sample: better / worse / same, with group columns
    results/error_summary.csv    mean ANLS per (question type, answer length) group

Usage
    python -m src.compare --baseline results/baseline.json --finetuned results/finetuned.json --results_dir results
"""
from __future__ import annotations

import argparse
import csv
import logging
from collections import defaultdict
from pathlib import Path

from src.utils import load_json, save_json, setup_logging

logger = logging.getLogger("docvqa")
TYPE_PRIORITY = ("table/list", "form")


def primary_type(types: list[str]) -> str:
    for t in TYPE_PRIORITY:
        if t in types:
            return t
    return types[0] if types else "unknown"


def answer_length_bucket(answer: str) -> str:
    n = len(answer.split())
    if n <= 1:
        return "1 word"
    if n <= 3:
        return "2-3 words"
    return "4+ words"


def compare(baseline: dict, finetuned: dict, eps: float = 1e-6) -> tuple[dict, list[dict], list[dict]]:
    base_by_id = {p["id"]: p for p in baseline["predictions"]}
    ft_by_id = {p["id"]: p for p in finetuned["predictions"]}
    common = [i for i in base_by_id if i in ft_by_id]
    if not common:
        raise ValueError("baseline and finetuned results share no sample ids (different splits?)")
    if len(common) != len(base_by_id) or len(common) != len(ft_by_id):
        logger.warning("Only %d common ids (baseline %d, finetuned %d) - comparing the intersection",
                       len(common), len(base_by_id), len(ft_by_id))

    rows = []
    for i in common:
        b, f = base_by_id[i], ft_by_id[i]
        delta = f["anls"] - b["anls"]
        rows.append({
            "id": i,
            "question_type": primary_type(b.get("question_types", [])),
            "answer_length": answer_length_bucket(b["answers"][0]),
            "status": "better" if delta > eps else "worse" if delta < -eps else "same",
            "delta_anls": round(delta, 4),
            "baseline_anls": round(b["anls"], 4),
            "finetuned_anls": round(f["anls"], 4),
            "question": b["question"],
            "gold": " | ".join(b["answers"]),
            "baseline_pred": b["pred"],
            "finetuned_pred": f["pred"],
        })
    rows.sort(key=lambda r: (r["question_type"], r["answer_length"], r["delta_anls"]))

    def mean(xs):
        return round(sum(xs) / len(xs), 4) if xs else None

    groups = defaultdict(list)
    for r in rows:
        groups[(r["question_type"], r["answer_length"])].append(r)
        groups[(r["question_type"], "ALL")].append(r)
        groups[("ALL", r["answer_length"])].append(r)
    summary_rows = []
    for (qt, al), rs in sorted(groups.items()):
        summary_rows.append({
            "question_type": qt, "answer_length": al, "n": len(rs),
            "baseline_anls": mean([r["baseline_anls"] for r in rs]),
            "finetuned_anls": mean([r["finetuned_anls"] for r in rs]),
            "better": sum(r["status"] == "better" for r in rs),
            "worse": sum(r["status"] == "worse" for r in rs),
        })

    ids = set(common)

    def subset_mean(res, key):
        return mean([p[key] for p in res["predictions"] if p["id"] in ids])

    table = {
        "n": len(common),
        "baseline": {"anls": subset_mean(baseline, "anls"), "exact_match": subset_mean(baseline, "exact_match"),
                     "avg_latency_ms": subset_mean(baseline, "latency_ms")},
        "finetuned": {"anls": subset_mean(finetuned, "anls"), "exact_match": subset_mean(finetuned, "exact_match"),
                      "avg_latency_ms": subset_mean(finetuned, "latency_ms")},
        "better": sum(r["status"] == "better" for r in rows),
        "worse": sum(r["status"] == "worse" for r in rows),
        "same": sum(r["status"] == "same" for r in rows),
        "baseline_meta": baseline.get("meta", {}),
        "finetuned_meta": finetuned.get("meta", {}),
    }
    return table, rows, summary_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def format_table(table: dict) -> str:
    b, f = table["baseline"], table["finetuned"]
    lines = [
        f"{'metric':<18}{'baseline':>12}{'fine-tuned':>12}{'delta':>10}",
        "-" * 52,
        f"{'ANLS':<18}{b['anls']:>12.4f}{f['anls']:>12.4f}{f['anls'] - b['anls']:>+10.4f}",
        f"{'Exact match':<18}{b['exact_match']:>12.4f}{f['exact_match']:>12.4f}{f['exact_match'] - b['exact_match']:>+10.4f}",
        f"{'Latency (ms)':<18}{b['avg_latency_ms']:>12.1f}{f['avg_latency_ms']:>12.1f}{f['avg_latency_ms'] - b['avg_latency_ms']:>+10.1f}",
        "-" * 52,
        f"n={table['n']} | better {table['better']} | worse {table['worse']} | same {table['same']}",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", default="results/baseline.json")
    parser.add_argument("--finetuned", default="results/finetuned.json")
    parser.add_argument("--results_dir", default="results")
    args = parser.parse_args()
    setup_logging()

    table, rows, summary_rows = compare(load_json(args.baseline), load_json(args.finetuned))
    out = Path(args.results_dir)
    save_json(out / "comparison.json", table)
    write_csv(out / "error_analysis.csv", rows)
    write_csv(out / "error_summary.csv", summary_rows)
    print("\n" + format_table(table) + "\n")
    print(f"{'group':<32}{'n':>5}{'base':>9}{'ft':>9}{'better':>8}{'worse':>7}")
    for s in summary_rows:
        print(f"{s['question_type'] + ' / ' + s['answer_length']:<32}{s['n']:>5}{s['baseline_anls']:>9.3f}"
              f"{s['finetuned_anls']:>9.3f}{s['better']:>8}{s['worse']:>7}")
    logger.info("Wrote comparison.json, error_analysis.csv, error_summary.csv to %s", out)


if __name__ == "__main__":
    main()
