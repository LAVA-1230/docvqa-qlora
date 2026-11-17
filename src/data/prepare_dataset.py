"""Phase 1 - build a ~3000-sample domain subset of DocVQA and split it BY DOCUMENT.

Steps
1. Stream HuggingFaceM4/DocumentVQA (train split only - test answers are hidden).
2. Keep QA pairs whose question_types intersect `keep_question_types` (table/list, form).
3. Verify every image (drop corrupt ones) and save each page image once (original bytes,
   no re-compression).
4. Split by `group_key` (ucsf_document_id) into train/val/test ~80/10/10 and ASSERT that
   no document appears in two splits.
5. Write {train,val,test}.jsonl + images/ + stats.json.

Usage
    python -m src.data.prepare_dataset --config configs/data.yaml
    python -m src.data.prepare_dataset --config configs/data.yaml --limit 50          # smoke test
    python -m src.data.prepare_dataset --config configs/data.yaml --list_types 2000   # only print types
"""
from __future__ import annotations

import argparse
import io
import logging
import random
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from src.utils import load_config, parse_set_args, save_json, set_seed, setup_logging, write_jsonl

logger = logging.getLogger("docvqa")
SPLITS = ("train", "val", "test")


# --------------------------------------------------------------------------- pure helpers (unit tested)
def split_by_group(records: list[dict], group_key: str, ratios: dict, seed: int) -> dict[str, list[dict]]:
    """Assign whole groups (documents) to splits so that the QUESTION counts approximate `ratios`.

    Groups are shuffled with a fixed seed, then filled greedily: train until it holds
    >= ratio_train of all questions, then val, the rest goes to test.
    """
    total_ratio = sum(ratios[s] for s in SPLITS)
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"split ratios must sum to 1, got {ratios}")
    groups: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        groups[str(rec[group_key])].append(rec)
    keys = sorted(groups)  # sort first so the shuffle is deterministic
    random.Random(seed).shuffle(keys)

    n = len(records)
    train_cut = ratios["train"] * n
    val_cut = (ratios["train"] + ratios["val"]) * n
    splits: dict[str, list[dict]] = {s: [] for s in SPLITS}
    assigned = 0
    for key in keys:
        if assigned < train_cut:
            target = "train"
        elif assigned < val_cut:
            target = "val"
        else:
            target = "test"
        splits[target].extend(groups[key])
        assigned += len(groups[key])
    return splits


def assert_no_overlap(splits: dict[str, list[dict]], keys: tuple[str, ...] = ("ucsf_document_id", "doc_id")) -> None:
    """Raise AssertionError if any document id appears in more than one split."""
    for key in keys:
        ids = {s: {str(r[key]) for r in rows} for s, rows in splits.items()}
        for i, a in enumerate(SPLITS):
            for b in SPLITS[i + 1:]:
                common = ids[a] & ids[b]
                assert not common, f"Leakage: {len(common)} '{key}' values in both {a} and {b}: {sorted(common)[:5]}"


def verify_image_bytes(data: bytes) -> tuple[int, int, str]:
    """Fully decode the image (verify() alone misses truncated files). Returns (w, h, format)."""
    with Image.open(io.BytesIO(data)) as im:
        im.verify()
    with Image.open(io.BytesIO(data)) as im:
        im.load()
        return im.width, im.height, (im.format or "PNG").lower()


def _describe(values: list[float]) -> dict:
    if not values:
        return {}
    return {
        "min": min(values),
        "median": statistics.median(values),
        "mean": round(statistics.mean(values), 2),
        "max": max(values),
    }


def compute_stats(splits: dict[str, list[dict]]) -> dict:
    stats: dict = {"counts": {}, "documents": {}, "images": {}}
    all_rows = [r for rows in splits.values() for r in rows]
    for s, rows in splits.items():
        stats["counts"][s] = len(rows)
        stats["documents"][s] = len({r["ucsf_document_id"] for r in rows})
    unique_imgs = {r["image"]: r for r in all_rows}.values()
    stats["images"]["unique"] = len(unique_imgs)
    stats["images"]["width"] = _describe([r["width"] for r in unique_imgs])
    stats["images"]["height"] = _describe([r["height"] for r in unique_imgs])
    stats["images"]["megapixels"] = _describe([round(r["width"] * r["height"] / 1e6, 3) for r in unique_imgs])
    stats["answer_chars"] = _describe([len(r["answers"][0]) for r in all_rows])
    stats["answer_words"] = _describe([len(r["answers"][0].split()) for r in all_rows])
    stats["question_types"] = dict(Counter(t for r in all_rows for t in r["question_types"]).most_common())
    return stats


# --------------------------------------------------------------------------- streaming
def _open_stream(cfg: dict, shuffle_buffer: int):
    from datasets import Image as HFImage
    from datasets import load_dataset

    logger.info("Opening %s split=%s (streaming)", cfg["dataset_name"], cfg["source_split"])
    ds = load_dataset(cfg["dataset_name"], split=cfg["source_split"], streaming=cfg.get("streaming", True))
    # decode=False -> we get the original encoded bytes; we verify + save them ourselves
    ds = ds.cast_column("image", HFImage(decode=False))
    if shuffle_buffer > 0:
        ds = ds.shuffle(seed=cfg["seed"], buffer_size=shuffle_buffer)
    return ds


def list_question_types(cfg: dict, n_rows: int) -> Counter:
    ds = _open_stream(cfg, shuffle_buffer=0)
    counter: Counter = Counter()
    for i, row in enumerate(ds):
        if i >= n_rows:
            break
        counter.update(row["question_types"])
    logger.info("Distinct question_types in the first %d rows:", n_rows)
    for t, c in counter.most_common():
        logger.info("  %-20s %d", t, c)
    return counter


def collect_samples(cfg: dict, out_dir: Path, target: int, shuffle_buffer: int) -> tuple[list[dict], dict]:
    keep = set(cfg["keep_question_types"])
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    seen_types: Counter = Counter()
    saved_images: dict[int, tuple[str, int, int]] = {}
    corrupt_docs: set[int] = set()
    scanned = dropped_corrupt = 0
    t0 = time.time()

    for row in _open_stream(cfg, shuffle_buffer):
        scanned += 1
        seen_types.update(row["question_types"])
        if scanned >= cfg["max_scan"]:
            logger.warning("Reached max_scan=%d before collecting %d samples", cfg["max_scan"], target)
            break
        if not keep & set(row["question_types"]):
            continue
        answers = [a.strip() for a in (row["answers"] or []) if a and a.strip()]
        if not answers or not row["question"].strip():
            continue

        doc_id = int(row["docId"])
        if doc_id in corrupt_docs:
            dropped_corrupt += 1
            continue
        if doc_id not in saved_images:
            data = (row["image"] or {}).get("bytes")
            try:
                if not data:
                    raise ValueError("no image bytes")
                w, h, fmt = verify_image_bytes(data)
            except Exception as e:  # corrupt / unreadable image -> drop the sample
                logger.warning("Dropping docId=%s: unreadable image (%s)", doc_id, e)
                corrupt_docs.add(doc_id)
                dropped_corrupt += 1
                continue
            ext = "jpg" if fmt in ("jpeg", "jpg") else fmt
            rel = f"images/{doc_id}.{ext}"
            (out_dir / rel).write_bytes(data)
            saved_images[doc_id] = (rel, w, h)

        rel, w, h = saved_images[doc_id]
        records.append(
            {
                "id": int(row["questionId"]),
                "question": row["question"].strip(),
                "answers": answers,
                "question_types": list(row["question_types"]),
                "doc_id": doc_id,
                "ucsf_document_id": str(row["ucsf_document_id"]),
                "page_no": str(row["ucsf_document_page_no"]),
                "image": rel,
                "width": w,
                "height": h,
            }
        )
        if len(records) % 250 == 0:
            logger.info("  kept %d / scanned %d rows (%.0fs)", len(records), scanned, time.time() - t0)
        if len(records) >= target:
            break

    info = {
        "scanned_rows": scanned,
        "kept": len(records),
        "dropped_corrupt": dropped_corrupt,
        "question_types_seen_while_scanning": dict(seen_types.most_common()),
        "seconds": round(time.time() - t0, 1),
    }
    return records, info


# --------------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--output_dir", default=None, help="override output_dir (e.g. a Google Drive path)")
    parser.add_argument("--limit", type=int, default=None, help="collect only N samples (smoke test)")
    parser.add_argument("--list_types", type=int, default=0, help="only print question_types of the first N rows")
    parser.add_argument("--set", nargs="*", default=[], help="config overrides key=value")
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config, parse_set_args(args.set))
    set_seed(cfg["seed"])

    if args.list_types:
        list_question_types(cfg, args.list_types)
        return

    out_dir = Path(args.output_dir or cfg["output_dir"])
    target = args.limit or cfg["target_samples"]
    shuffle_buffer = min(cfg["shuffle_buffer"], 100) if args.limit else cfg["shuffle_buffer"]
    logger.info("Collecting %d samples of types %s -> %s", target, cfg["keep_question_types"], out_dir.resolve())

    records, info = collect_samples(cfg, out_dir, target, shuffle_buffer)
    logger.info("Question types seen while scanning: %s", info["question_types_seen_while_scanning"])
    if not records:
        raise RuntimeError("No samples collected - check keep_question_types against the printed types.")
    if len(records) < target:
        logger.warning("Only %d samples collected (target %d)", len(records), target)

    splits = split_by_group(records, cfg["group_key"], cfg["split_ratios"], cfg["seed"])
    assert_no_overlap(splits)
    logger.info("Overlap check PASSED: no document appears in two splits")
    for s in SPLITS:
        if not splits[s]:
            logger.warning("Split '%s' is EMPTY (too few documents - only expected with a tiny --limit)", s)
    assert sum(len(v) for v in splits.values()) == len(records)

    for s in SPLITS:
        write_jsonl(out_dir / f"{s}.jsonl", splits[s])

    stats = compute_stats(splits)
    stats["collection"] = info
    stats["config"] = cfg
    save_json(out_dir / "stats.json", stats)

    logger.info("=" * 60)
    for s in SPLITS:
        logger.info("%-5s : %5d questions | %4d documents", s, stats["counts"][s], stats["documents"][s])
    logger.info("unique images: %d | dropped corrupt: %d", stats["images"]["unique"], info["dropped_corrupt"])
    logger.info("image width  %s", stats["images"]["width"])
    logger.info("image height %s", stats["images"]["height"])
    logger.info("Wrote %s/{train,val,test}.jsonl, images/, stats.json", out_dir)


if __name__ == "__main__":
    main()
