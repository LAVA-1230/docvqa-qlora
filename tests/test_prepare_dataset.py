import io

import pytest
from PIL import Image

from src.data.prepare_dataset import assert_no_overlap, compute_stats, split_by_group, verify_image_bytes


def _records(n_docs=50, questions_per_doc=(1, 2, 3, 4)):
    recs, qid = [], 0
    for d in range(n_docs):
        for _ in range(questions_per_doc[d % len(questions_per_doc)]):
            recs.append({"id": qid, "ucsf_document_id": f"doc{d}", "doc_id": d, "question": "q", "answers": ["a b"],
                         "question_types": ["form"], "image": f"images/{d}.png", "width": 100 + d, "height": 200})
            qid += 1
    return recs


RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}


def test_split_has_no_document_overlap_and_keeps_all_rows():
    recs = _records()
    splits = split_by_group(recs, "ucsf_document_id", RATIOS, seed=42)
    assert_no_overlap(splits)
    assert sum(len(v) for v in splits.values()) == len(recs)


def test_split_ratios_are_approximately_respected():
    recs = _records(n_docs=400)
    splits = split_by_group(recs, "ucsf_document_id", RATIOS, seed=0)
    n = len(recs)
    assert abs(len(splits["train"]) / n - 0.8) < 0.03
    assert abs(len(splits["val"]) / n - 0.1) < 0.03
    assert abs(len(splits["test"]) / n - 0.1) < 0.03


def test_split_is_deterministic():
    recs = _records()
    a = split_by_group(recs, "ucsf_document_id", RATIOS, seed=7)
    b = split_by_group(list(reversed(recs)), "ucsf_document_id", RATIOS, seed=7)
    assert {s: sorted(r["id"] for r in v) for s, v in a.items()} == {s: sorted(r["id"] for r in v) for s, v in b.items()}


def test_overlap_assertion_catches_leakage():
    recs = _records(n_docs=3)
    leaky = {"train": recs[:2], "val": recs[1:3], "test": []}
    with pytest.raises(AssertionError, match="Leakage"):
        assert_no_overlap(leaky)


def test_bad_ratios_raise():
    with pytest.raises(ValueError):
        split_by_group(_records(), "ucsf_document_id", {"train": 0.8, "val": 0.3, "test": 0.1}, seed=0)


def test_verify_image_bytes_accepts_good_and_rejects_corrupt():
    buf = io.BytesIO()
    Image.new("RGB", (40, 30), "white").save(buf, format="PNG")
    good = buf.getvalue()
    assert verify_image_bytes(good) == (40, 30, "png")
    with pytest.raises(Exception):
        verify_image_bytes(good[: len(good) // 2])  # truncated
    with pytest.raises(Exception):
        verify_image_bytes(b"not an image")


def test_compute_stats_counts():
    splits = split_by_group(_records(), "ucsf_document_id", RATIOS, seed=1)
    stats = compute_stats(splits)
    assert sum(stats["counts"].values()) == len(_records())
    assert stats["images"]["unique"] == 50
    assert stats["question_types"] == {"form": len(_records())}
