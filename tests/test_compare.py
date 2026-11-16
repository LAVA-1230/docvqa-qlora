from src.compare import answer_length_bucket, compare, format_table, primary_type


def _res(preds):
    return {"predictions": [
        {"id": i, "question": f"q{i}", "answers": [gold], "pred": pred, "anls": score, "exact_match": float(score == 1),
         "question_types": types, "latency_ms": 100.0}
        for i, (gold, pred, score, types) in enumerate(preds)]}


def test_compare_counts_better_worse_same():
    base = _res([("a", "x", 0.0, ["form"]), ("b c", "b c", 1.0, ["table/list"]), ("d", "d", 1.0, ["form"])])
    ft = _res([("a", "a", 1.0, ["form"]), ("b c", "zz", 0.0, ["table/list"]), ("d", "d", 1.0, ["form"])])
    table, rows, summary = compare(base, ft)
    assert (table["better"], table["worse"], table["same"]) == (1, 1, 1)
    assert table["baseline"]["anls"] == round(2 / 3, 4) and table["finetuned"]["anls"] == round(2 / 3, 4)
    assert {r["status"] for r in rows} == {"better", "worse", "same"}
    assert any(s["question_type"] == "form" and s["answer_length"] == "ALL" for s in summary)
    assert "ANLS" in format_table(table)


def test_helpers():
    assert primary_type(["handwritten", "form"]) == "form"
    assert primary_type(["layout"]) == "layout"
    assert answer_length_bucket("one") == "1 word"
    assert answer_length_bucket("two words") == "2-3 words"
    assert answer_length_bucket("this has four words") == "4+ words"
