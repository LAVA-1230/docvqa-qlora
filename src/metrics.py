"""ANLS (Average Normalized Levenshtein Similarity) - the official DocVQA metric.

For one question:   NL = levenshtein(pred, gold) / max(len(pred), len(gold))
                    score(gold) = 1 - NL  if NL < tau (0.5)  else 0
                    ANLS = max over all gold answers
Corpus ANLS = mean over questions.

Why ANLS instead of exact match? OCR-style answers often differ by a character
("$1,000" vs "$1000"); ANLS gives partial credit but cuts it off at tau so a
completely wrong answer gets 0 instead of a misleading partial score.
"""
from __future__ import annotations

import re
from typing import Sequence

from rapidfuzz.distance import Levenshtein

_WHITESPACE = re.compile(r"\s+")


def normalize(text: str | None) -> str:
    """Lowercase, collapse runs of whitespace, strip."""
    if text is None:
        return ""
    return _WHITESPACE.sub(" ", str(text).lower()).strip()


def _similarity(pred: str, gold: str, tau: float) -> float:
    if not pred and not gold:
        return 1.0
    nl = Levenshtein.distance(pred, gold) / max(len(pred), len(gold))
    return 1.0 - nl if nl < tau else 0.0


def anls(pred: str | None, gold_list: Sequence[str] | str, tau: float = 0.5) -> float:
    """ANLS of one prediction against all acceptable gold answers (max over golds)."""
    if isinstance(gold_list, str):
        gold_list = [gold_list]
    if not gold_list:
        raise ValueError("gold_list must contain at least one answer")
    p = normalize(pred)
    return max(_similarity(p, normalize(g), tau) for g in gold_list)


def exact_match(pred: str | None, gold_list: Sequence[str] | str) -> float:
    if isinstance(gold_list, str):
        gold_list = [gold_list]
    p = normalize(pred)
    return float(any(p == normalize(g) for g in gold_list))


def corpus_anls(preds: Sequence[str], golds: Sequence[Sequence[str]], tau: float = 0.5) -> float:
    if len(preds) != len(golds):
        raise ValueError(f"preds ({len(preds)}) and golds ({len(golds)}) differ in length")
    if not preds:
        raise ValueError("cannot compute ANLS of an empty corpus")
    return sum(anls(p, g, tau) for p, g in zip(preds, golds)) / len(preds)


def corpus_exact_match(preds: Sequence[str], golds: Sequence[Sequence[str]]) -> float:
    if len(preds) != len(golds):
        raise ValueError(f"preds ({len(preds)}) and golds ({len(golds)}) differ in length")
    if not preds:
        raise ValueError("cannot compute exact match of an empty corpus")
    return sum(exact_match(p, g) for p, g in zip(preds, golds)) / len(preds)
