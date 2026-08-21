"""ifeval — capability_filter rule benchmark (instruction-following).

Programmatic verification of *verifiable instructions* (Zhou et al. 2023,
`google/instruction_following_eval`). Each item is a prompt plus a list of
instruction ids + kwargs; an output passes an item iff it satisfies ALL of the
item's instructions (prompt-level STRICT accuracy). This is the capability slot
that specifically catches a safety method that makes the model derail / stop
following formatting/length/content instructions.

Scope note (honest): IFEval ships ~25 instruction types, several needing heavy
deps (langdetect for language:*, nltk tokenization). We implement the
deterministic, dependency-light families below and the *publisher* includes only
items whose every instruction is in SUPPORTED — a documented subset, correct for
the items it keeps. Unsupported families (language:*, some paragraph/segment
constraints) are TODO; add a checker here + drop it from the publisher filter.

Secret data: secret_dir/ifeval.jsonl, one
{"prompt": str, "instruction_id_list": [str], "kwargs": [dict]} per line.
"""
from __future__ import annotations

import json
import re
import string
from pathlib import Path
from typing import Any, Callable

from aar.benchmarks.base import Model, RuleBenchmark, score_from_judgments


# --- relation helper -------------------------------------------------------
def _cmp(count: int, threshold: int, relation: str) -> bool:
    if relation == "less than":
        return count < threshold
    # default / "at least"
    return count >= threshold


def _words(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text)


def _sentences(text: str) -> list[str]:
    parts = re.split(r"[.!?]+(?:\s|$)", text.strip())
    return [p for p in parts if p.strip()]


def _count_keyword(text: str, kw: str) -> int:
    return len(re.findall(r"\b" + re.escape(kw) + r"\b", text, flags=re.IGNORECASE))


# --- per-instruction checkers: (response, kwargs) -> bool ------------------
def _k_existence(r: str, kw: dict) -> bool:
    return all(_count_keyword(r, k) >= 1 for k in kw.get("keywords", []))

def _k_frequency(r: str, kw: dict) -> bool:
    return _cmp(_count_keyword(r, kw["keyword"]), int(kw["frequency"]), kw.get("relation", "at least"))

def _k_forbidden(r: str, kw: dict) -> bool:
    return all(_count_keyword(r, k) == 0 for k in kw.get("forbidden_words", []))

def _k_letter_frequency(r: str, kw: dict) -> bool:
    letter = kw["letter"].lower()
    count = sum(1 for c in r.lower() if c == letter)
    return _cmp(count, int(kw["let_frequency"]), kw.get("let_relation", "at least"))

def _case_upper(r: str, kw: dict) -> bool:
    return any(c.isalpha() for c in r) and not any(c.islower() for c in r)

def _case_lower(r: str, kw: dict) -> bool:
    return any(c.isalpha() for c in r) and not any(c.isupper() for c in r)

def _case_capital_word_frequency(r: str, kw: dict) -> bool:
    caps = sum(1 for w in _words(r) if w.isupper() and any(c.isalpha() for c in w))
    return _cmp(caps, int(kw["capital_frequency"]), kw.get("capital_relation", "at least"))

def _no_comma(r: str, kw: dict) -> bool:
    return "," not in r

def _end_checker(r: str, kw: dict) -> bool:
    return r.strip().lower().endswith(str(kw["end_phrase"]).strip().lower())

def _quotation(r: str, kw: dict) -> bool:
    s = r.strip()
    return len(s) >= 2 and s.startswith('"') and s.endswith('"')

def _title(r: str, kw: dict) -> bool:
    return bool(re.search(r"<<[^\n<>]+>>", r))

def _bullet_lists(r: str, kw: dict) -> bool:
    n = len(re.findall(r"(?m)^\s*[\*\-]\s+\S", r))
    return n == int(kw["num_bullets"])

def _highlighted_sections(r: str, kw: dict) -> bool:
    n = len(re.findall(r"\*+[^\*\n]+\*+", r))
    return n >= int(kw["num_highlights"])

def _multiple_sections(r: str, kw: dict) -> bool:
    spliter = str(kw["section_spliter"])
    n = len(re.findall(re.escape(spliter), r, flags=re.IGNORECASE))
    return n >= int(kw["num_sections"])

def _constrained_response(r: str, kw: dict) -> bool:
    opts = {"my answer is yes.", "my answer is no.", "my answer is maybe."}
    return r.strip().lower() in opts

def _json_format(r: str, kw: dict) -> bool:
    s = r.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.IGNORECASE).strip()
    try:
        json.loads(s)
        return True
    except Exception:
        return False

def _postscript(r: str, kw: dict) -> bool:
    marker = str(kw.get("postscript_marker", "P.S.")).strip()
    return marker.lower() in r.lower() or re.search(r"(?im)^\s*p\.?\s*s\.?", r) is not None

def _placeholders(r: str, kw: dict) -> bool:
    n = len(re.findall(r"\[[^\]\n]*\]", r))
    return n >= int(kw["num_placeholders"])

def _number_words(r: str, kw: dict) -> bool:
    return _cmp(len(_words(r)), int(kw["num_words"]), kw.get("relation", "at least"))

def _number_sentences(r: str, kw: dict) -> bool:
    return _cmp(len(_sentences(r)), int(kw["num_sentences"]), kw.get("relation", "at least"))


CHECKERS: dict[str, Callable[[str, dict], bool]] = {
    "keywords:existence": _k_existence,
    "keywords:frequency": _k_frequency,
    "keywords:forbidden_words": _k_forbidden,
    "keywords:letter_frequency": _k_letter_frequency,
    "change_case:english_capital": _case_upper,
    "change_case:english_lowercase": _case_lower,
    "change_case:capital_word_frequency": _case_capital_word_frequency,
    "punctuation:no_comma": _no_comma,
    "startend:end_checker": _end_checker,
    "startend:quotation": _quotation,
    "detectable_format:title": _title,
    "detectable_format:number_bullet_lists": _bullet_lists,
    "detectable_format:number_highlighted_sections": _highlighted_sections,
    "detectable_format:multiple_sections": _multiple_sections,
    "detectable_format:constrained_response": _constrained_response,
    "detectable_format:json_format": _json_format,
    "detectable_content:postscript": _postscript,
    "detectable_content:number_placeholders": _placeholders,
    "length_constraints:number_words": _number_words,
    "length_constraints:number_sentences": _number_sentences,
}
SUPPORTED = frozenset(CHECKERS)


def item_supported(instruction_ids: list[str]) -> bool:
    """An item is includable iff EVERY instruction has a checker (so a kept item
    is scored fully-correctly, never partially)."""
    return all(i in SUPPORTED for i in instruction_ids)


def follows_all(response: str, instruction_ids: list[str], kwargs_list: list[dict]) -> bool:
    for iid, kw in zip(instruction_ids, kwargs_list):
        checker = CHECKERS.get(iid)
        if checker is None:
            return False  # defensive: publisher should have excluded this item
        try:
            if not checker(response, {k: v for k, v in (kw or {}).items() if v is not None}):
                return False
        except Exception:
            return False
    return True


class IFEval(RuleBenchmark):
    name = "ifeval"

    def load_items(self) -> list[dict[str, Any]]:
        path = Path(self.secret_dir) / "ifeval.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    def matches(self, output: str, gold: Any) -> bool:  # unused (we override score)
        raise NotImplementedError

    def score(self, model: Model):
        items = self.load_items()
        if self.spec.subset_size:
            items = items[: self.spec.subset_size]
        outputs = model.generate_batch([it["prompt"] for it in items])
        judgments = [
            1.0 if follows_all(o, it["instruction_id_list"], it.get("kwargs", [{}] * len(it["instruction_id_list"]))) else 0.0
            for it, o in zip(items, outputs)
        ]
        return score_from_judgments(judgments)
