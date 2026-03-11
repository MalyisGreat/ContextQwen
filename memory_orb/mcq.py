from __future__ import annotations

import json
import re
from typing import Mapping

_CHOICE_RE = re.compile(r"\b([ABCD])\b", flags=re.IGNORECASE)
_MC_OPTION_RE = re.compile(r"(?im)^\s*([ABCD])\.\s+(.+?)\s*$")
_SCORECARD_RE = re.compile(
    r"^\s*([ABCD])\s*:\s*(?:score\s*=\s*)?(-?\d+(?:\.\d+)?)\s+"
    r"(supported|weak|contradicted|uncertain|mixed)\b",
    flags=re.IGNORECASE,
)
_BEST_RE = re.compile(r"^\s*(?:best|answer)\s*:\s*([ABCD]|uncertain)\b", flags=re.IGNORECASE)


def extract_multiple_choice_options(question: str) -> dict[str, str]:
    options: dict[str, str] = {}
    for match in _MC_OPTION_RE.finditer(question or ""):
        label = match.group(1).upper()
        text = " ".join(match.group(2).split()).strip()
        if label in {"A", "B", "C", "D"} and text:
            options[label] = text
    return options


def strip_multiple_choice_options(question: str) -> str:
    raw = (question or "").strip()
    if not raw:
        return ""
    if not extract_multiple_choice_options(raw):
        return raw
    kept_lines: list[str] = []
    option_block_started = False
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            if not option_block_started:
                kept_lines.append("")
            continue
        if stripped.lower() == "options:":
            option_block_started = True
            continue
        if _MC_OPTION_RE.match(stripped):
            option_block_started = True
            continue
        kept_lines.append(line.rstrip())
    collapsed = "\n".join(line for line in kept_lines).strip()
    return collapsed or raw


def build_multiple_choice_question_block(question: str, options: Mapping[str, str] | None = None) -> str:
    option_map = dict(options or extract_multiple_choice_options(question))
    stem = strip_multiple_choice_options(question)
    question_block = stem if stem.lower().startswith("question:") else f"Question:\n{stem}"
    if not option_map:
        return question_block
    ordered_lines = [f"{label}. {option_map[label]}" for label in ("A", "B", "C", "D") if label in option_map]
    return question_block + "\n\nOptions:\n" + "\n".join(ordered_lines)


def build_scorecard_instruction_block() -> str:
    return (
        "Score every option separately using only the provided evidence.\n"
        "Do not use outside knowledge.\n"
        "Do not default to option A when evidence is weak.\n"
        "Return exactly four lines and no extra text:\n"
        "A: <0.00-1.00> <supported|weak|contradicted>\n"
        "B: <0.00-1.00> <supported|weak|contradicted>\n"
        "C: <0.00-1.00> <supported|weak|contradicted>\n"
        "D: <0.00-1.00> <supported|weak|contradicted>"
    )


def parse_scorecard(raw: str) -> tuple[dict[str, float], str]:
    text = (raw or "").strip()
    if not text:
        return {}, ""
    scores: dict[str, float] = {}
    explicit_choice = ""
    for raw_line in text.splitlines():
        line = " ".join(raw_line.strip().split())
        if not line:
            continue
        best_match = _BEST_RE.match(line)
        if best_match:
            candidate = best_match.group(1).upper()
            if candidate in {"A", "B", "C", "D"}:
                explicit_choice = candidate
            continue
        score_match = _SCORECARD_RE.match(line)
        if score_match:
            label = score_match.group(1).upper()
            score = max(0.0, min(1.0, float(score_match.group(2))))
            verdict = score_match.group(3).lower()
            if verdict == "contradicted":
                score = min(score, 0.35)
            elif verdict in {"uncertain", "mixed"}:
                score = min(score, 0.55)
            scores[label] = score
            continue

        label_match = re.match(r"^\s*([ABCD])\s*:\s*(.+)$", line, flags=re.IGNORECASE)
        if not label_match:
            continue
        label = label_match.group(1).upper()
        payload = label_match.group(2).lower()
        score = _infer_score_from_text(payload)
        if score >= 0.0:
            scores[label] = score
    return scores, explicit_choice


def choose_best_option(scores: Mapping[str, float], explicit_choice: str = "") -> str:
    if not scores:
        return explicit_choice if explicit_choice in {"A", "B", "C", "D"} else ""
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    if len(ranked) == 1:
        return ranked[0][0]
    best_label, best_score = ranked[0]
    second_score = ranked[1][1]
    if best_score == second_score and explicit_choice:
        return explicit_choice
    return best_label


def extract_mc_choice(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        return ""

    if text.startswith("{") and text.endswith("}"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                for key in ("answer", "choice", "option", "value", "result"):
                    value = parsed.get(key)
                    if value is not None:
                        match = _CHOICE_RE.search(str(value))
                        if match:
                            return match.group(1).upper()
        except Exception:
            pass

    scores, explicit_choice = parse_scorecard(text)
    score_choice = choose_best_option(scores, explicit_choice=explicit_choice)
    if score_choice:
        return score_choice

    explicit = re.search(r"(?i)\b(?:answer|choice|option|best)\s*[:=]\s*['\"]?\s*([ABCD])\b", text)
    if explicit:
        return explicit.group(1).upper()

    stripped = text.upper()
    if stripped in {"A", "B", "C", "D"}:
        return stripped

    if len(text) <= 48:
        matches = _CHOICE_RE.findall(text)
        if matches:
            return matches[-1].upper()
    return ""


def _infer_score_from_text(text: str) -> float:
    lowered = text.lower()
    numeric = re.search(r"(-?\d+(?:\.\d+)?)", lowered)
    if numeric:
        return max(0.0, min(1.0, float(numeric.group(1))))
    if any(term in lowered for term in ("contradicted", "ruled out", "unsupported", "false")):
        return 0.0
    if any(term in lowered for term in ("weak", "uncertain", "mixed")):
        return 0.4
    if any(term in lowered for term in ("supported", "best", "strong")):
        return 0.9
    return -1.0
