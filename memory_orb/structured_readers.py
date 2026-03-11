from __future__ import annotations

import csv
import io
import json
import math
import re
import time
from dataclasses import dataclass
from typing import Any, Literal, Protocol, Sequence

from .adapters import ModelAdapter, SimpleTokenEstimator, TokenEstimator


_CHOICE_RE = re.compile(r"\b([ABCD])\b", flags=re.IGNORECASE)
_WORD_RE = re.compile(r"[a-z0-9]+")
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
_MANUAL_HINT_RE = re.compile(
    r"(?i)\b(?:manual|table of contents|preface|warning|danger|note|index|chapter|section|appendix|parameter|register|overview)\b"
)
_PAGE_HINT_RE = re.compile(r"^\s*(?:page\s+)?(\d{1,4})\s*$", flags=re.IGNORECASE)
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "based",
    "best",
    "by",
    "false",
    "following",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "not",
    "of",
    "on",
    "or",
    "provided",
    "the",
    "these",
    "this",
    "to",
    "true",
    "which",
    "with",
}
_COUNTRY_ALIASES = {
    "china": ("china", "cn"),
    "germany": ("germany", "de"),
    "italy": ("italy", "it"),
    "japan": ("japan", "jp"),
    "united kingdom": ("united kingdom", "uk", "gb"),
    "united states": ("united states", "us", "usa"),
}


@dataclass(slots=True)
class QuestionPacket:
    question: str
    options: dict[str, str] | None
    raw_context: str


@dataclass(slots=True)
class StructureProfile:
    shape: Literal["prose", "table", "manual"]
    confidence: float
    delimiter: str | None
    header_rows: int
    section_markers: tuple[str, ...]
    signals: dict[str, float]


@dataclass(slots=True)
class ReaderOutcome:
    answer_raw: str
    answer_pred: str
    evidence_lines: list[str]
    route_name: str
    deterministic: bool
    confidence: float
    latency_s: float


@dataclass(slots=True)
class SectionRecord:
    title: str
    page_hint: int | None
    body: str
    tokens: set[str]


@dataclass(slots=True)
class ParameterRecord:
    name: str
    values: tuple[str, ...]
    default: str | None
    effects: tuple[str, ...]
    constraints: tuple[str, ...]
    section_title: str


@dataclass(slots=True)
class ClaimEvidence:
    claim_text: str
    support_lines: list[str]
    contradiction_lines: list[str]
    confidence: float


class StructuredReader(Protocol):
    def match(self, packet: QuestionPacket, profile: StructureProfile) -> bool:
        ...

    def answer(self, packet: QuestionPacket, profile: StructureProfile, model: ModelAdapter) -> ReaderOutcome:
        ...


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
    explicit = re.search(r"(?i)\b(?:answer|choice|option)\s*[:=]\s*['\"]?\s*([ABCD])\b", text)
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


def profile_structure(packet: QuestionPacket) -> StructureProfile:
    lines = [line.rstrip() for line in packet.raw_context.splitlines() if line.strip()]
    delimiter, table_score = _detect_table_shape(lines)
    manual_markers = _collect_manual_markers(packet.raw_context)
    manual_score = min(
        1.0,
        0.12 * len(manual_markers)
        + 0.4 * _estimate_heading_density(lines)
        + 0.25 * _estimate_note_density(lines),
    )
    signals = {
        "table_score": round(table_score, 3),
        "manual_score": round(manual_score, 3),
        "line_count": float(len(lines)),
    }
    if table_score >= 0.80 and table_score >= manual_score:
        return StructureProfile("table", round(table_score, 3), delimiter, 1, tuple(manual_markers[:6]), signals)
    if manual_score >= 0.75:
        return StructureProfile("manual", round(manual_score, 3), None, 0, tuple(manual_markers[:6]), signals)
    return StructureProfile("prose", round(max(table_score, manual_score), 3), None, 0, tuple(manual_markers[:6]), signals)


def route_structured_reader(
    packet: QuestionPacket,
    model: ModelAdapter,
    *,
    token_estimator: TokenEstimator | None = None,
    structured_reader_ctx: int = 900,
    table_reader_max_rows: int = 6000,
    procedure_reader_max_sections: int = 6,
    procedure_reader_max_claims_per_option: int = 6,
) -> tuple[StructureProfile, ReaderOutcome | None]:
    estimator = token_estimator or SimpleTokenEstimator()
    working_packet = packet
    if estimator.count(packet.raw_context) > structured_reader_ctx:
        working_packet = QuestionPacket(
            question=packet.question,
            options=packet.options,
            raw_context=_truncate_context_window(packet.raw_context, estimator, structured_reader_ctx),
        )
    profile = profile_structure(working_packet)
    readers: list[StructuredReader] = [
        TableReader(estimator, table_reader_max_rows),
        ProcedureReader(estimator, procedure_reader_max_sections, procedure_reader_max_claims_per_option),
    ]
    for reader in readers:
        if reader.match(working_packet, profile):
            return profile, reader.answer(working_packet, profile, model)
    return profile, None


def _truncate_context_window(text: str, token_estimator: TokenEstimator, max_tokens: int) -> str:
    clean = (text or "").strip()
    if not clean:
        return ""
    if token_estimator.count(clean) <= max_tokens:
        return clean

    lines = [line.rstrip() for line in clean.splitlines()]
    if len(lines) <= 1:
        return _truncate_text_prefix(clean, token_estimator, max_tokens)

    ellipsis_line = "..."
    ellipsis_tokens = token_estimator.count(ellipsis_line)
    usable_tokens = max(1, max_tokens - ellipsis_tokens)
    head_budget = max(1, usable_tokens // 2)
    tail_budget = max(1, usable_tokens - head_budget)

    head: list[str] = []
    head_tokens = 0
    for line in lines:
        line_tokens = token_estimator.count(line)
        if head and head_tokens + line_tokens > head_budget:
            break
        head.append(line)
        head_tokens += line_tokens

    tail: list[str] = []
    tail_tokens = 0
    for line in reversed(lines[len(head) :]):
        line_tokens = token_estimator.count(line)
        if tail and tail_tokens + line_tokens > tail_budget:
            break
        tail.append(line)
        tail_tokens += line_tokens
    tail.reverse()

    merged = head[:]
    if tail:
        merged.append(ellipsis_line)
        merged.extend(tail)
    truncated = "\n".join(line for line in merged if line)
    if token_estimator.count(truncated) <= max_tokens:
        return truncated
    return _truncate_text_prefix(truncated, token_estimator, max_tokens)


def _truncate_text_prefix(text: str, token_estimator: TokenEstimator, max_tokens: int) -> str:
    clean = (text or "").strip()
    if not clean or max_tokens <= 0:
        return ""
    if token_estimator.count(clean) <= max_tokens:
        return clean

    low = 0
    high = len(clean)
    best = ""
    while low <= high:
        mid = (low + high) // 2
        candidate = clean[:mid].rstrip()
        if candidate and token_estimator.count(candidate) <= max_tokens:
            best = candidate
            low = mid + 1
        else:
            high = mid - 1
    return best or clean[: max(1, min(len(clean), max_tokens))]


def _keyword_tokens(text: str) -> set[str]:
    return {token for token in _WORD_RE.findall(text.lower()) if len(token) >= 3 and token not in _STOPWORDS}


def _is_heading_like(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 90 or stripped.count(",") >= 3:
        return False
    if _PAGE_HINT_RE.match(stripped):
        return False
    if re.match(r"^[A-Z](?:\.\d+)+", stripped):
        return True
    if re.match(r"^\d+(?:\.\d+)*\s+[A-Z]", stripped):
        return True
    words = stripped.replace("-", " ").split()
    if stripped.isupper() and len(words) <= 10:
        return True
    if len(words) <= 8:
        upperish = sum(1 for word in words if word[:1].isupper() or word.isupper())
        return upperish >= max(1, math.ceil(len(words) * 0.7))
    return False


def _is_numbered_heading(line: str) -> bool:
    stripped = line.strip()
    return bool(re.match(r"^\d+(?:\.\d+)*\s+[A-Z]", stripped))


def _detect_table_shape(lines: list[str]) -> tuple[str | None, float]:
    if len(lines) < 8:
        return None, 0.0
    best_delim = None
    best_score = 0.0
    sample = lines[: min(48, len(lines))]
    for delimiter in [",", "\t", "|"]:
        counts = [line.count(delimiter) for line in sample]
        dense = [count for count in counts if count >= 2]
        density_score = 0.0
        if dense:
            target = max(set(dense), key=dense.count)
            consistent = sum(1 for count in counts if abs(count - target) <= 1 and count >= 2)
            density_score = min(1.0, (consistent / len(sample)) * 0.75 + (len(dense) / len(sample)) * 0.25)
        row_shape_score = 0.0
        try:
            parsed = [
                row
                for row in csv.reader(io.StringIO("\n".join(sample)), delimiter=delimiter)
                if any(cell.strip() for cell in row)
            ]
            widths = [len(row) for row in parsed if len(row) >= 3]
            if widths:
                modal = max(set(widths), key=widths.count)
                width_consistent = sum(1 for width in widths if width == modal)
                row_shape_score = min(1.0, (width_consistent / len(widths)) * 0.8 + (modal >= 3) * 0.2)
        except Exception:
            row_shape_score = 0.0
        score = max(density_score, min(1.0, 0.45 * density_score + 0.75 * row_shape_score))
        if score > best_score:
            best_score = score
            best_delim = delimiter
    return best_delim, round(best_score, 3)


def _collect_manual_markers(text: str) -> list[str]:
    lowered = text.lower()
    return [marker for marker in ("manual", "table of contents", "preface", "warning", "danger", "note", "index", "parameter", "register", "overview") if marker in lowered]


def _estimate_heading_density(lines: list[str]) -> float:
    if not lines:
        return 0.0
    heading_like = sum(1 for line in lines[: min(240, len(lines))] if _is_heading_like(line))
    return min(1.0, heading_like / max(10, min(240, len(lines))))


def _estimate_note_density(lines: list[str]) -> float:
    if not lines:
        return 0.0
    note_like = sum(1 for line in lines[: min(240, len(lines))] if _MANUAL_HINT_RE.search(line))
    return min(1.0, note_like / max(12, min(240, len(lines))))


def _build_mc_prompt(packet: QuestionPacket) -> str:
    options = packet.options or {}
    return (
        f"Question:\n{packet.question}\n\nOptions:\n"
        + "\n".join(f"{label}. {text}" for label, text in options.items())
        + '\n\nReturn JSON only: {"answer":"A"} or B/C/D. Return one letter only.'
    )


class TableReader:
    def __init__(self, token_estimator: TokenEstimator | None = None, max_rows: int = 6000) -> None:
        self.token_estimator = token_estimator or SimpleTokenEstimator()
        self.max_rows = max_rows

    def match(self, packet: QuestionPacket, profile: StructureProfile) -> bool:
        del packet
        return profile.shape == "table" and profile.confidence >= 0.80

    def answer(self, packet: QuestionPacket, profile: StructureProfile, model: ModelAdapter) -> ReaderOutcome:
        started = time.perf_counter()
        headers, rows = _parse_table(packet.raw_context, profile.delimiter)
        if not headers or not rows:
            return ReaderOutcome("", "", [], "table/parse-failed", False, 0.0, round(time.perf_counter() - started, 3))

        deterministic = _try_rank_frequency_answer(packet, headers, rows)
        if deterministic is not None:
            answer_raw, evidence = deterministic
            return ReaderOutcome(
                answer_raw=answer_raw,
                answer_pred=extract_mc_choice(answer_raw),
                evidence_lines=evidence,
                route_name="table/deterministic-rank",
                deterministic=True,
                confidence=0.98,
                latency_s=round(time.perf_counter() - started, 3),
            )

        deterministic = _try_numeric_lookup_answer(packet, headers, rows)
        if deterministic is not None:
            answer_raw, evidence = deterministic
            return ReaderOutcome(
                answer_raw=answer_raw,
                answer_pred=extract_mc_choice(answer_raw),
                evidence_lines=evidence,
                route_name="table/deterministic-lookup",
                deterministic=True,
                confidence=0.9,
                latency_s=round(time.perf_counter() - started, 3),
            )

        if _is_temporal_trend_question(packet.question):
            evidence_lines = _build_temporal_trend_summary(packet, headers, rows[: min(len(rows), self.max_rows)])
            route_name = "table/temporal-trend"
            confidence = 0.68
        else:
            evidence_lines = _build_compact_table_summary(packet, headers, rows[: min(len(rows), self.max_rows)])
            route_name = "table/compact-summary"
            confidence = 0.62

        answer_raw = model.complete(
            [
                {
                    "role": "system",
                    "content": (
                        "Answer the multiple-choice question using only the supplied table-derived evidence.\n"
                        "Do not use outside knowledge.\n"
                        'Return JSON with one key: answer.'
                    ),
                },
                {"role": "system", "content": "\n".join(evidence_lines)},
                {"role": "user", "content": _build_mc_prompt(packet)},
            ]
        )
        return ReaderOutcome(
            answer_raw=answer_raw,
            answer_pred=extract_mc_choice(answer_raw),
            evidence_lines=evidence_lines[:12],
            route_name=route_name,
            deterministic=False,
            confidence=confidence,
            latency_s=round(time.perf_counter() - started, 3),
        )


def _parse_table(text: str, delimiter: str | None) -> tuple[list[str], list[list[str]]]:
    if delimiter is None:
        try:
            delimiter = csv.Sniffer().sniff(text[:4096], delimiters=",\t|").delimiter
        except Exception:
            delimiter = ","
    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    rows = [[cell.strip() for cell in row] for row in reader if any(cell.strip() for cell in row)]
    if len(rows) < 2:
        return [], []
    width = max(len(row) for row in rows)
    normalized = [row + [""] * (width - len(row)) for row in rows]
    return normalized[0], normalized[1:]


def _normalize_entity(text: str) -> str:
    return " ".join(_WORD_RE.findall(text.lower()))


def _candidate_aliases(text: str) -> tuple[str, ...]:
    normalized = _normalize_entity(text)
    if not normalized:
        return ()
    aliases = {normalized}
    aliases.update(_COUNTRY_ALIASES.get(normalized, ()))
    return tuple(sorted(alias for alias in aliases if alias))


def _extract_rank_entities(option_text: str) -> list[tuple[str, str]]:
    lowered = " ".join(option_text.lower().split())
    match = re.search(r"(?P<one>[a-z .-]+?)\s+as\s+the\s+second\s+and\s+(?P<two>[a-z .-]+?)\s+as\s+the\s+third", lowered)
    if match:
        return [("second", _normalize_entity(match.group("one"))), ("third", _normalize_entity(match.group("two")))]
    match = re.search(r"(?P<one>[a-z .-]+?)\s+as\s+the\s+first\s+and\s+(?P<two>[a-z .-]+?)\s+as\s+the\s+second", lowered)
    if match:
        return [("first", _normalize_entity(match.group("one"))), ("second", _normalize_entity(match.group("two")))]
    return []


def _select_relevant_columns(headers: Sequence[str], packet: QuestionPacket) -> list[int]:
    query_tokens = _keyword_tokens(packet.question) | _keyword_tokens(" ".join((packet.options or {}).values()))
    ranked: list[tuple[float, int]] = []
    for idx, header in enumerate(headers):
        header_tokens = _keyword_tokens(header)
        score = float(len(header_tokens & query_tokens))
        lowered = header.lower()
        if "address" in lowered:
            score += 2.0
        if "year" in lowered or "date" in lowered:
            score += 1.2
        if "language" in lowered:
            score += 0.8
        if "name" in lowered or "title" in lowered:
            score += 0.5
        ranked.append((score, idx))
    ranked.sort(reverse=True)
    chosen = [idx for score, idx in ranked if score > 0][:4]
    if not chosen:
        chosen = list(range(min(4, len(headers))))
    return chosen


def _try_rank_frequency_answer(packet: QuestionPacket, headers: list[str], rows: list[list[str]]) -> tuple[str, list[str]] | None:
    question_lower = packet.question.lower()
    if "frequently mentioned" not in question_lower and "most frequently" not in question_lower:
        return None
    specs = {label: _extract_rank_entities(text) for label, text in (packet.options or {}).items()}
    if not specs or not all(spec for spec in specs.values()):
        return None
    aliases = {entity: _candidate_aliases(entity) for spec in specs.values() for _, entity in spec}
    counts = {name: 0 for name in aliases}
    column_indexes = _select_relevant_columns(headers, packet)
    for row in rows[:12000]:
        text = " | ".join(row[idx] for idx in column_indexes if idx < len(row)).lower()
        for name, alias_group in aliases.items():
            for alias in alias_group:
                counts[name] += len(re.findall(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", text))
    ranked = [name for name, value in sorted(counts.items(), key=lambda item: (-item[1], item[0])) if value > 0]
    if len(ranked) < 3:
        return None
    slots = {"first": ranked[0], "second": ranked[1], "third": ranked[2]}
    matches = [label for label, spec in specs.items() if all(slots.get(slot) == entity for slot, entity in spec)]
    if len(matches) != 1:
        return None
    evidence = [
        f"Top frequency entities: {', '.join(f'{name}={counts[name]}' for name in ranked[:4])}",
        f"Resolved order: first={slots['first']}, second={slots['second']}, third={slots['third']}",
    ]
    return json.dumps({"answer": matches[0]}), evidence


def _try_numeric_lookup_answer(packet: QuestionPacket, headers: list[str], rows: list[list[str]]) -> tuple[str, list[str]] | None:
    question_lower = packet.question.lower()
    direction = None
    if any(word in question_lower for word in ("highest", "largest", "most")):
        direction = "max"
    if any(word in question_lower for word in ("lowest", "smallest", "least")):
        direction = "min"
    if direction is None:
        return None
    query_tokens = _keyword_tokens(packet.question)
    best_col = None
    best_score = -1.0
    for idx, header in enumerate(headers):
        score = float(len(_keyword_tokens(header) & query_tokens))
        if any(marker in header.lower() for marker in ("count", "price", "value", "score", "year", "amount", "total")):
            score += 0.8
        if score > best_score:
            best_score = score
            best_col = idx
    if best_col is None:
        return None
    scored_rows: list[tuple[float, list[str]]] = []
    for row in rows[:12000]:
        if best_col >= len(row):
            continue
        value = _safe_number(row[best_col])
        if value is not None:
            scored_rows.append((value, row))
    if not scored_rows:
        return None
    scored_rows.sort(key=lambda item: item[0], reverse=(direction == "max"))
    best_row = scored_rows[0][1]
    row_text = " | ".join(best_row[idx] for idx in _select_relevant_columns(headers, packet) if idx < len(best_row))
    matches = [label for label, text in (packet.options or {}).items() if len(_keyword_tokens(text) & _keyword_tokens(row_text)) >= 1]
    if len(matches) != 1:
        return None
    return json.dumps({"answer": matches[0]}), [f"Selected numeric column: {headers[best_col]}", f"Top row: {row_text}"]


def _safe_number(text: str) -> float | None:
    match = _NUMBER_RE.search((text or "").replace(",", ""))
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _is_temporal_trend_question(question: str) -> bool:
    lowered = question.lower()
    return any(marker in lowered for marker in ("trend", "evolution", "over time", "correlates", "from 200", "from 201"))


def _build_compact_table_summary(packet: QuestionPacket, headers: list[str], rows: list[list[str]]) -> list[str]:
    columns = _select_relevant_columns(headers, packet)
    projected_headers = [headers[idx] for idx in columns if idx < len(headers)]
    query_tokens = _keyword_tokens(packet.question) | _keyword_tokens(" ".join((packet.options or {}).values()))
    ranked_rows: list[tuple[float, list[str]]] = []
    for row in rows[:6000]:
        selected = [row[idx] for idx in columns if idx < len(row)]
        joined = " ".join(selected).lower()
        score = float(sum(1 for token in query_tokens if token in joined))
        ranked_rows.append((score, selected))
    ranked_rows.sort(key=lambda item: item[0], reverse=True)
    top_rows = [cells for score, cells in ranked_rows[:18] if score > 0]
    if not top_rows:
        top_rows = [[row[idx] for idx in columns if idx < len(row)] for row in rows[:18]]
    lines = ["Selected table view:", " | ".join(projected_headers)]
    lines.extend(" | ".join(cells) for cells in top_rows)
    return lines


def _find_column(headers: Sequence[str], markers: Sequence[str]) -> int | None:
    for idx, header in enumerate(headers):
        lowered = header.lower()
        if any(marker in lowered for marker in markers):
            return idx
    return None


def _extract_year(text: str) -> int | None:
    match = re.search(r"\b(19|20)\d{2}\b", text or "")
    if not match:
        return None
    return int(match.group(0))


def _build_temporal_trend_summary(packet: QuestionPacket, headers: list[str], rows: list[list[str]]) -> list[str]:
    year_idx = _find_column(headers, ("year", "date", "time"))
    if year_idx is None:
        return _build_compact_table_summary(packet, headers, rows)
    lines = [f"Temporal trend summary using {headers[year_idx]}:"]
    query_tokens = _keyword_tokens(packet.question) | _keyword_tokens(" ".join((packet.options or {}).values()))
    candidate_scores: list[tuple[float, int]] = []
    sample_rows = rows[: min(len(rows), 1200)]
    for idx, header in enumerate(headers):
        if idx == year_idx:
            continue
        lowered = header.lower()
        if any(marker in lowered for marker in ("id", "number", "amount", "value", "score")):
            continue
        column_values = [row[idx].strip() for row in sample_rows if idx < len(row) and row[idx].strip()]
        if not column_values:
            continue
        unique_ratio = len(set(column_values)) / max(1, len(column_values))
        score = float(len(_keyword_tokens(header) & query_tokens))
        if "language" in lowered:
            score += 2.6
        if any(marker in lowered for marker in ("cluster", "genre", "category", "topic", "type")):
            score += 1.9
        if any(marker in lowered for marker in ("title", "album", "artist", "name")):
            score -= 0.6
        if unique_ratio <= 0.35:
            score += 1.2
        elif unique_ratio <= 0.6:
            score += 0.5
        else:
            score -= 0.8
        candidate_scores.append((score, idx))
    candidate_scores.sort(reverse=True)
    candidate_indexes = [idx for score, idx in candidate_scores[:4] if score > -0.2]
    if not candidate_indexes:
        candidate_indexes = [
            idx
            for idx, header in enumerate(headers)
            if idx != year_idx and "language" in header.lower()
        ][:1]
    if not candidate_indexes:
        candidate_indexes = [
            idx
            for idx, header in enumerate(headers)
            if idx != year_idx and not any(marker in header.lower() for marker in ("id", "number"))
        ][:3]
    years = sorted({_extract_year(row[year_idx]) for row in rows if year_idx < len(row) and _extract_year(row[year_idx]) is not None})
    if not years:
        return _build_compact_table_summary(packet, headers, rows)
    for idx in candidate_indexes:
        per_year: dict[int, dict[str, int]] = {}
        for row in rows:
            if idx >= len(row) or year_idx >= len(row):
                continue
            year = _extract_year(row[year_idx])
            if year is None:
                continue
            value = row[idx].strip() or "<blank>"
            per_year.setdefault(year, {})
            per_year[year][value] = per_year[year].get(value, 0) + 1
        if not per_year:
            continue
        lines.append(f"- Candidate grouping column: {headers[idx]}")
        for year in years[:8]:
            counts = per_year.get(year, {})
            if not counts:
                continue
            total = sum(counts.values())
            ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:3]
            shares = [count / total for count in counts.values()]
            diversity = round(1.0 - sum(share * share for share in shares), 3)
            lines.append(
                f"  - {year}: total={total}, unique={len(counts)}, diversity={diversity}, top={', '.join(f'{name}:{count}' for name, count in ranked)}"
            )
        first_year = years[0]
        last_year = years[-1]
        first_counts = per_year.get(first_year, {})
        last_counts = per_year.get(last_year, {})
        if first_counts and last_counts:
            deltas: list[tuple[int, str]] = []
            labels = set(first_counts) | set(last_counts)
            for label in labels:
                deltas.append((last_counts.get(label, 0) - first_counts.get(label, 0), label))
            deltas.sort(reverse=True)
            increases = ", ".join(f"{label}:{delta:+d}" for delta, label in deltas[:3] if delta > 0)
            decreases = ", ".join(f"{label}:{delta:+d}" for delta, label in sorted(deltas)[:3] if delta < 0)
            if increases:
                lines.append(f"  - top increases from {first_year} to {last_year}: {increases}")
            if decreases:
                lines.append(f"  - top decreases from {first_year} to {last_year}: {decreases}")
    lines.extend(_build_compact_table_summary(packet, headers, rows)[:8])
    return lines


class ProcedureReader:
    def __init__(self, token_estimator: TokenEstimator | None = None, max_sections: int = 6, max_claims_per_option: int = 6) -> None:
        self.token_estimator = token_estimator or SimpleTokenEstimator()
        self.max_sections = max_sections
        self.max_claims_per_option = max_claims_per_option

    def match(self, packet: QuestionPacket, profile: StructureProfile) -> bool:
        if profile.shape != "manual" or profile.confidence < 0.75:
            return False
        query_text = " ".join([packet.question, *list((packet.options or {}).values())]).lower()
        strong_phrase_hits = sum(
            1
            for marker in (
                "according to the manual",
                "according to the guide",
                "should i",
                "how do i",
                "what should i",
                "if i",
                "i should",
            )
            if marker in query_text
        )
        statement_hits = sum(
            1
            for marker in ("which statement is false", "which statement is true")
            if marker in query_text
        )
        device_hits = sum(
            1
            for marker in ("password", "parameter", "menu", "button", "led", "pin", "switch", "camera", "device", "calculator", "controller")
            if re.search(rf"(?<![a-z0-9-]){re.escape(marker)}(?![a-z0-9-])", query_text)
        )
        structure_hits = sum(1 for marker in profile.section_markers if marker in {"manual", "table of contents", "parameter", "register"})
        return strong_phrase_hits >= 1 or device_hits >= 1 or (statement_hits >= 1 and structure_hits >= 1)

    def answer(self, packet: QuestionPacket, profile: StructureProfile, model: ModelAdapter) -> ReaderOutcome:
        del profile
        started = time.perf_counter()
        sections = _parse_manual_sections(packet.raw_context)
        parameters = _extract_parameter_records(sections)
        analyses: dict[str, dict[str, Any]] = {}
        evidence_lines: list[str] = []
        for label, option_text in (packet.options or {}).items():
            claims = _decompose_option_claims(option_text, model, self.max_claims_per_option)
            retrieved = _retrieve_manual_sections(sections, parameters, packet.question, option_text, claims, self.max_sections)
            evaluation = _evaluate_option_claims_with_model(model, packet.question, option_text, claims, retrieved)
            analyses[label] = evaluation
            evidence_lines.extend(_format_option_evidence(label, evaluation, retrieved))
        choice, deterministic = _select_manual_answer(packet.question.lower(), analyses)
        if not choice:
            choice = _final_manual_adjudication(model, packet, analyses)
            deterministic = False
        answer_raw = json.dumps({"answer": choice}) if choice else ""
        return ReaderOutcome(
            answer_raw=answer_raw,
            answer_pred=choice or extract_mc_choice(answer_raw),
            evidence_lines=evidence_lines[:16],
            route_name="manual/claim-matrix",
            deterministic=deterministic,
            confidence=_estimate_manual_confidence(choice, analyses),
            latency_s=round(time.perf_counter() - started, 3),
        )


def _parse_manual_sections(text: str) -> list[SectionRecord]:
    sections: list[SectionRecord] = []
    title = "Document Start"
    body: list[str] = []
    page_hint: int | None = None
    parent_heading: str | None = None

    def flush() -> None:
        body_text = "\n".join(line for line in body if line.strip()).strip()
        if not body_text and title == "Document Start":
            return
        sections.append(SectionRecord(title=title, page_hint=page_hint, body=body_text, tokens=_keyword_tokens(title + " " + body_text)))

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            if body and body[-1] != "":
                body.append("")
            continue
        page_match = _PAGE_HINT_RE.match(stripped)
        if page_match and len(stripped) <= 6:
            page_hint = int(page_match.group(1))
            continue
        if _is_heading_like(stripped):
            if body:
                flush()
                body = []
            if _is_numbered_heading(stripped) or stripped.isupper():
                parent_heading = stripped
                title = stripped
                continue
            if parent_heading and parent_heading != stripped:
                title = f"{parent_heading} :: {stripped}"
            else:
                title = stripped
            continue
        body.append(stripped)
    flush()
    return sections


def _extract_parameter_records(sections: list[SectionRecord]) -> list[ParameterRecord]:
    records: list[ParameterRecord] = []
    for section in sections:
        lines = [line.strip() for line in section.body.splitlines() if line.strip()]
        leaf_title = section.title.split("::")[-1].strip()
        if lines and _looks_parameter_name(leaf_title):
            window = [leaf_title, *lines[:4]]
            records.append(
                ParameterRecord(
                    name=leaf_title,
                    values=tuple(_extract_values(window)),
                    default=_extract_default(window),
                    effects=tuple(_select_window_lines(window, ("allow", "disable", "enable", "reset", "used", "reboot", "restart"))),
                    constraints=tuple(_select_window_lines(window, ("range:", "only when", "must", "if ", "unavailable"))),
                    section_title=section.title,
                )
            )
        for idx, line in enumerate(lines):
            if not _looks_parameter_name(line):
                continue
            window = lines[idx : idx + 5]
            values = tuple(_extract_values(window))
            records.append(
                ParameterRecord(
                    name=line,
                    values=values,
                    default=_extract_default(window),
                    effects=tuple(_select_window_lines(window, ("allow", "disable", "enable", "reset", "used"))),
                    constraints=tuple(_select_window_lines(window, ("range:", "only when", "must", "if "))),
                    section_title=section.title,
                )
            )
    return records


def _looks_parameter_name(line: str) -> bool:
    if not line or len(line) > 64 or line.count(",") >= 2:
        return False
    words = line.replace("-", " ").split()
    if len(words) > 8:
        return False
    upperish = sum(1 for word in words if word.isupper() or word[:1].isupper())
    return upperish >= max(1, math.ceil(len(words) * 0.7))


def _extract_values(lines: Sequence[str]) -> list[str]:
    values: list[str] = []
    for line in lines:
        for match in re.findall(r"\b(?:enable|disable|yes|no|open|closed|true|false|[A-Za-z0-9._/-]+)\b", line, flags=re.IGNORECASE):
            lowered = match.lower()
            if lowered in {"range", "save", "change", "enter", "password"}:
                continue
            if lowered not in {item.lower() for item in values}:
                values.append(match)
    return values[:8]


def _extract_default(lines: Sequence[str]) -> str | None:
    for line in lines:
        match = re.search(r"(?i)\bdefault(?: password)?\s*[:=]?\s*['\"]?([A-Za-z0-9._/-]+)", line)
        if match:
            return match.group(1)
    return None


def _select_window_lines(lines: Sequence[str], markers: Sequence[str]) -> list[str]:
    return [line for line in lines if any(marker in line.lower() for marker in markers)][:4]


def _decompose_option_claims(text: str, model: ModelAdapter, max_claims: int) -> list[str]:
    normalized = " ".join(text.split())
    parts = [part.strip(" ;,.") for part in re.split(r"(?<=[.;])\s+|,\s+and\s+|,\s+but\s+|\s+while\s+", normalized) if part.strip(" ;,.")]
    if len(parts) > 1:
        return parts[:max_claims]
    if len(normalized) <= 150:
        return [normalized]
    raw = model.complete(
        [
            {"role": "system", "content": 'Split the option into atomic factual claims. Return JSON: {"claims":["..."]}.'},
            {"role": "user", "content": normalized},
        ]
    )
    parsed = _json_load_loose(raw)
    claims = parsed.get("claims") if isinstance(parsed, dict) else None
    if isinstance(claims, list):
        cleaned = [str(item).strip() for item in claims if str(item).strip()]
        if cleaned:
            return cleaned[:max_claims]
    return [normalized]


def _retrieve_manual_sections(
    sections: list[SectionRecord],
    parameters: list[ParameterRecord],
    question: str,
    option_text: str,
    claims: list[str],
    max_sections: int,
) -> list[SectionRecord]:
    claim_token_sets = [_keyword_tokens(claim) for claim in claims]
    claim_tokens = set().union(*claim_token_sets) if claim_token_sets else set()
    query_tokens = _keyword_tokens(question) | _keyword_tokens(option_text) | claim_tokens
    scored: list[tuple[float, int, SectionRecord]] = []
    parameter_names = {_normalize_entity(record.name) for record in parameters}
    for idx, section in enumerate(sections):
        title_overlap = len(_keyword_tokens(section.title) & query_tokens)
        body_overlap = len(section.tokens & query_tokens)
        lowered = f"{section.title}\n{section.body}".lower()
        bonus = 0.0
        if any(marker in lowered for marker in ("note", "warning", "danger")):
            bonus += 0.4
        if any(name and name in lowered for name in parameter_names):
            bonus += 0.3
        scored.append((title_overlap * 1.4 + body_overlap + bonus, idx, section))
    scored.sort(key=lambda item: (-item[0], item[1]))
    chosen: list[SectionRecord] = []
    used: set[int] = set()
    for score, idx, _ in scored:
        if score <= 0 and chosen:
            break
        for neighbor in [idx - 1, idx, idx + 1]:
            if neighbor < 0 or neighbor >= len(sections) or neighbor in used:
                continue
            used.add(neighbor)
            chosen.append(sections[neighbor])
            if len(chosen) >= max_sections:
                return chosen[:max_sections]
    return chosen[:max_sections]


def _evaluate_option_claims_with_model(
    model: ModelAdapter,
    question: str,
    option_text: str,
    claims: list[str],
    sections: list[SectionRecord],
) -> dict[str, Any]:
    excerpt = _render_section_excerpt(sections)
    raw = model.complete(
        [
            {
                "role": "system",
                "content": (
                    "Assess each claim only from the supplied manual excerpts.\n"
                    'Return JSON with keys: claims and overall_status.\n'
                    'Each claim entry must include claim, status (supported|contradicted|unresolved), and evidence.'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{question}\n\nOption:\n{option_text}\n\nClaims:\n"
                    + "\n".join(f"- {claim}" for claim in claims)
                    + "\n\nManual excerpts:\n"
                    + excerpt
                ),
            },
        ]
    )
    parsed = _json_load_loose(raw)
    if isinstance(parsed, dict) and isinstance(parsed.get("claims"), list):
        payload = []
        for item in parsed["claims"]:
            if not isinstance(item, dict):
                continue
            payload.append(
                {
                    "claim": str(item.get("claim", "")).strip(),
                    "status": _normalize_status(str(item.get("status", "unresolved"))),
                    "evidence": str(item.get("evidence", "")).strip(),
                }
            )
        return {"claims": payload, "overall_status": _normalize_status(str(parsed.get("overall_status", "unresolved")))}
    return _heuristic_claim_evaluation(claims, sections)


def _heuristic_claim_evaluation(claims: list[str], sections: list[SectionRecord]) -> dict[str, Any]:
    rendered_tokens = _keyword_tokens(_render_section_excerpt(sections))
    payload = []
    for claim in claims:
        claim_tokens = _keyword_tokens(claim)
        overlap = len(claim_tokens & rendered_tokens)
        status = "supported" if overlap >= max(2, len(claim_tokens) // 2) else "unresolved"
        payload.append({"claim": claim, "status": status, "evidence": ""})
    supported = sum(1 for item in payload if item["status"] == "supported")
    return {"claims": payload, "overall_status": "supported" if supported else "unresolved"}


def _render_section_excerpt(sections: list[SectionRecord], max_chars: int = 2600) -> str:
    parts: list[str] = []
    used = 0
    for section in sections:
        block = f"[Section] {section.title}\n{section.body}".strip()
        if used + len(block) > max_chars and parts:
            break
        parts.append(block[: max(0, max_chars - used)])
        used += len(parts[-1]) + 2
    return "\n\n".join(parts)


def _format_option_evidence(label: str, evaluation: dict[str, Any], sections: list[SectionRecord]) -> list[str]:
    lines = [f"{label}: overall={evaluation.get('overall_status', 'unresolved')}"]
    for item in evaluation.get("claims", [])[:4]:
        lines.append(f"{label}: {item.get('status', 'unresolved')} :: {item.get('claim', '')} :: {item.get('evidence', '')}")
    if sections:
        lines.append(f"{label}: sections={', '.join(section.title for section in sections[:3])}")
    return lines


def _normalize_status(status: str) -> str:
    lowered = status.lower()
    if "support" in lowered or lowered == "true":
        return "supported"
    if "contradict" in lowered or lowered == "false":
        return "contradicted"
    return "unresolved"


def _select_manual_answer(question_lower: str, analyses: dict[str, dict[str, Any]]) -> tuple[str, bool]:
    false_question = "false" in question_lower or "not true" in question_lower
    best_label = ""
    best_score = -10_000.0
    second_best = -10_000.0
    for label, analysis in analyses.items():
        statuses = [item.get("status", "unresolved") for item in analysis.get("claims", [])]
        supported = sum(1 for status in statuses if status == "supported")
        contradicted = sum(1 for status in statuses if status == "contradicted")
        unresolved = sum(1 for status in statuses if status == "unresolved")
        score = (2.0 * contradicted - supported - 0.25 * unresolved) if false_question else (2.0 * supported - 2.5 * contradicted - 0.2 * unresolved)
        if score > best_score:
            second_best = best_score
            best_score = score
            best_label = label
        elif score > second_best:
            second_best = score
    return best_label, bool(best_label and (best_score - second_best) >= 0.8)


def _final_manual_adjudication(model: ModelAdapter, packet: QuestionPacket, analyses: dict[str, dict[str, Any]]) -> str:
    lines: list[str] = []
    for label, analysis in analyses.items():
        lines.append(f"{label}: overall={analysis.get('overall_status', 'unresolved')}")
        for item in analysis.get("claims", [])[:4]:
            lines.append(f"{label}: {item.get('status', 'unresolved')} :: {item.get('claim', '')} :: {item.get('evidence', '')}")
    raw = model.complete(
        [
            {"role": "system", "content": 'Choose the best multiple-choice answer using only the claim matrix below. Return JSON with one key: answer.'},
            {"role": "system", "content": "\n".join(lines)},
            {"role": "user", "content": _build_mc_prompt(packet)},
        ]
    )
    return extract_mc_choice(raw)


def _estimate_manual_confidence(choice: str, analyses: dict[str, dict[str, Any]]) -> float:
    if not choice or choice not in analyses:
        return 0.0
    statuses = [item.get("status", "unresolved") for item in analyses[choice].get("claims", [])]
    supported = sum(1 for status in statuses if status == "supported")
    contradicted = sum(1 for status in statuses if status == "contradicted")
    return round(min(0.95, 0.45 + 0.12 * supported + 0.08 * contradicted), 3)


def _json_load_loose(raw: str) -> Any:
    text = (raw or "").strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except Exception:
                return {}
    return {}
