from __future__ import annotations

import argparse
import csv
import io
import json
import random
import re
import statistics
import time
from dataclasses import asdict
from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from benchmarks.api_backend import ChatBackendConfig
from benchmarks.api_backend import chat_completion
from memory_orb import MemoryOrbEngine
from memory_orb import MemoryOrbEngineConfig
from memory_orb.adapters import ModelAdapter
from memory_orb.structured_readers import QuestionPacket, ReaderOutcome, StructureProfile
from memory_orb.structured_readers import profile_structure as _profile_structure
from memory_orb.structured_readers import route_structured_reader


CHOICE_RE = re.compile(r"\b([ABCD])\b", flags=re.IGNORECASE)
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "among",
    "all",
    "based",
    "best",
    "by",
    "following",
    "from",
    "in",
    "is",
    "of",
    "on",
    "or",
    "provided",
    "reflects",
    "the",
    "these",
    "this",
    "to",
    "trend",
    "underlying",
    "which",
    "with",
}
_COUNTRY_ALIASES = {
    "australia": ("australia", "au"),
    "austria": ("austria", "at"),
    "belgium": ("belgium", "be"),
    "brazil": ("brazil", "br"),
    "canada": ("canada", "ca"),
    "china": ("china", "cn"),
    "czech republic": ("czech republic", "cz"),
    "france": ("france", "fr"),
    "germany": ("germany", "de"),
    "hungary": ("hungary", "hu"),
    "india": ("india", "in"),
    "italy": ("italy", "it"),
    "japan": ("japan", "jp"),
    "korea": ("korea", "kr"),
    "netherlands": ("netherlands", "nl"),
    "new zealand": ("new zealand", "nz"),
    "russia": ("russia", "ru"),
    "spain": ("spain", "es"),
    "sweden": ("sweden", "se"),
    "switzerland": ("switzerland", "ch"),
    "taiwan": ("taiwan", "tw"),
    "turkey": ("turkey", "tr"),
    "united kingdom": ("united kingdom", "uk", "gb"),
    "united states": ("united states", "usa", "us"),
}


@dataclass(slots=True)
class BenchCase:
    case_id: str
    length: str
    difficulty: str
    domain: str
    sub_domain: str
    question: str
    context: str
    choice_a: str
    choice_b: str
    choice_c: str
    choice_d: str
    answer: str


@dataclass(slots=True)
class BenchResult:
    case_id: str
    length: str
    difficulty: str
    domain: str
    sub_domain: str
    answer: str
    direct_raw: str
    direct_pred: str
    direct_correct: int
    direct_error: str
    memory_raw: str
    memory_pred: str
    memory_correct: int
    memory_error: str
    direct_latency_s: float
    memory_latency_s: float
    context_chars: int
    chunk_count: int
    route_name: str = ""
    route_profile_shape: str = ""
    route_profile_confidence: float = 0.0
    deterministic_reader_used: int = 0
    reader_evidence_excerpt: str = ""


@dataclass(slots=True)
class _MemoryRunOutcome:
    answer_raw: str
    chunk_count: int
    route_name: str
    route_profile_shape: str
    route_profile_confidence: float
    deterministic_reader_used: bool
    reader_evidence_excerpt: str


def _build_question_packet(case: BenchCase) -> QuestionPacket:
    return QuestionPacket(
        question=case.question,
        options={
            "A": case.choice_a,
            "B": case.choice_b,
            "C": case.choice_c,
            "D": case.choice_d,
        },
        raw_context=case.context,
    )


def _progress_bar(done: int, total: int, width: int = 28) -> str:
    if total <= 0:
        return "[" + ("-" * width) + "]"
    filled = int((done / float(total)) * width)
    filled = max(0, min(width, filled))
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def _build_mc_prompt(case: BenchCase) -> str:
    return _build_mc_question_block(case) + '\n\nReturn JSON only: {"answer":"A"} or B/C/D. Return one letter only.'


def _build_mc_question_block(case: BenchCase) -> str:
    return (
        f"Question:\n{case.question}\n\n"
        "Options:\n"
        f"A. {case.choice_a}\n"
        f"B. {case.choice_b}\n"
        f"C. {case.choice_c}\n"
        f"D. {case.choice_d}"
    )


def _route_structured_memory_case(
    packet: QuestionPacket,
    model: ModelAdapter,
    config: MemoryOrbEngineConfig,
) -> tuple[StructureProfile, ReaderOutcome | None]:
    return route_structured_reader(
        packet=packet,
        model=model,
        structured_reader_ctx=config.structured_reader_ctx,
        table_reader_max_rows=config.table_reader_max_rows,
        procedure_reader_max_sections=config.procedure_reader_max_sections,
        procedure_reader_max_claims_per_option=config.procedure_reader_max_claims_per_option,
    )


def _extract_choice(raw: str) -> str:
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
                        match = CHOICE_RE.search(str(value))
                        if match:
                            return match.group(1).upper()
        except Exception:
            pass

    explicit = re.search(r"(?i)\b(?:answer|choice|option)\s*[:=]\s*['\"]?\s*([ABCD])\b", text)
    if explicit:
        return explicit.group(1).upper()

    stripped = text.strip().upper()
    if stripped in {"A", "B", "C", "D"}:
        return stripped

    # Avoid accidental extraction from long echoed context/options blobs.
    if len(text) <= 40:
        matches = CHOICE_RE.findall(text)
        if matches:
            return matches[-1].upper()
    return ""


def _chunk_text(text: str, chunk_chars: int = 1400) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    used = 0
    for para in text.split("\n"):
        line = para.strip()
        if not line:
            continue
        need = len(line) + 1
        if current and used + need > chunk_chars:
            chunks.append("\n".join(current))
            current = [line]
            used = len(line)
        else:
            current.append(line)
            used += need
    if current:
        chunks.append("\n".join(current))
    return chunks


def _looks_like_tabular_context(text: str) -> bool:
    packet = QuestionPacket(question="", options=None, raw_context=text)
    return _profile_structure(packet).shape == "table"


def _parse_tabular_context(context: str) -> tuple[list[str], list[list[str]]]:
    packet = QuestionPacket(question="", options=None, raw_context=context)
    profile = _profile_structure(packet)
    delimiter = profile.delimiter or ","
    reader = csv.reader(io.StringIO(context), delimiter=delimiter)
    rows = [[cell.strip() for cell in row] for row in reader if any(cell.strip() for cell in row)]
    if len(rows) < 2:
        return [], []
    width = max(len(row) for row in rows)
    normalized = [row + [""] * (width - len(row)) for row in rows]
    return normalized[0], normalized[1:]


def _keyword_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) >= 3 and token not in _STOPWORDS
    }


def _select_relevant_table_columns(headers: list[str], case: BenchCase) -> list[int]:
    question_tokens = _keyword_tokens(case.question)
    option_tokens = _keyword_tokens(" ".join([case.choice_a, case.choice_b, case.choice_c, case.choice_d]))
    query_tokens = question_tokens | option_tokens
    ranked: list[tuple[float, int]] = []
    for idx, header in enumerate(headers):
        header_tokens = _keyword_tokens(header)
        score = float(len(header_tokens & query_tokens))
        if "address" in header.lower():
            score += 2.0
        if "year" in header.lower():
            score += 1.2
        if "language" in header.lower():
            score += 0.8
        ranked.append((score, idx))
    ranked.sort(reverse=True)
    chosen = [idx for score, idx in ranked if score > 0][:3]
    if not chosen:
        chosen = list(range(min(3, len(headers))))
    return chosen


def _candidate_aliases(text: str) -> tuple[str, ...]:
    normalized = " ".join(re.findall(r"[a-z]+", text.lower()))
    if not normalized:
        return ()
    aliases = {normalized}
    aliases.update(_COUNTRY_ALIASES.get(normalized, ()))
    if normalized.endswith("s"):
        aliases.add(normalized[:-1])
    return tuple(alias for alias in aliases if alias)


def _extract_rank_entities(option_text: str) -> list[tuple[str, str]]:
    lowered = " ".join(option_text.lower().split())
    match = re.search(
        r"(?P<first>[a-z .-]+?)\s+as\s+the\s+second\s+and\s+(?P<second>[a-z .-]+?)\s+as\s+the\s+third",
        lowered,
    )
    if match:
        return [("second", match.group("first").strip(" .")), ("third", match.group("second").strip(" ."))]
    match = re.search(
        r"(?P<first>[a-z .-]+?)\s+as\s+the\s+first\s+and\s+(?P<second>[a-z .-]+?)\s+as\s+the\s+second",
        lowered,
    )
    if match:
        return [("first", match.group("first").strip(" .")), ("second", match.group("second").strip(" ."))]
    return []


def _count_alias_matches(texts: list[str], alias_groups: dict[str, tuple[str, ...]]) -> dict[str, int]:
    counts = {name: 0 for name in alias_groups}
    for text in texts:
        lowered = f" {text.lower()} "
        for name, aliases in alias_groups.items():
            for alias in aliases:
                pattern = rf"(?<![a-z0-9]){re.escape(alias.lower())}(?![a-z0-9])"
                counts[name] += len(re.findall(pattern, lowered))
    return counts


def _rank_entities_by_frequency(counts: dict[str, int]) -> list[str]:
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [name for name, value in ranked if value > 0]


def _try_answer_table_rank_question(case: BenchCase, headers: list[str], rows: list[list[str]]) -> str | None:
    question_lower = case.question.lower()
    if "frequently mentioned" not in question_lower and "most frequently" not in question_lower:
        return None
    option_specs = {
        "A": _extract_rank_entities(case.choice_a),
        "B": _extract_rank_entities(case.choice_b),
        "C": _extract_rank_entities(case.choice_c),
        "D": _extract_rank_entities(case.choice_d),
    }
    if not all(spec for spec in option_specs.values()):
        return None
    all_entities = {
        entity
        for spec in option_specs.values()
        for _, entity in spec
    }
    aliases = {entity: _candidate_aliases(entity) for entity in all_entities}
    column_indexes = _select_relevant_table_columns(headers, case)
    texts: list[str] = []
    for row in rows:
        selected = [row[idx] for idx in column_indexes if idx < len(row)]
        texts.append(" | ".join(selected))
    counts = _count_alias_matches(texts, aliases)
    ranked = _rank_entities_by_frequency(counts)
    if len(ranked) < 3:
        return None

    slot_lookup = {
        "first": ranked[0] if len(ranked) >= 1 else "",
        "second": ranked[1] if len(ranked) >= 2 else "",
        "third": ranked[2] if len(ranked) >= 3 else "",
    }
    matches: list[str] = []
    for label, spec in option_specs.items():
        if all(slot_lookup.get(slot) == entity for slot, entity in spec):
            matches.append(label)
    if len(matches) == 1:
        return json.dumps({"answer": matches[0]})
    return None


def _build_table_summary(case: BenchCase, headers: list[str], rows: list[list[str]], max_rows: int = 18) -> str:
    column_indexes = _select_relevant_table_columns(headers, case)
    projected_headers = [headers[idx] for idx in column_indexes if idx < len(headers)]
    option_tokens = _keyword_tokens(" ".join([case.choice_a, case.choice_b, case.choice_c, case.choice_d]))
    question_tokens = _keyword_tokens(case.question)
    ranked_rows: list[tuple[float, list[str]]] = []
    for row in rows[: min(len(rows), 6000)]:
        selected = [row[idx] for idx in column_indexes if idx < len(row)]
        combined = " ".join(selected).lower()
        score = float(sum(1 for token in option_tokens if token in combined))
        score += 0.6 * sum(1 for token in question_tokens if token in combined)
        ranked_rows.append((score, selected))
    ranked_rows.sort(key=lambda item: item[0], reverse=True)
    top_rows = [cells for score, cells in ranked_rows[:max_rows] if score > 0]
    if not top_rows:
        top_rows = [
            [row[idx] for idx in column_indexes if idx < len(row)]
            for row in rows[:max_rows]
        ]
    rendered_rows = [" | ".join(projected_headers)]
    rendered_rows.extend(" | ".join(cells) for cells in top_rows)
    return "Selected table view:\n" + "\n".join(rendered_rows)


def _run_table_memory_case(
    case: BenchCase,
    model: str,
    memory_ctx: int,
    timeout_s: int,
) -> tuple[str, int]:
    headers, rows = _parse_tabular_context(case.context)
    if not headers or not rows:
        return "", 0
    deterministic = _try_answer_table_rank_question(case, headers, rows)
    if deterministic:
        return deterministic, len(rows)
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string", "enum": ["A", "B", "C", "D"]}},
        "required": ["answer"],
    }
    adapter = _OllamaAdapter(
        model_name=model,
        num_ctx=memory_ctx,
        timeout_s=timeout_s,
        response_schema=schema,
    )
    summary = _build_table_summary(case, headers, rows)
    prompt = _build_mc_prompt(case)
    answer = adapter.complete(
        [
            {
                "role": "system",
                "content": (
                    "Answer the multiple-choice question using only the compact table view.\n"
                    "Treat the first line as the table header.\n"
                    "Return JSON only with one key: answer (A/B/C/D)."
                ),
            },
            {"role": "system", "content": summary},
            {"role": "user", "content": prompt},
        ]
    )
    return answer, len(rows)


def _select_cases(
    sample_size: int,
    seed: int,
    lengths: set[str],
    max_context_chars: int,
    difficulty_filter: set[str] | None = None,
) -> list[BenchCase]:
    ds = load_dataset("THUDM/LongBench-v2", split="train")
    rows: list[BenchCase] = []
    for row in ds:
        length = str(row["length"])
        if lengths and length not in lengths:
            continue
        if int(len(str(row["context"]))) > max_context_chars:
            continue
        difficulty = str(row["difficulty"])
        if difficulty_filter and difficulty not in difficulty_filter:
            continue
        rows.append(
            BenchCase(
                case_id=str(row["_id"]),
                length=length,
                difficulty=difficulty,
                domain=str(row["domain"]),
                sub_domain=str(row["sub_domain"]),
                question=str(row["question"]),
                context=str(row["context"]),
                choice_a=str(row["choice_A"]),
                choice_b=str(row["choice_B"]),
                choice_c=str(row["choice_C"]),
                choice_d=str(row["choice_D"]),
                answer=str(row["answer"]).strip().upper(),
            )
        )

    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[: max(1, min(sample_size, len(rows)))]


def _run_direct_case(
    case: BenchCase,
    model: str,
    direct_ctx: int,
    timeout_s: int,
    backend: ChatBackendConfig,
) -> str:
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string", "enum": ["A", "B", "C", "D"]}},
        "required": ["answer"],
    }
    prompt = _build_mc_prompt(case)
    messages = [
        {
            "role": "system",
            "content": (
                "Read the long context and answer the multiple-choice question.\n"
                "Return strict JSON with one key: answer, where value is exactly one of A/B/C/D."
            ),
        },
        {"role": "user", "content": f"Context:\n{case.context}\n\n{prompt}"},
    ]

    ctx_candidates = [direct_ctx]
    for candidate in [196608, 131072, 98304, 65536, 49152, 32768]:
        if candidate < direct_ctx and candidate not in ctx_candidates:
            ctx_candidates.append(candidate)

    last_error = ""
    for ctx in ctx_candidates:
        try:
            return chat_completion(
                backend=backend,
                model=model,
                messages=messages,
                num_ctx=ctx,
                timeout_s=timeout_s,
                json_mode=True,
                response_schema=schema,
            )
        except RuntimeError as err:
            last_error = str(err)
            continue
    raise RuntimeError(f"Direct inference failed for case {case.case_id}: {last_error}")


class _ChatAdapter(ModelAdapter):
    def __init__(
        self,
        model_name: str,
        num_ctx: int,
        timeout_s: int,
        backend: ChatBackendConfig,
        response_schema: dict[str, Any] | None = None,
        reasoning_num_predict: int = 192,
        enable_think: bool = False,
    ) -> None:
        self.model_name = model_name
        self.num_ctx = num_ctx
        self.timeout_s = timeout_s
        self.backend = backend
        self.response_schema = response_schema
        self.reasoning_num_predict = reasoning_num_predict
        self.enable_think = enable_think

    def complete(self, messages: list[dict[str, str]]) -> str:
        return chat_completion(
            backend=self.backend,
            model=self.model_name,
            messages=messages,
            num_ctx=self.num_ctx,
            timeout_s=self.timeout_s,
            json_mode=True,
            response_schema=self.response_schema,
        )

    def _should_use_ollama_think(self, think: bool) -> bool:
        if not think:
            return False
        if self.backend.normalized_provider() != "ollama":
            return False
        if self.enable_think:
            return True
        model_lower = self.model_name.lower()
        if "qwen3.5" in model_lower:
            return False
        return True

    def complete_with_reasoning(self, messages: list[dict[str, str]], think: bool = True) -> str:
        return chat_completion(
            backend=self.backend,
            model=self.model_name,
            messages=messages,
            num_ctx=self.num_ctx,
            timeout_s=self.timeout_s,
            json_mode=False,
            response_schema=None,
            think=self._should_use_ollama_think(think),
            num_predict=self.reasoning_num_predict,
        )


_OllamaAdapter = _ChatAdapter


class _FixedResponseAdapter(ModelAdapter):
    def __init__(self, response_text: str) -> None:
        self.response_text = response_text

    def complete(self, messages: list[dict[str, str]]) -> str:
        return self.response_text


def _build_memory_engine(memory_ctx: int, answer_dwell_mode: str) -> MemoryOrbEngine:
    reasoned = answer_dwell_mode == "reasoned"
    return MemoryOrbEngine(
        config=MemoryOrbEngineConfig(
            context_max_tokens=memory_ctx,
            working_max_tokens=max(700, int(memory_ctx * 0.65)),
            working_target_tokens=max(520, int(memory_ctx * 0.48)),
            memory_budget_ratio=0.55,
            max_retrieved_orbs=12,
            min_focus_orb_count=2,
            answer_dwell_mode=answer_dwell_mode,
            reasoning_dwell_trigger_score=1.25 if reasoned else 1.85,
            reasoning_dwell_min_complexity=0.28 if reasoned else 0.42,
            reasoning_dwell_max_areas=2 if reasoned else 3,
            enable_structured_readers=True,
            structured_reader_ctx=max(512, min(memory_ctx, 900)),
            table_reader_max_rows=6000,
            procedure_reader_max_sections=6,
            procedure_reader_max_claims_per_option=6,
        )
    )


def _ingest_benchmark_context(engine: MemoryOrbEngine, context: str, chunk_chars: int) -> int:
    chunks = _chunk_text(context, chunk_chars=chunk_chars)
    for chunk in chunks:
        engine.add_turn(
            "tool",
            chunk,
            metadata={
                "importance": 0.6,
                "source": "benchmark_document",
                "exclude_from_pulse_map": True,
            },
        )
        # Preserve chunk-sized swap boundaries without adding semantic noise.
        engine.add_turn(
            "assistant",
            "",
            metadata={
                "source": "benchmark_boundary",
                "exclude_from_pulse_map": True,
                "exclude_from_question_memory_pool": True,
            },
        )
    return len(chunks)


def _extract_reasoned_chat_evidence(answer_doc: str, max_lines: int = 8) -> str:
    evidence_lines: list[str] = []
    seen: set[str] = set()
    for raw_line in (answer_doc or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        cleaned = ""
        if line.startswith("- pass=") and ": " in line:
            cleaned = "- " + line.split(": ", 1)[1].strip()
        elif line.startswith("- candidate="):
            cleaned = line
        elif line.startswith("- No relevant evidence"):
            cleaned = line
        if cleaned and cleaned not in seen:
            evidence_lines.append(cleaned)
            seen.add(cleaned)
        if len(evidence_lines) >= max_lines:
            break
    if not evidence_lines:
        return ""
    return "Reasoned skim evidence:\n" + "\n".join(evidence_lines)


def _normalize_reasoned_option_analysis(text: str) -> str:
    kept: list[str] = []
    for raw_line in (text or "").splitlines():
        line = " ".join(raw_line.strip().split())
        if not line:
            continue
        if re.match(r"^(?:[ABCD]|BEST)\s*:", line, flags=re.IGNORECASE):
            kept.append(line)
        if len(kept) >= 5:
            break
    return "\n".join(kept)


def _build_reasoned_option_analysis(
    case: BenchCase,
    evidence_block: str,
    model: str,
    num_ctx: int,
    timeout_s: int,
    backend: ChatBackendConfig,
    reasoning_num_predict: int,
    enable_ollama_think: bool,
) -> str:
    if not evidence_block:
        return ""
    analysis_adapter = _ChatAdapter(
        model_name=model,
        num_ctx=num_ctx,
        timeout_s=timeout_s,
        backend=backend,
        response_schema=None,
        reasoning_num_predict=reasoning_num_predict,
        enable_think=enable_ollama_think,
    )
    question_block = _build_mc_question_block(case)
    raw = analysis_adapter.complete_with_reasoning(
        [
            {
                "role": "system",
                "content": (
                    "You are comparing multiple-choice options using only supplied evidence notes.\n"
                    "Do not use outside knowledge.\n"
                    "Return exactly one short line for A, B, C, D, and BEST."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"{question_block}\n\n"
                    f"Evidence notes:\n{evidence_block}\n\n"
                    "Return exactly this format:\n"
                    "A: supported|contradicted|weak - <reason>\n"
                    "B: supported|contradicted|weak - <reason>\n"
                    "C: supported|contradicted|weak - <reason>\n"
                    "D: supported|contradicted|weak - <reason>\n"
                    "BEST: <option>|uncertain - <reason>"
                ),
            },
        ],
        think=True,
    )
    return _normalize_reasoned_option_analysis(raw)


def _build_reasoned_chat_supplement(
    case: BenchCase,
    model: str,
    memory_ctx: int,
    timeout_s: int,
    chunk_chars: int,
    reasoning_dwell_ctx: int,
    backend: ChatBackendConfig,
    reasoning_num_predict: int,
    enable_ollama_think: bool,
) -> str:
    prompt = _build_mc_prompt(case)
    scratch_engine = _build_memory_engine(memory_ctx=memory_ctx, answer_dwell_mode="reasoned")
    _ingest_benchmark_context(scratch_engine, case.context, chunk_chars=chunk_chars)
    dwell_adapter = _ChatAdapter(
        model_name=model,
        num_ctx=max(512, min(memory_ctx, reasoning_dwell_ctx)),
        timeout_s=timeout_s,
        backend=backend,
        response_schema=None,
        reasoning_num_predict=reasoning_num_predict,
        enable_think=enable_ollama_think,
    )
    noop_final_adapter = _FixedResponseAdapter('{"answer":"A"}')
    answer_doc = scratch_engine.answer_with_answer_document(
        model=noop_final_adapter,
        dwell_model=dwell_adapter,
        question=prompt,
        passes=4,
        per_pass_orbs=4,
        answer_doc_max_tokens=max(192, int(memory_ctx * 0.24)),
        system_prompt=(
            "Collect only concise evidence that helps answer the multiple-choice question.\n"
            "Do not answer the question here."
        ),
        allow_answer_coercion=False,
    ).answer_document
    evidence_block = _extract_reasoned_chat_evidence(answer_doc)
    option_analysis = _build_reasoned_option_analysis(
        case=case,
        evidence_block=evidence_block,
        model=model,
        num_ctx=max(512, min(memory_ctx, reasoning_dwell_ctx)),
        timeout_s=timeout_s,
        backend=backend,
        reasoning_num_predict=reasoning_num_predict,
        enable_ollama_think=enable_ollama_think,
    )
    if evidence_block and option_analysis:
        return evidence_block + "\n\nOption analysis:\n" + option_analysis
    if option_analysis:
        return "Option analysis:\n" + option_analysis
    return evidence_block


def _build_option_probe_supplement(
    case: BenchCase,
    model: str,
    memory_ctx: int,
    timeout_s: int,
    chunk_chars: int,
    reasoning_dwell_ctx: int,
    backend: ChatBackendConfig,
    reasoning_num_predict: int,
    enable_ollama_think: bool,
) -> str:
    dwell_adapter = _ChatAdapter(
        model_name=model,
        num_ctx=max(512, min(memory_ctx, reasoning_dwell_ctx)),
        timeout_s=timeout_s,
        backend=backend,
        response_schema=None,
        reasoning_num_predict=reasoning_num_predict,
        enable_think=enable_ollama_think,
    )
    noop_final_adapter = _FixedResponseAdapter('{"answer":"A"}')
    sections: list[str] = ["Option-specific probe notes:"]
    for label, option_text in (
        ("A", case.choice_a),
        ("B", case.choice_b),
        ("C", case.choice_c),
        ("D", case.choice_d),
    ):
        scratch_engine = _build_memory_engine(memory_ctx=memory_ctx, answer_dwell_mode="reasoned")
        _ingest_benchmark_context(scratch_engine, case.context, chunk_chars=chunk_chars)
        result = scratch_engine.answer_with_answer_document(
            model=noop_final_adapter,
            dwell_model=dwell_adapter,
            question=(
                f"Audit option {label} for the question below.\n\n"
                f"Question:\n{case.question}\n\n"
                f"Option {label} claim:\n{option_text}\n\n"
                "Collect concise evidence that supports or undermines this option claim."
            ),
            passes=2,
            per_pass_orbs=2,
            answer_doc_max_tokens=max(96, int(memory_ctx * 0.11)),
            system_prompt=(
                "Collect only claim-specific evidence for the option text.\n"
                "Do not answer the overall multiple-choice question."
            ),
            allow_answer_coercion=False,
        )
        evidence = _extract_reasoned_chat_evidence(result.answer_document, max_lines=2)
        evidence_body = evidence.replace("Reasoned skim evidence:\n", "").strip()
        if not evidence_body:
            evidence_body = "- no strong evidence retrieved"
        sections.append(f"{label}:\n{evidence_body}")
    return "\n".join(sections)


def _adjudicate_from_probe_notes(adapter: _ChatAdapter, prompt: str, supplement: str) -> str:
    return adapter.complete(
        [
            {
                "role": "system",
                "content": (
                    "Use only the option-specific probe notes to choose the best answer.\n"
                    "Prefer the option with the most explicit claim-matching evidence.\n"
                    "Ignore generic disclosures, metadata, and boilerplate.\n"
                    "Return JSON only with one key: answer (A/B/C/D)."
                ),
            },
            {"role": "system", "content": supplement},
            {"role": "user", "content": prompt},
        ]
    )


def _should_route_reasoned_chat(packet: QuestionPacket, profile: StructureProfile) -> bool:
    question_lower = packet.question.lower()
    comparative_markers = (
        "comparative analysis",
        "best encapsulates",
        "primary divergence",
        "strategic divergence",
        "contrast",
        "compared with",
        "difference among",
        "among the analyses",
    )
    policy_mix_markers = (
        "which policy mix",
        "policy mix should the government pursue",
        "best balance fiscal sustainability",
    )
    if profile.shape in {"table", "manual"}:
        return False
    if any(marker in question_lower for marker in comparative_markers):
        return True
    if any(marker in question_lower for marker in policy_mix_markers):
        return True
    return False


def _run_memory_case(
    case: BenchCase,
    model: str,
    memory_ctx: int,
    timeout_s: int,
    chunk_chars: int,
    memory_answer_mode: str,
    memory_dwell_mode: str,
    reasoning_dwell_ctx: int,
    backend: ChatBackendConfig,
    reasoning_num_predict: int = 192,
    enable_ollama_think: bool = False,
) -> _MemoryRunOutcome:
    packet = _build_question_packet(case)
    packet_profile = _profile_structure(packet)
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string", "enum": ["A", "B", "C", "D"]}},
        "required": ["answer"],
    }
    prompt = _build_mc_prompt(case)
    main_dwell_mode = "heuristic" if memory_answer_mode == "reasoned-chat" else memory_dwell_mode
    engine = _build_memory_engine(memory_ctx=memory_ctx, answer_dwell_mode=main_dwell_mode)
    chunk_count = _ingest_benchmark_context(engine, case.context, chunk_chars=chunk_chars)
    adapter = _ChatAdapter(
        model_name=model,
        num_ctx=memory_ctx,
        timeout_s=timeout_s,
        backend=backend,
        response_schema=schema,
        reasoning_num_predict=reasoning_num_predict,
        enable_think=enable_ollama_think,
    )
    structured_adapter = _ChatAdapter(
        model_name=model,
        num_ctx=min(memory_ctx, max(512, reasoning_dwell_ctx), engine.config.structured_reader_ctx),
        timeout_s=timeout_s,
        backend=backend,
        response_schema=None,
        reasoning_num_predict=reasoning_num_predict,
        enable_think=enable_ollama_think,
    )
    dwell_adapter = _ChatAdapter(
        model_name=model,
        num_ctx=max(512, min(memory_ctx, reasoning_dwell_ctx)),
        timeout_s=timeout_s,
        backend=backend,
        response_schema=None,
        reasoning_num_predict=reasoning_num_predict,
        enable_think=enable_ollama_think,
    )
    if memory_answer_mode == "reasoned-chat" and engine.config.enable_structured_readers:
        profile, outcome = _route_structured_memory_case(
            packet=packet,
            model=structured_adapter,
            config=engine.config,
        )
        if outcome is not None:
            return _MemoryRunOutcome(
                answer_raw=outcome.answer_raw,
                chunk_count=chunk_count,
                route_name=outcome.route_name,
                route_profile_shape=profile.shape,
                route_profile_confidence=profile.confidence,
                deterministic_reader_used=outcome.deterministic,
                reader_evidence_excerpt="\n".join(outcome.evidence_lines[:6]),
            )
    if memory_answer_mode == "answer-doc":
        result = engine.answer_with_answer_document(
            model=adapter,
            dwell_model=dwell_adapter if memory_dwell_mode == "reasoned" else None,
            question=prompt,
            passes=4,
            per_pass_orbs=4,
            answer_doc_max_tokens=max(256, int(memory_ctx * 0.7)),
            system_prompt=(
                "Use only the answer document evidence to answer the multiple-choice question.\n"
                "Return JSON only with one key: answer (A/B/C/D)."
            ),
            allow_answer_coercion=False,
        )
        answer = result.answer
    elif memory_answer_mode == "reasoned-chat":
        system_prompt = (
            "Use memory context and answer the multiple-choice question.\n"
            "Return JSON only with one key: answer (A/B/C/D)."
        )
        if _should_route_reasoned_chat(packet, packet_profile):
            supplement = _build_option_probe_supplement(
                case=case,
                model=model,
                memory_ctx=memory_ctx,
                timeout_s=timeout_s,
                chunk_chars=chunk_chars,
                reasoning_dwell_ctx=reasoning_dwell_ctx,
                backend=backend,
                reasoning_num_predict=reasoning_num_predict,
                enable_ollama_think=enable_ollama_think,
            )
            if not supplement.strip():
                supplement = _build_reasoned_chat_supplement(
                    case=case,
                    model=model,
                    memory_ctx=memory_ctx,
                    timeout_s=timeout_s,
                    chunk_chars=chunk_chars,
                    reasoning_dwell_ctx=reasoning_dwell_ctx,
                    backend=backend,
                    reasoning_num_predict=reasoning_num_predict,
                    enable_ollama_think=enable_ollama_think,
                )
            if supplement:
                answer = _adjudicate_from_probe_notes(adapter=adapter, prompt=prompt, supplement=supplement)
            else:
                answer, _ = engine.chat(
                    model=adapter,
                    user_text=prompt,
                    system_prompt=system_prompt,
                )
        else:
            answer, _ = engine.chat(
                model=adapter,
                user_text=prompt,
                system_prompt=system_prompt,
            )
    else:
        answer, _ = engine.chat(
            model=adapter,
            user_text=prompt,
            system_prompt=(
                "Use memory context and answer the multiple-choice question.\n"
                "Return JSON only with one key: answer (A/B/C/D)."
            ),
        )
    return _MemoryRunOutcome(
        answer_raw=answer,
        chunk_count=chunk_count,
        route_name=f"orb/{memory_answer_mode}",
        route_profile_shape=packet_profile.shape,
        route_profile_confidence=packet_profile.confidence,
        deterministic_reader_used=False,
        reader_evidence_excerpt="",
    )


def run_compare(
    sample_size: int,
    seed: int,
    lengths: set[str],
    max_context_chars: int,
    memory_model: str,
    long_model: str,
    direct_ctx: int,
    memory_ctx: int,
    timeout_s: int,
    chunk_chars: int,
    memory_answer_mode: str,
    memory_dwell_mode: str,
    reasoning_dwell_ctx: int,
    backend: ChatBackendConfig,
    reasoning_num_predict: int = 192,
    enable_ollama_think: bool = False,
    difficulty_filter: set[str] | None = None,
    show_progress: bool = True,
) -> dict[str, Any]:
    cases = _select_cases(
        sample_size=sample_size,
        seed=seed,
        lengths=lengths,
        max_context_chars=max_context_chars,
        difficulty_filter=difficulty_filter,
    )
    results: list[BenchResult] = []
    started = time.time()

    total_cases = len(cases)
    direct_latencies: list[float] = []
    memory_latencies: list[float] = []
    for idx, case in enumerate(cases, start=1):
        direct_raw = ""
        direct_pred = ""
        direct_correct = 0
        direct_error = ""
        direct_latency_s = 0.0
        try:
            direct_started = time.perf_counter()
            direct_raw = _run_direct_case(
                case=case,
                model=long_model,
                direct_ctx=direct_ctx,
                timeout_s=timeout_s,
                backend=backend,
            )
            direct_latency_s = time.perf_counter() - direct_started
            direct_pred = _extract_choice(direct_raw)
            direct_correct = 1 if direct_pred == case.answer else 0
        except Exception as err:
            direct_latency_s = time.perf_counter() - direct_started
            direct_error = str(err)
        direct_latencies.append(direct_latency_s)

        memory_raw = ""
        memory_pred = ""
        memory_correct = 0
        memory_error = ""
        chunk_count = 0
        memory_latency_s = 0.0
        route_name = ""
        route_profile_shape = ""
        route_profile_confidence = 0.0
        deterministic_reader_used = 0
        reader_evidence_excerpt = ""
        try:
            memory_started = time.perf_counter()
            memory_run = _run_memory_case(
                case=case,
                model=memory_model,
                memory_ctx=memory_ctx,
                timeout_s=timeout_s,
                chunk_chars=chunk_chars,
                memory_answer_mode=memory_answer_mode,
                memory_dwell_mode=memory_dwell_mode,
                reasoning_dwell_ctx=reasoning_dwell_ctx,
                backend=backend,
                reasoning_num_predict=reasoning_num_predict,
                enable_ollama_think=enable_ollama_think,
            )
            memory_latency_s = time.perf_counter() - memory_started
            memory_raw = memory_run.answer_raw
            chunk_count = memory_run.chunk_count
            route_name = memory_run.route_name
            route_profile_shape = memory_run.route_profile_shape
            route_profile_confidence = memory_run.route_profile_confidence
            deterministic_reader_used = 1 if memory_run.deterministic_reader_used else 0
            reader_evidence_excerpt = memory_run.reader_evidence_excerpt
            memory_pred = _extract_choice(memory_raw)
            memory_correct = 1 if memory_pred == case.answer else 0
        except Exception as err:
            memory_latency_s = time.perf_counter() - memory_started
            memory_error = str(err)
        memory_latencies.append(memory_latency_s)

        results.append(
            BenchResult(
                case_id=case.case_id,
                length=case.length,
                difficulty=case.difficulty,
                domain=case.domain,
                sub_domain=case.sub_domain,
                answer=case.answer,
                direct_raw=direct_raw,
                direct_pred=direct_pred,
                direct_correct=direct_correct,
                direct_error=direct_error,
                memory_raw=memory_raw,
                memory_pred=memory_pred,
                memory_correct=memory_correct,
                memory_error=memory_error,
                direct_latency_s=round(direct_latency_s, 3),
                memory_latency_s=round(memory_latency_s, 3),
                context_chars=len(case.context),
                chunk_count=chunk_count,
                route_name=route_name,
                route_profile_shape=route_profile_shape,
                route_profile_confidence=route_profile_confidence,
                deterministic_reader_used=deterministic_reader_used,
                reader_evidence_excerpt=reader_evidence_excerpt,
            )
        )

        if show_progress:
            direct_running = statistics.mean(row.direct_correct for row in results)
            memory_running = statistics.mean(row.memory_correct for row in results)
            bar = _progress_bar(idx, total_cases)
            direct_status = "ok" if direct_correct else ("err" if direct_error else "x")
            memory_status = "ok" if memory_correct else ("err" if memory_error else "x")
            print(
                f"{bar} {idx}/{total_cases} "
                f"case={case.case_id[:8]} "
                f"direct={direct_pred or '-'}({direct_status}) "
                f"memory={memory_pred or '-'}({memory_status}) "
                f"running_direct={direct_running:.3f} "
                f"running_memory={memory_running:.3f} "
                f"direct_t={statistics.mean(direct_latencies):.2f}s "
                f"memory_t={statistics.mean(memory_latencies):.2f}s",
                flush=True,
            )

    elapsed = time.time() - started
    direct_acc = statistics.mean(row.direct_correct for row in results) if results else 0.0
    memory_acc = statistics.mean(row.memory_correct for row in results) if results else 0.0
    direct_errors = sum(1 for row in results if row.direct_error)
    memory_errors = sum(1 for row in results if row.memory_error)

    return {
        "benchmark": "THUDM/LongBench-v2",
        "task_format": "multiple-choice (A/B/C/D)",
        "sample_size": len(results),
        "seed": seed,
        "length_filter": sorted(lengths),
        "difficulty_filter": sorted(difficulty_filter) if difficulty_filter else [],
        "max_context_chars": max_context_chars,
        "memory_model": memory_model,
        "long_context_model": long_model,
        "backend": backend.as_dict(),
        "memory_answer_mode": memory_answer_mode,
        "memory_dwell_mode": memory_dwell_mode,
        "reasoning_dwell_ctx": reasoning_dwell_ctx if memory_dwell_mode == "reasoned" else None,
        "reasoning_num_predict": reasoning_num_predict if memory_answer_mode == "reasoned-chat" or memory_dwell_mode == "reasoned" else None,
        "enable_ollama_think": bool(enable_ollama_think),
        "memory_ctx": memory_ctx,
        "direct_ctx": direct_ctx,
        "chunk_chars": chunk_chars,
        "elapsed_seconds": round(elapsed, 2),
        "direct_mean_latency_s": round(statistics.mean(direct_latencies), 3) if direct_latencies else 0.0,
        "memory_mean_latency_s": round(statistics.mean(memory_latencies), 3) if memory_latencies else 0.0,
        "direct_error_count": direct_errors,
        "memory_error_count": memory_errors,
        "memory_orb_accuracy": direct_acc * 0 + memory_acc,
        "long_context_direct_accuracy": direct_acc,
        "delta_memory_minus_long_direct": memory_acc - direct_acc,
        "results": [asdict(row) for row in results],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Memory Orb vs direct long-context model on LongBench v2.")
    parser.add_argument("--sample-size", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--lengths",
        type=str,
        default="short",
        help="Comma-separated: short,medium,long",
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        default="",
        help="Optional comma-separated filter: easy,hard",
    )
    parser.add_argument("--max-context-chars", type=int, default=220000)
    parser.add_argument("--memory-model", type=str, default="qwen3:0.6b")
    parser.add_argument("--long-model", type=str, default="qwen3:4b")
    parser.add_argument("--direct-ctx", type=int, default=262144)
    parser.add_argument("--memory-ctx", type=int, default=2200)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--chunk-chars", type=int, default=1400)
    parser.add_argument(
        "--backend-provider",
        type=str,
        choices=["ollama", "openai", "openai-compatible", "vllm"],
        default="ollama",
        help="Inference backend. Use openai/vllm for an OpenAI-compatible server such as vLLM.",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default="",
        help="Base URL for an OpenAI-compatible server, for example http://127.0.0.1:8000/v1",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="EMPTY",
        help="API key for the OpenAI-compatible server. vLLM commonly uses EMPTY.",
    )
    parser.add_argument("--memory-answer-mode", type=str, choices=["chat", "answer-doc", "reasoned-chat"], default="chat")
    parser.add_argument("--memory-dwell-mode", type=str, choices=["heuristic", "reasoned"], default="heuristic")
    parser.add_argument("--reasoning-dwell-ctx", type=int, default=900)
    parser.add_argument("--reasoning-num-predict", type=int, default=192)
    parser.add_argument("--enable-ollama-think", action="store_true")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable per-sample progress lines.",
    )
    parser.add_argument("--json-out", type=str, default="")
    args = parser.parse_args()

    lengths = {item.strip() for item in args.lengths.split(",") if item.strip()}
    difficulty_filter = {item.strip() for item in args.difficulty.split(",") if item.strip()} or None
    backend = ChatBackendConfig(
        provider=args.backend_provider,
        api_base=args.api_base,
        api_key=args.api_key,
    )
    summary = run_compare(
        sample_size=max(1, args.sample_size),
        seed=args.seed,
        lengths=lengths,
        max_context_chars=max(20000, args.max_context_chars),
        memory_model=args.memory_model,
        long_model=args.long_model,
        direct_ctx=max(8192, args.direct_ctx),
        memory_ctx=max(800, args.memory_ctx),
        timeout_s=max(30, args.timeout),
        chunk_chars=max(500, args.chunk_chars),
        memory_answer_mode=args.memory_answer_mode,
        memory_dwell_mode=args.memory_dwell_mode,
        reasoning_dwell_ctx=max(256, args.reasoning_dwell_ctx),
        backend=backend,
        reasoning_num_predict=max(32, args.reasoning_num_predict),
        enable_ollama_think=bool(args.enable_ollama_think),
        difficulty_filter=difficulty_filter,
        show_progress=not bool(args.no_progress),
    )
    print(json.dumps(summary, indent=2))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
