from __future__ import annotations

import json
import math
import re
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from .adapters import Embedder, HashingEmbedder, ModelAdapter, SimpleTokenEstimator, TokenEstimator
from .types import AnswerDocumentResult, ContextPacket, FocusLatch, MemoryOrb, SemanticCard, Turn
from .utils import cosine_similarity, extract_anchors, merge_synopsis, mmr_rank, summarize_text


@dataclass(slots=True)
class MemoryOrbEngineConfig:
    context_max_tokens: int = 1600
    working_max_tokens: int = 900
    working_target_tokens: int = 680
    memory_budget_ratio: float = 0.45
    semantic_budget_ratio: float = 0.35
    max_retrieved_orbs: int = 8
    pulse_window_turns: int = 8
    recency_tau_turns: float = 200.0
    mmr_lambda: float = 0.76
    focus_budget_ratio: float = 0.22
    selective_pulse_threshold: float = 1.1
    focus_relevance_boost: float = 0.22
    focus_ttl_turns: int = 320
    focus_hold_turns: int = 6
    focus_hold_boost: float = 0.16
    focus_dwell_tau_turns: float = 20.0
    focus_dwell_boost: float = 0.14
    min_focus_orb_count: int = 1
    anchor_aliases: dict[str, list[str]] = field(default_factory=dict)
    stale_context_penalty: float = 0.14
    skim_area_max_chars: int = 300
    skim_top_area_count: int = 28
    skim_neighbor_window: int = 1
    skim_max_area_evals_per_pass: int = 14
    skim_cursor_chain_length: int = 4
    skim_source_repeat_penalty: float = 0.18
    area_dwell_base_reads: int = 1
    area_dwell_max_reads: int = 3
    area_dwell_trigger_score: float = 1.45
    area_dwell_neighbor_boost: float = 0.26
    area_dwell_complexity_threshold: float = 0.58
    area_dwell_complexity_boost: float = 0.32
    area_dwell_target_boost: float = 0.55
    answer_dwell_mode: str = "heuristic"
    reasoning_dwell_trigger_score: float = 1.85
    reasoning_dwell_min_complexity: float = 0.42
    reasoning_dwell_max_fact_lines: int = 4
    reasoning_dwell_evidence_tail: int = 3
    reasoning_dwell_max_areas: int = 3
    enable_structured_readers: bool = True
    structured_reader_ctx: int = 900
    table_reader_max_rows: int = 6000
    procedure_reader_max_sections: int = 6
    procedure_reader_max_claims_per_option: int = 6

    def __post_init__(self) -> None:
        if self.context_max_tokens <= 64:
            raise ValueError("context_max_tokens must be > 64")
        if self.working_target_tokens <= 0:
            raise ValueError("working_target_tokens must be > 0")
        if self.working_max_tokens <= self.working_target_tokens:
            raise ValueError("working_max_tokens must be greater than working_target_tokens")
        if not (0.0 < self.memory_budget_ratio < 0.9):
            raise ValueError("memory_budget_ratio must be in (0.0, 0.9)")
        if not (0.0 < self.semantic_budget_ratio < 1.0):
            raise ValueError("semantic_budget_ratio must be in (0.0, 1.0)")
        if self.max_retrieved_orbs <= 0:
            raise ValueError("max_retrieved_orbs must be > 0")
        if not (0.05 <= self.focus_budget_ratio <= 0.6):
            raise ValueError("focus_budget_ratio must be in [0.05, 0.6]")
        if self.selective_pulse_threshold <= 0.0:
            raise ValueError("selective_pulse_threshold must be > 0.0")
        if not (0.0 <= self.focus_relevance_boost <= 0.6):
            raise ValueError("focus_relevance_boost must be in [0.0, 0.6]")
        if self.focus_ttl_turns <= 0:
            raise ValueError("focus_ttl_turns must be > 0")
        if self.focus_hold_turns <= 0:
            raise ValueError("focus_hold_turns must be > 0")
        if not (0.0 <= self.focus_hold_boost <= 0.5):
            raise ValueError("focus_hold_boost must be in [0.0, 0.5]")
        if self.focus_dwell_tau_turns <= 0.0:
            raise ValueError("focus_dwell_tau_turns must be > 0.0")
        if not (0.0 <= self.focus_dwell_boost <= 0.5):
            raise ValueError("focus_dwell_boost must be in [0.0, 0.5]")
        if self.min_focus_orb_count < 0:
            raise ValueError("min_focus_orb_count must be >= 0")
        if not isinstance(self.anchor_aliases, dict):
            raise ValueError("anchor_aliases must be a dict[str, list[str]]")
        if not (0.0 <= self.stale_context_penalty <= 0.5):
            raise ValueError("stale_context_penalty must be in [0.0, 0.5]")
        if self.skim_area_max_chars < 80:
            raise ValueError("skim_area_max_chars must be >= 80")
        if self.skim_top_area_count <= 0:
            raise ValueError("skim_top_area_count must be > 0")
        if self.skim_neighbor_window < 0:
            raise ValueError("skim_neighbor_window must be >= 0")
        if self.skim_max_area_evals_per_pass <= 0:
            raise ValueError("skim_max_area_evals_per_pass must be > 0")
        if self.skim_cursor_chain_length <= 0:
            raise ValueError("skim_cursor_chain_length must be > 0")
        if not (0.0 <= self.skim_source_repeat_penalty <= 1.0):
            raise ValueError("skim_source_repeat_penalty must be in [0.0, 1.0]")
        if self.area_dwell_base_reads <= 0:
            raise ValueError("area_dwell_base_reads must be > 0")
        if self.area_dwell_max_reads < self.area_dwell_base_reads:
            raise ValueError("area_dwell_max_reads must be >= area_dwell_base_reads")
        if self.area_dwell_trigger_score <= 0.0:
            raise ValueError("area_dwell_trigger_score must be > 0.0")
        if not (0.0 <= self.area_dwell_neighbor_boost <= 1.0):
            raise ValueError("area_dwell_neighbor_boost must be in [0.0, 1.0]")
        if not (0.0 <= self.area_dwell_complexity_threshold <= 1.0):
            raise ValueError("area_dwell_complexity_threshold must be in [0.0, 1.0]")
        if not (0.0 <= self.area_dwell_complexity_boost <= 1.5):
            raise ValueError("area_dwell_complexity_boost must be in [0.0, 1.5]")
        if not (0.0 <= self.area_dwell_target_boost <= 2.0):
            raise ValueError("area_dwell_target_boost must be in [0.0, 2.0]")
        if self.answer_dwell_mode not in {"heuristic", "reasoned"}:
            raise ValueError("answer_dwell_mode must be 'heuristic' or 'reasoned'")
        if self.reasoning_dwell_trigger_score <= 0.0:
            raise ValueError("reasoning_dwell_trigger_score must be > 0.0")
        if not (0.0 <= self.reasoning_dwell_min_complexity <= 1.0):
            raise ValueError("reasoning_dwell_min_complexity must be in [0.0, 1.0]")
        if self.reasoning_dwell_max_fact_lines <= 0:
            raise ValueError("reasoning_dwell_max_fact_lines must be > 0")
        if self.reasoning_dwell_evidence_tail < 0:
            raise ValueError("reasoning_dwell_evidence_tail must be >= 0")
        if self.reasoning_dwell_max_areas <= 0:
            raise ValueError("reasoning_dwell_max_areas must be > 0")
        if self.structured_reader_ctx < 256:
            raise ValueError("structured_reader_ctx must be >= 256")
        if self.table_reader_max_rows <= 0:
            raise ValueError("table_reader_max_rows must be > 0")
        if self.procedure_reader_max_sections <= 0:
            raise ValueError("procedure_reader_max_sections must be > 0")
        if self.procedure_reader_max_claims_per_option <= 0:
            raise ValueError("procedure_reader_max_claims_per_option must be > 0")


@dataclass(slots=True)
class _SkimArea:
    area_id: str
    source_orb_id: str
    source_turn_ids: list[str]
    source_orb: MemoryOrb
    area_index: int
    text: str
    anchors: list[str]
    embedding: list[float]
    tokens: int
    created_turn: int
    salience: float
    complexity: float


@dataclass(slots=True)
class _AreaReadResult:
    facts: list[str]
    score: float
    reader: str
    followup_anchors: list[str] = field(default_factory=list)


class MemoryOrbEngine:
    """
    Model-agnostic memory manager for long-running chats.

    "Memory Orb" design:
    - Working memory: bounded recent turns.
    - Episodic memory (orbs): compressed chunks swapped out of working memory.
    - Semantic cards: lightweight anchor summaries built from recurrent orbs.
    - Focus latch: preserves current and predecessor sub-goals in a compact form.
    - Focus dwell: keeps high-signal memories "sticky" for multiple subsequent turns.
    - Context assembler: strict token budget split between recent turns and memory.
    """

    _SUBGOAL_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
    _SUBGOAL_MARKERS = (
        "goal",
        "subgoal",
        "objective",
        "priority",
        "need to",
        "must",
        "should",
        "focus",
        "deliver",
        "finish",
        "before",
    )
    _CODE_PATTERN = re.compile(r"\b[a-z]{2,}(?:-[a-z0-9]{2,}){2,}\b", flags=re.IGNORECASE)
    _FINAL_APPROVAL_MARKERS = (
        "finally approved",
        "approved for production",
        "approved",
        "authoritative production control",
        "production operations",
        "final governance session",
        "closeout memo",
    )
    _TRANSITIONAL_MARKERS = (
        "placeholder",
        "temporary",
        "pilot-only",
        "pilot",
        "legacy",
        "switched to",
        "not final",
        "retired",
    )
    _ORB_ANCHOR_BLACKLIST = {
        "assistant",
        "reply",
        "response",
        "stored",
        "system",
        "tool",
        "user",
    }
    _BOILERPLATE_MARKERS = (
        "important disclosures",
        "company-specific disclosures",
        "distribution of ratings",
        "investment banking",
        "makes a market",
        "client relationship",
        "non-securities services",
        "price target",
        "52 week range",
        "market cap",
        "price charts",
        "debt securities",
        "reuters:",
        "bloomberg:",
        "deutsche bank ag/",
        "goldman sachs does and seeks to do business",
        "monetary authority of singapore",
        "accredited investors",
        "expert investors",
        "institutional investors",
        "securities and futures act",
        "this material is issued and distributed",
        "accepts no liability",
        "for the contents hereof",
        "for the exclusive use of",
        "member of the singapore exchange",
        "subject company(ies)",
        "mci (p)",
    )
    _MC_OPTION_RE = re.compile(r"(?im)^\s*([ABCD])\.\s+(.+?)\s*$")

    def __init__(
        self,
        config: MemoryOrbEngineConfig | None = None,
        token_estimator: TokenEstimator | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self.config = config or MemoryOrbEngineConfig()
        self.token_estimator = token_estimator or SimpleTokenEstimator()
        self.embedder = embedder or HashingEmbedder()

        self.turn_index = 0
        self._working_turns: deque[Turn] = deque()
        self._working_tokens = 0
        self._orbs: list[MemoryOrb] = []
        self._semantic_cards: dict[str, SemanticCard] = {}
        self._focus_latch: FocusLatch | None = None
        self._skim_area_cache: dict[str, list[_SkimArea]] = {}
        self._last_route_audit: dict[str, Any] | None = None
        self._alias_forward, self._alias_reverse = self._build_alias_indices(self.config.anchor_aliases)

    @property
    def orb_count(self) -> int:
        return len(self._orbs)

    @property
    def working_tokens(self) -> int:
        return self._working_tokens

    def add_turn(self, role: str, content: str, metadata: dict[str, Any] | None = None) -> Turn:
        role = role.strip().lower()
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"unsupported role: {role}")
        content = (content or "").strip()
        token_count = self.token_estimator.count(content)
        self.turn_index += 1
        turn = Turn(
            turn_id=f"t-{self.turn_index}",
            turn_index=self.turn_index,
            role=role,
            content=content,
            tokens=token_count,
            metadata=metadata or {},
        )
        self._working_turns.append(turn)
        self._working_tokens += token_count
        if role == "user":
            self._update_focus_latch_from_turn(turn)
        self._apply_swap()
        return turn

    def chat(
        self,
        model: ModelAdapter,
        user_text: str,
        system_prompt: str = "You are a helpful assistant.",
    ) -> tuple[str, ContextPacket]:
        self.add_turn("user", user_text)
        packet = self.build_context(user_query=user_text, system_prompt=system_prompt)
        assistant_text = model.complete(packet.messages)
        self.add_turn("assistant", assistant_text)
        return assistant_text, packet

    def answer_with_answer_document(
        self,
        model: ModelAdapter,
        question: str,
        passes: int = 3,
        per_pass_orbs: int = 3,
        answer_doc_max_tokens: int | None = None,
        system_prompt: str = "You are a factual QA assistant.",
        allow_answer_coercion: bool = True,
        update_memory_state: bool = False,
        dwell_model: ModelAdapter | None = None,
    ) -> AnswerDocumentResult:
        """
        Multi-pass factual answering workflow:
        1) read question
        2) retrieve related facts
        3) append to answer document
        4) reread question + current answer document
        5) final answer from the answer document
        """
        question_clean = (question or "").strip()
        if not question_clean:
            raise ValueError("question must be non-empty")
        passes = max(1, passes)
        per_pass_orbs = max(1, per_pass_orbs)
        doc_budget = answer_doc_max_tokens or max(256, int(self.config.context_max_tokens * 0.62))

        question_profile = self._build_question_profile(question_clean)
        question_anchors = list(question_profile["anchors"])
        target_record = str(question_profile["target_record"])
        target_field = str(question_profile["target_field"])
        excluded_codes = list(question_profile["excluded_codes"])
        prefer_final_code = bool(question_profile["prefer_final_code"])
        active_anchors = list(question_anchors)
        if not active_anchors:
            active_anchors = self._expand_anchors(extract_anchors(question_clean), max_items=24)

        selected_ids: list[str] = []
        selected_orb_set: set[str] = set()
        used_area_ids: set[str] = set()
        evidence_lines: list[str] = []
        evidence_tokens = 0
        code_evidence_scores: dict[str, float] = {}
        running_query = question_clean
        completed_passes = 0
        screen_cursor: tuple[str, int] | None = None
        area_reader_model = dwell_model or model
        reasoned_area_reads_used = 0

        for _ in range(passes):
            memory_pool = self._build_question_memory_pool()
            skim_areas = self._build_skim_areas(memory_pool)
            area_plan, screen_cursor = self._plan_area_skim_sweep(
                areas=skim_areas,
                question=running_query,
                anchors=active_anchors,
                used_area_ids=used_area_ids,
                max_candidates=max(self.config.skim_top_area_count, per_pass_orbs * 3),
                max_evals=max(self.config.skim_max_area_evals_per_pass, per_pass_orbs * 2),
                screen_cursor=screen_cursor,
                target_record=target_record,
                target_field=target_field,
                excluded_codes=excluded_codes,
                prefer_final_code=prefer_final_code,
            )
            if not area_plan:
                break

            pass_added = 0
            for area, skim_score in area_plan:
                if area.area_id in used_area_ids:
                    continue
                dwell_reads = self._compute_area_dwell_reads(
                    skim_score=skim_score,
                    area=area,
                    target_record=target_record,
                    target_field=target_field,
                )
                area_result = self._read_area_heuristically(
                    area=area,
                    skim_score=skim_score,
                    dwell_reads=dwell_reads,
                    active_anchors=active_anchors,
                    target_record=target_record,
                    target_field=target_field,
                    excluded_codes=excluded_codes,
                    prefer_final_code=prefer_final_code,
                )
                if self._should_use_reasoned_dwell(
                    model=area_reader_model,
                    area=area,
                    skim_score=skim_score,
                    target_record=target_record,
                    target_field=target_field,
                    prefer_final_code=prefer_final_code,
                    reasoned_area_reads_used=reasoned_area_reads_used,
                    question=question_clean,
                ):
                    reasoned = self._read_area_with_reasoning(
                        model=area_reader_model,
                        question=question_clean,
                        area=area,
                        skim_score=skim_score,
                        active_anchors=active_anchors,
                        evidence_lines=evidence_lines,
                        target_record=target_record,
                        target_field=target_field,
                        excluded_codes=excluded_codes,
                        prefer_final_code=prefer_final_code,
                    )
                    if reasoned is not None and reasoned.score >= area_result.score:
                        area_result = reasoned
                        reasoned_area_reads_used += 1

                if not area_result.facts:
                    continue
                line = (
                    f"- pass={completed_passes + 1} area={area.area_id} source={area.source_orb_id} "
                    f"skim={skim_score:.2f} dwell={dwell_reads} reader={area_result.reader} "
                    f"complexity={area.complexity:.2f} score={area_result.score:.2f}: "
                    + " | ".join(area_result.facts)
                )
                line_tokens = self.token_estimator.count(line)
                if evidence_tokens + line_tokens > doc_budget and evidence_lines:
                    continue
                evidence_lines.append(line)
                evidence_tokens += line_tokens
                used_area_ids.add(area.area_id)
                if area.source_orb_id not in selected_orb_set:
                    selected_orb_set.add(area.source_orb_id)
                    selected_ids.append(area.source_orb_id)
                if update_memory_state:
                    self._apply_question_focus_dwell(area.source_orb, area_result.score)
                code_updates = self._score_code_candidates_from_text(
                    text=" ".join(area_result.facts),
                    excluded_codes=excluded_codes,
                    prefer_final_code=prefer_final_code,
                )
                for code, score in code_updates.items():
                    code_evidence_scores[code] = code_evidence_scores.get(code, 0.0) + score
                if area_result.followup_anchors:
                    active_anchors = self._merge_anchors(active_anchors, area_result.followup_anchors, max_items=42)
                pass_added += 1
                if pass_added >= per_pass_orbs:
                    break

            if pass_added == 0:
                break
            completed_passes += 1
            new_anchor_source = " ".join(evidence_lines[-pass_added:])
            active_anchors = self._expand_anchors(
                self._merge_anchors(question_anchors, extract_anchors(new_anchor_source, max_anchors=18), max_items=42),
                max_items=42,
            )
            running_query = (
                f"{question_clean}\n"
                "Known evidence so far (continue skim-and-dwell scan):\n"
                + "\n".join(evidence_lines[-min(4, len(evidence_lines)) :])
            )

        doc_lines = [
            "Answer Document:",
            f"- Question: {question_clean}",
            "- Evidence collected:",
        ]
        if evidence_lines:
            doc_lines.extend(evidence_lines)
        else:
            doc_lines.append("- No relevant evidence was retrieved from memory.")
        ranked_codes = sorted(code_evidence_scores.items(), key=lambda item: item[1], reverse=True)
        if ranked_codes:
            doc_lines.append("- Code candidate ranking:")
            for code, score in ranked_codes[:5]:
                doc_lines.append(f"- candidate={code.upper()} score={score:.2f}")
        answer_doc = "\n".join(doc_lines)

        steering = (
            "Use only the answer document evidence when possible.\n"
            "If evidence is insufficient, say you are uncertain."
        )
        if prefer_final_code:
            steering = (
                "Use only the answer document evidence when possible.\n"
                "Question asks for the final approved production code.\n"
                "Do not return retired, legacy, or replaced pilot codes.\n"
                "If evidence is insufficient, say you are uncertain."
            )
        if excluded_codes:
            steering += "\nExcluded replaced codes: " + ", ".join(code.upper() for code in excluded_codes)

        system_block = self._compose_system_message(
            [
                (system_prompt or "You are a factual QA assistant.").strip(),
                steering,
                answer_doc,
            ]
        )
        messages = [
            {"role": "system", "content": system_block},
            {"role": "user", "content": question_clean},
        ]
        total_tokens = sum(self.token_estimator.count(msg["content"]) for msg in messages)
        messages, total_tokens = self._enforce_context_cap(messages, total_tokens)
        answer = model.complete(messages)
        if prefer_final_code and ranked_codes and allow_answer_coercion:
            answer = self._coerce_answer_with_code_preference(answer, ranked_codes, excluded_codes)
        if update_memory_state:
            question_metadata = {
                "source": "answer_document_question",
                "exclude_from_pulse_map": True,
                "exclude_from_question_memory_pool": True,
            }
            answer_metadata = {
                "source": "answer_document",
                "exclude_from_pulse_map": True,
                "exclude_from_question_memory_pool": True,
            }
            # Persist the exchange only when the caller explicitly opts in.
            self.add_turn("user", question_clean, metadata=question_metadata)
            self.add_turn("assistant", answer, metadata=answer_metadata)

        return AnswerDocumentResult(
            question=question_clean,
            answer_document=answer_doc,
            answer=answer,
            selected_orb_ids=selected_ids,
            passes_completed=completed_passes,
            total_tokens=total_tokens,
        )

    def build_context(self, user_query: str, system_prompt: str = "") -> ContextPacket:
        system_prompt = (system_prompt or "").strip()
        system_tokens = self.token_estimator.count(system_prompt) if system_prompt else 0
        usable_budget = max(64, self.config.context_max_tokens - system_tokens)

        memory_budget = int(usable_budget * self.config.memory_budget_ratio)
        focus_budget = max(0, int(memory_budget * self.config.focus_budget_ratio))
        retrieval_budget = max(0, memory_budget - focus_budget)
        recent_budget = max(32, usable_budget - memory_budget)
        semantic_budget = int(retrieval_budget * self.config.semantic_budget_ratio)
        orb_budget = max(0, retrieval_budget - semantic_budget)

        query_anchors = self._expand_anchors(extract_anchors(user_query))
        focus_block, focus_tokens, focus_anchors = self._build_focus_block(query_anchors, focus_budget)

        semantic_cards = self._select_semantic_cards(query_anchors, focus_anchors, semantic_budget)
        semantic_tokens = sum(card.tokens for card in semantic_cards)

        selected_orbs = self._retrieve_orbs(
            user_query=user_query,
            query_anchors=query_anchors,
            focus_anchors=focus_anchors,
            budget_tokens=max(0, orb_budget),
        )

        memory_block = self._format_memory_block(semantic_cards, selected_orbs)
        memory_tokens = self.token_estimator.count(memory_block) if memory_block else 0
        memory_tokens += focus_tokens

        recent_turns = self._select_recent_turns(budget_tokens=recent_budget)
        recent_tokens = sum(turn.tokens for turn in recent_turns)

        messages: list[dict[str, str]] = []
        system_block = self._compose_system_message([system_prompt, focus_block, memory_block])
        if system_block:
            messages.append({"role": "system", "content": system_block})
        for turn in recent_turns:
            messages.append({"role": turn.role, "content": turn.content})
        query_text = user_query.strip()
        if query_text:
            last_user_content = ""
            for msg in reversed(messages):
                if msg["role"] == "user":
                    last_user_content = msg["content"].strip()
                    break
            if last_user_content != query_text:
                messages.append({"role": "user", "content": query_text})

        total_tokens = sum(self.token_estimator.count(msg["content"]) for msg in messages)
        messages, total_tokens = self._enforce_context_cap(messages, total_tokens)

        return ContextPacket(
            messages=messages,
            total_tokens=total_tokens,
            memory_tokens=memory_tokens,
            recent_tokens=recent_tokens,
            selected_orb_ids=[orb.orb_id for orb in selected_orbs],
            selected_anchors=[card.anchor for card in semantic_cards],
            latched_subgoal=self._focus_latch.summary if self._focus_latch else "",
            background_subgoal=self._focus_latch.predecessor_summary if self._focus_latch else "",
        )

    def _compose_system_message(self, sections: list[str]) -> str:
        cleaned = [section.strip() for section in sections if section and section.strip()]
        return "\n\n".join(cleaned)

    def stats(self) -> dict[str, int]:
        return {
            "turn_index": self.turn_index,
            "working_turn_count": len(self._working_turns),
            "working_tokens": self._working_tokens,
            "orb_count": len(self._orbs),
            "semantic_card_count": len(self._semantic_cards),
            "focus_latch_active": 1 if self._focus_latch else 0,
            "focus_pinned_orb_count": sum(1 for orb in self._orbs if orb.pinned_until_turn >= self.turn_index),
            "route_audit_active": 1 if self._last_route_audit else 0,
        }

    def record_route_audit(self, audit: dict[str, Any] | None) -> None:
        self._last_route_audit = dict(audit) if audit else None

    def consume_route_audit(self) -> dict[str, Any] | None:
        audit = self._last_route_audit
        self._last_route_audit = None
        return dict(audit) if audit else None

    def save_state(self, file_path: str | Path) -> None:
        path = Path(file_path)
        payload = {
            "config": asdict(self.config),
            "embedder_meta": self._embedder_metadata(),
            "turn_index": self.turn_index,
            "working_turns": [asdict(turn) for turn in self._working_turns],
            "orbs": [asdict(orb) for orb in self._orbs],
            "semantic_cards": [asdict(card) for card in self._semantic_cards.values()],
            "focus_latch": asdict(self._focus_latch) if self._focus_latch else None,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load_state(
        cls,
        file_path: str | Path,
        token_estimator: TokenEstimator | None = None,
        embedder: Embedder | None = None,
    ) -> "MemoryOrbEngine":
        path = Path(file_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        engine = cls(
            config=MemoryOrbEngineConfig(**payload["config"]),
            token_estimator=token_estimator,
            embedder=embedder,
        )
        engine.turn_index = int(payload.get("turn_index", 0))
        for raw_turn in payload.get("working_turns", []):
            turn = Turn(**raw_turn)
            engine._working_turns.append(turn)
            engine._working_tokens += turn.tokens
        engine._orbs = [MemoryOrb(**raw_orb) for raw_orb in payload.get("orbs", [])]
        for orb in engine._orbs:
            orb.embedding = engine.embedder.embed(orb.summary)
        raw_cards = payload.get("semantic_cards", [])
        if raw_cards:
            for raw_card in raw_cards:
                card = SemanticCard(**raw_card)
                engine._semantic_cards[card.anchor] = card
        else:
            engine._rebuild_semantic_cards()
        raw_focus = payload.get("focus_latch")
        if raw_focus:
            engine._focus_latch = FocusLatch(**raw_focus)
        return engine

    def _apply_swap(self) -> None:
        while self._working_tokens > self.config.working_max_tokens and self._working_turns:
            evicted: list[Turn] = []
            evicted_tokens = 0

            while self._working_turns and (self._working_tokens - evicted_tokens) > self.config.working_target_tokens:
                turn = self._working_turns.popleft()
                evicted.append(turn)
                evicted_tokens += turn.tokens
                if len(evicted) >= 2 and turn.role == "assistant":
                    break

            if not evicted:
                break

            self._working_tokens -= evicted_tokens
            orb = self._build_orb(evicted)
            self._orbs.append(orb)
            self._update_semantic_cards(orb)

    def _build_orb(self, turns: list[Turn]) -> MemoryOrb:
        joined = "\n".join(f"{turn.role}: {turn.content}" for turn in turns).strip()
        content_only = "\n".join(turn.content for turn in turns if turn.content).strip()
        summary_source = content_only or joined
        summary = summarize_text(summary_source, max_chars=340)
        anchors = self._filter_orb_anchors(self._expand_anchors(extract_anchors(summary)))
        if not anchors:
            anchors = self._filter_orb_anchors(
                self._expand_anchors(extract_anchors(summary_source, max_anchors=12))
            )

        lower_source = summary_source.lower()
        question_bias = 0.18 if "?" in summary_source else 0.0
        instruction_bias = 0.12 if any(k in lower_source for k in ("must", "should", "need", "todo")) else 0.0
        metadata_values: list[float] = []
        for turn in turns:
            raw = turn.metadata.get("importance")
            if raw is None:
                continue
            try:
                metadata_values.append(max(0.0, min(1.0, float(raw))))
            except (TypeError, ValueError):
                continue
        metadata_importance = (sum(metadata_values) / len(metadata_values)) if metadata_values else 0.0
        salience = min(1.0, 0.3 + question_bias + instruction_bias + 0.02 * len(anchors) + 0.28 * metadata_importance)

        return MemoryOrb(
            orb_id=f"orb-{uuid4().hex[:10]}",
            summary=summary,
            raw_excerpt=joined[:2600],
            anchors=anchors,
            embedding=self.embedder.embed(summary),
            source_turn_ids=[turn.turn_id for turn in turns],
            created_turn=self.turn_index,
            salience=salience,
            tokens=self.token_estimator.count(summary),
            focus_strength=min(1.0, 0.8 * metadata_importance),
            last_focus_turn=self.turn_index if metadata_importance >= 0.25 else 0,
            pinned_until_turn=self.turn_index + 4 if metadata_importance >= 0.7 else 0,
        )

    def _filter_orb_anchors(self, anchors: list[str]) -> list[str]:
        filtered: list[str] = []
        for anchor in anchors:
            lowered = anchor.lower()
            if lowered in self._ORB_ANCHOR_BLACKLIST:
                continue
            if lowered not in filtered:
                filtered.append(lowered)
        return filtered

    def _update_semantic_cards(self, orb: MemoryOrb) -> None:
        if not self._orb_is_semantic_card_eligible(orb):
            return
        for anchor in orb.anchors:
            existing = self._semantic_cards.get(anchor)
            if existing is None:
                synopsis = summarize_text(orb.summary, max_chars=260)
                card = SemanticCard(
                    anchor=anchor,
                    synopsis=synopsis,
                    frequency=1,
                    tokens=self.token_estimator.count(synopsis),
                    last_turn=self.turn_index,
                    evidence_orb_ids=[orb.orb_id],
                )
                self._semantic_cards[anchor] = card
                continue

            existing.frequency += 1
            existing.last_turn = self.turn_index
            if orb.orb_id not in existing.evidence_orb_ids and len(existing.evidence_orb_ids) < 8:
                existing.evidence_orb_ids.append(orb.orb_id)
            existing.synopsis = merge_synopsis(existing.synopsis, summarize_text(orb.summary, max_chars=180))
            existing.tokens = self.token_estimator.count(existing.synopsis)

    def _rebuild_semantic_cards(self) -> None:
        self._semantic_cards = {}
        for orb in self._orbs:
            self._update_semantic_cards(orb)

    def _embedder_metadata(self) -> dict[str, Any]:
        dimensions = getattr(self.embedder, "dimensions", None)
        return {
            "class": self.embedder.__class__.__name__,
            "module": self.embedder.__class__.__module__,
            "dimensions": int(dimensions) if isinstance(dimensions, int) else None,
        }

    def _orb_is_semantic_card_eligible(self, orb: MemoryOrb) -> bool:
        text = f"{orb.summary}\n{orb.raw_excerpt}".lower()
        mutable_markers = (
            "record=",
            "current_value=",
            "value=",
            "owner=",
            "priority=",
            "status=",
            "region=",
            "version=",
            "source=",
            "id=",
            "old_value=",
            "not_current=true",
        )
        if any(marker in text for marker in mutable_markers):
            return False
        if re.search(r"\b[a-z_][a-z0-9_]*=", text):
            return False
        if self._extract_code_tokens(text) and any(
            marker in text for marker in self._FINAL_APPROVAL_MARKERS + self._TRANSITIONAL_MARKERS
        ):
            return False
        return True

    def _pulse_map(self) -> dict[str, float]:
        pulses: dict[str, float] = {}
        if not self._working_turns:
            return pulses

        recent = list(self._working_turns)[-self.config.pulse_window_turns :]
        for distance, turn in enumerate(reversed(recent), start=1):
            if bool(turn.metadata.get("exclude_from_pulse_map")):
                continue
            decay = 1.0 / float(distance)
            role_weight = 1.25 if turn.role == "user" else 1.0
            anchors = self._expand_anchors(extract_anchors(turn.content, max_anchors=12))
            for anchor in anchors:
                pulses[anchor] = pulses.get(anchor, 0.0) + decay * role_weight
        return pulses

    def _select_semantic_cards(
        self, query_anchors: list[str], focus_anchors: list[str], budget_tokens: int
    ) -> list[SemanticCard]:
        if budget_tokens <= 0 or not self._semantic_cards:
            return []

        candidates: list[tuple[float, SemanticCard]] = []
        qset = set(query_anchors)
        fset = set(focus_anchors)
        for card in self._semantic_cards.values():
            direct = 1.0 if card.anchor in qset else 0.0
            focus_direct = 1.0 if card.anchor in fset else 0.0
            recency = math.exp(-max(0, self.turn_index - card.last_turn) / max(1.0, self.config.recency_tau_turns))
            score = direct * 2.1 + focus_direct * 1.45 + 0.35 * math.log1p(card.frequency) + 0.2 * recency
            if direct > 0.0 or focus_direct > 0.0 or card.frequency >= 2:
                candidates.append((score, card))

        selected: list[SemanticCard] = []
        used_tokens = 0
        for _, card in sorted(candidates, key=lambda item: item[0], reverse=True):
            if used_tokens + card.tokens > budget_tokens:
                continue
            selected.append(card)
            used_tokens += card.tokens
        return selected

    def _retrieve_orbs(
        self,
        user_query: str,
        query_anchors: list[str],
        focus_anchors: list[str],
        budget_tokens: int,
    ) -> list[MemoryOrb]:
        if budget_tokens <= 0 or not self._orbs:
            return []

        pulses = self._pulse_map()
        query_embedding = self.embedder.embed(user_query)
        qset = set(query_anchors)
        fset = set(focus_anchors)
        pulse_hits = {anchor for anchor, weight in pulses.items() if weight >= self.config.selective_pulse_threshold}
        query_lower = user_query.lower()
        action_query = any(
            marker in query_lower
            for marker in ("step", "steps", "execute", "execution", "next", "plan", "fix", "mitigation", "action")
        )

        candidates: list[MemoryOrb] = []
        scores: list[float] = []
        vectors: list[list[float]] = []
        focus_signals: list[float] = []
        for orb in self._orbs:
            semantic_score = cosine_similarity(query_embedding, orb.embedding)
            resonance = 0.0
            if orb.anchors:
                resonance = sum(pulses.get(anchor, 0.0) for anchor in orb.anchors) / math.sqrt(len(orb.anchors))
            overlap = 0.0
            if qset and orb.anchors:
                overlap = len(qset.intersection(orb.anchors)) / len(qset.union(orb.anchors))
            focus_overlap = 0.0
            if fset and orb.anchors:
                focus_overlap = len(fset.intersection(orb.anchors)) / len(fset.union(orb.anchors))
            selective_gate = 0.0
            if pulse_hits and orb.anchors:
                selective_hits = len(pulse_hits.intersection(orb.anchors))
                selective_gate = min(1.0, selective_hits / 2.0)
            hold_bonus = self.config.focus_hold_boost if orb.pinned_until_turn >= self.turn_index else 0.0
            dwell_decay = math.exp(
                -max(0, self.turn_index - orb.last_focus_turn) / max(1.0, self.config.focus_dwell_tau_turns)
            )
            dwell_bonus = self.config.focus_dwell_boost * orb.focus_strength * dwell_decay
            importance_bonus = 0.12 * orb.focus_strength
            orb_text_lower = orb.summary.lower()
            stale_penalty = 0.0
            if action_query and any(marker in orb_text_lower for marker in ("historical", "archived", "not tied")):
                stale_penalty = self.config.stale_context_penalty
            recency = math.exp(
                -max(0, self.turn_index - orb.created_turn) / max(1.0, self.config.recency_tau_turns)
            )
            usage_bonus = 0.06 * math.log1p(orb.accesses)
            synergy = 0.0
            if focus_overlap > 0.0 and selective_gate > 0.0:
                synergy = 1.0
            score = (
                0.34 * semantic_score
                + 0.2 * resonance
                + 0.1 * overlap
                + self.config.focus_relevance_boost * focus_overlap
                + 0.14 * selective_gate
                + 0.08 * synergy
                + 0.08 * recency
                + 0.12 * orb.salience
                + hold_bonus
                + dwell_bonus
                + importance_bonus
                + usage_bonus
                - stale_penalty
            )
            focus_signal = max(focus_overlap, selective_gate, orb.focus_strength)
            candidates.append(orb)
            scores.append(score)
            vectors.append(orb.embedding)
            focus_signals.append(focus_signal)

        ranked_indices = mmr_rank(
            vectors=vectors,
            base_scores=scores,
            max_items=self.config.max_retrieved_orbs,
            lambda_weight=self.config.mmr_lambda,
        )

        selected_indices: list[int] = []
        used_tokens = 0
        for idx in ranked_indices:
            orb = candidates[idx]
            if used_tokens + orb.tokens > budget_tokens:
                continue
            selected_indices.append(idx)
            used_tokens += orb.tokens

        min_focus_needed = max(0, self.config.min_focus_orb_count)
        if min_focus_needed > 0:
            def focus_count(indices: list[int]) -> int:
                return sum(1 for i in indices if focus_signals[i] >= 0.2 or candidates[i].pinned_until_turn >= self.turn_index)

            ranked_focus = sorted(
                range(len(candidates)),
                key=lambda i: (focus_signals[i], scores[i]),
                reverse=True,
            )
            while focus_count(selected_indices) < min_focus_needed:
                added = False
                for idx in ranked_focus:
                    if idx in selected_indices:
                        continue
                    if focus_signals[idx] <= 0.0:
                        continue
                    orb = candidates[idx]
                    if used_tokens + orb.tokens <= budget_tokens:
                        selected_indices.append(idx)
                        used_tokens += orb.tokens
                        added = True
                        break

                    replace_idx = -1
                    replace_score = float("inf")
                    for cur in selected_indices:
                        cur_is_focus = focus_signals[cur] >= 0.2 or candidates[cur].pinned_until_turn >= self.turn_index
                        if cur_is_focus:
                            continue
                        if scores[cur] < replace_score:
                            replace_score = scores[cur]
                            replace_idx = cur
                    if replace_idx >= 0:
                        new_total = used_tokens - candidates[replace_idx].tokens + orb.tokens
                        if new_total <= budget_tokens:
                            selected_indices.remove(replace_idx)
                            selected_indices.append(idx)
                            used_tokens = new_total
                            added = True
                            break
                if not added:
                    break

        selected: list[MemoryOrb] = []
        for idx in selected_indices:
            orb = candidates[idx]
            focus_signal = focus_signals[idx]
            if focus_signal >= 0.2:
                orb.focus_strength = min(1.0, 0.72 * orb.focus_strength + 0.42 * focus_signal + 0.18)
                orb.last_focus_turn = self.turn_index
                hold_extension = self.config.focus_hold_turns + (1 if focus_signal >= 0.55 else 0)
                orb.pinned_until_turn = max(orb.pinned_until_turn, self.turn_index + hold_extension)
            elif orb.focus_strength > 0.0:
                orb.focus_strength = max(0.0, 0.86 * orb.focus_strength)
            orb.accesses += 1
            orb.last_access_turn = self.turn_index
            selected.append(orb)
        return selected

    def _format_memory_block(self, cards: list[SemanticCard], orbs: list[MemoryOrb]) -> str:
        if not cards and not orbs:
            return ""

        lines = [
            "Memory Orb Sync:",
            "- Treat memory as context hints, not guaranteed truth.",
            "- Prioritize recent user intent when memory conflicts.",
        ]
        if cards:
            lines.append("Semantic anchors:")
            for card in cards:
                lines.append(f"- {card.anchor}: {card.synopsis}")
        if orbs:
            lines.append("Episodic orbs:")
            for orb in orbs:
                anchor_preview = ", ".join(orb.anchors[:6])
                lines.append(f"- {orb.orb_id} | anchors={anchor_preview} | {orb.summary}")
        return "\n".join(lines)

    def _build_focus_block(self, query_anchors: list[str], budget_tokens: int) -> tuple[str, int, list[str]]:
        if budget_tokens <= 0 or self._focus_latch is None:
            return "", 0, []
        latch = self._focus_latch
        if (self.turn_index - latch.last_refresh_turn) > self.config.focus_ttl_turns:
            return "", 0, []

        qset = set(query_anchors)
        primary_summary = latch.summary
        if qset and latch.anchors:
            overlap = len(qset.intersection(latch.anchors))
            if overlap == 0:
                primary_summary = summarize_text(primary_summary, max_chars=160)
            else:
                primary_summary = summarize_text(primary_summary, max_chars=220)
        else:
            primary_summary = summarize_text(primary_summary, max_chars=190)

        background_summary = summarize_text(latch.predecessor_summary, max_chars=140) if latch.predecessor_summary else ""
        lines = [
            "Selective Attention Latch:",
            "- Keep this active objective in mind while focusing on local context.",
            f"- Primary sub-goal: {primary_summary}",
        ]
        if background_summary:
            lines.append(f"- Background sub-goal: {background_summary}")

        block = "\n".join(lines)
        token_count = self.token_estimator.count(block)
        if token_count <= budget_tokens:
            return block, token_count, self._merge_anchors(latch.anchors, latch.predecessor_anchors)

        max_chars = max(80, int(budget_tokens * 4))
        trimmed_primary = summarize_text(primary_summary, max_chars=max_chars)
        trimmed_background = summarize_text(background_summary, max_chars=max(0, max_chars // 2))
        lines = [
            "Selective Attention Latch:",
            f"- Primary sub-goal: {trimmed_primary}",
        ]
        if trimmed_background:
            lines.append(f"- Background sub-goal: {trimmed_background}")
        block = "\n".join(lines)
        token_count = self.token_estimator.count(block)
        return block, token_count, self._merge_anchors(latch.anchors, latch.predecessor_anchors)

    def _update_focus_latch_from_turn(self, turn: Turn) -> None:
        candidate = self._extract_subgoal_candidate(turn.content)
        if candidate is None:
            return
        summary, anchors, confidence, marker_strength = candidate
        if not anchors:
            return
        if marker_strength == 0:
            if self._focus_latch is None:
                return
            implicit_overlap = len(set(self._focus_latch.anchors).intersection(anchors))
            if implicit_overlap == 0:
                return
            confidence *= 0.78

        if self._focus_latch is None:
            self._focus_latch = FocusLatch(
                summary=summary,
                anchors=anchors,
                confidence=confidence,
                created_turn=turn.turn_index,
                last_refresh_turn=turn.turn_index,
            )
            return

        current = self._focus_latch
        overlap = len(set(current.anchors).intersection(anchors))
        should_rotate = marker_strength >= 2 and overlap == 0
        if should_rotate:
            self._focus_latch = FocusLatch(
                summary=summary,
                anchors=anchors,
                confidence=confidence,
                created_turn=turn.turn_index,
                last_refresh_turn=turn.turn_index,
                predecessor_summary=current.summary,
                predecessor_anchors=current.anchors,
            )
            return

        current.summary = merge_synopsis(current.summary, summary, max_chars=280)
        current.anchors = self._merge_anchors(current.anchors, anchors)
        current.last_refresh_turn = turn.turn_index
        current.confidence = min(1.0, 0.65 * current.confidence + 0.35 * confidence)

    def _extract_subgoal_candidate(self, text: str) -> tuple[str, list[str], float, int] | None:
        cleaned = " ".join((text or "").split())
        if not cleaned:
            return None

        sentences = [s.strip() for s in self._SUBGOAL_SPLIT_RE.split(cleaned) if s.strip()]
        if not sentences:
            sentences = [cleaned]

        best_sentence = ""
        marker_strength = 0
        for sentence in sentences:
            lower = sentence.lower()
            score = 0
            for marker in self._SUBGOAL_MARKERS:
                if marker in lower:
                    score += 1
            if score > marker_strength:
                marker_strength = score
                best_sentence = sentence

        if marker_strength == 0:
            best_sentence = summarize_text(cleaned, max_chars=170)
        summary = summarize_text(best_sentence, max_chars=220)
        anchors = self._expand_anchors(extract_anchors(summary, max_anchors=10))
        confidence = min(1.0, 0.38 + 0.12 * marker_strength + 0.03 * len(anchors))
        return summary, anchors, confidence, marker_strength

    def _build_question_profile(self, question: str) -> dict[str, Any]:
        clean = " ".join((question or "").split())
        lower = clean.lower()
        anchors = self._expand_anchors(extract_anchors(clean, max_anchors=18), max_items=30)

        target_record = ""
        for pattern in (r"\brecord\s*=\s*([a-z0-9._\-]+)", r"\brecord\s+([a-z0-9._\-]+)"):
            match = re.search(pattern, lower)
            if match:
                target_record = match.group(1).strip(".,;:()[]{}")
                break

        target_field = ""
        known_fields = ("current_value", "value", "status", "owner", "priority", "region", "version", "source", "id")
        for field in known_fields:
            if re.search(rf"\b{re.escape(field)}\b", lower):
                target_field = field
                break
        if not target_field:
            match = re.search(r"\bwhat\s+is\s+the\s+([a-z_][a-z0-9_]*)", lower)
            if match:
                target_field = match.group(1)
        prefer_final_code = any(
            marker in lower
            for marker in (
                "final",
                "finally approved",
                "production control code",
                "approved on",
                "authoritative",
                "after replacing",
            )
        )
        excluded_codes: list[str] = []
        replacing_match = re.search(r"after\s+replacing\s+([a-z0-9\-]{6,})", lower)
        if replacing_match:
            token = replacing_match.group(1).strip(".,;:()[]{}")
            excluded_codes.append(token)
        for code in self._extract_code_tokens(lower):
            if code not in excluded_codes and "replacing" in lower:
                excluded_codes.append(code)

        merged = list(anchors)
        if target_record:
            record_anchors = self._expand_anchors(extract_anchors(target_record, max_anchors=6), max_items=8)
            merged = self._merge_anchors(merged, record_anchors, max_items=34)
        if target_field:
            field_anchors = self._expand_anchors(extract_anchors(target_field, max_anchors=6), max_items=8)
            merged = self._merge_anchors(merged, field_anchors, max_items=34)

        return {
            "anchors": merged,
            "target_record": target_record,
            "target_field": target_field,
            "excluded_codes": excluded_codes,
            "prefer_final_code": prefer_final_code,
        }

    def _extract_multiple_choice_options(self, question: str) -> dict[str, str]:
        options: dict[str, str] = {}
        for match in self._MC_OPTION_RE.finditer(question or ""):
            label = match.group(1).upper()
            text = " ".join(match.group(2).split()).strip()
            if label in {"A", "B", "C", "D"} and text:
                options[label] = text
        return options

    def _is_comparative_question(self, question: str) -> bool:
        lower = (question or "").lower()
        markers = (
            "comparative analysis",
            "primary divergence",
            "divergence among",
            "strategic divergence",
            "best encapsulates",
            "contrast",
            "compared with",
            "difference among",
            "diverge",
            "which of the following best",
        )
        return any(marker in lower for marker in markers)

    def _estimate_area_noise(self, text: str) -> float:
        lower = (text or "").lower()
        hits = sum(1 for marker in self._BOILERPLATE_MARKERS if marker in lower)
        if hits <= 0:
            return 0.0
        return min(1.0, 0.16 * hits + 0.12)

    def _segment_text_into_scan_areas(self, text: str, max_chars: int) -> list[str]:
        clean = " ".join((text or "").split())
        if not clean:
            return []
        if len(clean) <= max_chars:
            return [clean]

        parts = [p.strip() for p in re.split(r"\n+|(?<=[.!?])\s+", text) if p.strip()]
        if not parts:
            parts = [clean]

        areas: list[str] = []
        current: list[str] = []
        used = 0
        for part in parts:
            block = " ".join(part.split())
            if not block:
                continue
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", block) if s.strip()]
            if not sentences:
                sentences = [block]

            for sent in sentences:
                wrapped_parts = self._hard_wrap_scan_block(sent, max_chars=max_chars)
                for wrapped in wrapped_parts:
                    need = len(wrapped) + (1 if current else 0)
                    if current and (used + need) > max_chars:
                        areas.append(" ".join(current).strip())
                        current = [wrapped]
                        used = len(wrapped)
                    else:
                        current.append(wrapped)
                        used += need
        if current:
            areas.append(" ".join(current).strip())
        return [area for area in areas if area]

    def _hard_wrap_scan_block(self, text: str, max_chars: int) -> list[str]:
        clean = " ".join((text or "").split())
        if not clean:
            return []
        if len(clean) <= max_chars:
            return [clean]

        wrapped: list[str] = []
        remaining = clean
        while len(remaining) > max_chars:
            split_at = remaining.rfind(" ", 0, max_chars)
            if split_at < int(max_chars * 0.4):
                split_at = max_chars
            chunk = remaining[:split_at].strip()
            if chunk:
                wrapped.append(chunk)
            remaining = remaining[split_at:].strip()
        if remaining:
            wrapped.append(remaining)
        return wrapped

    def _estimate_area_complexity(self, text: str) -> float:
        clean = " ".join((text or "").split())
        if not clean:
            return 0.0

        token_count = len(clean.split())
        sentence_count = len([s for s in re.split(r"(?<=[.!?])\s+", clean) if s.strip()])
        clause_count = len([s for s in re.split(r"[,:;]", clean) if s.strip()])
        numeric_hits = len(re.findall(r"\d", clean))
        code_hits = len(self._extract_code_tokens(clean))
        anchor_count = len(extract_anchors(clean, max_anchors=28))

        complexity = (
            0.26 * min(1.0, token_count / 90.0)
            + 0.2 * min(1.0, sentence_count / 5.0)
            + 0.2 * min(1.0, clause_count / 7.0)
            + 0.14 * min(1.0, numeric_hits / 10.0)
            + 0.1 * min(1.0, code_hits / 4.0)
            + 0.1 * min(1.0, anchor_count / 18.0)
        )
        return max(0.0, min(1.0, complexity))

    def _build_skim_areas(self, pool: list[MemoryOrb]) -> list[_SkimArea]:
        if not pool:
            return []
        areas: list[_SkimArea] = []
        max_per_source = max(6, min(18, self.config.skim_top_area_count))
        for orb in pool:
            cache_key = self._skim_area_cache_key(orb)
            if cache_key is not None and cache_key in self._skim_area_cache:
                areas.extend(self._clone_cached_skim_areas(self._skim_area_cache[cache_key], orb))
                continue
            source_text = (orb.raw_excerpt or orb.summary or "").strip()
            if not source_text:
                continue
            segments = self._segment_text_into_scan_areas(source_text, max_chars=self.config.skim_area_max_chars)
            if not segments:
                segments = [summarize_text(source_text, max_chars=self.config.skim_area_max_chars)]
            built_areas: list[_SkimArea] = []
            for idx, segment in enumerate(segments[:max_per_source]):
                anchors = self._expand_anchors(extract_anchors(segment, max_anchors=12), max_items=20)
                if not anchors:
                    anchors = list(orb.anchors[:10])
                built_areas.append(
                    _SkimArea(
                        area_id=f"{orb.orb_id}-area-{idx}",
                        source_orb_id=orb.orb_id,
                        source_turn_ids=list(orb.source_turn_ids),
                        source_orb=orb,
                        area_index=idx,
                        text=segment,
                        anchors=anchors,
                        embedding=self.embedder.embed(segment),
                        tokens=self.token_estimator.count(segment),
                        created_turn=orb.created_turn,
                        salience=orb.salience,
                        complexity=self._estimate_area_complexity(segment),
                    )
                )
            if cache_key is not None:
                self._skim_area_cache[cache_key] = built_areas
            areas.extend(self._clone_cached_skim_areas(built_areas, orb))
        return areas

    def _skim_area_cache_key(self, orb: MemoryOrb) -> str | None:
        if orb.orb_id.startswith("working-"):
            return None
        source_text = (orb.raw_excerpt or orb.summary or "").strip()
        return f"{orb.orb_id}:{orb.created_turn}:{len(source_text)}:{self._embedder_metadata()['class']}"

    def _clone_cached_skim_areas(self, cached: list[_SkimArea], orb: MemoryOrb) -> list[_SkimArea]:
        cloned: list[_SkimArea] = []
        for area in cached:
            cloned.append(
                _SkimArea(
                    area_id=area.area_id,
                    source_orb_id=area.source_orb_id,
                    source_turn_ids=list(area.source_turn_ids),
                    source_orb=orb,
                    area_index=area.area_index,
                    text=area.text,
                    anchors=list(area.anchors),
                    embedding=list(area.embedding),
                    tokens=area.tokens,
                    created_turn=area.created_turn,
                    salience=orb.salience,
                    complexity=area.complexity,
                )
            )
        return cloned

    def _plan_area_skim_sweep(
        self,
        areas: list[_SkimArea],
        question: str,
        anchors: list[str],
        used_area_ids: set[str],
        max_candidates: int,
        max_evals: int,
        screen_cursor: tuple[str, int] | None = None,
        target_record: str = "",
        target_field: str = "",
        excluded_codes: list[str] | None = None,
        prefer_final_code: bool = False,
    ) -> tuple[list[tuple[_SkimArea, float]], tuple[str, int] | None]:
        if not areas or max_candidates <= 0 or max_evals <= 0:
            return [], screen_cursor

        query_embedding = self.embedder.embed(question)
        anchor_set = set(anchors)
        mc_options = self._extract_multiple_choice_options(question)
        option_anchor_set: set[str] = set()
        if mc_options:
            option_text = " ".join(mc_options.values())
            option_anchor_set = set(self._expand_anchors(extract_anchors(option_text, max_anchors=22), max_items=32))
        comparative_question = self._is_comparative_question(question)
        excluded = {code.lower() for code in (excluded_codes or [])}
        by_source: dict[str, list[_SkimArea]] = {}
        for area in areas:
            by_source.setdefault(area.source_orb_id, []).append(area)
        for source_areas in by_source.values():
            source_areas.sort(key=lambda item: item.area_index)

        ranked: list[tuple[float, _SkimArea]] = []
        for area in areas:
            if area.area_id in used_area_ids:
                continue
            semantic_score = cosine_similarity(query_embedding, area.embedding)
            overlap = 0.0
            if anchor_set and area.anchors:
                overlap = len(anchor_set.intersection(area.anchors)) / max(1, len(anchor_set))
            option_overlap = 0.0
            if option_anchor_set and area.anchors:
                option_overlap = len(option_anchor_set.intersection(area.anchors)) / max(1, len(option_anchor_set))
            text_lower = area.text.lower()
            noise_penalty = 0.74 * self._estimate_area_noise(area.text)
            record_bonus = 1.0 if target_record and target_record in text_lower else 0.0
            field_bonus = 0.9 if target_field and target_field in text_lower else 0.0
            complexity_bonus = 0.16 * area.complexity
            approval_hits = sum(1 for marker in self._FINAL_APPROVAL_MARKERS if marker in text_lower)
            transitional_hits = sum(1 for marker in self._TRANSITIONAL_MARKERS if marker in text_lower)
            approval_bonus = (0.36 * approval_hits) if prefer_final_code else 0.0
            transition_penalty = (0.26 * transitional_hits) if prefer_final_code else 0.0
            excluded_penalty = 0.0
            if excluded:
                excluded_penalty = 0.9 * sum(1 for code in excluded if code in text_lower)

            cursor_bonus = 0.0
            if screen_cursor and screen_cursor[0] == area.source_orb_id:
                distance = abs(screen_cursor[1] - area.area_index)
                if distance <= max(1, self.config.skim_neighbor_window + 1):
                    cursor_bonus = self.config.area_dwell_neighbor_boost * (1.0 / (1.0 + distance))

            recency = math.exp(-max(0, self.turn_index - area.created_turn) / max(1.0, self.config.recency_tau_turns))
            score = (
                0.44 * semantic_score
                + 0.28 * overlap
                + 0.24 * option_overlap
                + 0.16 * area.salience
                + 0.08 * recency
                + 0.33 * record_bonus
                + 0.22 * field_bonus
                + complexity_bonus
                + approval_bonus
                + cursor_bonus
                - transition_penalty
                - excluded_penalty
                - noise_penalty
            )
            ranked.append((score, area))

        if not ranked:
            return [], screen_cursor
        ranked.sort(key=lambda item: item[0], reverse=True)
        seed_limit = max(1, min(max_candidates, len(ranked)))
        seed_pool = ranked[:seed_limit]
        if comparative_question and len(by_source) > 1:
            diverse_seeds: list[tuple[float, _SkimArea]] = []
            seen_sources: set[str] = set()
            for score, area in ranked:
                if area.source_orb_id in seen_sources:
                    continue
                diverse_seeds.append((score + 0.04, area))
                seen_sources.add(area.source_orb_id)
                if len(diverse_seeds) >= seed_limit:
                    break
            merged_seed_pool: list[tuple[float, _SkimArea]] = []
            seen_areas: set[str] = set()
            for score, area in diverse_seeds + seed_pool:
                if area.area_id in seen_areas:
                    continue
                merged_seed_pool.append((score, area))
                seen_areas.add(area.area_id)
                if len(merged_seed_pool) >= seed_limit:
                    break
            seed_pool = merged_seed_pool
        seed_vectors = [area.embedding for _, area in seed_pool]
        seed_scores = [score for score, _ in seed_pool]
        seed_order = mmr_rank(seed_vectors, seed_scores, max_items=len(seed_pool), lambda_weight=self.config.mmr_lambda)
        if not seed_order:
            seed_order = list(range(len(seed_pool)))
        seeds = [seed_pool[idx] for idx in seed_order if 0 <= idx < len(seed_pool)]
        score_by_area_id = {area.area_id: score for score, area in ranked}

        sweep: list[tuple[_SkimArea, float]] = []
        local_used: set[str] = set()
        source_hits: dict[str, int] = {}
        window = max(0, self.config.skim_neighbor_window)
        if comparative_question and len(by_source) > 1:
            source_cap = 2
        else:
            source_cap = max(2, max_evals // 2) if len(by_source) > 1 else max_evals

        def push_area(area: _SkimArea, score: float) -> None:
            if area.area_id in local_used or area.area_id in used_area_ids:
                return
            hits = source_hits.get(area.source_orb_id, 0)
            if hits >= source_cap and len(by_source) > 1:
                return
            penalty = self.config.skim_source_repeat_penalty * hits
            sweep.append((area, score - penalty))
            local_used.add(area.area_id)
            source_hits[area.source_orb_id] = hits + 1

        if screen_cursor:
            cursor_source, cursor_idx = screen_cursor
            source_areas = by_source.get(cursor_source, [])
            index_lookup = {area.area_index: area for area in source_areas}
            offsets: list[int] = [0]
            chain = max(1, self.config.skim_cursor_chain_length)
            for distance in range(1, chain + 1):
                offsets.append(distance)
                offsets.append(-distance)
            for offset in offsets:
                area = index_lookup.get(cursor_idx + offset)
                if area is None:
                    continue
                base_score = score_by_area_id.get(area.area_id, -0.25)
                cursor_bonus = self.config.area_dwell_neighbor_boost * (1.0 / (1.0 + abs(offset)))
                push_area(area, base_score + cursor_bonus)
                if len(sweep) >= max_evals:
                    break

        for base_score, seed in seeds:
            if len(sweep) >= max_evals:
                break
            push_area(seed, base_score)
            if window <= 0:
                continue
            if len(sweep) >= max_evals:
                break

            forward_first = True
            if screen_cursor and screen_cursor[0] == seed.source_orb_id and screen_cursor[1] > seed.area_index:
                forward_first = False
            ordered_offsets: list[int] = []
            for distance in range(1, window + 1):
                if forward_first:
                    ordered_offsets.extend([distance, -distance])
                else:
                    ordered_offsets.extend([-distance, distance])

            source_areas = by_source.get(seed.source_orb_id, [])
            index_lookup = {area.area_index: area for area in source_areas}
            for offset in ordered_offsets:
                neighbor = index_lookup.get(seed.area_index + offset)
                if neighbor is None:
                    continue
                neighbor_score = base_score + (self.config.area_dwell_neighbor_boost * (1.0 / (1.0 + abs(offset))))
                push_area(neighbor, neighbor_score)
                if len(sweep) >= max_evals:
                    break

        if not sweep:
            return [], screen_cursor
        last = sweep[-1][0]
        return sweep[:max_evals], (last.source_orb_id, last.area_index)

    def _compute_area_dwell_reads(
        self,
        skim_score: float,
        area: _SkimArea,
        target_record: str = "",
        target_field: str = "",
    ) -> int:
        text_lower = area.text.lower()
        dwell_signal = skim_score
        dwell_signal += 0.22 * area.salience
        dwell_signal += 0.16 * area.source_orb.focus_strength
        dwell_signal += self.config.area_dwell_complexity_boost * area.complexity
        if area.complexity >= self.config.area_dwell_complexity_threshold:
            dwell_signal += 0.14
        if target_record and target_record in text_lower:
            dwell_signal += self.config.area_dwell_target_boost
        if target_field and target_field in text_lower:
            dwell_signal += 0.72 * self.config.area_dwell_target_boost

        reads = self.config.area_dwell_base_reads
        if dwell_signal >= self.config.area_dwell_trigger_score:
            reads += 1
        if dwell_signal >= (self.config.area_dwell_trigger_score + 0.9):
            reads += 1
        return max(self.config.area_dwell_base_reads, min(self.config.area_dwell_max_reads, reads))

    def _supports_reasoning_completion(self, model: ModelAdapter) -> bool:
        return callable(getattr(model, "complete_with_reasoning", None))

    def _complete_model(self, model: ModelAdapter, messages: list[dict[str, str]], think: bool = False) -> str:
        if think and self._supports_reasoning_completion(model):
            completion = getattr(model, "complete_with_reasoning")
            try:
                return str(completion(messages, think=True))
            except TypeError:
                return str(completion(messages))
        return str(model.complete(messages))

    def _estimate_reasoning_dwell_signal(
        self,
        area: _SkimArea,
        skim_score: float,
        target_record: str,
        target_field: str,
    ) -> float:
        text_lower = area.text.lower()
        signal = skim_score
        signal += 0.22 * area.salience
        signal += 0.18 * area.complexity
        signal += 0.16 * area.source_orb.focus_strength
        signal -= 0.46 * self._estimate_area_noise(area.text)
        if target_record and target_record in text_lower:
            signal += 0.78
        if target_field and target_field in text_lower:
            signal += 0.56
        return signal

    def _should_use_reasoned_dwell(
        self,
        model: ModelAdapter,
        area: _SkimArea,
        skim_score: float,
        target_record: str,
        target_field: str,
        prefer_final_code: bool,
        reasoned_area_reads_used: int,
        question: str = "",
    ) -> bool:
        if self.config.answer_dwell_mode != "reasoned":
            return False
        if not self._supports_reasoning_completion(model):
            return False
        if reasoned_area_reads_used >= self.config.reasoning_dwell_max_areas:
            return False
        text_lower = area.text.lower()
        mc_options = self._extract_multiple_choice_options(question)
        is_multiple_choice = bool(mc_options)
        option_anchor_set: set[str] = set()
        if mc_options:
            option_anchor_set = set(
                self._expand_anchors(extract_anchors(" ".join(mc_options.values()), max_anchors=22), max_items=32)
            )
        option_overlap = 0.0
        if option_anchor_set and area.anchors:
            option_overlap = len(option_anchor_set.intersection(area.anchors)) / max(1, len(option_anchor_set))
        comparative_question = self._is_comparative_question(question)
        has_target = bool(target_record and target_record in text_lower) or bool(target_field and target_field in text_lower)
        signal = self._estimate_reasoning_dwell_signal(
            area=area,
            skim_score=skim_score,
            target_record=target_record,
            target_field=target_field,
        )
        signal += 0.22 * option_overlap
        if (
            is_multiple_choice
            and area.complexity >= max(0.18, self.config.reasoning_dwell_min_complexity * 0.65)
            and signal >= (self.config.reasoning_dwell_trigger_score - 0.35)
        ):
            return True
        if (
            is_multiple_choice
            and comparative_question
            and option_overlap > 0.0
            and area.complexity >= max(0.16, self.config.reasoning_dwell_min_complexity * 0.55)
            and signal >= (self.config.reasoning_dwell_trigger_score - 0.45)
        ):
            return True
        if not prefer_final_code and area.complexity < self.config.reasoning_dwell_min_complexity:
            return False
        if prefer_final_code and has_target and signal >= (self.config.reasoning_dwell_trigger_score - 0.35):
            return True
        if area.complexity >= self.config.reasoning_dwell_min_complexity and signal >= self.config.reasoning_dwell_trigger_score:
            return True
        return False

    def _score_extracted_facts(
        self,
        facts: list[str],
        skim_score: float,
        area: _SkimArea,
        target_record: str,
        target_field: str,
        excluded_codes: list[str] | None,
        prefer_final_code: bool,
        read_idx: int = 0,
        confidence: float = 0.0,
    ) -> float:
        if not facts:
            return float("-inf")
        facts_blob = " ".join(facts).lower()
        evidence_score = (
            0.2 * len(facts)
            + 0.28 * skim_score
            + 0.18 * area.salience
            + 0.08 * area.complexity
            + 0.12 * area.source_orb.focus_strength
            + (0.06 * read_idx)
            + (0.42 * max(0.0, min(1.0, confidence)))
        )
        if target_record and target_record in facts_blob:
            evidence_score += 1.15
        if target_field and target_field in facts_blob:
            evidence_score += 0.9
        if "source=canonical" in facts_blob:
            evidence_score += 0.55
        if any(marker in facts_blob for marker in ("archived", "old_value=", "not_current=true")):
            evidence_score -= 0.65
        noise_score = self._estimate_area_noise(facts_blob)
        if noise_score > 0.0:
            evidence_score -= 0.58 * noise_score
        if prefer_final_code:
            approval_hits = sum(1 for marker in self._FINAL_APPROVAL_MARKERS if marker in facts_blob)
            transitional_hits = sum(1 for marker in self._TRANSITIONAL_MARKERS if marker in facts_blob)
            evidence_score += 0.34 * approval_hits
            evidence_score -= 0.24 * transitional_hits
        if excluded_codes:
            evidence_score -= 0.1 * sum(1 for code in excluded_codes if code and code.lower() in facts_blob)
        return evidence_score

    def _read_area_heuristically(
        self,
        area: _SkimArea,
        skim_score: float,
        dwell_reads: int,
        active_anchors: list[str],
        target_record: str,
        target_field: str,
        excluded_codes: list[str] | None,
        prefer_final_code: bool,
    ) -> _AreaReadResult:
        best_facts: list[str] = []
        best_score = float("-inf")
        for read_idx in range(dwell_reads):
            read_anchors = active_anchors
            if read_idx > 0:
                read_anchors = self._merge_anchors(active_anchors, area.anchors, max_items=40)
            facts = self._extract_fact_lines_from_text(
                text=area.text,
                anchors=read_anchors,
                max_lines=max(2, 2 + read_idx),
                target_record=target_record,
                target_field=target_field,
                excluded_codes=excluded_codes,
                prefer_final_code=prefer_final_code,
            )
            score = self._score_extracted_facts(
                facts=facts,
                skim_score=skim_score,
                area=area,
                target_record=target_record,
                target_field=target_field,
                excluded_codes=excluded_codes,
                prefer_final_code=prefer_final_code,
                read_idx=read_idx,
            )
            if score > best_score:
                best_score = score
                best_facts = facts
        followup_anchors = self._expand_anchors(extract_anchors(" ".join(best_facts), max_anchors=10), max_items=16)
        return _AreaReadResult(
            facts=best_facts,
            score=best_score,
            reader="heuristic",
            followup_anchors=followup_anchors,
        )

    def _parse_reasoned_area_response(self, raw: str) -> tuple[bool, float, list[str], list[str]]:
        text = (raw or "").strip()
        if not text:
            return False, 0.0, [], []
        relevant_match = re.search(r"(?im)^RELEVANT:\s*(yes|no)\s*$", text)
        confidence_match = re.search(r"(?im)^CONFIDENCE:\s*([0-9]*\.?[0-9]+)\s*$", text)
        facts_match = re.search(
            r"(?is)^FACTS:\s*(.*?)(?:^\s*FOLLOWUP_ANCHORS:\s*|\Z)",
            text,
            flags=re.MULTILINE,
        )
        followup_match = re.search(r"(?is)^FOLLOWUP_ANCHORS:\s*(.*)\Z", text, flags=re.MULTILINE)

        relevant = bool(relevant_match and relevant_match.group(1).strip().lower() == "yes")
        confidence = 0.0
        if confidence_match:
            try:
                confidence = max(0.0, min(1.0, float(confidence_match.group(1).strip())))
            except ValueError:
                confidence = 0.0

        def parse_bullets(block: str) -> list[str]:
            items: list[str] = []
            for line in (block or "").splitlines():
                cleaned = line.strip()
                if not cleaned:
                    continue
                if cleaned.startswith("-"):
                    cleaned = cleaned[1:].strip()
                if cleaned and cleaned.lower() not in {"none", "n/a"} and cleaned not in items:
                    items.append(cleaned)
            return items

        facts = parse_bullets(facts_match.group(1) if facts_match else "")
        followup = parse_bullets(followup_match.group(1) if followup_match else "")
        return relevant, confidence, facts, followup

    def _read_area_with_reasoning(
        self,
        model: ModelAdapter,
        question: str,
        area: _SkimArea,
        skim_score: float,
        active_anchors: list[str],
        evidence_lines: list[str],
        target_record: str,
        target_field: str,
        excluded_codes: list[str] | None,
        prefer_final_code: bool,
    ) -> _AreaReadResult | None:
        evidence_tail = evidence_lines[-self.config.reasoning_dwell_evidence_tail :] if evidence_lines else []
        evidence_preview = "\n".join(evidence_tail) if evidence_tail else "- none"
        anchor_preview = ", ".join(active_anchors[:14]) if active_anchors else "(none)"
        exclusions = ", ".join(code.upper() for code in (excluded_codes or []) if code)
        mc_options = self._extract_multiple_choice_options(question)
        instructions = [
            "You are a local evidence reader for one scan area.",
            "Read only the provided area text.",
            "Extract concise facts that directly help answer the question.",
            "If the area is not relevant, say RELEVANT: no and leave FACTS empty.",
        ]
        if mc_options:
            instructions.append(
                "For multiple-choice questions, use FACTS lines to say which option this area supports or rules out."
            )
        if prefer_final_code:
            instructions.append("Prefer final approved production facts over pilot, retired, or transitional facts.")
        if exclusions:
            instructions.append(f"Exclude replaced codes: {exclusions}.")
        system_prompt = "\n".join(instructions)
        user_prompt = (
            f"Question:\n{question}\n\n"
            f"Active anchors:\n{anchor_preview}\n\n"
            f"Current evidence tail:\n{evidence_preview}\n\n"
            f"Area text:\n{area.text}\n\n"
            "Return exactly this format:\n"
            "RELEVANT: yes|no\n"
            "CONFIDENCE: 0.00-1.00\n"
            "FACTS:\n"
            "- <fact>\n"
            "FOLLOWUP_ANCHORS:\n"
            "- <anchor>\n"
            "Use short bullet lines. If none, write - none."
        )
        if mc_options:
            option_lines = "\n".join(f"{label}. {text}" for label, text in sorted(mc_options.items()))
            user_prompt += (
                "\n\nMultiple-choice options:\n"
                f"{option_lines}\n\n"
                "If possible, include facts like 'supports option C because ...' or 'rules out option A because ...'."
            )
        raw = self._complete_model(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            think=True,
        )
        relevant, confidence, facts, followup = self._parse_reasoned_area_response(raw)
        facts = [summarize_text(item, max_chars=220) for item in facts[: self.config.reasoning_dwell_max_fact_lines]]
        if (not relevant or not facts) and confidence < 0.45:
            return None
        score = self._score_extracted_facts(
            facts=facts,
            skim_score=skim_score,
            area=area,
            target_record=target_record,
            target_field=target_field,
            excluded_codes=excluded_codes,
            prefer_final_code=prefer_final_code,
            confidence=confidence,
        )
        followup_anchors = self._expand_anchors(followup, max_items=16)
        return _AreaReadResult(
            facts=facts,
            score=score,
            reader="reasoned",
            followup_anchors=followup_anchors,
        )

    def _question_first_skim_orbs(
        self,
        question: str,
        anchors: list[str],
        used_ids: set[str],
        max_candidates: int,
        target_record: str = "",
        target_field: str = "",
        excluded_codes: list[str] | None = None,
        prefer_final_code: bool = False,
        candidate_pool: list[MemoryOrb] | None = None,
    ) -> list[MemoryOrb]:
        pool = candidate_pool if candidate_pool is not None else self._orbs
        if max_candidates <= 0 or not pool:
            return []
        question_embedding = self.embedder.embed(question)
        anchor_set = set(anchors)
        excluded = {code.lower() for code in (excluded_codes or [])}
        ranked: list[tuple[float, MemoryOrb]] = []
        for orb in pool:
            if orb.orb_id in used_ids:
                continue
            semantic_score = cosine_similarity(question_embedding, orb.embedding)
            overlap = 0.0
            if anchor_set and orb.anchors:
                overlap = len(anchor_set.intersection(orb.anchors)) / max(1, len(anchor_set))
            orb_text_lower = f"{orb.summary}\n{orb.raw_excerpt}".lower()
            record_bonus = 1.0 if target_record and target_record in orb_text_lower else 0.0
            field_bonus = 1.0 if target_field and target_field in orb_text_lower else 0.0
            canonical_bonus = 0.55 if "source=canonical" in orb_text_lower else 0.0
            stale_penalty = 0.55 if any(marker in orb_text_lower for marker in ("archived", "old_value=", "not_current=true")) else 0.0
            approval_hits = sum(1 for marker in self._FINAL_APPROVAL_MARKERS if marker in orb_text_lower)
            transitional_hits = sum(1 for marker in self._TRANSITIONAL_MARKERS if marker in orb_text_lower)
            approval_bonus = (0.34 * approval_hits) if prefer_final_code else 0.0
            transition_penalty = (0.2 * transitional_hits) if prefer_final_code else 0.0
            excluded_penalty = 0.0
            if excluded:
                excluded_hits = sum(1 for code in excluded if code and code in orb_text_lower)
                excluded_penalty = 0.85 * excluded_hits
            recency = math.exp(-max(0, self.turn_index - orb.created_turn) / max(1.0, self.config.recency_tau_turns))
            score = (
                0.42 * semantic_score
                + 0.28 * overlap
                + 0.2 * orb.salience
                + 0.12 * orb.focus_strength
                + 0.08 * recency
                + 0.4 * record_bonus
                + 0.28 * field_bonus
                + canonical_bonus
                + approval_bonus
                - stale_penalty
                - transition_penalty
                - excluded_penalty
            )
            ranked.append((score, orb))

        ranked.sort(key=lambda item: item[0], reverse=True)
        return [orb for _, orb in ranked[:max_candidates]]

    def _apply_question_focus_dwell(self, orb: MemoryOrb, evidence_score: float) -> None:
        normalized = max(0.0, min(1.0, evidence_score / 2.6))
        if normalized <= 0.0:
            return
        if orb.orb_id.startswith("working-"):
            return
        orb.focus_strength = min(1.0, 0.58 * orb.focus_strength + 0.42 * normalized + 0.16)
        orb.last_focus_turn = self.turn_index
        hold_extension = self.config.focus_hold_turns
        if normalized >= 0.75:
            hold_extension += 2
        elif normalized >= 0.5:
            hold_extension += 1
        orb.pinned_until_turn = max(orb.pinned_until_turn, self.turn_index + hold_extension)
        orb.accesses += 1
        orb.last_access_turn = self.turn_index

    def _merge_anchors(self, base: list[str], extra: list[str], max_items: int = 16) -> list[str]:
        merged: list[str] = []
        for anchor in base + extra:
            if anchor not in merged:
                merged.append(anchor)
            if len(merged) >= max_items:
                break
        return merged

    def _extract_relevant_fact_lines(
        self,
        orb: MemoryOrb,
        anchors: list[str],
        max_lines: int = 3,
        target_record: str = "",
        target_field: str = "",
        excluded_codes: list[str] | None = None,
        prefer_final_code: bool = False,
    ) -> list[str]:
        text = f"{orb.raw_excerpt}\n{orb.summary}".strip()
        return self._extract_fact_lines_from_text(
            text=text,
            anchors=anchors,
            max_lines=max_lines,
            target_record=target_record,
            target_field=target_field,
            excluded_codes=excluded_codes,
            prefer_final_code=prefer_final_code,
        )

    def _extract_fact_lines_from_text(
        self,
        text: str,
        anchors: list[str],
        max_lines: int = 3,
        target_record: str = "",
        target_field: str = "",
        excluded_codes: list[str] | None = None,
        prefer_final_code: bool = False,
    ) -> list[str]:
        clean = (text or "").strip()
        if not clean:
            return []
        chunks = [c.strip() for c in re.split(r"(?<=[.!?])\s+|\n+", clean) if c.strip()]
        if not chunks:
            return []

        aset = set(a.lower() for a in anchors)
        excluded = {code.lower() for code in (excluded_codes or [])}
        markers = ("record=", "current_value=", "value=", "id=", "key=", "fact", "source=", "current")
        scored: list[tuple[float, str]] = []
        for chunk in chunks:
            lower = chunk.lower()
            overlap = sum(1 for a in aset if a in lower)
            marker_bonus = 1.0 if any(marker in lower for marker in markers) else 0.0
            numeric_bonus = 0.35 if re.search(r"[a-z]+-[a-z0-9]+|\d+\.\d+\.\d+", lower) else 0.0
            record_bonus = 1.4 if target_record and target_record in lower else 0.0
            field_bonus = 1.2 if target_field and target_field in lower else 0.0
            canonical_bonus = 0.8 if "source=canonical" in lower else 0.0
            archived_penalty = 1.1 if any(marker in lower for marker in ("archived", "old_value=", "not_current=true")) else 0.0
            key_value_bonus = 0.45 if "=" in lower else 0.0
            approval_hits = sum(1 for marker in self._FINAL_APPROVAL_MARKERS if marker in lower)
            transitional_hits = sum(1 for marker in self._TRANSITIONAL_MARKERS if marker in lower)
            approval_bonus = (0.48 * approval_hits) if prefer_final_code else 0.0
            transition_penalty = (0.34 * transitional_hits) if prefer_final_code else 0.0
            excluded_hits = sum(1 for code in excluded if code and code in lower)
            excluded_penalty = (0.9 * excluded_hits) if prefer_final_code else 0.0
            score = (
                (1.1 * overlap)
                + marker_bonus
                + numeric_bonus
                + record_bonus
                + field_bonus
                + canonical_bonus
                + key_value_bonus
                + approval_bonus
                - archived_penalty
                - transition_penalty
                - excluded_penalty
            )
            if target_record and target_field and target_record in lower and target_field in lower:
                score += 0.85
            if score <= 0.0:
                continue
            scored.append((score, summarize_text(chunk, max_chars=220)))

        if not scored:
            return []
        scored.sort(key=lambda item: item[0], reverse=True)
        selected: list[str] = []
        for _, line in scored:
            if line not in selected:
                selected.append(line)
            if len(selected) >= max(1, max_lines):
                break
        return selected

    def _extract_code_tokens(self, text: str) -> list[str]:
        if not text:
            return []
        found = [item.group(0).lower() for item in self._CODE_PATTERN.finditer(text)]
        unique: list[str] = []
        for code in found:
            if code not in unique:
                unique.append(code)
        return unique

    def _build_question_memory_pool(self) -> list[MemoryOrb]:
        pool: list[MemoryOrb] = list(self._orbs)
        if not self._working_turns:
            return pool

        for turn in list(self._working_turns):
            if turn.role not in {"user", "assistant", "tool"}:
                continue
            if bool(turn.metadata.get("exclude_from_question_memory_pool")):
                continue
            text = (turn.content or "").strip()
            if not text:
                continue
            source_text = f"role={turn.role} {text}"
            summary = summarize_text(source_text, max_chars=340)
            anchors = self._expand_anchors(extract_anchors(text, max_anchors=14), max_items=24)
            metadata_importance = 0.0
            raw_importance = turn.metadata.get("importance")
            if raw_importance is not None:
                try:
                    metadata_importance = max(0.0, min(1.0, float(raw_importance)))
                except (TypeError, ValueError):
                    metadata_importance = 0.0
            role_weight = 1.0 if turn.role == "user" else (0.92 if turn.role == "tool" else 0.72)
            if turn.role == "assistant" and turn.metadata.get("source") == "answer_document":
                role_weight = 0.0
            if role_weight <= 0.0:
                continue
            salience = min(1.0, role_weight * (0.36 + 0.06 * len(anchors) + 0.28 * metadata_importance))
            pool.append(
                MemoryOrb(
                    orb_id=f"working-{turn.turn_id}",
                    summary=summary,
                    raw_excerpt=source_text[:2600],
                    anchors=anchors,
                    embedding=self.embedder.embed(summary),
                    source_turn_ids=[turn.turn_id],
                    created_turn=turn.turn_index,
                    salience=salience,
                    tokens=self.token_estimator.count(summary),
                    focus_strength=min(1.0, role_weight * 0.72 * metadata_importance),
                    last_focus_turn=turn.turn_index if metadata_importance >= 0.25 else 0,
                    pinned_until_turn=0,
                )
            )
        return pool

    def _score_code_candidates_from_text(
        self, text: str, excluded_codes: list[str], prefer_final_code: bool
    ) -> dict[str, float]:
        if not text:
            return {}
        lower = text.lower()
        codes = self._extract_code_tokens(lower)
        if not codes:
            return {}
        excluded = {code.lower() for code in excluded_codes}
        approval_hits = sum(1 for marker in self._FINAL_APPROVAL_MARKERS if marker in lower)
        transitional_hits = sum(1 for marker in self._TRANSITIONAL_MARKERS if marker in lower)

        updates: dict[str, float] = {}
        for code in codes:
            score = 0.22
            if prefer_final_code:
                score += 0.36 * approval_hits
                score -= 0.18 * transitional_hits
            if code in excluded:
                score -= 1.35
            if re.search(rf"\bretired\s+{re.escape(code)}\b", lower):
                score -= 1.15
            if re.search(rf"\b{re.escape(code)}\b.*\bpilot-only\b", lower):
                score -= 1.05
            if re.search(rf"\bapproved\s+{re.escape(code)}\b", lower):
                score += 1.65
            if re.search(rf"\b{re.escape(code)}\b.*\bproduction operations\b", lower):
                score += 1.25
            updates[code] = updates.get(code, 0.0) + score
        return updates

    def _coerce_answer_with_code_preference(
        self, answer: str, ranked_codes: list[tuple[str, float]], excluded_codes: list[str]
    ) -> str:
        if not ranked_codes:
            return answer
        excluded = {code.lower() for code in excluded_codes}
        best = ""
        for code, _ in ranked_codes:
            if code.lower() not in excluded:
                best = code.upper()
                break
        if not best:
            return answer

        predicted_codes = self._extract_code_tokens(answer)
        if predicted_codes and predicted_codes[0].lower() not in excluded:
            return answer

        stripped = (answer or "").strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, dict):
                    if "answer" in parsed:
                        parsed["answer"] = best
                    elif "value" in parsed:
                        parsed["value"] = best
                    else:
                        parsed = {"answer": best}
                    return json.dumps(parsed)
            except Exception:
                return best
        return best

    def _build_alias_indices(self, alias_map: dict[str, list[str]]) -> tuple[dict[str, list[str]], dict[str, str]]:
        forward: dict[str, list[str]] = {}
        reverse: dict[str, str] = {}
        for canonical_raw, aliases in alias_map.items():
            canonical_terms = extract_anchors(str(canonical_raw), max_anchors=6)
            if not canonical_terms:
                continue
            canonical = canonical_terms[0]
            alias_terms: list[str] = []
            for alias in aliases:
                for token in extract_anchors(str(alias), max_anchors=10):
                    if token not in alias_terms and token != canonical:
                        alias_terms.append(token)
                        reverse[token] = canonical
            if canonical not in reverse:
                reverse[canonical] = canonical
            forward[canonical] = alias_terms
        return forward, reverse

    def _expand_anchors(self, anchors: list[str], max_items: int = 24) -> list[str]:
        if not anchors:
            return []
        expanded: list[str] = []
        for raw in anchors:
            anchor = raw.lower()
            canonical = self._alias_reverse.get(anchor, anchor)
            for token in [canonical, anchor]:
                if token and token not in expanded:
                    expanded.append(token)
                    if len(expanded) >= max_items:
                        return expanded
            for alias in self._alias_forward.get(canonical, []):
                if alias not in expanded:
                    expanded.append(alias)
                    if len(expanded) >= max_items:
                        return expanded
        return expanded

    def _select_recent_turns(self, budget_tokens: int) -> list[Turn]:
        if budget_tokens <= 0:
            return []
        selected: list[Turn] = []
        used_tokens = 0
        for turn in reversed(self._working_turns):
            if used_tokens + turn.tokens > budget_tokens and selected:
                break
            selected.append(turn)
            used_tokens += turn.tokens
        selected.reverse()
        return selected

    def _enforce_context_cap(
        self, messages: list[dict[str, str]], total_tokens: int
    ) -> tuple[list[dict[str, str]], int]:
        if total_tokens <= self.config.context_max_tokens:
            return messages, total_tokens

        records = [
            {
                "idx": idx,
                "role": message["role"],
                "content": message["content"],
                "tokens": self.token_estimator.count(message["content"]),
                "removed": False,
            }
            for idx, message in enumerate(messages)
        ]
        cap = self.config.context_max_tokens

        latest_user_idx = -1
        for idx, record in enumerate(records):
            if record["role"] == "user":
                latest_user_idx = idx

        def current_total() -> int:
            return sum(int(record["tokens"]) for record in records if not bool(record["removed"]))

        # Drop oldest non-system messages first, but keep latest user.
        for idx, record in enumerate(records):
            if current_total() <= cap:
                break
            if record["role"] == "system":
                continue
            if idx == latest_user_idx:
                continue
            record["removed"] = True

        # Drop system memory blocks before trimming user content.
        if current_total() > cap:
            system_indices = [idx for idx, record in enumerate(records) if record["role"] == "system" and not record["removed"]]

            def system_drop_rank(target_idx: int) -> tuple[int, int]:
                content = str(records[target_idx]["content"])
                if content.startswith("Memory Orb Sync:"):
                    return (0, target_idx)
                if content.startswith("Selective Attention Latch:"):
                    return (1, target_idx)
                return (2, target_idx)

            for idx in sorted(system_indices, key=system_drop_rank):
                if current_total() <= cap:
                    break
                # Keep at least one system message when no user message exists.
                if latest_user_idx < 0 and len(
                    [
                        i
                        for i, record in enumerate(records)
                        if record["role"] == "system" and not record["removed"] and i != idx
                    ]
                ) == 0:
                    break
                records[idx]["removed"] = True

        # Trim remaining system messages to fit the hard cap.
        if current_total() > cap:
            remaining_system = [idx for idx, record in enumerate(records) if record["role"] == "system" and not record["removed"]]

            def system_trim_rank(target_idx: int) -> tuple[int, int]:
                content = str(records[target_idx]["content"])
                if content.startswith("Memory Orb Sync:"):
                    return (0, target_idx)
                if content.startswith("Selective Attention Latch:"):
                    return (1, target_idx)
                return (2, target_idx)

            for idx in sorted(remaining_system, key=system_trim_rank):
                overflow = current_total() - cap
                if overflow <= 0:
                    break
                current_tokens = int(records[idx]["tokens"])
                target_tokens = max(8, current_tokens - overflow)
                new_text = self._truncate_text_to_token_budget(str(records[idx]["content"]), target_tokens)
                new_tokens = self.token_estimator.count(new_text)
                records[idx]["content"] = new_text
                records[idx]["tokens"] = new_tokens

        # Ensure latest user survives by truncating it if needed.
        if current_total() > cap and latest_user_idx >= 0 and not records[latest_user_idx]["removed"]:
            overflow = current_total() - cap
            current_tokens = int(records[latest_user_idx]["tokens"])
            target_tokens = max(8, current_tokens - overflow)
            new_text = self._truncate_text_to_token_budget(str(records[latest_user_idx]["content"]), target_tokens)
            new_tokens = self.token_estimator.count(new_text)
            records[latest_user_idx]["content"] = new_text
            records[latest_user_idx]["tokens"] = new_tokens

        # Final fallback: trim every remaining message to force exact cap.
        if current_total() > cap:
            for idx in range(len(records)):
                if current_total() <= cap:
                    break
                if records[idx]["removed"]:
                    continue
                overflow = current_total() - cap
                current_tokens = int(records[idx]["tokens"])
                target_tokens = max(1, current_tokens - overflow)
                new_text = self._truncate_text_to_token_budget(str(records[idx]["content"]), target_tokens)
                new_tokens = self.token_estimator.count(new_text)
                records[idx]["content"] = new_text
                records[idx]["tokens"] = new_tokens

        final_messages = [
            {"role": str(record["role"]), "content": str(record["content"])}
            for record in records
            if not bool(record["removed"]) and str(record["content"]).strip()
        ]
        total_tokens = sum(self.token_estimator.count(msg["content"]) for msg in final_messages)
        return final_messages, total_tokens

    def _truncate_text_to_token_budget(self, text: str, max_tokens: int) -> str:
        if max_tokens <= 0:
            return ""
        clean = (text or "").strip()
        if not clean:
            return ""
        if self.token_estimator.count(clean) <= max_tokens:
            return clean

        low = 0
        high = len(clean)
        best = ""
        while low <= high:
            mid = (low + high) // 2
            candidate = clean[:mid].rstrip()
            if candidate and self.token_estimator.count(candidate) <= max_tokens:
                best = candidate
                low = mid + 1
            else:
                high = mid - 1

        if not best:
            return clean[:1]

        with_ellipsis = f"{best}..."
        if self.token_estimator.count(with_ellipsis) <= max_tokens:
            return with_ellipsis
        return best
