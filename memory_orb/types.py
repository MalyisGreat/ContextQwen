from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Turn:
    turn_id: str
    turn_index: int
    role: str
    content: str
    tokens: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MemoryOrb:
    orb_id: str
    summary: str
    raw_excerpt: str
    anchors: list[str]
    embedding: list[float]
    source_turn_ids: list[str]
    created_turn: int
    salience: float
    tokens: int
    accesses: int = 0
    last_access_turn: int = 0
    focus_strength: float = 0.0
    last_focus_turn: int = 0
    pinned_until_turn: int = 0


@dataclass(slots=True)
class SemanticCard:
    anchor: str
    synopsis: str
    frequency: int
    tokens: int
    last_turn: int
    evidence_orb_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ContextPacket:
    messages: list[dict[str, str]]
    total_tokens: int
    memory_tokens: int
    recent_tokens: int
    selected_orb_ids: list[str]
    selected_anchors: list[str]
    latched_subgoal: str = ""
    background_subgoal: str = ""


@dataclass(slots=True)
class AnswerDocumentResult:
    question: str
    answer_document: str
    answer: str
    selected_orb_ids: list[str]
    passes_completed: int
    total_tokens: int


@dataclass(slots=True)
class FocusLatch:
    summary: str
    anchors: list[str]
    confidence: float
    created_turn: int
    last_refresh_turn: int
    predecessor_summary: str = ""
    predecessor_anchors: list[str] = field(default_factory=list)
