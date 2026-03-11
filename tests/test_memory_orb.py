from __future__ import annotations

import re
from pathlib import Path
from tempfile import TemporaryDirectory

from benchmarks.selective_attention_benchmark import generate_long_writing
from memory_orb import MemoryOrbEngine, MemoryOrbEngineConfig
from memory_orb.adapters import HashingEmbedder, ModelAdapter
from memory_orb.engine import _SkimArea
from memory_orb.types import MemoryOrb


class DeterministicModel(ModelAdapter):
    def complete(self, messages):
        last_user = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user = msg.get("content", "")
                break
        return f"ACK::{last_user[:80]}"


class ValueExtractorModel(ModelAdapter):
    def complete(self, messages):
        text = "\n".join(msg.get("content", "") for msg in messages)
        match = re.search(r"current_value=([a-z0-9._\\-]+)", text.lower())
        if match:
            return match.group(1)
        return "unknown"


class RecordScopedExtractorModel(ModelAdapter):
    def complete(self, messages):
        text = "\n".join(msg.get("content", "") for msg in messages).lower()
        qmatch = re.search(r"for record\s+([a-z0-9._\\-]+)", text)
        target = qmatch.group(1) if qmatch else ""
        if target:
            scoped = re.search(rf"record={re.escape(target)}[^\\n]*?current_value=([a-z0-9._\\-]+)", text)
            if scoped:
                return scoped.group(1)
        match = re.search(r"current_value=([a-z0-9._\\-]+)", text)
        if match:
            return match.group(1)
        return "unknown"


class ReasoningRecordScopedModel(RecordScopedExtractorModel):
    def complete_with_reasoning(self, messages, think=True):
        text = "\n".join(msg.get("content", "") for msg in messages).lower()
        qmatch = re.search(r"for record\s+([a-z0-9._\\-]+)", text)
        target = qmatch.group(1) if qmatch else ""
        scoped = ""
        if target:
            match = re.search(rf"(record={re.escape(target)}[^\\n]*?current_value=([a-z0-9._\\-]+)[^\\n]*)", text)
            if match:
                scoped = match.group(1).strip()
        if not scoped:
            return "RELEVANT: no\nCONFIDENCE: 0.15\nFACTS:\n- none\nFOLLOWUP_ANCHORS:\n- none"
        return (
            "RELEVANT: yes\n"
            "CONFIDENCE: 0.91\n"
            f"FACTS:\n- {scoped}\n"
            "FOLLOWUP_ANCHORS:\n"
            "- current_value\n"
            f"- {target}"
        )


class PilotBiasModel(ModelAdapter):
    def complete(self, messages):
        return '{"answer":"MNT-FLOW-246"}'


def test_context_size_is_bounded_during_long_run():
    config = MemoryOrbEngineConfig(
        context_max_tokens=520,
        working_max_tokens=260,
        working_target_tokens=180,
        max_retrieved_orbs=5,
    )
    engine = MemoryOrbEngine(config=config)
    model = DeterministicModel()

    for i in range(90):
        user_text = (
            f"Turn {i}: Remember PostgreSQL replication lag, wal tuning, and oncall mitigation. "
            f"Capture owner, due dates, and unresolved blockers in the memory layer."
        )
        _, packet = engine.chat(model=model, user_text=user_text, system_prompt="You are a memory test assistant.")
        assert packet.total_tokens <= config.context_max_tokens

    stats = engine.stats()
    assert stats["orb_count"] > 0
    assert stats["working_tokens"] <= config.working_max_tokens


def test_anchor_driven_retrieval_exposes_relevant_memory():
    config = MemoryOrbEngineConfig(
        context_max_tokens=420,
        working_max_tokens=190,
        working_target_tokens=120,
        max_retrieved_orbs=4,
    )
    engine = MemoryOrbEngine(config=config)

    for i in range(30):
        engine.add_turn(
            "user",
            f"Session {i}: Postgres replication lag remains high in shard-{i % 3}. "
            "Need WAL retention and checkpoint tuning plan.",
        )
        engine.add_turn("assistant", "Acknowledged. Tracking replication lag and WAL checkpoint strategy.")

    packet = engine.build_context(
        user_query="How should we reduce postgres replication lag this week?",
        system_prompt="You are an SRE copilot.",
    )
    system_text = " ".join(msg["content"] for msg in packet.messages if msg["role"] == "system").lower()

    assert "memory orb sync" in system_text
    assert "replication" in system_text
    assert packet.total_tokens <= config.context_max_tokens


def test_subgoal_latch_keeps_background_goal_during_focus_shift():
    config = MemoryOrbEngineConfig(
        context_max_tokens=430,
        working_max_tokens=210,
        working_target_tokens=130,
        max_retrieved_orbs=4,
    )
    engine = MemoryOrbEngine(config=config)
    model = DeterministicModel()

    engine.chat(
        model=model,
        user_text="Primary goal: ship the analytics dashboard this sprint with reliable reporting.",
        system_prompt="You are a planning assistant.",
    )
    engine.chat(
        model=model,
        user_text=(
            "New objective and priority before launch: focus on oauth callback signature mismatch "
            "and fix it before release."
        ),
        system_prompt="You are a planning assistant.",
    )

    packet = engine.build_context(
        user_query="Focus on oauth callback handling and give next steps.",
        system_prompt="You are a planning assistant.",
    )
    system_text = " ".join(msg["content"] for msg in packet.messages if msg["role"] == "system").lower()

    assert "selective attention latch" in system_text
    assert "primary sub-goal" in system_text
    assert "background sub-goal" in system_text
    assert "oauth callback" in packet.latched_subgoal.lower()
    assert "analytics dashboard" in packet.background_subgoal.lower()
    assert packet.total_tokens <= config.context_max_tokens


def test_build_context_combines_memory_sections_into_one_system_message():
    config = MemoryOrbEngineConfig(
        context_max_tokens=430,
        working_max_tokens=80,
        working_target_tokens=40,
        max_retrieved_orbs=4,
    )
    engine = MemoryOrbEngine(config=config)
    for idx in range(10):
        engine.add_turn("user", f"Planning note {idx} about analytics dashboard, oauth callback handling, and release readiness.")
        engine.add_turn("assistant", "stored acknowledgement")

    packet = engine.build_context(
        user_query="Focus on oauth callback handling and give next steps.",
        system_prompt="You are a planning assistant.",
    )

    system_messages = [msg for msg in packet.messages if msg["role"] == "system"]

    assert len(system_messages) == 1
    assert "Memory Orb Sync:" in system_messages[0]["content"]


def test_build_context_adds_multiple_choice_guidance_block():
    config = MemoryOrbEngineConfig(
        context_max_tokens=520,
        working_max_tokens=120,
        working_target_tokens=70,
        max_retrieved_orbs=4,
    )
    engine = MemoryOrbEngine(config=config)
    for idx in range(8):
        engine.add_turn(
            "user",
            (
                f"Policy note {idx}: visa-free travel and inbound tourism expanded in later years, "
                "while outbound-only framing is contradicted by the later summaries."
            ),
        )
        engine.add_turn("assistant", "stored acknowledgement")

    packet = engine.build_context(
        user_query=(
            "Question:\nWhich option best matches the trend?\n\n"
            "Options:\n"
            "A. Outbound travel dominates throughout.\n"
            "B. Inbound travel and visa-free policy are highlighted.\n"
            "C. The reports reject any tourism growth.\n"
            "D. The documents discuss only domestic rail travel."
        ),
        system_prompt="You are a policy analyst.",
    )

    system_text = " ".join(msg["content"] for msg in packet.messages if msg["role"] == "system").lower()

    assert "multiple-choice comparison hints" in system_text
    assert "do not default to option a" in system_text
    assert "evidence:" in system_text
    assert "visa-free" in system_text


def test_strip_multiple_choice_options_keeps_question_stem_only():
    engine = MemoryOrbEngine()

    stripped = engine._strip_multiple_choice_options(
        "Question:\nWhich approach is approved?\n\nOptions:\nA. Legacy pilot workflow.\nB. Final production workflow."
    )

    assert "legacy pilot workflow" not in stripped.lower()
    assert "final production workflow" not in stripped.lower()
    assert "which approach is approved?" in stripped.lower()


def test_selective_attention_uses_recent_pulses_to_retrieve_matching_orbs():
    config = MemoryOrbEngineConfig(
        context_max_tokens=500,
        working_max_tokens=130,
        working_target_tokens=90,
        max_retrieved_orbs=5,
    )
    engine = MemoryOrbEngine(config=config)

    for i in range(18):
        engine.add_turn(
            "user",
            f"Billing session {i}: investigate reconciliation checksum drift in ledger shard-{i % 4}.",
        )
        engine.add_turn("assistant", "Tracking billing reconciliation drift and ledger checksums.")

    for i in range(6):
        engine.add_turn(
            "user",
            "Objective and priority: resolve oauth callback signature mismatch before release window.",
        )
        engine.add_turn("assistant", f"Captured oauth callback fix task {i}.")

    packet = engine.build_context(
        user_query="What should we execute next this week?",
        system_prompt="You are an engineering assistant.",
    )
    selected_orbs = {orb.orb_id: orb for orb in engine._orbs if orb.orb_id in set(packet.selected_orb_ids)}

    assert packet.selected_orb_ids
    assert any("oauth" in " ".join(orb.anchors) for orb in selected_orbs.values())
    assert packet.total_tokens <= config.context_max_tokens


def test_hard_cap_enforced_with_oversized_system_prompt():
    config = MemoryOrbEngineConfig(
        context_max_tokens=220,
        working_max_tokens=160,
        working_target_tokens=100,
    )
    engine = MemoryOrbEngine(config=config)
    engine.add_turn("user", "Need deployment steps for shard migration.")

    packet = engine.build_context(
        user_query="what should we do first?",
        system_prompt=("CRITICAL_POLICY " * 500).strip(),
    )

    assert packet.total_tokens <= config.context_max_tokens
    assert any(msg["role"] == "user" for msg in packet.messages)


def test_latest_user_is_retained_when_latest_turn_is_huge():
    config = MemoryOrbEngineConfig(
        context_max_tokens=260,
        working_max_tokens=180,
        working_target_tokens=110,
    )
    engine = MemoryOrbEngine(config=config)

    huge_user = " ".join(["payload"] * 1500)
    engine.add_turn("user", huge_user)
    packet = engine.build_context(user_query=huge_user, system_prompt="You are concise.")

    assert packet.total_tokens <= config.context_max_tokens
    user_messages = [msg for msg in packet.messages if msg["role"] == "user"]
    assert user_messages
    assert len(user_messages[-1]["content"]) < len(huge_user)


def test_subgoal_latch_ignores_irrelevant_unmarked_chatter():
    config = MemoryOrbEngineConfig(
        context_max_tokens=460,
        working_max_tokens=220,
        working_target_tokens=140,
    )
    engine = MemoryOrbEngine(config=config)

    engine.add_turn("user", "Primary goal: finish SOC2 control mapping and evidence review this quarter.")
    for idx in range(5):
        engine.add_turn("assistant", "ack")
        engine.add_turn("user", f"tell me a random cat joke number {idx}")

    packet = engine.build_context(user_query="status update", system_prompt="You are a program manager.")
    assert "soc2" in packet.latched_subgoal.lower()
    assert "cat joke" not in packet.latched_subgoal.lower()


def test_focus_dwell_keeps_relevant_orb_for_followup_turns():
    config = MemoryOrbEngineConfig(
        context_max_tokens=560,
        working_max_tokens=160,
        working_target_tokens=100,
        max_retrieved_orbs=4,
        focus_hold_turns=6,
    )
    engine = MemoryOrbEngine(config=config)

    for i in range(18):
        engine.add_turn("user", f"Billing stream {i}: ledger reconciliation drift and invoice mismatch investigation.")
        engine.add_turn("assistant", "Tracking billing drift.")
    for i in range(8):
        engine.add_turn("user", f"Objective: oauth callback signature mismatch fix before launch window {i}.")
        engine.add_turn("assistant", "Tracking oauth callback fix.")

    first = engine.build_context(
        user_query="Focus on oauth callback signature and execution plan.",
        system_prompt="You are a planner.",
    )
    assert first.selected_orb_ids

    first_orbs = set(first.selected_orb_ids)
    engine.add_turn("user", "status heartbeat check")
    engine.add_turn("assistant", "ack")
    followup = engine.build_context(
        user_query="What should we do next?",
        system_prompt="You are a planner.",
    )

    assert followup.selected_orb_ids
    assert first_orbs.intersection(set(followup.selected_orb_ids))


def test_semantic_benchmark_mode_uses_paraphrases_for_high_importance():
    rows = generate_long_writing(target_word="oauth", seed=31, segments=90, mode="semantic")
    high_rows = [row for row in rows if row.kind == "high"]
    decoy_rows = [row for row in rows if row.kind == "decoy"]

    high_hits = sum(1 for row in high_rows if "oauth" in row.text.lower())
    decoy_hits = sum(1 for row in decoy_rows if "oauth" in row.text.lower())

    assert high_rows
    assert decoy_rows
    assert high_hits <= max(1, len(high_rows) // 4)
    assert decoy_hits >= max(1, (len(decoy_rows) * 3) // 4)


def test_anchor_aliases_enable_semantic_focus_match():
    config = MemoryOrbEngineConfig(
        context_max_tokens=480,
        working_max_tokens=170,
        working_target_tokens=110,
        max_retrieved_orbs=4,
        anchor_aliases={"oauth": ["callback signature", "identity handshake"]},
        min_focus_orb_count=1,
    )
    engine = MemoryOrbEngine(config=config)

    for i in range(16):
        engine.add_turn(
            "user",
            f"Task {i}: callback signature workflow fails on redirect validation and needs action plan.",
            metadata={"importance": 0.8},
        )
        engine.add_turn("assistant", "Tracking callback signature workflow.")

    packet = engine.build_context(
        user_query="Focus on oauth issue and provide next actions.",
        system_prompt="You are a reliability assistant.",
    )
    selected_orbs = [orb for orb in engine._orbs if orb.orb_id in set(packet.selected_orb_ids)]

    assert selected_orbs
    assert any("oauth" in orb.anchors for orb in selected_orbs)


def test_metadata_importance_increases_orb_salience():
    config = MemoryOrbEngineConfig(
        context_max_tokens=360,
        working_max_tokens=20,
        working_target_tokens=10,
    )
    engine = MemoryOrbEngine(config=config)

    engine.add_turn(
        "user",
        "high importance item with concrete mitigation steps and urgent owner assignment now",
        metadata={"importance": 1.0},
    )
    engine.add_turn("assistant", "ack")
    engine.add_turn("user", "low importance background chatter and routine note", metadata={"importance": 0.0})
    engine.add_turn("assistant", "ack")
    engine.add_turn("user", "another low importance line with minor detail", metadata={"importance": 0.0})
    engine.add_turn("assistant", "ack")
    engine.add_turn("user", "extra line to trigger swap and force orb creation", metadata={"importance": 0.0})
    engine.add_turn("assistant", "ack")

    assert len(engine._orbs) >= 1
    max_salience = max(orb.salience for orb in engine._orbs)
    assert max_salience >= 0.55


def test_action_query_penalizes_historical_orbs():
    config = MemoryOrbEngineConfig(
        context_max_tokens=520,
        working_max_tokens=120,
        working_target_tokens=80,
        max_retrieved_orbs=2,
        anchor_aliases={"oauth": ["callback signature"]},
    )
    engine = MemoryOrbEngine(config=config)

    for i in range(10):
        engine.add_turn(
            "user",
            f"Historical note {i}: oauth is archived and not tied to current execution plan.",
            metadata={"importance": 0.1},
        )
        engine.add_turn("assistant", "ack")
    for i in range(10):
        engine.add_turn(
            "user",
            f"Action note {i}: callback signature workflow needs immediate mitigation and rollout steps.",
            metadata={"importance": 1.0},
        )
        engine.add_turn("assistant", "ack")

    packet = engine.build_context(
        user_query="Give me execution steps to fix oauth now.",
        system_prompt="You are an operator assistant.",
    )
    selected = [orb for orb in engine._orbs if orb.orb_id in set(packet.selected_orb_ids)]

    assert selected
    assert any("action note" in orb.summary.lower() or "immediate mitigation" in orb.summary.lower() for orb in selected)


def test_answer_document_loop_collects_evidence_for_fact_question():
    config = MemoryOrbEngineConfig(
        context_max_tokens=600,
        working_max_tokens=180,
        working_target_tokens=120,
        max_retrieved_orbs=5,
    )
    engine = MemoryOrbEngine(config=config)
    model = ValueExtractorModel()

    for i in range(20):
        if i == 9:
            text = "FACT record=REC-42 current_value=queue-gold source=canonical."
            importance = 1.0
        else:
            text = f"noise segment {i} with status updates and backlog movement."
            importance = 0.0
        engine.add_turn("user", text, metadata={"importance": importance})
        engine.add_turn("assistant", "stored")

    result = engine.answer_with_answer_document(
        model=model,
        question="For record REC-42, what is the current_value? Return value.",
        passes=4,
        per_pass_orbs=3,
    )

    assert result.passes_completed >= 1
    assert "rec-42" in result.answer_document.lower()
    assert "queue-gold" in result.answer_document.lower()
    assert result.answer.lower() == "queue-gold"
    assert result.total_tokens <= config.context_max_tokens


def test_answer_document_is_question_first_under_recent_noise():
    config = MemoryOrbEngineConfig(
        context_max_tokens=620,
        working_max_tokens=170,
        working_target_tokens=110,
        max_retrieved_orbs=6,
    )
    engine = MemoryOrbEngine(config=config)
    model = RecordScopedExtractorModel()
    target_record = "REC-900"

    for i in range(16):
        if i == 2:
            text = f"FACT record={target_record} current_value=queue-gold source=canonical."
            importance = 1.0
        else:
            text = f"older noise block {i} with planning chatter and standup summaries."
            importance = 0.0
        engine.add_turn("user", text, metadata={"importance": importance})
        engine.add_turn("assistant", "stored")

    for i in range(12):
        engine.add_turn(
            "user",
            f"recent noisy fact record=REC-{i+100} current_value=queue-red source=recent.",
            metadata={"importance": 0.8},
        )
        engine.add_turn("assistant", "stored")

    result = engine.answer_with_answer_document(
        model=model,
        question=f"For record {target_record}, what is the current_value? Return value.",
        passes=4,
        per_pass_orbs=4,
    )

    assert f"record={target_record.lower()}" in result.answer_document.lower()
    assert "queue-gold" in result.answer_document.lower()
    assert result.answer.lower() == "queue-gold"


def test_answer_document_prefers_final_approved_code_over_replaced_code():
    config = MemoryOrbEngineConfig(
        context_max_tokens=620,
        working_max_tokens=36,
        working_target_tokens=20,
        max_retrieved_orbs=6,
    )
    engine = MemoryOrbEngine(config=config)
    model = PilotBiasModel()

    narrative = [
        "The original charter used MNT-FLOW-163 as a placeholder label.",
        "Pilot runs switched to MNT-FLOW-246 and marked it as temporary.",
        (
            "In the final governance session, the committee retired MNT-FLOW-163, "
            "marked MNT-FLOW-246 as pilot-only, and approved MNT-FLOW-284 for production operations."
        ),
        "The closeout memo confirms MNT-FLOW-284 is the authoritative production control.",
    ]
    for line in narrative:
        engine.add_turn("user", line, metadata={"importance": 1.0})
        engine.add_turn("assistant", "stored")

    result = engine.answer_with_answer_document(
        model=model,
        question=(
            "For Mesa Fare Reconciliation Upgrade, which production control code was finally approved "
            "after replacing MNT-FLOW-246? Return code only."
        ),
        passes=4,
        per_pass_orbs=4,
    )

    assert "mnt-flow-284" in result.answer_document.lower()
    assert "candidate=mnt-flow-284" in result.answer_document.lower()
    assert "mnt-flow-246" in result.answer_document.lower()
    assert "mnt-flow-284" in result.answer.lower()


def test_segment_text_into_scan_areas_hard_wraps_long_sentence():
    engine = MemoryOrbEngine()
    text = " ".join(["segment"] * 260)
    areas = engine._segment_text_into_scan_areas(text, max_chars=120)

    assert len(areas) >= 4
    assert all(len(area) <= 120 for area in areas)


def test_compute_area_dwell_reads_prefers_complex_targeted_area():
    config = MemoryOrbEngineConfig(
        area_dwell_base_reads=1,
        area_dwell_max_reads=3,
        area_dwell_trigger_score=1.45,
        area_dwell_complexity_boost=0.6,
        area_dwell_target_boost=0.8,
    )
    engine = MemoryOrbEngine(config=config)
    orb = MemoryOrb(
        orb_id="orb-1",
        summary="summary",
        raw_excerpt="raw",
        anchors=["rec", "value"],
        embedding=engine.embedder.embed("summary"),
        source_turn_ids=["t-1"],
        created_turn=1,
        salience=0.8,
        tokens=10,
        focus_strength=0.7,
    )
    high_area = _SkimArea(
        area_id="orb-1-area-1",
        source_orb_id="orb-1",
        source_turn_ids=["t-1"],
        source_orb=orb,
        area_index=1,
        text="FACT record=REC-7 current_value=queue-gold source=canonical and governance approval notes.",
        anchors=["rec", "current_value"],
        embedding=engine.embedder.embed("FACT record=REC-7 current_value=queue-gold"),
        tokens=28,
        created_turn=1,
        salience=0.8,
        complexity=0.92,
    )
    low_area = _SkimArea(
        area_id="orb-1-area-2",
        source_orb_id="orb-1",
        source_turn_ids=["t-1"],
        source_orb=orb,
        area_index=2,
        text="short status note.",
        anchors=["status"],
        embedding=engine.embedder.embed("short status note"),
        tokens=4,
        created_turn=1,
        salience=0.3,
        complexity=0.1,
    )

    high_reads = engine._compute_area_dwell_reads(
        skim_score=1.3,
        area=high_area,
        target_record="rec-7",
        target_field="current_value",
    )
    low_reads = engine._compute_area_dwell_reads(
        skim_score=1.3,
        area=low_area,
        target_record="rec-7",
        target_field="current_value",
    )

    assert high_reads == config.area_dwell_max_reads
    assert low_reads < high_reads


def test_load_state_reembeds_orbs_for_current_embedder():
    config = MemoryOrbEngineConfig(
        context_max_tokens=420,
        working_max_tokens=60,
        working_target_tokens=30,
        max_retrieved_orbs=4,
    )
    with TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "state.json"
        engine = MemoryOrbEngine(config=config, embedder=HashingEmbedder(dimensions=64))
        for i in range(12):
            engine.add_turn("user", f"alpha topic {i} with enough detail to force orb creation and retrieval.")
            engine.add_turn("assistant", "stored acknowledgement for alpha topic")
        assert engine._orbs
        engine.save_state(path)

        loaded = MemoryOrbEngine.load_state(path, embedder=HashingEmbedder(dimensions=128))
        packet = loaded.build_context(
            user_query="alpha topic retrieval query",
            system_prompt="You are a retrieval assistant.",
        )

    assert packet.total_tokens <= config.context_max_tokens
    assert packet.selected_orb_ids


def test_load_state_preserves_semantic_card_recency():
    config = MemoryOrbEngineConfig(
        context_max_tokens=420,
        working_max_tokens=36,
        working_target_tokens=18,
        max_retrieved_orbs=4,
    )
    with TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "state.json"
        engine = MemoryOrbEngine(config=config)
        for i in range(6):
            engine.add_turn("user", f"atlas migration planning block {i} with postgres replication details")
            engine.add_turn("assistant", f"stored acknowledgement {i}")
        original_last_turns = {anchor: card.last_turn for anchor, card in engine._semantic_cards.items()}
        assert original_last_turns
        engine.save_state(path)

        loaded = MemoryOrbEngine.load_state(path)

    loaded_last_turns = {anchor: card.last_turn for anchor, card in loaded._semantic_cards.items()}
    assert loaded_last_turns == original_last_turns


def test_answer_document_uses_recent_assistant_and_tool_facts():
    config = MemoryOrbEngineConfig(
        context_max_tokens=640,
        working_max_tokens=2000,
        working_target_tokens=1500,
        max_retrieved_orbs=5,
    )
    engine = MemoryOrbEngine(config=config)
    model = ValueExtractorModel()

    engine.add_turn("assistant", "FACT record=REC-1 current_value=queue-gold source=canonical.")
    engine.add_turn("tool", "FACT record=REC-2 current_value=queue-silver source=canonical.")

    first = engine.answer_with_answer_document(
        model=model,
        question="For record REC-1, what is the current_value? Return value.",
        passes=3,
        per_pass_orbs=2,
    )
    second = engine.answer_with_answer_document(
        model=model,
        question="For record REC-2, what is the current_value? Return value.",
        passes=3,
        per_pass_orbs=2,
    )

    assert first.answer.lower() == "queue-gold"
    assert second.answer.lower() == "queue-silver"


def test_answer_document_does_not_mutate_state_by_default():
    engine = MemoryOrbEngine()

    result = engine.answer_with_answer_document(
        model=DeterministicModel(),
        question="What is project Atlas?",
    )

    assert result.answer.startswith("ACK::What is project Atlas?")
    assert engine.turn_index == 0
    assert list(engine._working_turns) == []


def test_answer_document_uses_single_system_message_for_backend_compatibility():
    engine = MemoryOrbEngine()
    captured_messages: list[dict[str, str]] = []

    class CaptureModel(ModelAdapter):
        def complete(self, messages):
            captured_messages.extend(messages)
            return "captured"

    result = engine.answer_with_answer_document(
        model=CaptureModel(),
        question="What is project Atlas?",
    )

    system_messages = [msg for msg in captured_messages if msg["role"] == "system"]

    assert result.answer == "captured"
    assert len(system_messages) == 1
    assert "You are a factual QA assistant." in system_messages[0]["content"]
    assert "Answer Document:" in system_messages[0]["content"]


def test_semantic_cards_skip_mutable_fact_summaries():
    config = MemoryOrbEngineConfig(
        context_max_tokens=420,
        working_max_tokens=60,
        working_target_tokens=30,
    )
    engine = MemoryOrbEngine(config=config)

    engine.add_turn("user", "FACT record=REC-1 current_value=queue-gold source=canonical.")
    engine.add_turn("assistant", "stored")
    engine.add_turn("user", "FACT record=REC-1 current_value=queue-red source=canonical.")
    engine.add_turn("assistant", "stored")

    card_text = " ".join(card.synopsis.lower() for card in engine._semantic_cards.values())

    assert "current_value=" not in card_text
    assert "queue-gold" not in card_text
    assert "queue-red" not in card_text


def test_semantic_cards_do_not_use_role_or_boundary_tokens_as_anchors():
    config = MemoryOrbEngineConfig(
        context_max_tokens=420,
        working_max_tokens=36,
        working_target_tokens=18,
    )
    engine = MemoryOrbEngine(config=config)

    for i in range(6):
        engine.add_turn("user", f"Atlas planning note {i} covering postgres replication and owner details.")
        engine.add_turn("assistant", f"stored acknowledgement {i}")

    anchors = set(engine._semantic_cards)

    assert anchors
    assert not ({"assistant", "user", "stored", "tool", "system"} & anchors)


def test_answer_document_does_not_mutate_orb_focus_state_by_default():
    config = MemoryOrbEngineConfig(
        context_max_tokens=620,
        working_max_tokens=80,
        working_target_tokens=40,
        max_retrieved_orbs=4,
    )
    engine = MemoryOrbEngine(config=config)
    model = ValueExtractorModel()

    for i in range(10):
        if i == 4:
            text = "FACT record=REC-77 current_value=queue-gold source=canonical."
            importance = 1.0
        else:
            text = f"noise segment {i} with unrelated narrative details."
            importance = 0.0
        engine.add_turn("user", text, metadata={"importance": importance})
        engine.add_turn("assistant", "stored")

    before = {
        orb.orb_id: (orb.focus_strength, orb.pinned_until_turn, orb.accesses, orb.last_focus_turn)
        for orb in engine._orbs
    }
    result = engine.answer_with_answer_document(
        model=model,
        question="For record REC-77, what is the current_value? Return value.",
        passes=3,
        per_pass_orbs=2,
    )
    after = {
        orb.orb_id: (orb.focus_strength, orb.pinned_until_turn, orb.accesses, orb.last_focus_turn)
        for orb in engine._orbs
        if orb.orb_id in before
    }

    assert result.answer.lower() == "queue-gold"
    assert after == before


def test_answer_document_can_persist_a_coherent_exchange_when_enabled():
    engine = MemoryOrbEngine()
    question = "What is project Borealis?"

    result = engine.answer_with_answer_document(
        model=DeterministicModel(),
        question=question,
        update_memory_state=True,
    )

    assert result.answer.startswith("ACK::What is project Borealis?")
    assert engine.turn_index == 2
    assert [turn.role for turn in engine._working_turns] == ["user", "assistant"]
    assert engine._working_turns[0].content == question
    assert bool(engine._working_turns[0].metadata.get("exclude_from_question_memory_pool")) is True
    assert bool(engine._working_turns[1].metadata.get("exclude_from_question_memory_pool")) is True


def test_pulse_map_skips_excluded_document_turns():
    engine = MemoryOrbEngine()
    engine.add_turn(
        "tool",
        "benchmark document about queue-gold and record REC-77",
        metadata={"exclude_from_pulse_map": True},
    )
    engine.add_turn("user", "Need the status for REC-77 now.")

    pulses = engine._pulse_map()

    assert "need" not in pulses
    assert "queue-gold" not in pulses
    assert "benchmark" not in pulses
    assert "rec-77" in pulses or "rec" in pulses


def test_reasoned_dwell_uses_reasoning_reader_for_high_signal_area():
    config = MemoryOrbEngineConfig(
        context_max_tokens=640,
        working_max_tokens=180,
        working_target_tokens=120,
        max_retrieved_orbs=5,
        answer_dwell_mode="reasoned",
        reasoning_dwell_trigger_score=1.0,
        reasoning_dwell_min_complexity=0.0,
    )
    engine = MemoryOrbEngine(config=config)
    model = ReasoningRecordScopedModel()

    for i in range(18):
        if i == 7:
            text = "FACT record=REC-55 current_value=queue-gold source=canonical with detailed governance notes."
            importance = 1.0
        else:
            text = f"noise segment {i} with unrelated planning chatter and backlog movement."
            importance = 0.0
        engine.add_turn("user", text, metadata={"importance": importance})
        engine.add_turn("assistant", "stored")

    result = engine.answer_with_answer_document(
        model=model,
        dwell_model=model,
        question="For record REC-55, what is the current_value? Return value.",
        passes=4,
        per_pass_orbs=3,
    )

    assert result.answer.lower() == "queue-gold"
    assert "reader=reasoned" in result.answer_document.lower()


def test_reasoned_mode_falls_back_to_heuristic_without_reasoning_adapter():
    config = MemoryOrbEngineConfig(
        context_max_tokens=620,
        working_max_tokens=180,
        working_target_tokens=120,
        max_retrieved_orbs=5,
        answer_dwell_mode="reasoned",
        reasoning_dwell_trigger_score=1.0,
        reasoning_dwell_min_complexity=0.0,
    )
    engine = MemoryOrbEngine(config=config)
    model = RecordScopedExtractorModel()

    for i in range(16):
        if i == 4:
            text = "FACT record=REC-88 current_value=queue-gold source=canonical."
            importance = 1.0
        else:
            text = f"noise segment {i} with unrelated operational narrative."
            importance = 0.0
        engine.add_turn("user", text, metadata={"importance": importance})
        engine.add_turn("assistant", "stored")

    result = engine.answer_with_answer_document(
        model=model,
        question="For record REC-88, what is the current_value? Return value.",
        passes=3,
        per_pass_orbs=3,
    )

    assert result.answer.lower() == "queue-gold"
    assert "reader=heuristic" in result.answer_document.lower()


def test_comparative_multiple_choice_area_can_trigger_reasoned_dwell():
    config = MemoryOrbEngineConfig(
        context_max_tokens=620,
        working_max_tokens=180,
        working_target_tokens=120,
        answer_dwell_mode="reasoned",
        reasoning_dwell_trigger_score=1.55,
        reasoning_dwell_min_complexity=0.42,
    )
    engine = MemoryOrbEngine(config=config)
    model = ReasoningRecordScopedModel()
    orb = MemoryOrb(
        orb_id="orb-comp",
        summary="comparative analyst notes",
        raw_excerpt="HSBC versus Deutsche Bank travel outlook memo",
        anchors=["hsbc", "deutsche", "travel"],
        embedding=engine.embedder.embed("HSBC versus Deutsche Bank travel outlook memo"),
        source_turn_ids=["t-1"],
        created_turn=1,
        salience=0.8,
        tokens=22,
        focus_strength=0.35,
    )
    area = _SkimArea(
        area_id="orb-comp-area-1",
        source_orb_id="orb-comp",
        source_turn_ids=["t-1"],
        source_orb=orb,
        area_index=1,
        text="HSBC emphasized domestic recovery while Deutsche Bank highlighted inbound travel and visa-free entry.",
        anchors=["hsbc", "domestic", "deutsche", "inbound", "travel"],
        embedding=engine.embedder.embed(
            "HSBC emphasized domestic recovery while Deutsche Bank highlighted inbound travel and visa-free entry."
        ),
        tokens=24,
        created_turn=1,
        salience=0.8,
        complexity=0.24,
    )

    should_reason = engine._should_use_reasoned_dwell(
        model=model,
        area=area,
        skim_score=1.22,
        target_record="",
        target_field="",
        prefer_final_code=False,
        reasoned_area_reads_used=0,
        question=(
            "Based on a comparative analysis of the reports, which option best captures the divergence?\n"
            "A. HSBC emphasizes domestic recovery while Deutsche Bank highlights outbound travel.\n"
            "B. Goldman Sachs focuses on package tours.\n"
            "C. Deutsche Bank underscores inbound travel to China through visa-free entry while HSBC stresses outbound demand.\n"
            "D. Goldman Sachs and J.P. Morgan agree on outbound demand."
        ),
    )

    assert should_reason is True


def test_skim_plan_penalizes_disclosure_boilerplate():
    engine = MemoryOrbEngine()
    orb = MemoryOrb(
        orb_id="orb-report",
        summary="analyst report",
        raw_excerpt="report text",
        anchors=["trip", "deutsche", "travel"],
        embedding=engine.embedder.embed("Trip.com analyst report"),
        source_turn_ids=["t-1"],
        created_turn=1,
        salience=0.7,
        tokens=24,
        focus_strength=0.2,
    )
    useful = _SkimArea(
        area_id="orb-report-area-1",
        source_orb_id="orb-report",
        source_turn_ids=["t-1"],
        source_orb=orb,
        area_index=1,
        text="Deutsche Bank highlights inbound travel to China and visa-free entry policies as a growth driver.",
        anchors=["deutsche", "travel", "china", "visa-free"],
        embedding=engine.embedder.embed(
            "Deutsche Bank highlights inbound travel to China and visa-free entry policies as a growth driver."
        ),
        tokens=22,
        created_turn=1,
        salience=0.7,
        complexity=0.36,
    )
    boilerplate = _SkimArea(
        area_id="orb-report-area-2",
        source_orb_id="orb-report",
        source_turn_ids=["t-1"],
        source_orb=orb,
        area_index=2,
        text="Important disclosures and company-specific disclosures. Deutsche Bank AG/Hong Kong price target and market cap details.",
        anchors=["deutsche", "price", "target"],
        embedding=engine.embedder.embed(
            "Important disclosures and company-specific disclosures. Deutsche Bank AG/Hong Kong price target and market cap details."
        ),
        tokens=20,
        created_turn=1,
        salience=0.7,
        complexity=0.24,
    )

    sweep, _ = engine._plan_area_skim_sweep(
        areas=[boilerplate, useful],
        question=(
            "Based on a comparative analysis, which option best captures Deutsche Bank's inbound travel thesis?\n"
            "A. Domestic recovery.\n"
            "B. Package tours.\n"
            "C. Inbound travel and visa-free entry.\n"
            "D. Margin caution."
        ),
        anchors=["deutsche", "travel", "inbound", "visa-free"],
        used_area_ids=set(),
        max_candidates=4,
        max_evals=2,
    )

    assert sweep
    assert sweep[0][0].area_id == "orb-report-area-1"
