from __future__ import annotations

from benchmarks.api_backend import ChatBackendConfig
from benchmarks.longbench_v2_compare import _build_mc_prompt
from benchmarks.longbench_v2_compare import _build_question_packet
from benchmarks.longbench_v2_compare import _compose_system_message
from benchmarks.longbench_v2_compare import _chunk_text
from benchmarks.longbench_v2_compare import _extract_choice
from benchmarks.longbench_v2_compare import _extract_reasoned_chat_evidence
from benchmarks.longbench_v2_compare import _OllamaAdapter
from benchmarks.longbench_v2_compare import _looks_like_tabular_context
from benchmarks.longbench_v2_compare import _map_permuted_choice
from benchmarks.longbench_v2_compare import _normalize_reasoned_option_analysis
from benchmarks.longbench_v2_compare import _parse_tabular_context
from benchmarks.longbench_v2_compare import _permute_case_labels
from benchmarks.longbench_v2_compare import _should_route_reasoned_chat
from benchmarks.longbench_v2_compare import _try_answer_table_rank_question
from benchmarks.longbench_v2_compare import BenchCase
from memory_orb.structured_readers import profile_structure


def test_extract_choice_from_json_and_plain_text():
    assert _extract_choice('{"answer":"C"}') == "C"
    assert _extract_choice('{"value":"b"}') == "B"
    assert _extract_choice("A: 0.21 weak\nB: 0.92 supported\nC: 0.11 contradicted\nD: 0.18 weak") == "B"
    assert _extract_choice("Answer: D") == "D"
    assert _extract_choice("A") == "A"
    assert _extract_choice("No valid answer") == ""


def test_chunk_text_preserves_content_non_empty():
    text = "alpha " * 400 + "\n" + "beta " * 400 + "\n" + "gamma " * 400
    chunks = _chunk_text(text, chunk_chars=300)
    assert len(chunks) >= 3
    combined = " ".join(chunks)
    assert "alpha" in combined
    assert "beta" in combined
    assert "gamma" in combined


def test_build_mc_prompt_contains_all_options():
    case = BenchCase(
        case_id="x",
        length="short",
        difficulty="hard",
        domain="domain",
        sub_domain="sub",
        question="What is correct?",
        context="ctx",
        choice_a="opt1",
        choice_b="opt2",
        choice_c="opt3",
        choice_d="opt4",
        answer="B",
    )
    prompt = _build_mc_prompt(case)
    assert "A. opt1" in prompt
    assert "B. opt2" in prompt
    assert "C. opt3" in prompt
    assert "D. opt4" in prompt
    assert "Return exactly four lines" in prompt


def test_permute_case_labels_changes_option_order_and_maps_back():
    case = BenchCase(
        case_id="perm-1",
        length="medium",
        difficulty="hard",
        domain="domain",
        sub_domain="sub",
        question="Which answer is correct?",
        context="ctx",
        choice_a="alpha",
        choice_b="beta",
        choice_c="gamma",
        choice_d="delta",
        answer="C",
    )

    permuted, permuted_to_original = _permute_case_labels(case, "seed:perm-1")

    assert (permuted.choice_a, permuted.choice_b, permuted.choice_c, permuted.choice_d) != (
        case.choice_a,
        case.choice_b,
        case.choice_c,
        case.choice_d,
    )
    assert _map_permuted_choice(permuted.answer, permuted_to_original) == case.answer


def test_extract_reasoned_chat_evidence_condenses_answer_doc_lines():
    answer_doc = "\n".join(
        [
            "Answer Document:",
            "- Question: Which option is correct?",
            "- Evidence collected:",
            "- pass=1 area=a1 source=o1 skim=1.40 dwell=2 reader=reasoned complexity=0.44 score=2.13: supports option C because the policy was retired after June",
            "- pass=2 area=a2 source=o2 skim=1.10 dwell=1 reader=heuristic complexity=0.31 score=1.54: rules out option A because the contract remained active",
            "- Code candidate ranking:",
            "- candidate=OPT-C score=2.80",
        ]
    )

    supplement = _extract_reasoned_chat_evidence(answer_doc, max_lines=4)

    assert supplement.startswith("Reasoned skim evidence:")
    assert "supports option C because the policy was retired after June" in supplement
    assert "rules out option A because the contract remained active" in supplement
    assert "- candidate=OPT-C score=2.80" in supplement
    assert "area=a1" not in supplement


def test_normalize_reasoned_option_analysis_keeps_only_option_lines():
    raw = "\n".join(
        [
            "A: contradicted - demand remained flat.",
            "noise line",
            "B: weak - margin commentary is indirect.",
            "C: supported - inbound travel and visa-free policy are highlighted.",
            "D: contradicted - both reports do not agree on outbound dominance.",
            "BEST: C - strongest explicit evidence.",
            "extra trailing note",
        ]
    )

    normalized = _normalize_reasoned_option_analysis(raw)

    assert "noise line" not in normalized
    assert "extra trailing note" not in normalized
    assert "BEST: C - strongest explicit evidence." in normalized


def test_should_route_reasoned_chat_only_for_comparative_cases():
    table_context = "\n".join(
        [
            "country,address,count",
            "US,addr1,10",
            "DE,addr2,9",
            "IT,addr3,8",
            "JP,addr4,7",
            "CN,addr5,6",
            "FR,addr6,5",
            "GB,addr7,4",
            "CA,addr8,3",
        ]
    )
    comparative = BenchCase(
        case_id="cmp",
        length="medium",
        difficulty="hard",
        domain="Multi-Document QA",
        sub_domain="Financial",
        question="Based on a comparative analysis of the reports, which option best encapsulates the primary divergence?",
        context="ctx",
        choice_a="a",
        choice_b="b",
        choice_c="c",
        choice_d="d",
        answer="A",
    )
    table_case = BenchCase(
        case_id="tbl",
        length="medium",
        difficulty="easy",
        domain="Long Structured Data Understanding",
        sub_domain="Table QA",
        question="Which row has the highest value?",
        context=table_context,
        choice_a="a",
        choice_b="b",
        choice_c="c",
        choice_d="d",
        answer="B",
    )
    policy_mix_case = BenchCase(
        case_id="pol",
        length="medium",
        difficulty="hard",
        domain="Single-Document QA",
        sub_domain="Financial",
        question="Which policy mix should the government pursue to best balance fiscal sustainability and the energy transition?",
        context="ctx",
        choice_a="a",
        choice_b="b",
        choice_c="c",
        choice_d="d",
        answer="B",
    )

    comparative_packet = _build_question_packet(comparative)
    comparative_profile = profile_structure(comparative_packet)
    table_packet = _build_question_packet(table_case)
    table_profile = profile_structure(table_packet)
    policy_packet = _build_question_packet(policy_mix_case)
    policy_profile = profile_structure(policy_packet)

    assert _should_route_reasoned_chat(comparative_packet, comparative_profile) is True
    assert _should_route_reasoned_chat(table_packet, table_profile) is False
    assert _should_route_reasoned_chat(policy_packet, policy_profile) is True


def test_looks_like_tabular_context_and_rank_solver():
    context = "\n".join(
        [
            "Registry,Assignment,Organization Name,Organization Address",
            "MA-S,0,Z,Street Boston US 09999",
            "MA-S,1,A,Street Berlin DE 10000",
            "MA-S,2,B,Street Munich DE 10001",
            "MA-S,3,C,Street Rome IT 10002",
            "MA-S,4,D,Street New York US 10003",
            "MA-S,5,E,Street Tokyo JP 10004",
            "MA-S,6,F,Street Turin IT 10005",
            "MA-S,7,G,Street Seattle US 10006",
            "MA-S,8,H,Street Osaka JP 10007",
        ]
    )
    case = BenchCase(
        case_id="tbl-rank",
        length="medium",
        difficulty="hard",
        domain="Long Structured Data Understanding",
        sub_domain="Table QA",
        question="Which two countries are the second and third frequently mentioned among all these organization addresses?",
        context=context,
        choice_a="Italy as the second and China as the third.",
        choice_b="Germany as the second and Italy as the third.",
        choice_c="Japan as the second and Germany as the third.",
        choice_d="United States as the second and China as the third.",
        answer="B",
    )

    assert _looks_like_tabular_context(context) is True
    headers, rows = _parse_tabular_context(context)
    assert headers[:2] == ["Registry", "Assignment"]
    assert _try_answer_table_rank_question(case, headers, rows) == '{"answer": "B"}'


def test_ollama_adapter_think_can_be_forced_for_qwen35():
    default_adapter = _OllamaAdapter(
        model_name="qwen3.5:0.8b",
        num_ctx=900,
        timeout_s=30,
        backend=ChatBackendConfig(provider="ollama"),
    )
    forced_adapter = _OllamaAdapter(
        model_name="qwen3.5:0.8b",
        num_ctx=900,
        timeout_s=30,
        backend=ChatBackendConfig(provider="ollama"),
        enable_think=True,
    )

    assert default_adapter._should_use_ollama_think(True) is False
    assert forced_adapter._should_use_ollama_think(True) is True


def test_non_ollama_backend_disables_think_flag():
    adapter = _OllamaAdapter(
        model_name="Qwen/Qwen3.5-0.8B",
        num_ctx=900,
        timeout_s=30,
        backend=ChatBackendConfig(provider="openai", api_base="http://127.0.0.1:8000/v1"),
        enable_think=True,
    )

    assert adapter._should_use_ollama_think(True) is False


def test_compose_system_message_skips_empty_sections():
    combined = _compose_system_message("base instructions", "", "memory section")

    assert combined == "base instructions\n\nmemory section"
