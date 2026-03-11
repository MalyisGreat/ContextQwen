from __future__ import annotations

import inspect
import json

from benchmarks import longbench_v2_compare as compare
from memory_orb.adapters import SimpleTokenEstimator
from memory_orb.structured_readers import ProcedureReader
from memory_orb.structured_readers import QuestionPacket
from memory_orb.structured_readers import StructureProfile
from memory_orb.structured_readers import TableReader
from memory_orb.structured_readers import _truncate_context_window
from memory_orb.structured_readers import _decompose_option_claims
from memory_orb.structured_readers import _extract_parameter_records
from memory_orb.structured_readers import _parse_manual_sections
from memory_orb.structured_readers import _select_manual_answer
from memory_orb.structured_readers import profile_structure
from memory_orb.structured_readers import route_structured_reader


class _FixedModel:
    def __init__(self, response_text: str) -> None:
        self.response_text = response_text

    def complete(self, messages: list[dict[str, str]]) -> str:
        return self.response_text


class _ManualEvalModel:
    def complete(self, messages: list[dict[str, str]]) -> str:
        prompt = messages[-1]["content"]
        if "overall_status" in messages[0]["content"]:
            if "disabled, Device Reset is available" in prompt:
                return json.dumps(
                    {
                        "claims": [
                            {
                                "claim": "When Secure Boot is disabled, Device Reset is available.",
                                "status": "contradicted",
                                "evidence": "Secure Boot disables Device Reset only when locked mode is active.",
                            }
                        ],
                        "overall_status": "contradicted",
                    }
                )
            if "default password is admin123" in prompt:
                return json.dumps(
                    {
                        "claims": [
                            {
                                "claim": "The default password is admin123.",
                                "status": "supported",
                                "evidence": "Default password: admin123",
                            }
                        ],
                        "overall_status": "supported",
                    }
                )
        if "claim matrix" in messages[0]["content"].lower():
            return '{"answer":"B"}'
        return '{"claims":["fallback"]}'


def _country_frequency_packet() -> QuestionPacket:
    return QuestionPacket(
        question="Which two countries are the second and third frequently mentioned among all these organization addresses?",
        options={
            "A": "Italy as the second and China as the third.",
            "B": "Germany as the second and Italy as the third.",
            "C": "Japan as the second and Germany as the third.",
            "D": "United States as the second and China as the third.",
        },
        raw_context="\n".join(
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
        ),
    )


def _music_trend_packet() -> QuestionPacket:
    return QuestionPacket(
        question="Which option best reflects the underlying trend in how genre and language evolve over time?",
        options={
            "A": "Genres become less diverse over time and English disappears after 2018.",
            "B": "Genres stay fixed year to year and language is unrelated.",
            "C": "Genre mix shifts over time while language diversity expands in later years.",
            "D": "The table shows no year-based trend at all.",
        },
        raw_context="\n".join(
            [
                "cluster,title,artist,album,year,language,genre",
                "1,track a,artist a,album a,2016,English,Pop",
                "1,track b,artist b,album b,2016,English,Pop",
                "2,track c,artist c,album c,2017,Spanish,Latin",
                "2,track d,artist d,album d,2017,English,Pop",
                "3,track e,artist e,album e,2018,Korean,K-Pop",
                "3,track f,artist f,album f,2018,Japanese,J-Pop",
                "4,track g,artist g,album g,2019,Spanish,Reggaeton",
                "4,track h,artist h,album h,2019,English,Hip-Hop",
                "5,track i,artist i,album i,2020,Korean,K-Pop",
                "5,track j,artist j,album j,2020,English,Indie",
            ]
        ),
    )


def _manual_packet() -> QuestionPacket:
    return QuestionPacket(
        question="Which statement is false according to the manual?",
        options={
            "A": "The audit log stores the last 200 events.",
            "B": "When Secure Boot is disabled, Device Reset is available.",
            "C": "The default password is admin123.",
            "D": "Warning text states a reboot is required after changing Mode Lock.",
        },
        raw_context="\n".join(
            [
                "TABLE OF CONTENTS",
                "1 Overview",
                "2 Security Parameters",
                "3 Warning Notes",
                "",
                "1 Overview",
                "Overview",
                "The device manual covers register setup and parameter defaults.",
                "NOTE: The audit log stores the last 200 events.",
                "",
                "2 Security Parameters",
                "Mode Lock",
                "Default: enabled",
                "If Mode Lock changes, a reboot is required.",
                "WARNING: Changing Mode Lock requires restart before Device Reset is re-enabled.",
                "",
                "Secure Boot",
                "Default password: admin123",
                "When Secure Boot is disabled, Device Reset remains unavailable while locked mode is active.",
                "Parameter Register",
                "Overview of password and register behavior.",
            ]
        ),
    )


def test_profile_structure_detects_table_positive():
    profile = profile_structure(_country_frequency_packet())
    assert profile.shape == "table"
    assert profile.confidence >= 0.8
    assert profile.delimiter == ","


def test_profile_structure_detects_quoted_csv_table():
    packet = QuestionPacket(
        question="Which country appears second most frequently?",
        options={"A": "Germany", "B": "Italy", "C": "Japan", "D": "China"},
        raw_context="\n".join(
            [
                "Registry,Assignment,Organization Name,Organization Address",
                'MA-S,1,Foo,"Main st., 3 Berlin DE 10000"',
                'MA-S,2,Bar,"High st., 4 Rome IT 10002"',
                'MA-S,3,Baz,"Lake st., 5 Tokyo JP 10003"',
                'MA-S,4,Qux,"Broadway, 7 New York US 10004"',
                'MA-S,5,Norf,"Another st., 8 Berlin DE 10005"',
                'MA-S,6,Plugh,"Third st., 9 Rome IT 10006"',
                'MA-S,7,Xyzzy,"Fourth st., 10 Seattle US 10007"',
                'MA-S,8,Thud,"Fifth st., 11 Osaka JP 10008"',
            ]
        ),
    )

    profile = profile_structure(packet)

    assert profile.shape == "table"
    assert profile.confidence >= 0.8


def test_profile_structure_detects_manual_positive():
    profile = profile_structure(_manual_packet())
    assert profile.shape == "manual"
    assert profile.confidence >= 0.75
    assert "manual" in profile.section_markers


def test_profile_structure_keeps_plain_prose_out_of_specialized_paths():
    packet = QuestionPacket(
        question="What is the main argument?",
        options={"A": "a", "B": "b", "C": "c", "D": "d"},
        raw_context="This is plain narrative prose.\nIt has paragraphs and examples.\nNothing is formatted as a manual or table.",
    )
    profile = profile_structure(packet)
    assert profile.shape == "prose"


def test_route_structured_reader_uses_table_reader_for_case_66f95e11bb02136c067c5370():
    profile, outcome = route_structured_reader(
        packet=_country_frequency_packet(),
        model=_FixedModel('{"answer":"A"}'),
    )
    assert profile.shape == "table"
    assert outcome is not None
    assert outcome.route_name == "table/deterministic-rank"
    assert outcome.answer_pred == "B"
    assert outcome.deterministic is True


def test_table_reader_temporal_trend_summary_for_case_67039cfabb02136c067cd04e():
    packet = _music_trend_packet()
    profile = profile_structure(packet)
    reader = TableReader()

    outcome = reader.answer(packet, profile, _FixedModel('{"answer":"C"}'))

    assert outcome.route_name == "table/temporal-trend"
    assert outcome.answer_pred == "C"
    assert any("Temporal trend summary" in line for line in outcome.evidence_lines)
    assert any("candidate grouping column: language" in line.lower() for line in outcome.evidence_lines)
    assert any("2018" in line or "2019" in line for line in outcome.evidence_lines)


def test_parse_manual_sections_and_parameter_records():
    sections = _parse_manual_sections(_manual_packet().raw_context)
    parameters = _extract_parameter_records(sections)

    assert any(section.title.startswith("2 Security Parameters") for section in sections)
    assert any(record.name == "Mode Lock" for record in parameters)
    assert any(record.default == "enabled" for record in parameters)


def test_decompose_option_claims_uses_model_fallback_only_when_needed():
    long_option = (
        "The controller automatically resets itself after a lockout event because the unattended recovery workflow "
        "propagates every audit log to remote supervisors without any explicit operator prompt during validation."
    )
    claims = _decompose_option_claims(long_option, _FixedModel('{"claims":["claim one","claim two"]}'), 6)

    assert claims == ["claim one", "claim two"]


def test_procedure_reader_finds_false_option_for_case_66ec4370821e116aacb1c905():
    packet = _manual_packet()
    profile = profile_structure(packet)
    reader = ProcedureReader()

    outcome = reader.answer(packet, profile, _ManualEvalModel())

    assert outcome.route_name == "manual/claim-matrix"
    assert outcome.answer_pred == "B"
    assert any(line.startswith("B: overall=") for line in outcome.evidence_lines)


def test_procedure_reader_rejects_finance_report_false_positive():
    packet = QuestionPacket(
        question="Which policy mix best stabilizes debt while preserving export revenue?",
        options={
            "A": "Raise carbon taxes and expand green bonds.",
            "B": "Issue guarantees for private lenders.",
            "C": "Cut spending while delaying renewable investment.",
            "D": "Maintain the current export-led fiscal regime.",
        },
        raw_context="\n".join(
            [
                "PAGE | 31 :: Finance",
                "SUSTAINABLE FINANCE: BRIDGING THE GAP IN ASIA AND THE PACIFIC",
                "NOTE: MDB guarantees can lower the cost of capital.",
                "PAGE | 32 :: Finance",
                "Transition pathways are constrained by fiscal buffers and export demand.",
                "PAGE | 33 :: Finance",
                "WARNING: This report contains forward-looking statements.",
                "PAGE | 34 :: Finance",
                "Comparative discussion of debt sustainability and export earnings.",
            ]
        ),
    )
    manual_like_profile = StructureProfile(
        shape="manual",
        confidence=0.91,
        delimiter=None,
        header_rows=0,
        section_markers=("note", "warning"),
        signals={"manual_score": 0.91},
    )

    assert ProcedureReader().match(packet, manual_like_profile) is False


def test_select_manual_answer_penalizes_contradicted_clause():
    choice, deterministic = _select_manual_answer(
        "which statement is false according to the manual?",
        {
            "A": {"claims": [{"status": "supported"}]},
            "B": {"claims": [{"status": "contradicted"}]},
            "C": {"claims": [{"status": "supported"}]},
            "D": {"claims": [{"status": "unresolved"}]},
        },
    )

    assert choice == "B"
    assert deterministic is True


def test_runtime_routing_does_not_reference_benchmark_metadata():
    route_source = inspect.getsource(compare._should_route_reasoned_chat)
    structured_route_source = inspect.getsource(compare._route_structured_memory_case)
    for banned in ("case.answer", "case.domain", "case.sub_domain"):
        assert banned not in route_source
        assert banned not in structured_route_source


def test_runtime_routing_uses_question_packet_not_bench_case():
    signature = inspect.signature(compare._route_structured_memory_case)
    assert "packet" in signature.parameters
    assert "case" not in signature.parameters


def test_reasoned_answer_doc_paths_disable_answer_coercion():
    supplement_source = inspect.getsource(compare._build_reasoned_chat_supplement)
    probe_source = inspect.getsource(compare._build_option_probe_supplement)
    assert "allow_answer_coercion=False" in supplement_source
    assert "allow_answer_coercion=False" in probe_source


def test_truncate_context_window_preserves_head_and_tail_within_budget():
    text = "\n".join(f"row {i} value with repeated context markers" for i in range(120))
    estimator = SimpleTokenEstimator()

    truncated = _truncate_context_window(text, estimator, 80)

    assert estimator.count(truncated) <= 80
    assert "row 0 value" in truncated
    assert "row 119 value" in truncated
    assert "..." in truncated
