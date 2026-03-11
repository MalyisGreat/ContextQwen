from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime
from datetime import timedelta
from typing import Any

try:
    from benchmarks.linked_longform_benchmark import BLUEPRINTS
    from benchmarks.linked_longform_benchmark import _chunk_text
    from benchmarks.linked_longform_benchmark import _ollama_chat
except ImportError:
    from linked_longform_benchmark import BLUEPRINTS
    from linked_longform_benchmark import _chunk_text
    from linked_longform_benchmark import _ollama_chat
from memory_orb import MemoryOrbEngine
from memory_orb import MemoryOrbEngineConfig


CODE_PATTERN = re.compile(r"\b[A-Z]{2,}(?:-[A-Z0-9]{2,}){2,}\b")
DATE_PATTERN = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b",
    flags=re.IGNORECASE,
)
NORMALIZE_RE = re.compile(r"[^a-z0-9]+")


@dataclass(slots=True)
class HardLinkedCase:
    case_id: str
    title: str
    context: str
    question: str
    expected: dict[str, str]
    legacy_code: str
    pilot_code: str
    decoy_code: str
    decoy_approver: str
    decoy_date: str


@dataclass(slots=True)
class HardLinkedCaseResult:
    case_id: str
    title: str
    expected: dict[str, str]
    predicted: dict[str, str]
    raw_response: str
    final_code_exact: int
    approver_exact: int
    final_date_exact: int
    constraint_match: int
    all_fields_match: int


def _normalize(text: str) -> str:
    return NORMALIZE_RE.sub(" ", (text or "").lower()).strip()


def _format_date(date_obj: datetime) -> str:
    return date_obj.strftime("%B %d, %Y").replace(" 0", " ")


def _make_decoy_date(final_date: str, days_back: int = 19) -> str:
    parsed = datetime.strptime(final_date, "%B %d, %Y")
    return _format_date(parsed - timedelta(days=days_back))


def _make_decoy_code(legacy_code: str, idx: int) -> str:
    prefix = legacy_code.split("-")[0]
    return f"{prefix}-DRAFT-{740 + idx * 7}"


def _build_hard_context(idx: int) -> str:
    bp = BLUEPRINTS[idx]
    approvers = [item.approver for item in BLUEPRINTS]
    decoy_approver = approvers[(idx + 3) % len(approvers)]
    if decoy_approver == bp.approver:
        decoy_approver = approvers[(idx + 4) % len(approvers)]
    decoy_date = _make_decoy_date(bp.final_date, days_back=17 + (idx % 4))
    decoy_code = _make_decoy_code(bp.legacy_code, idx)

    paragraphs = [
        (
            f"{bp.organization} launched the {bp.program} after an uncomfortable quarter in which shift leaders were forced to narrate, in very long handoff notes, "
            f"how {bp.asset} drifted during low-staff hours in {bp.region}, because the reported rollups looked deceptively calm whenever teams summarized incidents "
            f"in short bullet points rather than writing complete operating narratives with sequence, timing, and control labels."
        ),
        (
            f"Program facilitators asked each team to preserve old labels and temporary labels in context, rather than deleting them, so future reviewers could see that "
            f"{bp.legacy_code} had been introduced as a placeholder and that {decoy_code} appeared in one procurement draft that was never approved for production, even "
            "though the draft circulated long enough to confuse onboarding packets and one internal scorecard."
        ),
        (
            f"In one interim governance rehearsal, {decoy_approver} signed a temporary extension memo on {decoy_date} that allowed a pilot rhythm to continue while controls "
            f"were tested under edge conditions; that temporary extension explicitly referred to {bp.pilot_code}, and the text warned that the extension should not be read "
            "as final approval, yet several copied excerpts dropped the warning clause and made later recall tasks harder."
        ),
        (
            f"Engineering reports then explained, with unusually long sentence chains that connected queue pressure, handoff timing, and regulatory windows, that "
            f"{bp.pilot_code} reduced obvious failures but still could not handle {bp.constraint}, so the pilot could stabilize operations only when supervisors added manual "
            "buffers, which made it operationally helpful but governance-incomplete."
        ),
        (
            "Because teams repeatedly mixed draft facts and final facts, the review committee required every status update to list three categories in order: retired labels, "
            "pilot-only labels, and candidate production labels; this ordering constraint, while tedious, made it possible to inspect whether a statement referred to past "
            "state, temporary state, or approved state without trusting a short summary sentence."
        ),
        (
            f"In the final governance session, which was recorded in both legal minutes and engineering release notes to prevent wording drift, the committee retired "
            f"{bp.legacy_code}, marked {bp.pilot_code} as pilot-only, and approved {bp.final_code} for production operations; the sign-off block was executed by "
            f"{bp.approver} on {bp.final_date}, and distribution instructions said legacy and pilot references must remain only in historical sections."
        ),
        (
            f"Immediately after that session, deployment coordinators wrote a closeout memo stating that only {bp.final_code} should appear in active runbooks, and that any "
            f"reference to {bp.pilot_code}, {bp.legacy_code}, or {decoy_code} must be interpreted as historical context; this was repeated because search results sometimes "
            "surfaced archived pages before current procedures when people searched by code fragment."
        ),
        (
            "To reduce ambiguity for future incident drills, trainers produced scenario narratives in which a single sentence intentionally mentioned an old label, a pilot label, "
            "and the production label in close proximity, then required participants to annotate which clause represented historical context and which clause represented active policy, "
            "because many previous errors came from people identifying a correct token but attaching it to the wrong timeline state."
        ),
        (
            "The records team also preserved verbose meeting transcripts where speakers corrected each other in real time, and those transcripts showed that shorthand references such as "
            "the last three digits of a code were dangerous when two nearby labels shared a prefix, so every compliance packet began with a rule that labels must be copied in full and "
            "paired with status words like retired, pilot-only, or approved-for-production before decisions were recorded."
        ),
        (
            "During rollout, on-call supervisors traced anomalies with full narrative paragraphs instead of terse notes, and these paragraphs intentionally repeated names, dates, "
            "and code labels in different clause orders so that an auditor reading only one fragment would still be able to infer timeline direction, even if two non-final labels "
            "appeared near each other in a sentence."
        ),
        (
            f"A later cross-functional review mapped evidence from legal notes, engineering runbooks, and procurement amendments, and while the legal appendix still carried {decoy_code} "
            f"as a rejected draft identifier and one staffing slide still quoted {bp.pilot_code} as a transitional safeguard, the merged review matrix tied production authority only to "
            f"{bp.final_code} with the signer {bp.approver} and the date {bp.final_date}, ensuring every audit trail converged on the same tuple."
        ),
        (
            "Another appendix listed false-positive examples from previous audits where reviewers extracted the first code-like token in a paragraph and missed the clause that explicitly "
            "negated production authority, so the appendix recommended clause-aware reading that tracks conjunctions such as while, although, and after, because those conjunctions often "
            "mark the boundary between temporary context and final decision statements in long operational prose."
        ),
        (
            f"One appendix, which many readers skimmed too quickly, contrasted the interim memo signed by {decoy_approver} on {decoy_date} with the final approval signed by "
            f"{bp.approver} on {bp.final_date}; it stressed that the interim memo prolonged pilot safeguards and did not grant production authority, while the final signature "
            "completed the legal transition to the production control."
        ),
        (
            "Post-deployment quality checks then required supervisors to answer structured recall prompts that mixed decoy names, decoy dates, and decoy codes into long composite "
            "sentences; evaluators observed that teams performed best when they first identified the sentence segment that contained explicit production language and then verified "
            "that the associated signer and date matched the same segment rather than nearby transitional clauses."
        ),
        (
            "Finally, the governance archivist added a prose note explaining that the canonical tuple must be reconstructed from semantically linked fragments rather than from the "
            "first convenient token match, because the archive intentionally preserves contradictory historical fragments for transparency, and reliable extraction depends on tracking "
            "status words, signer identity, and temporal markers across the same logical decision statement."
        ),
        (
            "The final retrospective emphasized a practical extraction rule for future automation: if a paragraph includes both a pilot label and an approved label, the approved "
            "label must be paired with explicit production language, while pilot references should appear near words like temporary, extension, or pilot-only; this rule exists to "
            "reduce false positives when systems ingest long narrative text with multiple historical markers."
        ),
        (
            f"As a result, the authoritative tuple for this case is: final production code {bp.final_code}, approver {bp.approver}, final date {bp.final_date}, and critical "
            f"constraint phrase \"{bp.constraint}\"; all other similarly formatted facts in this document are decoys, historical references, or intermediate artifacts."
        ),
    ]
    return "\n\n".join(paragraphs)


def build_hard_cases() -> list[HardLinkedCase]:
    approvers = [item.approver for item in BLUEPRINTS]
    cases: list[HardLinkedCase] = []
    for idx, bp in enumerate(BLUEPRINTS):
        decoy_approver = approvers[(idx + 3) % len(approvers)]
        if decoy_approver == bp.approver:
            decoy_approver = approvers[(idx + 4) % len(approvers)]
        decoy_date = _make_decoy_date(bp.final_date, days_back=17 + (idx % 4))
        decoy_code = _make_decoy_code(bp.legacy_code, idx)
        context = _build_hard_context(idx)
        question = (
            f"For {bp.program}, after replacing {bp.pilot_code} and retiring {bp.legacy_code}, return JSON with keys "
            "{\"final_code\",\"approver\",\"final_date\",\"constraint\"}. Use the final approved production tuple only."
        )
        expected = {
            "final_code": bp.final_code,
            "approver": bp.approver,
            "final_date": bp.final_date,
            "constraint": bp.constraint,
        }
        cases.append(
            HardLinkedCase(
                case_id=bp.case_id,
                title=bp.title,
                context=context,
                question=question,
                expected=expected,
                legacy_code=bp.legacy_code,
                pilot_code=bp.pilot_code,
                decoy_code=decoy_code,
                decoy_approver=decoy_approver,
                decoy_date=decoy_date,
            )
        )
    return cases


def _extract_hard_answer(raw: str) -> dict[str, str]:
    result = {"final_code": "", "approver": "", "final_date": "", "constraint": ""}
    text = (raw or "").strip()
    if not text:
        return result

    if text.startswith("{") and text.endswith("}"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                key_map = {
                    "final_code": ("final_code", "code", "answer_code", "production_code", "value"),
                    "approver": ("approver", "approved_by", "signed_by", "signatory"),
                    "final_date": ("final_date", "approval_date", "approved_on", "date"),
                    "constraint": ("constraint", "operational_constraint", "constraint_phrase"),
                }
                for field, candidates in key_map.items():
                    for key in candidates:
                        value = parsed.get(key)
                        if value is not None:
                            result[field] = str(value).strip()
                            break
        except Exception:
            pass

    upper = text.upper()
    if not result["final_code"]:
        code_match = CODE_PATTERN.search(upper)
        if code_match:
            result["final_code"] = code_match.group(0)
    if not result["final_date"]:
        date_match = DATE_PATTERN.search(text)
        if date_match:
            result["final_date"] = date_match.group(0)
    if not result["approver"]:
        by_match = re.search(r"\bby\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", text)
        if by_match:
            result["approver"] = by_match.group(1).strip()
    return result


def _constraint_match(expected: str, predicted: str) -> int:
    exp_tokens = {tok for tok in re.findall(r"[a-z0-9]+", expected.lower()) if len(tok) >= 4}
    pred_tokens = {tok for tok in re.findall(r"[a-z0-9]+", (predicted or "").lower()) if len(tok) >= 4}
    if not exp_tokens or not pred_tokens:
        return 0
    overlap = len(exp_tokens.intersection(pred_tokens)) / float(len(exp_tokens))
    return 1 if overlap >= 0.7 else 0


def _score_case_fields(expected: dict[str, str], predicted: dict[str, str]) -> dict[str, int]:
    final_code_exact = 1 if _normalize(expected["final_code"]) == _normalize(predicted.get("final_code", "")) else 0
    approver_exact = 1 if _normalize(expected["approver"]) == _normalize(predicted.get("approver", "")) else 0
    final_date_exact = 1 if _normalize(expected["final_date"]) == _normalize(predicted.get("final_date", "")) else 0
    constraint_hit = _constraint_match(expected["constraint"], predicted.get("constraint", ""))
    all_fields = 1 if (final_code_exact and approver_exact and final_date_exact and constraint_hit) else 0
    return {
        "final_code_exact": final_code_exact,
        "approver_exact": approver_exact,
        "final_date_exact": final_date_exact,
        "constraint_match": constraint_hit,
        "all_fields_match": all_fields,
    }


def _run_case_direct(case: HardLinkedCase, model: str, num_ctx: int, timeout_s: int) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a strict long-context extractor. Return JSON only with keys "
                "{\"final_code\",\"approver\",\"final_date\",\"constraint\"}."
            ),
        },
        {
            "role": "user",
            "content": f"Passage:\n{case.context}\n\nQuestion: {case.question}",
        },
    ]
    return _ollama_chat(model=model, messages=messages, num_ctx=num_ctx, timeout_s=timeout_s, json_mode=True)


def _run_case_memory_orb(
    case: HardLinkedCase,
    model: str,
    timeout_s: int,
    allow_post_correction: bool = False,
    memory_dwell_mode: str = "heuristic",
    reasoning_dwell_ctx: int = 900,
) -> str:
    class _Adapter:
        def __init__(self, model_name: str, timeout: int, context_window: int) -> None:
            self.model_name = model_name
            self.timeout = timeout
            self.context_window = context_window

        def complete(self, messages: list[dict[str, str]]) -> str:
            return _ollama_chat(
                model=self.model_name,
                messages=messages,
                num_ctx=self.context_window,
                timeout_s=self.timeout,
                json_mode=True,
            )

        def complete_with_reasoning(self, messages: list[dict[str, str]], think: bool = True) -> str:
            return _ollama_chat(
                model=self.model_name,
                messages=messages,
                num_ctx=self.context_window,
                timeout_s=self.timeout,
                json_mode=False,
                think=think,
            )

    engine = MemoryOrbEngine(
        config=MemoryOrbEngineConfig(
            context_max_tokens=2200,
            working_max_tokens=1100,
            working_target_tokens=820,
            memory_budget_ratio=0.56,
            max_retrieved_orbs=12,
            min_focus_orb_count=2,
            answer_dwell_mode=memory_dwell_mode,
        )
    )
    chunks = _chunk_text(case.context, max_chars=900)
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
        engine.add_turn(
            "assistant",
            "",
            metadata={
                "source": "benchmark_boundary",
                "exclude_from_pulse_map": True,
                "exclude_from_question_memory_pool": True,
            },
        )

    adapter = _Adapter(model_name=model, timeout=timeout_s, context_window=2200)
    dwell_adapter = _Adapter(
        model_name=model,
        timeout=timeout_s,
        context_window=max(512, min(2200, reasoning_dwell_ctx)),
    )
    result = engine.answer_with_answer_document(
        model=adapter,
        question=case.question,
        passes=5,
        per_pass_orbs=5,
        answer_doc_max_tokens=1500,
        system_prompt=(
            "You are a strict factual extractor.\n"
            "Return JSON only with keys {\"final_code\",\"approver\",\"final_date\",\"constraint\"}."
        ),
        allow_answer_coercion=allow_post_correction,
        dwell_model=dwell_adapter if memory_dwell_mode == "reasoned" else None,
    )
    return result.answer


def evaluate_hard_model(
    model: str,
    cases: list[HardLinkedCase],
    mode: str,
    direct_ctx: int = 32768,
    timeout_s: int = 150,
    allow_post_correction: bool = False,
    memory_dwell_mode: str = "heuristic",
    reasoning_dwell_ctx: int = 900,
) -> dict[str, Any]:
    rows: list[HardLinkedCaseResult] = []
    for case in cases:
        if mode == "direct":
            raw = _run_case_direct(case=case, model=model, num_ctx=direct_ctx, timeout_s=timeout_s)
        else:
            raw = _run_case_memory_orb(
                case=case,
                model=model,
                timeout_s=timeout_s,
                allow_post_correction=allow_post_correction,
                memory_dwell_mode=memory_dwell_mode,
                reasoning_dwell_ctx=reasoning_dwell_ctx,
            )

        predicted = _extract_hard_answer(raw)
        scores = _score_case_fields(case.expected, predicted)
        rows.append(
            HardLinkedCaseResult(
                case_id=case.case_id,
                title=case.title,
                expected=case.expected,
                predicted=predicted,
                raw_response=raw,
                final_code_exact=scores["final_code_exact"],
                approver_exact=scores["approver_exact"],
                final_date_exact=scores["final_date_exact"],
                constraint_match=scores["constraint_match"],
                all_fields_match=scores["all_fields_match"],
            )
        )

    count = len(rows)
    def mean(attr: str) -> float:
        return (sum(getattr(row, attr) for row in rows) / float(count)) if count else 0.0

    return {
        "model": model,
        "mode": mode,
        "allow_post_correction": allow_post_correction if mode == "memory-orb" else None,
        "memory_dwell_mode": memory_dwell_mode if mode == "memory-orb" else None,
        "reasoning_dwell_ctx": reasoning_dwell_ctx if mode == "memory-orb" and memory_dwell_mode == "reasoned" else None,
        "cases": count,
        "metrics": {
            "final_code_accuracy": mean("final_code_exact"),
            "approver_accuracy": mean("approver_exact"),
            "final_date_accuracy": mean("final_date_exact"),
            "constraint_match_accuracy": mean("constraint_match"),
            "all_fields_exact_accuracy": mean("all_fields_match"),
        },
        "results": [asdict(row) for row in rows],
    }


def hard_dataset_summary(cases: list[HardLinkedCase]) -> dict[str, Any]:
    words = [len(case.context.split()) for case in cases]
    return {
        "case_count": len(cases),
        "avg_words_per_passage": (sum(words) / len(words)) if words else 0.0,
        "min_words_per_passage": min(words) if words else 0,
        "max_words_per_passage": max(words) if words else 0,
        "question_preview": cases[0].question if cases else "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Hard long-form linked benchmark with multi-field tracking over long passages."
        )
    )
    parser.add_argument("--models", type=str, default="qwen3:0.6b")
    parser.add_argument("--mode", choices=["direct", "memory-orb", "both"], default="both")
    parser.add_argument("--direct-ctx", type=int, default=32768)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--allow-post-correction", action="store_true")
    parser.add_argument("--memory-dwell-mode", type=str, choices=["heuristic", "reasoned"], default="heuristic")
    parser.add_argument("--reasoning-dwell-ctx", type=int, default=900)
    parser.add_argument(
        "--limit-cases",
        type=int,
        default=0,
        help="Optional limit for quick smoke runs. 0 means all cases.",
    )
    parser.add_argument("--write-dataset", type=str, default="")
    parser.add_argument("--json-out", type=str, default="")
    args = parser.parse_args()

    cases = build_hard_cases()
    if args.limit_cases and args.limit_cases > 0:
        cases = cases[: args.limit_cases]

    summary: dict[str, Any] = {"dataset": hard_dataset_summary(cases), "runs": []}
    if args.write_dataset:
        with open(args.write_dataset, "w", encoding="utf-8") as f:
            json.dump([asdict(case) for case in cases], f, indent=2)

    models = [item.strip() for item in args.models.split(",") if item.strip()]
    modes = ["direct", "memory-orb"] if args.mode == "both" else [args.mode]
    for model in models:
        for mode in modes:
            run = evaluate_hard_model(
                model=model,
                cases=cases,
                mode=mode,
                direct_ctx=max(4096, args.direct_ctx),
                timeout_s=max(30, args.timeout),
                allow_post_correction=bool(args.allow_post_correction),
                memory_dwell_mode=args.memory_dwell_mode,
                reasoning_dwell_ctx=max(256, args.reasoning_dwell_ctx),
            )
            summary["runs"].append(run)

    print(json.dumps(summary, indent=2))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
