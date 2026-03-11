from __future__ import annotations

import argparse
import json
import re
import socket
from dataclasses import asdict, dataclass
from typing import Any
from urllib.error import HTTPError
from urllib.error import URLError
from urllib.request import Request
from urllib.request import urlopen

from memory_orb import MemoryOrbEngine
from memory_orb import MemoryOrbEngineConfig


CODE_PATTERN = re.compile(r"\b[a-z]{2,}(?:-[a-z0-9]{2,}){2,}\b", flags=re.IGNORECASE)
NORMALIZE_RE = re.compile(r"[^a-z0-9]+")


@dataclass(slots=True)
class LinkedCase:
    case_id: str
    title: str
    context: str
    question: str
    answer: str
    aliases: list[str]
    legacy_code: str
    pilot_code: str
    approver: str
    final_date: str


@dataclass(slots=True)
class LinkedCaseResult:
    case_id: str
    title: str
    expected: str
    predicted: str
    raw_response: str
    exact_match: int


@dataclass(slots=True)
class _CaseBlueprint:
    case_id: str
    title: str
    organization: str
    program: str
    asset: str
    region: str
    constraint: str
    legacy_code: str
    pilot_code: str
    final_code: str
    approver: str
    final_date: str


BLUEPRINTS: list[_CaseBlueprint] = [
    _CaseBlueprint(
        case_id="case-01",
        title="Harborlight Cold Chain",
        organization="North Shore Health Alliance",
        program="Harborlight Cold Chain Refresh",
        asset="vaccine storage telemetry",
        region="Maine coastal hospitals",
        constraint="generator transfer times on storm nights",
        legacy_code="HBR-LINK-219",
        pilot_code="HBR-LINK-384",
        final_code="HBR-LOCK-731",
        approver="Elena Park",
        final_date="September 18, 2025",
    ),
    _CaseBlueprint(
        case_id="case-02",
        title="Mesa Fare Reconciliation",
        organization="Mesa Transit Authority",
        program="Mesa Fare Reconciliation Upgrade",
        asset="bus fare clearing jobs",
        region="Phoenix commuter corridors",
        constraint="overnight settlement before 4:30 a.m. dispatch",
        legacy_code="MNT-FLOW-163",
        pilot_code="MNT-FLOW-246",
        final_code="MNT-FLOW-284",
        approver="Darius Molina",
        final_date="October 2, 2025",
    ),
    _CaseBlueprint(
        case_id="case-03",
        title="Arcadia Archive Migration",
        organization="Arcadia Public Records Office",
        program="Arcadia Archive Migration Program",
        asset="zoning archive ingestion",
        region="three county permit districts",
        constraint="legal hold windows during filing season",
        legacy_code="ARC-NODE-402",
        pilot_code="ARC-NODE-518",
        final_code="ARC-NODE-553",
        approver="Maya Jennings",
        final_date="August 27, 2025",
    ),
    _CaseBlueprint(
        case_id="case-04",
        title="Delta Grid Dispatch",
        organization="Delta Valley Utility Cooperative",
        program="Delta Grid Dispatch Modernization",
        asset="substation demand balancing",
        region="Sacramento delta feeder network",
        constraint="peaker unit spin-up latency under heat alerts",
        legacy_code="DELTA-ROUTE-771",
        pilot_code="DELTA-ROUTE-846",
        final_code="DELTA-ROUTE-909",
        approver="Noah Finley",
        final_date="July 11, 2025",
    ),
    _CaseBlueprint(
        case_id="case-05",
        title="Quay Freight Ledger",
        organization="Quayline Maritime Logistics",
        program="Quay Freight Ledger Stabilization",
        asset="port handoff reconciliation",
        region="Gulf transshipment terminals",
        constraint="cross-midnight customs manifests",
        legacy_code="QUAY-BATCH-301",
        pilot_code="QUAY-BATCH-419",
        final_code="QUAY-BATCH-447",
        approver="Rina Talbot",
        final_date="November 6, 2025",
    ),
    _CaseBlueprint(
        case_id="case-06",
        title="Pine Forest Incident Index",
        organization="Pine County Emergency Office",
        program="Pine Forest Incident Index Rebuild",
        asset="wildfire dispatch indexing",
        region="Sierra foothill response zones",
        constraint="radio dead spots in canyon sectors",
        legacy_code="PINE-INDEX-488",
        pilot_code="PINE-INDEX-577",
        final_code="PINE-INDEX-612",
        approver="Gavin Chen",
        final_date="June 20, 2025",
    ),
    _CaseBlueprint(
        case_id="case-07",
        title="Lumen Clinic Slotting",
        organization="Lumen Care Partners",
        program="Lumen Clinic Slotting Initiative",
        asset="specialty referral scheduling",
        region="multi-state outpatient network",
        constraint="insurance cutoff timestamps in local time",
        legacy_code="LUMEN-SLOT-211",
        pilot_code="LUMEN-SLOT-332",
        final_code="LUMEN-SLOT-358",
        approver="Priya Nandakumar",
        final_date="May 14, 2025",
    ),
    _CaseBlueprint(
        case_id="case-08",
        title="Crown Parcel Mapping",
        organization="Crown Ridge Distribution",
        program="Crown Parcel Mapping Overhaul",
        asset="last-mile route mapping",
        region="upper Midwest warehouse ring",
        constraint="frozen-road detours in winter dispatch",
        legacy_code="CROWN-MAP-622",
        pilot_code="CROWN-MAP-731",
        final_code="CROWN-MAP-774",
        approver="Luis Herrera",
        final_date="December 1, 2025",
    ),
    _CaseBlueprint(
        case_id="case-09",
        title="Ridge Claims Traceability",
        organization="Ridge Mutual Assurance",
        program="Ridge Claims Traceability Reform",
        asset="catastrophe claim triage",
        region="Atlantic coastal policy book",
        constraint="regulatory callback windows after major storms",
        legacy_code="RIDGE-TRACE-144",
        pilot_code="RIDGE-TRACE-223",
        final_code="RIDGE-TRACE-265",
        approver="Avery Bishop",
        final_date="April 29, 2025",
    ),
    _CaseBlueprint(
        case_id="case-10",
        title="Tidal Queue Orchestration",
        organization="Tidal BioManufacturing",
        program="Tidal Queue Orchestration Program",
        asset="batch release sequencing",
        region="two-campus biologics operation",
        constraint="cold-room handoff deadlines before assay expiration",
        legacy_code="TIDAL-QUEUE-690",
        pilot_code="TIDAL-QUEUE-755",
        final_code="TIDAL-QUEUE-801",
        approver="Sofia Alvarez",
        final_date="March 19, 2025",
    ),
]


def _build_context(bp: _CaseBlueprint) -> str:
    paragraphs = [
        (
            f"{bp.organization} opened the {bp.program} after two quarters of incident review and executive pressure to stabilize "
            f"{bp.asset} across {bp.region}. The kickoff memo was written in plain operational language because site supervisors had to "
            "carry procedures in paper binders during outages. Program managers asked each team to separate hard facts from assumptions, "
            "document conflicting terms, and keep a running table of any label that might later be mistaken for a production control value. "
            "The first project workshop framed the work as reliability engineering rather than software modernization, and participants were "
            "told to preserve old terms in notes even when they looked obsolete."
        ),
        (
            "During the discovery phase, facilitators interviewed dispatch leads, network technicians, compliance officers, and payroll staff "
            "because every overnight workflow crossed boundaries that were normally invisible in monthly reports. Interview transcripts described "
            "how one delayed handoff propagated through staffing plans, inventory confirmations, and daily dashboards. Analysts built a timeline "
            "that compared paper procedures to what operators actually did at 2:00 a.m., then tagged each mismatch by severity and recurrence. "
            "The pattern that mattered most was repeat drift during edge conditions, especially when local overrides were passed verbally."
        ),
        (
            f"The original charter used the control label {bp.legacy_code} as a placeholder for the first rewrite milestone. That legacy label "
            "appeared in change tickets, on whiteboard photos, and in one budget spreadsheet exported from an old finance macro. Engineers later "
            "confirmed that the label represented a sequencing idea from an earlier era and did not encode the constraints discovered in the new "
            f"field interviews. Even so, the string {bp.legacy_code} remained embedded in several ad hoc scripts, so reviewers kept it in every "
            "meeting packet to avoid accidental deletion without migration notes."
        ),
        (
            f"When pilot runs started, the operations team switched to {bp.pilot_code} because the old placeholder could not handle {bp.constraint}. "
            "The pilot label was practical but explicitly temporary. Site coordinators accepted the temporary code to keep training momentum while "
            "data engineers compared error rates between old and new sequencing rules. The pilot handbook included a warning banner that said the "
            "code must not be interpreted as final policy and that legal approval had not yet been granted for permanent use. Several team leads "
            "still copied the pilot label into local playbooks, which later created reconciliation noise."
        ),
        (
            "Mid-project status updates looked calm on the surface, yet appendix notes showed persistent friction in handoff timing. Supervisors "
            "reported that technical fixes worked when everyone followed the same clock, but partner teams used overlapping time conventions that "
            "introduced hidden offsets. In response, analysts added narrative explanations beside each metric instead of publishing tables alone. "
            "They wanted reviewers to see why a small percentage change could still imply operational risk. This narrative-first reporting style "
            "produced longer documents, but it reduced false confidence in summary charts and made unresolved assumptions easier to spot."
        ),
        (
            "Compliance review happened in two rounds. The first round focused on retention and access controls, and the second tested whether "
            "night-shift exceptions were documented in a way external auditors could replicate. Audit staff asked for plain-language statements that "
            "connected requirements to concrete controls. They also asked teams to state which previous labels had been retired, which labels were "
            "temporary, and which one would become authoritative when legal and finance signatures were complete. This forced everyone to align terms "
            "before the final governance meeting."
        ),
        (
            "A rehearsal weekend exposed one more source of ambiguity: archived wiki pages were still indexed by internal search and often surfaced "
            "ahead of current runbooks. To prevent accidental rollback, engineering managers added date stamps, retirement tags, and a one-line summary "
            "at the top of each critical page. They also wrote a short FAQ for on-call staff explaining how to distinguish archival references from "
            "active policy. This became important because a handful of archived pages still referenced both the legacy and pilot labels without context."
        ),
        (
            "Finance and procurement joined late, but their review changed the decision timeline. Contract language required a single control label in "
            "support agreements, and insurers required consistency between procedure text and incident notification templates. That legal packaging work "
            "did not change technical logic directly, yet it determined when the new control could be called production-ready. Program leadership agreed "
            "that no label would be declared final until signatures landed and all customer-facing language was synchronized with the approved operating "
            "definition."
        ),
        (
            f"In the final governance session, the committee explicitly retired {bp.legacy_code}, marked {bp.pilot_code} as pilot-only, and approved "
            f"{bp.final_code} for production operations. Meeting minutes tied that approval to signed risk acceptance and a completed rollback checklist. "
            f"The approval block was signed by {bp.approver} on {bp.final_date}, and the distribution note instructed every team to remove pilot references "
            "from active dashboards within one release cycle. This section was duplicated in both legal minutes and the engineering release note to avoid "
            "interpretation drift."
        ),
        (
            "After approval, deployment teams ran a controlled cutover with paired observers so that each step had an independent witness. Incident "
            "commanders monitored queue depth, handoff latency, and acknowledgment lag, then compared those traces against pre-cutover rehearsal logs. "
            "The first production week produced minor anomalies, but none required rollback. Most anomalies came from stale bookmarks to archive pages, "
            "not from behavior of the new control itself. By the end of the week, training staff replaced screenshots in onboarding materials and removed "
            "the old pilot banner from local printouts."
        ),
        (
            f"The closeout memo repeated one practical rule for future audits: only {bp.final_code} is considered the authoritative production control "
            f"for {bp.program}. The memo warned that references to {bp.legacy_code} and {bp.pilot_code} may still appear in historical notes and should "
            "be interpreted as prior states, not active configuration. Program archivists attached a glossary to the closeout packet so later reviewers "
            "could verify each label in context instead of guessing from isolated snippets. This guidance was intended to make later recall tasks "
            "deterministic, even when records were read out of order."
        ),
    ]
    return "\n\n".join(paragraphs)


def build_linked_cases() -> list[LinkedCase]:
    cases: list[LinkedCase] = []
    for bp in BLUEPRINTS:
        context = _build_context(bp)
        question = (
            f"For {bp.program}, which production control code was finally approved on {bp.final_date} "
            f"after replacing {bp.pilot_code}? Return the code only."
        )
        cases.append(
            LinkedCase(
                case_id=bp.case_id,
                title=bp.title,
                context=context,
                question=question,
                answer=bp.final_code,
                aliases=[bp.final_code.lower(), bp.final_code.replace("-", "").lower()],
                legacy_code=bp.legacy_code,
                pilot_code=bp.pilot_code,
                approver=bp.approver,
                final_date=bp.final_date,
            )
        )
    return cases


def _normalize(text: str) -> str:
    return NORMALIZE_RE.sub("", (text or "").lower())


def _extract_answer(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        return ""

    if text.startswith("{") and text.endswith("}"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                for key in ("answer", "value", "final_code", "current_value", "result"):
                    value = parsed.get(key)
                    if value is not None:
                        text = str(value).strip()
                        break
                else:
                    for value in parsed.values():
                        if isinstance(value, (str, int, float)):
                            text = str(value).strip()
                            break
        except Exception:
            pass

    match = CODE_PATTERN.search(text)
    if match:
        return match.group(0).upper()
    if text:
        return text.splitlines()[0].strip()
    return ""


def _is_exact_match(expected: str, aliases: list[str], predicted: str) -> int:
    accepted = {_normalize(expected)}
    for alias in aliases:
        normalized = _normalize(alias)
        if normalized:
            accepted.add(normalized)
    pred = _normalize(predicted)
    if not pred:
        return 0
    if pred in accepted:
        return 1
    for item in accepted:
        if item and item in pred:
            return 1
    return 0


def _chunk_text(text: str, max_chars: int = 900) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    used = 0
    for paragraph in text.split("\n\n"):
        para = paragraph.strip()
        if not para:
            continue
        need = len(para) + 2
        if current and used + need > max_chars:
            chunks.append("\n\n".join(current))
            current = [para]
            used = len(para)
        else:
            current.append(para)
            used += need
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def _ollama_chat(
    model: str,
    messages: list[dict[str, str]],
    num_ctx: int,
    timeout_s: int = 120,
    json_mode: bool = True,
    think: bool = False,
) -> str:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": think,
        "options": {
            "temperature": 0,
            "top_p": 0.1,
            "num_ctx": num_ctx,
            "num_predict": 120,
        },
        "keep_alive": "10m",
    }
    if json_mode:
        payload["format"] = "json"
    req = Request(
        "http://127.0.0.1:11434/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8")
    except HTTPError as err:
        detail = ""
        try:
            detail = err.read().decode("utf-8")
        except Exception:
            detail = str(err)
        raise RuntimeError(f"Ollama chat failed (HTTP {err.code}): {detail}") from err
    except URLError as err:
        raise RuntimeError(f"Ollama chat failed: {err}") from err
    except TimeoutError as err:
        raise RuntimeError(f"Ollama chat timed out: {err}") from err
    except socket.timeout as err:
        raise RuntimeError(f"Ollama socket timeout: {err}") from err

    parsed: dict[str, Any] = json.loads(body)
    message = parsed.get("message") or {}
    content = str(message.get("content", "")).strip()
    if content:
        return content
    return str(message.get("thinking", "")).strip()


def _run_case_direct(case: LinkedCase, model: str, num_ctx: int, timeout_s: int) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are an exact factual retrieval assistant. Return JSON only: {\"answer\":\"...\"}."
                " Output exactly one code."
            ),
        },
        {
            "role": "user",
            "content": f"Passage:\n{case.context}\n\nQuestion: {case.question}",
        },
    ]
    return _ollama_chat(model=model, messages=messages, num_ctx=num_ctx, timeout_s=timeout_s, json_mode=True)


def _run_case_memory_orb(
    case: LinkedCase,
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
            context_max_tokens=1900,
            working_max_tokens=980,
            working_target_tokens=760,
            memory_budget_ratio=0.52,
            max_retrieved_orbs=10,
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

    adapter = _Adapter(model_name=model, timeout=timeout_s, context_window=1900)
    dwell_adapter = _Adapter(
        model_name=model,
        timeout=timeout_s,
        context_window=max(512, min(1900, reasoning_dwell_ctx)),
    )
    result = engine.answer_with_answer_document(
        model=adapter,
        question=case.question,
        passes=4,
        per_pass_orbs=4,
        answer_doc_max_tokens=1200,
        system_prompt=(
            "You answer with strict factual retrieval.\n"
            "Return JSON only: {\"answer\":\"...\"}."
        ),
        allow_answer_coercion=allow_post_correction,
        dwell_model=dwell_adapter if memory_dwell_mode == "reasoned" else None,
    )
    return result.answer


def evaluate_model(
    model: str,
    cases: list[LinkedCase],
    mode: str,
    direct_ctx: int = 32768,
    timeout_s: int = 120,
    allow_post_correction: bool = False,
    memory_dwell_mode: str = "heuristic",
    reasoning_dwell_ctx: int = 900,
) -> dict[str, Any]:
    rows: list[LinkedCaseResult] = []
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
        predicted = _extract_answer(raw)
        exact = _is_exact_match(case.answer, case.aliases, predicted)
        rows.append(
            LinkedCaseResult(
                case_id=case.case_id,
                title=case.title,
                expected=case.answer,
                predicted=predicted,
                raw_response=raw,
                exact_match=exact,
            )
        )

    total = len(rows)
    accuracy = (sum(row.exact_match for row in rows) / total) if total else 0.0
    return {
        "model": model,
        "mode": mode,
        "allow_post_correction": allow_post_correction if mode == "memory-orb" else None,
        "memory_dwell_mode": memory_dwell_mode if mode == "memory-orb" else None,
        "reasoning_dwell_ctx": reasoning_dwell_ctx if mode == "memory-orb" and memory_dwell_mode == "reasoned" else None,
        "cases": total,
        "exact_match_accuracy": accuracy,
        "results": [asdict(row) for row in rows],
    }


def dataset_summary(cases: list[LinkedCase]) -> dict[str, Any]:
    words = [len(case.context.split()) for case in cases]
    return {
        "case_count": len(cases),
        "avg_words_per_passage": (sum(words) / len(words)) if words else 0.0,
        "min_words_per_passage": min(words) if words else 0,
        "max_words_per_passage": max(words) if words else 0,
        "question_examples": [case.question for case in cases[:2]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Long-form linked QA benchmark: 10 natural ~2-page passages with deterministic answer keys."
        )
    )
    parser.add_argument(
        "--models",
        type=str,
        default="qwen3:0.6b",
        help="Comma-separated model list, e.g. qwen3:0.6b,qwen3:4b",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["direct", "memory-orb", "both"],
        default="both",
        help="Evaluate direct long-context, memory-orb, or both.",
    )
    parser.add_argument("--direct-ctx", type=int, default=32768, help="Context window for direct mode.")
    parser.add_argument("--timeout", type=int, default=150, help="Timeout in seconds per model request.")
    parser.add_argument(
        "--write-dataset",
        type=str,
        default="",
        help="Optional JSON path to dump the full 10-case benchmark dataset.",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default="",
        help="Optional JSON path to write benchmark evaluation output.",
    )
    parser.add_argument(
        "--allow-post-correction",
        action="store_true",
        help=(
            "Allow memory-orb answer post-correction. Off by default for fair model-to-model comparison."
        ),
    )
    parser.add_argument("--memory-dwell-mode", type=str, choices=["heuristic", "reasoned"], default="heuristic")
    parser.add_argument("--reasoning-dwell-ctx", type=int, default=900)
    args = parser.parse_args()

    cases = build_linked_cases()
    summary: dict[str, Any] = {
        "dataset": dataset_summary(cases),
        "runs": [],
    }

    if args.write_dataset:
        with open(args.write_dataset, "w", encoding="utf-8") as f:
            json.dump([asdict(case) for case in cases], f, indent=2)

    models = [item.strip() for item in args.models.split(",") if item.strip()]
    modes = ["direct", "memory-orb"] if args.mode == "both" else [args.mode]
    for model in models:
        for mode in modes:
            run = evaluate_model(
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
