# Project

- Name: Memory Orb
- Date started: 2026-02-27
- Owner: joshj + Codex
- Status: active
- Scope: Model-agnostic long-term memory architecture that keeps LLM context bounded while preserving useful history.

# Vision and Context

- Problem statement: Long-running LLM sessions grow token usage linearly if full history is replayed. This increases latency/cost and eventually exceeds model limits.
- In-scope:
  - Strict context budget enforcement.
  - Swap policy from short-term turns into long-term memory artifacts.
  - Retrieval and rehydration of relevant long-term memory for each prompt.
  - Portable interfaces for any LLM backend.
- Out-of-scope:
  - Training-time memory changes to base models.
  - Provider-specific SDK wrappers for every API vendor.
  - Perfect factual guarantees for recalled memories.
- Primary assumptions:
  - Approximate token counting is acceptable for provider-agnostic operation.
  - Retrieval quality can be improved later with stronger embedding backends.
  - Context growth must be bounded even if memory store grows indefinitely.
- Constraints:
  - Keep runtime dependency-light.
  - Maintain compatibility with Python 3.10+.
  - No architecture should require model fine-tuning.

# Architecture Overview

- Components and boundaries:
  - `Working Memory`: deque of recent turns with hard token budget.
  - `Orb Store (Episodic)`: compressed conversation chunks swapped out of working memory.
  - `Semantic Card Store`: anchor-based summaries distilled across multiple orbs.
  - `Selective Attention Latch`: compact active sub-goal and predecessor sub-goal state.
  - `Context Assembler`: composes final prompt packet under fixed budget.
  - `Adapter Layer`: token estimator, embedder, and model adapter protocols.
- Responsibility map:
  - `engine.py` owns lifecycle, swap, retrieval, packing, and persistence.
  - `utils.py` owns anchor extraction, summarization, similarity, and MMR ranking.
  - `adapters.py` owns pluggable backend interfaces and local fallbacks.
- Data flow:
  - Turn ingestion -> working memory append -> budget check -> swap into orb.
  - User turn ingestion -> sub-goal extraction -> latch refresh or rotate predecessor.
  - Orb creation -> semantic card updates.
  - Query -> anchor pulse extraction + focus-latch read + semantic retrieval + orb retrieval.
  - Fact QA -> question-first profile -> global skim of orbs -> deep-read evidence extraction -> answer document synthesis.
  - Context assembly -> system + memory block + recent turns under cap.
- External dependencies:
  - None required at runtime.
  - Optional: provider SDKs in downstream apps using `ModelAdapter`.

# Technical Design

- Language/framework stack: Python 3.10+, standard library.
- Core interfaces:
  - `ModelAdapter.complete(messages) -> str`
  - `TokenEstimator.count(text) -> int`
  - `Embedder.embed(text) -> list[float]`
- Persistence/storage model:
  - In-memory by default.
  - JSON snapshot for save/load state.
- State and consistency model:
  - Single-process mutable state.
  - Deterministic retrieval ranking given same state and query.
  - Eventual semantic consolidation as orbs accumulate.
- Deployment topology:
  - Embedded library inside app server, agent runtime, or local process.
- Non-functional requirements (availability, latency, scale, cost):
  - O(N) scoring over stored orbs per query; bounded context output.
  - Cost remains stable with long conversations due to hard context cap.
  - Designed for low operational overhead and no mandatory external service.

# Decision Log

- Decision ID: ADR-001
- Date: 2026-02-27
- Owner: Codex
- Decision: Use three memory layers (working, episodic orbs, semantic cards).
- Rationale: Separates recency, narrative history, and distilled facts without overloading context.
- Alternatives considered: Single vector store only; pure rolling summary.
- Risks: Distillation errors in semantic cards.
- Status: active

- Decision ID: ADR-002
- Date: 2026-02-27
- Owner: Codex
- Decision: Enforce explicit token budget split across memory and recent turns.
- Rationale: Guarantees context size stability independent of session length.
- Alternatives considered: Soft truncation only; provider-side auto-trimming.
- Risks: Hard caps can omit relevant context if scoring is weak.
- Status: active

- Decision ID: ADR-003
- Date: 2026-02-27
- Owner: Codex
- Decision: Use orbital resonance (anchor pulses from short-term memory) in retrieval scoring.
- Rationale: Connects long-term recall to active thread state, not just embedding similarity.
- Alternatives considered: Similarity-only retrieval; recency-only retrieval.
- Risks: Noisy anchor extraction can bias retrieval.
- Status: active

- Decision ID: ADR-004
- Date: 2026-02-27
- Owner: Codex
- Decision: Keep default implementation dependency-light with hashing embedder.
- Rationale: Enables immediate adoption for any model stack without extra services.
- Alternatives considered: Mandatory external vector DB / embedding API.
- Risks: Lower retrieval quality than SOTA dense embeddings.
- Status: active

- Decision ID: ADR-005
- Date: 2026-02-27
- Owner: Codex
- Decision: Add selective-attention sub-goal latch with predecessor retention.
- Rationale: Allows narrow local focus while retaining the prior sub-goal as background intent.
- Alternatives considered: Drop predecessor goal; rely only on recency retrieval.
- Risks: Incorrect sub-goal extraction can carry stale priorities.
- Status: active

- Decision ID: ADR-006
- Date: 2026-02-27
- Owner: Codex
- Decision: Add focus dwell and pinning for retrieved episodic orbs.
- Rationale: Important areas should stay prioritized across several follow-up turns, not only on the turn they are first detected.
- Alternatives considered: No persistence; larger static retrieval budget.
- Risks: Sticky focus can over-prioritize stale topics if hold durations are too long.
- Status: active

- Decision ID: ADR-007
- Date: 2026-02-27
- Owner: Codex
- Decision: Add explicit question-first skim/deep-read pipeline for factual recall.
- Rationale: For long documents, first identify question targets (record/field), skim all orbs quickly, then spend more dwell on high-evidence orbs before final answer generation.
- Alternatives considered: Generic retrieval with no question parsing; single-pass extraction.
- Risks: Regex target parsing can miss unusual question formats; over-weighting target tokens can suppress useful supporting context.
- Status: active

- Decision ID: ADR-008
- Date: 2026-03-04
- Owner: Codex
- Decision: Add cursor-chain sweep and adaptive dwell allocation at scan-area level.
- Rationale: Mimics human scan behavior by continuing near the last viewed region while still diversifying sources, and allocates extra rereads for dense/targeted evidence areas.
- Alternatives considered: Static dwell count per area; pure global rank without cursor continuity.
- Risks: Cursor momentum can over-bias local neighborhoods if source diversity penalties are too weak.
- Status: active

- Decision ID: ADR-009
- Date: 2026-03-06
- Owner: Codex
- Decision: Restrict semantic cards to durable thematic memory and keep fact-QA retrieval read-only by default.
- Rationale: Mutable facts and one-off QA probes should not rewrite long-term retrieval priors or collapse contradictory state into a single semantic synopsis.
- Alternatives considered: Keep semantic cards fully generic; allow QA dwell to mutate normal chat retrieval state.
- Risks: Less aggressive semantic distillation can reduce recall for some recent factual snippets unless episodic retrieval remains strong.
- Status: active

- Decision ID: ADR-010
- Date: 2026-03-06
- Owner: Codex
- Decision: Preserve heuristic skim/dwell as the baseline and add an optional reasoned dwell mode that routes only high-signal scan areas through a low-context reasoning-capable model adapter.
- Rationale: Pure heuristic skim is fast but can miss semantic subtleties, while model-per-area reading is too expensive. A selective deep-dwell path concentrates test-time compute on complex or ambiguous evidence regions and keeps the original path available for benchmarking.
- Alternatives considered: Replace heuristic dwell entirely; run a reasoning model on every scan area.
- Risks: If trigger thresholds are too eager, reasoned dwell can add latency without improving accuracy; if too strict, it may fail to activate on the cases that need deeper semantic reading.
- Status: active

- Decision ID: ADR-011
- Date: 2026-03-06
- Owner: Codex
- Decision: Route model-based deep-dwell selectively for comparative multiple-choice QA, and bias skim planning toward cross-source coverage when the question asks for divergences across analyses.
- Rationale: LongBench-style comparative questions fail when the sweep overfocuses one source or when reasoned dwell is invoked on tasks like table QA that do not benefit from extra local model passes. Question-aware routing and source diversity are better uses of limited test-time compute than globally lowering thresholds.
- Alternatives considered: Global lower reasoned-dwell thresholds; answer-document replacement for all QA; uniform deep-dwell across every case.
- Risks: Simple routing heuristics can miss cases that would benefit from deeper local reading, and source diversity may reduce precision when the correct answer truly lives in one document.
- Status: active

- Decision ID: ADR-012
- Date: 2026-03-06
- Owner: Codex
- Decision: For comparative and policy-mix MCQ cases, use option-specific probe notes plus a supplement-only adjudicator, and penalize disclosure-style boilerplate during skim scoring.
- Rationale: Broad chat prompts were washing out targeted option evidence, and report boilerplate was hijacking retrieval because it shared entity names with real evidence. Option probes force explicit search for each claim, while the focused adjudicator chooses from the probe notes without reintroducing the full noisy memory context.
- Alternatives considered: Keep final answering in the broad chat path; rely on one generic reasoned skim summary; clean benchmark text externally.
- Risks: Probe-based adjudication adds extra local calls on routed cases, and boilerplate heuristics may suppress useful metadata in edge cases.
- Status: active

- Decision ID: ADR-013
- Date: 2026-03-07
- Owner: Codex
- Decision: Add input-shaped structured readers for tables and manuals, and restrict routing to raw question/options/context signals only.
- Rationale: Table QA and user-guide QA fail when treated as generic prose retrieval problems. A `TableReader` can operate over headers, rows, ranks, and temporal groupings, while a `ProcedureReader` can operate over headings, parameters, warnings, and contradiction matrices. These readers stay generic by triggering only from observed input structure, not benchmark metadata.
- Alternatives considered: Keep all long-context QA on the generic reasoned-chat path; route from dataset domain labels; build benchmark-only table/manual hacks inside the harness.
- Risks: Structure detection can misclassify noisy contexts, and specialized readers can overfit shallow formatting cues if their confidence thresholds are too loose.
- Status: active

# Update Cadence

- Trigger: changes in architecture, requirements, or infrastructure
- Review cadence: every feature milestone or major retrieval policy change
- Who approves updates: project owner (joshj) with implementation owner
- Last reviewed: 2026-03-06

# Open Risks and Issues

- Risk, impact, mitigation, owner, due date:
  - Approximate token counting mismatch vs provider tokenizer; can still overflow in edge cases; mitigate by adding provider tokenizer adapters; owner: Codex; due: next iteration.
  - Anchor extraction is regex-based; may miss domain entities; mitigate with NER plugin option; owner: Codex; due: next iteration.
  - Sub-goal extraction can misclassify intent shifts; mitigate with optional structured goal API hook; owner: Codex; due: next iteration.
  - Focus dwell pinning can hold stale context too long; mitigate with shorter hold and adaptive decay controls; owner: Codex; due: next iteration.
  - Question target parsing may fail for implicit/ambiguous queries; can degrade fact recall; mitigate with model-assisted query parser fallback; owner: Codex; due: next iteration.
  - Cursor-chain sweep can overfocus one source and miss distant evidence; mitigate with source repeat penalties and seed diversity via MMR; owner: Codex; due: next iteration.
  - Recent assistant messages may contain hallucinated facts while still being useful evidence; mitigate with role-aware weighting and explicit provenance markers in working-memory question pools; owner: Codex; due: next iteration.
  - O(N) orb scoring may degrade with very large memory stores; mitigate with ANN index adapter; owner: Codex; due: future scale milestone.
  - Structured reader routing may drift toward benchmark-specific heuristics; mitigate with fairness tests that forbid runtime use of answer keys and dataset domain metadata, and keep route decisions bound to raw context/question structure only; owner: Codex; due: current iteration.

# Change History

- 2026-02-27 | Initial architecture plan created | Kickoff for Memory Orb design and implementation
- 2026-02-27 | Added selective-attention latch architecture | Preserve active and prior sub-goals while context is aggressively trimmed
- 2026-02-27 | Added focus dwell and pinning policy | Keep important memory active for multiple turns after retrieval
- 2026-02-27 | Added question-first skim/deep-read policy | Fact QA now starts from question targets before evidence scan and dwell
- 2026-03-04 | Added scan-area cursor-chain and adaptive dwell policy | Skim now moves across nearby areas and dwell increases on complex/targeted text regions
- 2026-03-06 | Hardened persistence and factual retrieval boundaries | Reload now re-embeds orbs, semantic cards skip mutable facts, and answer-document QA no longer mutates retrieval state by default
- 2026-03-06 | Added selective reasoned dwell mode | High-signal scan areas can now invoke low-context `think:true` reads while the heuristic dwell path remains the benchmark baseline
- 2026-03-06 | Added comparative QA routing and source-diverse skim bias | Reasoned deep-dwell is now routed to comparison-heavy MCQ cases, and skim planning spreads attention across sources when the question asks for divergences across analyses
- 2026-03-06 | Added option-probe adjudication and boilerplate suppression | Comparative MCQ benchmarks now gather claim-specific evidence per option and discount disclosure-style report noise before final adjudication
- 2026-03-07 | Added structured table/manual readers with fairness guardrails | Table QA and User-guide QA can now bypass generic prose retrieval through structure-only routing, and benchmark audit fields record which route was used
