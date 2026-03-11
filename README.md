# Memory Orb

Memory Orb is a model-agnostic memory layer for LLM applications. It prevents
unbounded context growth by swapping old conversation turns into compressed
"orbs" and re-injecting only the most relevant long-term memory under a strict
token budget.

## What is novel here

- Orbital resonance retrieval: long-term memories are scored by both semantic
  similarity and live "anchor pulses" extracted from short-term memory.
- Selective-attention latch: preserves the current sub-goal plus predecessor
  sub-goal so focus can narrow without losing background intent.
- Focus dwell and pinning: high-signal memory stays prioritized for several
  turns instead of disappearing immediately after one retrieval.
- Question-first answer document loop: for fact recall, the engine parses the
  question target, performs a fast global skim, then deep-reads top regions.
- Cursor-chain skim + adaptive dwell: skim follows a moving scan cursor across
  neighboring regions and spends more reread budget on dense target-bearing text.
- Dual dwell modes for fact QA: keep the current heuristic reader as a baseline,
  or enable reasoned dwell so a low-context model spends `think:true` compute only
  on the highest-signal scan areas.
- Hybrid memory layers:
  - Working memory (bounded recent turns)
  - Episodic orb store (compressed swap chunks)
  - Semantic anchor cards (cross-orb distilled summaries)
  - Focus latch (active + background sub-goals)
- Hard context cap: context packet assembly never exceeds `context_max_tokens`.

## Install

```bash
cd memory-orb
python -m pip install -e ".[dev]"
```

## Quick start

```python
from memory_orb import EchoModelAdapter, MemoryOrbEngine, MemoryOrbEngineConfig

engine = MemoryOrbEngine(
    config=MemoryOrbEngineConfig(
        context_max_tokens=900,
        working_max_tokens=500,
        working_target_tokens=360,
    )
)
model = EchoModelAdapter()

reply, packet = engine.chat(
    model=model,
    user_text="Remember that project Atlas uses PostgreSQL logical replication.",
    system_prompt="You are an engineering copilot.",
)

print(reply)
print(packet.total_tokens, packet.selected_orb_ids, packet.selected_anchors)
print(engine.stats())
```

## Adapter pattern (for any model)

Implement one method:

```python
class MyModelAdapter:
    def complete(self, messages):
        # messages: list[{"role": "...", "content": "..."}]
        # call your provider SDK here and return assistant text
        return "..."
```

Then call `engine.chat(model=MyModelAdapter(), user_text=...)`.

Optional for answer-document deep dwell:

```python
class MyReasoningAdapter(MyModelAdapter):
    def complete_with_reasoning(self, messages, think=True):
        # same backend, but low-context local reads can enable reasoning mode
        return "..."
```

If `MemoryOrbEngineConfig(answer_dwell_mode="reasoned")` is enabled and
`complete_with_reasoning(...)` is available, the engine will use that adapter
only for selected high-signal scan areas. The final answer still uses
`complete(...)`.

## Files

- `memory_orb/engine.py`: swap manager + retrieval + bounded context assembler
- `memory_orb/adapters.py`: pluggable token estimator/embedder/model interfaces
- `memory_orb/utils.py`: anchor extraction, MMR, similarity, summarization
- `tests/test_memory_orb.py`: behavior tests for bounded context and retrieval
- `examples/simulation.py`: long-run simulation
- `benchmarks/selective_attention_benchmark.py`: synthetic importance benchmark

## Benchmark

Run a synthetic selective-attention benchmark that compares Memory Orb against a
recent-window baseline, plus quick-search chunk scoring with `qwen3:0.6b` and
a keyword baseline. Use `semantic` dataset mode to test beyond literal keyword
matching.

```bash
python benchmarks/selective_attention_benchmark.py --trials 4 --seed 17 --qwen-model qwen3:0.6b --qwen-segments 60 --dataset-mode semantic
```

Fact-recall benchmark (direct compare of Memory Orb vs long-context Qwen):

```bash
python benchmarks/fact_recall_compare.py --trials 3 --memory-model qwen3:0.6b --long-model qwen3:4b --long-model-ctx 262144 --records 260 --noise 700
```

To benchmark the new dwell mode:

```bash
python benchmarks/fact_recall_compare.py --trials 3 --memory-model qwen3.5:0.8b --long-model qwen3.5:0.8b --memory-dwell-mode reasoned --reasoning-dwell-ctx 900
```

Long-form linked benchmark (10 natural multi-paragraph passages, one verifiable
question each):

```bash
python benchmarks/linked_longform_benchmark.py --models qwen3:0.6b,qwen3:4b --mode both --direct-ctx 32768 --write-dataset benchmarks/linked_longform_dataset.json
```

`linked_longform_benchmark.py` runs memory-orb in strict fair mode by default
(`allow_answer_coercion=False`). To enable post-correction explicitly, add
`--allow-post-correction`.

Hard long-form benchmark (longer passages + multi-field tracking in one answer):

```bash
python benchmarks/linked_longform_hard_benchmark.py --models qwen3:0.6b,qwen3:4b --mode both --direct-ctx 32768
```

Memory-only smoke compare of heuristic vs reasoned dwell:

```bash
python benchmarks/linked_longform_hard_benchmark.py --models qwen3.5:0.8b --mode memory-orb --memory-dwell-mode reasoned --reasoning-dwell-ctx 900 --limit-cases 1
```

Official benchmark compare (LongBench v2 sampled subset):

```bash
python benchmarks/longbench_v2_compare.py --sample-size 12 --lengths short --max-context-chars 220000 --memory-model qwen3:0.6b --long-model qwen3:4b --json-out benchmarks/longbench_v2_compare_results.json
```
