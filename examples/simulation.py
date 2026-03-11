from __future__ import annotations

from memory_orb import EchoModelAdapter, MemoryOrbEngine, MemoryOrbEngineConfig


def run_simulation(turns: int = 60) -> None:
    engine = MemoryOrbEngine(
        config=MemoryOrbEngineConfig(
            context_max_tokens=700,
            working_max_tokens=340,
            working_target_tokens=240,
            max_retrieved_orbs=6,
        )
    )
    model = EchoModelAdapter()

    themes = [
        "postgres replication lag",
        "vector retrieval latency",
        "token budget policy",
        "incident runbook escalation",
        "pricing forecast assumptions",
        "frontend release checklist",
    ]

    for idx in range(turns):
        theme = themes[idx % len(themes)]
        user_text = (
            f"Turn {idx}: Keep track of {theme}. "
            f"We need concrete steps, blockers, and tradeoffs for sprint planning."
        )
        _, packet = engine.chat(model=model, user_text=user_text, system_prompt="You are an ops copilot.")
        stats = engine.stats()
        print(
            f"turn={idx:02d} context_tokens={packet.total_tokens:03d} "
            f"working_tokens={stats['working_tokens']:03d} orbs={stats['orb_count']:03d}"
        )


if __name__ == "__main__":
    run_simulation()
