# Memory Research Notes (2026-02-27)

Primary-source references used to shape Memory Orb:

1. Transformer-XL (arXiv:1901.02860): segment-level recurrence to reuse hidden
   states across segments, improving long-context modeling.
2. Compressive Transformer (arXiv:1911.05507): compresses older memories instead
   of dropping them.
3. kNN-LM / Generalization through Memorization (arXiv:1911.00172): augments
   language models with nearest-neighbor datastore lookups.
4. Retrieval-Augmented Generation (RAG, arXiv:2005.11401): combines parametric
   LM with retrieved non-parametric memory.
5. RETRO (arXiv:2112.04426): retrieves external chunks at generation time for
   factual grounding and scale benefits.
6. Memorizing Transformers (arXiv:2203.08913): introduces external memory for
   approximate retrieval over long contexts.
7. Generative Agents (arXiv:2304.03442): memory stream scoring with relevance,
   recency, and importance.
8. LongMem (arXiv:2306.07174): decouples long-term memory from finite context by
   plugging in retrieval memory.
9. MemGPT (arXiv:2310.08560): virtual-context management inspired by OS paging.
10. Infini-attention (arXiv:2404.07143): bounded attention with infinite-context
    style memory compression mechanisms.

## Design implications extracted

- Pure context replay does not scale; explicit memory hierarchy is required.
- Retrieval should include recency and salience signals, not only embedding
  similarity.
- Compression/distillation is necessary because long-term stores also grow.
- OS-inspired swap boundaries are practical for long-running agent sessions.
- Hard token budgeting is a non-negotiable runtime guarantee.

## Memory Orb response

Memory Orb combines:

- bounded working memory for near-term coherence,
- episodic compressed orbs for swap history,
- semantic anchor cards for distilled cross-orb memory,
- and orbital resonance scoring that links long-term retrieval to short-term
  anchor pulses before context assembly.
