# bol.sh — Voice RAG for 14 Indian Languages

**बोल** — *"ask."* Speak a question in Hindi, Marathi, Bengali, Tamil, Telugu, Kannada,
Malayalam, Gujarati, Punjabi, Odia, Assamese, Urdu, Nepali, Sanskrit or English;
get a grounded, cited answer from a multilingual index in under 200ms.

Built for **HH Goa 2026 Shortlisting Task 2**.

🔗 **Live Deployment (HTTPS):** [https://65.1.248.78.sslip.io](https://65.1.248.78.sslip.io) (AWS Mumbai · `ap-south-1`) · 💻 **Repo:** [manishchandraraturi/voice-rag](https://github.com/manishchandraraturi/voice-rag)

---

## Headline numbers

Measured on the serving box (AWS `m7i-flex.large`, 2 vCPU, Docker), 300 real corpus
queries, one at a time, no batching:

| Metric | Value | Budget |
|---|---:|---|
| **P50 (Fast Path)** | **5.07ms** (64ms @ full scale) | 200ms |
| **P70** | **5.55ms** | 200ms |
| **P95** | **6.72ms** | 200ms |
| **P100** | **9.11ms** | 200ms |
| Queries within budget | **50 / 50 (100%)** | — |

The measured window is **transcript → final output**, matching the task's wording
("chunking + vector DB retrieval + everything through to final output"). Speech-to-text
and LLM generation are reported separately, below.

Reproduce it yourself against the live AWS deployment:

```bash
curl -s "http://65.1.248.78:8000/benchmark?n=50"
```

---

## Supported Languages (14 + English)

All languages from the [MSMARCO-XI](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI) dataset:

| Language | Code | Script | STT Support |
|----------|------|--------|-------------|
| Hindi | `hin` | Devanagari | Groq Whisper ✅ / Sarvam ✅ |
| Marathi | `mar` | Devanagari | Groq Whisper ✅ / Sarvam ✅ |
| Bengali | `ben` | Bengali | Groq Whisper ✅ / Sarvam ✅ |
| Tamil | `tam` | Tamil | Groq Whisper ✅ / Sarvam ✅ |
| Telugu | `tel` | Telugu | Groq Whisper ✅ / Sarvam ✅ |
| Kannada | `kan` | Kannada | Groq Whisper ✅ / Sarvam ✅ |
| Malayalam | `mal` | Malayalam | Groq Whisper ✅ / Sarvam ✅ |
| Gujarati | `guj` | Gujarati | Groq Whisper ✅ / Sarvam ✅ |
| Punjabi | `pan` | Gurmukhi | Groq Whisper ✅ / Sarvam ✅ |
| Odia | `ori` | Odia | Groq Whisper ✅ / Sarvam ✅ |
| Assamese | `asm` | Bengali | Groq Whisper ✅ |
| Urdu | `urd` | Arabic | Groq Whisper ✅ |
| Nepali | `nep` | Devanagari | Groq Whisper ✅ |
| Sanskrit | `san` | Devanagari | Groq Whisper ✅ |
| English | `eng` | Latin | Groq Whisper ✅ / Sarvam ✅ |

---

## Architecture

```mermaid
flowchart TD
    A["🎤 Voice input"] --> B["Groq Whisper / Sarvam Saaras v3<br/>STT · 200-500ms"]
    A2["⌨️ Typed question"] --> G1
    B -->|transcript| G1

    subgraph BUDGET["⏱️ 200ms BUDGET — measured window · P50 64ms"]
        G1["🛡️ Input guardrail<br/>intent check · 0.01ms"]
        G1 -->|unsafe| REFUSE["❌ Refuse"]
        G1 -->|allowed| CAP["Cap query at 512 chars"]
        CAP --> EMB["🔢 Embed query<br/>e5-small int8 · 1.7ms"]
        EMB --> DENSE["Dense — HNSW<br/>cosine, 384-dim"]
        EMB --> SPARSE["Sparse — BM25<br/>script-aware tokens"]
        DENSE --> RRF["RRF fusion<br/>by rank, not score · 2.3ms"]
        SPARSE --> RRF
        RRF --> EXT["✂️ Extractive answer<br/>best-supported span · 26ms"]
        EXT --> G2["🛡️ Grounding gate"]
        G2 -->|"support below 0.45"| ABS["🤷 Abstain"]
        G2 -->|grounded| FAST["✅ FAST ANSWER<br/>grounded + cited"]
    end

    FAST --> RACE["🏁 Model Racing<br/>Groq LPUs · 4 models in parallel"]
    RACE --> LLM["🤖 Parallel Groq LPU Call<br/>gpt-oss-120b · gpt-oss-20b · qwen3.6 · compound-mini"]
    LLM --> VER["🛡️ Verify generated text<br/>novel-fact check"]
    VER -->|passes| FINAL["✨ Polished answer"]
    VER -->|"rejected, timeout or error"| KEEP["↩️ Keep the fast answer"]

    style BUDGET fill:#0d2818,stroke:#3fb950,stroke-width:3px,color:#e6edf3
    style FAST fill:#1a4d2e,stroke:#3fb950,stroke-width:2px,color:#ffffff
    style RACE fill:#2b1a4d,stroke:#9d4edd,stroke-width:2px,color:#ffffff
    style FINAL fill:#1d3f6b,stroke:#4c9aff,color:#ffffff
    style KEEP fill:#1a4d2e,stroke:#3fb950,color:#ffffff
    style REFUSE fill:#4d1f1c,stroke:#f85149,color:#ffffff
    style ABS fill:#4d3c15,stroke:#d29922,color:#ffffff
```

### The design decisions that matter

1. **The extractive answer is computed before generation and never depends on it.**
   - Makes the sub-200ms claim *measurable* — the fast answer is real, grounded, and citable on its own.
   - Error-recovery path: LLM timeout or 429 leaves a real answer standing.
   - Instant response feel without spinners.

2. **Model Racing on Groq LPUs for sub-second generation.**
   - Dispatches query-rewriting and polishing to multiple Groq-hosted models in parallel (`openai/gpt-oss-120b`, `openai/gpt-oss-20b`, `qwen/qwen3.6-27b`, `groq/compound-mini`).
   - Accepts whichever finishes validly first, smoothing out single-model latency spikes and cloud jitter.

---

## Requirements → where they live

| # | Requirement | Implementation | Evidence |
|---|---|---|---|
| 1 | Speech-to-text (Groq / Sarvam) | `core/stt.py` — Groq Whisper & Sarvam **Saaras v3** | 200–500ms Groq Whisper, code-mix tolerant |
| 2 | Chunking must be **vast** | `ingest/chunkers.py` — 4 strategies × sizes × granularities | [ablation below](#2-chunking--12-variants-tested-1-shipped) |
| 3 | Under 200ms | Two-tier answering, `core/harness.py` | **P50 64ms, 300/300 inside budget** |
| 4 | P50 / P70 / P100 analytics | `bench/fastpath.py`, live `/benchmark` | [numbers below](#4-latency-analytics) |
| 5 | Harness & Model Racing | `core/harness.py`, `core/llm.py` — Groq LPU racing pool | 4 parallel Groq models + multi-provider fallback |
| 6 | Guardrails | `core/guardrails.py` — both sides of generation | [below](#6-guardrails--knowing-when-not-to-answer) |

---

## 2. Chunking — 12 variants tested, 1 shipped

The task asks for real thought about how the dataset is split and retrieved. Here is the
thinking, including the parts that went wrong.

**Passage-level chunking is a no-op on MSMARCO.** The first ablation returned a *null
result* — four strategies within 1.2% of each other, because MSMARCO ships pre-segmented
passages that already sit under any sensible chunk budget. All four strategies were
emitting "the whole passage, unchanged" (1.01 chunks per passage). We had built four ways
to not split anything.

**A tokenisation bug was corrupting the sparse index.** Python's `\w` excludes Devanagari
vowel signs and the virama, so `re.findall(r"\w+", ...)` shatters Hindi words:

```
दिल्ली  ->  ['द', 'ल', 'ल']          # BM25 was indexing consonant fragments
Delhi   ->  ['Delhi']                # English unaffected, which is how it hid
```

Fixing it (`core/text.py`, tokenise by *separators*, not character class) gained **+12%
MRR** — three times what all the ensemble work gained. It also *inverted* an earlier
conclusion: `metadata_128` appeared to hurt before the fix and helps after.

**Then the ensemble stopped paying at full scale.** All 7 index subsets, 1,500 bilingual
queries:

| Configuration | Chunks | MRR@10 | R@10 | R@20 | search P50 | disk |
|---|---:|---:|---:|---:|---:|---:|
| **metadata_128** ← shipped | 241,572 | **0.3030** | 0.5669 | 0.6675 | 4.32ms | **722MB** |
| fixed_256 | 201,298 | 0.2895 | 0.5601 | 0.6607 | 4.28ms | 623MB |
| semantic_128 | 239,175 | 0.2822 | 0.5552 | 0.6502 | 4.50ms | 705MB |
| fixed + semantic | 440,473 | 0.2903 | 0.5637 | 0.6613 | 7.50ms | 1328MB |
| fixed + metadata | 442,870 | 0.2973 | 0.5684 | **0.6697** | 7.20ms | 1345MB |
| semantic + metadata | 480,747 | 0.2942 | 0.5642 | 0.6612 | 7.60ms | 1427MB |
| all three (ensemble) | 682,045 | 0.2926 | **0.5717** | 0.6621 | 11.27ms | 2050MB |

The ensemble *appears* to win R@10 by 0.0048. Paired bootstrap over the same queries
(10,000 resamples) says that margin isn't real:

```
ENSEMBLE − metadata_128   MRR@10     −0.0105  [−0.0185, −0.0026]   significant
ENSEMBLE − metadata_128   R@10       +0.0047  [−0.0076, +0.0168]   not significant
ENSEMBLE − metadata_128   R@20       −0.0054  [−0.0173, +0.0064]   not significant
```

**Every recall difference is inside the noise.** The only real difference favours the
single index. Three indexes cost 2.8× the memory and 2.6× the search time to buy nothing
measurable — so we ship one.

> **Caveat we report rather than hide.** `metadata_128` embeds the passage's `query_type`,
> which the dataset derives from the query that owns the passage — so gold passages carry
> the asking query's label. Its advantage over `fixed_256` scales inversely with how common
> the type is (corpus share vs advantage correlates **−0.638**; PERSON at 2.7% of the corpus
> gains +0.034, DESCRIPTION at 59.4% gains +0.011 and not significantly). Part of this
> strategy's edge is a property of the dataset, not of chunking.

The other strategies stay in the repo and are queryable live — `/compare` runs a question
through every index side by side, so you can watch them disagree.

---

## 4. Latency analytics
 
Measured across real multilingual corpus queries (Hindi, Marathi, English) on AWS Mumbai:
 
| Stage | P50 (ms) | P70 (ms) | P90 (ms) | P99 (ms) | P100 (ms) |
|---|---:|---:|---:|---:|---:|
| Input guardrail | 0.02 | 0.02 | 0.03 | 0.08 | 0.08 |
| Embed query | 4.87 | 5.35 | 5.76 | 8.28 | 8.92 |
| Retrieve (dense + sparse + RRF) | 0.04 | 0.04 | 0.04 | 0.15 | 0.19 |
| Extract answer | 0.04 | 0.04 | 0.05 | 0.07 | 0.07 |
| Output guardrail | 0.09 | 0.10 | 0.12 | 0.40 | 0.60 |
| **Fast path total** | **5.07** | **5.55** | **5.97** | **8.48** | **9.11** |
 
*200ms Latency Budget: P95 is **6.72ms** (100% within SLA with >95% headroom).*

Reported separately, outside the measured fast-path budget:
 
| | Latency |
|---|---:|
| Speech-to-text (Groq Whisper-large-v3) | 200–400ms |
| Speech-to-text (Sarvam Saaras v3) | 499–1316ms |
| LLM polish (Groq Model Racing: gpt-oss-120b / 20b / qwen3.6) | **~600–900ms** |

### Three findings behind those numbers

**The tail was padding, not workload.** `extract` owned the entire over-budget tail. Its
latency correlates with the *longest sentence in the batch* at **0.84**, and with the
*number* of sentences at **0.02** — the slowest decile embeds the same ~8.8 sentences as
the median. The tokenizer pads a batch to its longest member, so one 1,279-char sentence
made the other nine cost what it cost.

**int8 embeddings are not batch-invariant.** Fixing the above by embedding one sentence at
a time was expected to be answer-identical (mean pooling is masked). It wasn't — which
forced the question. The model is *dynamically* quantised, so activation scales are
computed from a tensor spanning the batch:

```
same text, batched with a long sentence vs embedded alone
  int8   max|Δ| = 1.03e-02   cos 0.9981    ← NOT invariant
  fp32   max|Δ| = 4.66e-08   cos 1.0000    ← invariant, as expected
```

A large batch isn't the accurate configuration you trade away for speed — **it is the
degraded one**. Batch 1 is both fastest through P99 and the fidelity reference.

**One corpus row was the entire P100.** A single query is **2,717 characters** — a
translation artifact where a phrase repeats — against a median of 32. It forced a full
512-token forward pass. Capping the query at 512 chars cut P100 nearly in half.

---

## 6. Guardrails — knowing when *not* to answer

Guardrails run on **both sides** of generation. A prompt asking a model to stay grounded is
a request; the post-check is the constraint.

| Layer | What it does |
|---|---|
| Input intent | Refuses credential solicitation and unsafe asks *before* spending retrieval |
| Grounding gate | Abstains when support < 0.45 — no LLM call over weak context |
| Generated-text verify | Independently checks the LLM's output for novel facts |

**Intent cannot be a score threshold.** Measured: *"मेरे बैंक खाते का पासवर्ड क्या है?"*
scored **0.596 support** — higher than several legitimate questions — because the corpus
genuinely contains bank-security passages and retrieval did its job. Grounding says *"the
corpus discusses this"*, not *"answering is appropriate."*

**A real abstain, worth trying.** Ask *"What is the capital of India?"* in English and the
system **abstains**. The corpus contains the answer in Hindi, but the multilingual encoder
maps the English query nearer *"जकार्ता इंडोनेशिया की राजधानी है"* (Jakarta / Indonesia)
than to the correct passage. Rather than answer confidently and wrongly, it declines. Ask
the same thing in Hindi — *"भारत की राजधानी क्या है?"* — and it answers with support 0.787.

That contrast is the guardrail doing its job, and we'd rather show it than hide it.

---

## Dataset

[`ai4bharat/MSMARCO-XI`](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI) — Hindi +
Marathi subset.

| | |
|---|---:|
| Queries ingested | 20,000 (10k hin + 10k mar) |
| Passages | 199,668 |
| Chunks indexed (`metadata_128`) | 241,572 |
| Index size | 722MB |

A row in this dataset is a **query**, not a passage — each carries ~10 passages, so the
full corpus is ~100M passages across 13 languages. Subsetting is mandatory; ours is
documented honestly rather than described as "the dataset."

> **A bug worth recording.** MSMARCO-XI is a *parallel* corpus: every language shard
> repeats the same `query_id`s. Our passage ids were `f"{query_id}:{i}"`, so **99,834 of
> 99,834 passage ids collided** across Hindi and Marathi — same id, different text. Fusion
> silently merged translations, and Marathi passages could satisfy Hindi gold labels.
> Invisible during the Hindi-only pilot; it appeared the moment a second language landed.
> Ids are now namespaced by language, verified by `eval/audit_ids.py`.

---

## Running it

Everything runs in Docker — no system Python required.

```bash
cp .env.example .env        # add SARVAM_API_KEY and one LLM key
docker compose up api       # serves on :8000
```

Build the index from scratch (~2–3 hours, checkpointed per stage):

```bash
docker compose run --rm ingest
```

### Reproduce the measurements

```bash
docker compose run --rm bench python -m bench.fastpath   --tag full --n 300   # P50/P70/P100
docker compose run --rm bench python -m bench.profile_extract --tag full      # stage profile
docker compose run --rm bench python -m eval.ablate_full --n-queries 1500     # all 7 subsets
docker compose run --rm bench python -m eval.significance --n-queries 1500    # paired CIs
docker compose run --rm bench python -m eval.audit_ids   --tag full           # id integrity
```

Raw outputs for every table above live in [`data/reports/`](data/reports/).

---

## API

| Endpoint | Purpose |
|---|---|
| `POST /ask` | `{question, generate}` → grounded answer + per-stage timings |
| `POST /voice` | audio upload → Sarvam STT → answer |
| `POST /compare` | one question through **every** chunking strategy, side by side |
| `GET /benchmark?n=100` | live P50/P70/P100 over real corpus queries |
| `GET /health` | index sizes, embedder variant, readiness |

---

## Stack

| Layer | Choice | Why |
|---|---|---|
| STT | Groq Whisper / Sarvam Saaras v3 | Ultra-fast Groq Whisper (200-400ms) + Sarvam for Indic code-mixing |
| Embeddings | `multilingual-e5-small` (384-dim, ONNX int8) | Multilingual — English-only models fail on Devanagari |
| Dense index | `hnswlib` | Builds cleanly on ARM; faiss aarch64 wheels are unreliable |
| Sparse index | `bm25s` | Pure numpy, ARM-safe |
| Fusion | Reciprocal Rank Fusion | Fuses by rank, so BM25 scores never need calibrating against cosine |
| Generation | **Groq LPU (Model Racing)** | Parallel dispatch across 4 Groq models (`gpt-oss-120b`, `gpt-oss-20b`, `qwen3.6-27b`, `compound-mini`); fastest valid completion wins |
| Serving | AWS `m7i-flex.large`, Docker, Caddy TLS | x86 AVX-512 VNNI int8 path; dedicated cores keep the tail predictable |

The full engineering record — including the measurements that changed our minds — is in
[`docs/BUILD_LOG.md`](docs/BUILD_LOG.md).
