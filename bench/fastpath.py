"""End-to-end fast-path latency: question -> grounded extractive answer.

`bench/latency.py` measures a *single* ChunkIndex. That is not the served path.
`AdaptiveRetriever` queries three indexes, and at pilot scale the fan-out was free
(3 x ~1ms), so the difference never showed. At full scale BM25 costs ~30ms per
index because `bm25s` scores by sparse matmul -- O(corpus) -- while HNSW is
logarithmic, and a serial fan-out therefore sums into the budget:

    3 x ~60ms retrieval + ~40ms extract  ~=  220ms   -> over

This measures the real harness path (guardrail -> embed -> retrieve -> extract ->
guardrail) so the reported P50/P70/P100 belongs to the system that actually serves,
not to one component of it. Generation is excluded by construction: the task scopes
the budget as "chunking + vector DB retrieval + everything through to final output",
and the extractive answer is complete and grounded on its own.

    python -m bench.fastpath --tag full --n 300
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np

from core.embedder import Embedder, EmbedderConfig
from core.harness import RAGHarness
from core.retriever import DEFAULT_ENSEMBLE
from ingest.pipeline import DATA_ROOT, read_jsonl

PERCENTILES = {"p50": 50, "p70": 70, "p90": 90, "p99": 99}


class _NoLLM:
    """Generation is out of scope here; the harness must never call this."""

    def generate(self, *a, **kw):  # pragma: no cover - defensive
        raise AssertionError("fast-path benchmark must not invoke generation")


def pct(values: list[float]) -> dict[str, float]:
    a = np.array(values)
    out = {k: round(float(np.percentile(a, p)), 3) for k, p in PERCENTILES.items()}
    out["p100"] = round(float(a.max()), 3)
    out["mean"] = round(float(a.mean()), 3)
    return out


def run(tag: str, ensemble: list[str], n_per_lang: int = 100, warmup_per_lang: int = 10, top_k: int = 10) -> dict:
    rng = np.random.default_rng(42)
    hin_raw = read_jsonl(DATA_ROOT / "raw" / "hin_train_queries.jsonl")
    mar_raw = read_jsonl(DATA_ROOT / "raw" / "mar_train_queries.jsonl")

    total_need = n_per_lang + warmup_per_lang
    hin_idx = rng.choice(len(hin_raw), total_need, replace=False)
    mar_idx = rng.choice(len(mar_raw), total_need, replace=False)

    # 30 Warmup queries (10 per language)
    hin_warmup = [hin_raw[i]["query"] for i in hin_idx[:warmup_per_lang]]
    mar_warmup = [mar_raw[i]["query"] for i in mar_idx[:warmup_per_lang]]
    eng_warmup = [hin_raw[i]["query_eng"] for i in hin_idx[:warmup_per_lang]]
    warmup_queries = hin_warmup + mar_warmup + eng_warmup

    # 300 Measured queries (100 per language)
    hin_test = [hin_raw[i]["query"] for i in hin_idx[warmup_per_lang:]]
    mar_test = [mar_raw[i]["query"] for i in mar_idx[warmup_per_lang:]]
    eng_test = [hin_raw[i]["query_eng"] for i in hin_idx[warmup_per_lang:]]
    test_queries = [(q, "Hindi") for q in hin_test] + [(q, "Marathi") for q in mar_test] + [(q, "English") for q in eng_test]

    # Deterministic shuffle across languages
    rng_test = np.random.default_rng(42)
    rng_test.shuffle(test_queries)

    threads = int(os.getenv("ORT_THREADS", "3"))
    embedder = Embedder(EmbedderConfig(threads=threads))
    index_root = DATA_ROOT / "index" / tag

    eng_ret = None
    if (index_root / "english_256" / "hnsw.bin").exists():
        from core.retriever import AdaptiveRetriever
        eng_ret = AdaptiveRetriever.load(index_root, ["english_256"])

    h = RAGHarness(index_root, ensemble, llm=_NoLLM(), embedder=embedder, top_k=top_k, english_retriever=eng_ret)

    # Warmup pass (30 queries)
    cold = {}
    for i, q in enumerate(warmup_queries):
        r = h.answer(q, generate=False)
        if i == 0:
            cold = {"fast_path_ms": r.fast_path_ms, **r.timings_ms}

    # Measurement pass (300 queries)
    stages: dict[str, list[float]] = defaultdict(list)
    fast: list[float] = []
    by_lang: dict[str, list[float]] = defaultdict(list)
    decisions: dict[str, int] = defaultdict(int)

    for q, lang in test_queries:
        r = h.answer(q, generate=False)
        fast.append(r.fast_path_ms)
        by_lang[lang].append(r.fast_path_ms)
        for k, v in r.timings_ms.items():
            stages[k].append(v)
        decisions[r.answer_source] += 1

    return {
        "tag": tag,
        "ensemble": ensemble,
        "n_queries": len(fast),
        "warmup_queries": len(warmup_queries),
        "threads": threads,
        "cold_first_query_ms": cold,
        "fast_path_ms": pct(fast),
        "by_lang_ms": {k: pct(v) for k, v in by_lang.items()},
        "stages_ms": {k: pct(v) for k, v in stages.items()},
        "answer_source": dict(decisions),
        "over_budget": sum(1 for v in fast if v > 200),
        "budget_ms": 200,
    }


def render(r: dict) -> str:
    f = r["fast_path_ms"]
    under_200 = r["n_queries"] - r["over_budget"]
    pass_pct = (under_200 / r["n_queries"]) * 100
    status = "PASS" if r["over_budget"] == 0 else "FAIL"

    lines = [
        "============================================================",
        "        VOICE RAG — FINAL LATENCY BENCHMARK",
        "============================================================",
        "",
        "Configuration",
        "  Languages       : Hindi + Marathi + English",
        f"  Queries         : {r['n_queries']} (100 / 100 / 100)",
        f"  Warmup          : {r['warmup_queries']} queries (excluded from latency)",
        "  MAX_EMBED       : 5",
        f"  ORT_THREADS     : {r['threads']}",
        "  Mode            : Full grounded RAG fast-path",
        "",
        "------------------------------------------------------------",
        "OVERALL LATENCY",
        "------------------------------------------------------------",
        "",
        f"  P50   : {f['p50']:6.2f} ms",
        f"  P70   : {f['p70']:6.2f} ms",
        f"  P90   : {f['p90']:6.2f} ms",
        f"  P99   : {f['p99']:6.2f} ms",
        f"  P100  : {f['p100']:6.2f} ms",
        f"  Mean  : {f['mean']:6.2f} ms",
        "",
        f"  Under 200 ms : {under_200} / {r['n_queries']} ({pass_pct:.1f}%)",
        f"  STATUS       : {status}",
        "",
        "------------------------------------------------------------",
        "PIPELINE BREAKDOWN",
        "------------------------------------------------------------",
        "",
        f"  {'Stage':<20} {'P50':>8} {'P70':>8} {'P90':>8} {'P99':>8} {'P100':>8}",
        f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}",
    ]

    stage_display_names = [
        ("guardrail_in", "Input Guardrail"),
        ("embed_query", "Query Embedding"),
        ("retrieve", "Hybrid Retrieval"),
        ("extract", "Extraction & Embed"),
        ("guardrail_out", "Output Guardrail"),
    ]

    for key, label in stage_display_names:
        if key in r["stages_ms"]:
            s = r["stages_ms"][key]
            lines.append(f"  {label:<20} {s['p50']:8.2f} {s['p70']:8.2f} {s['p90']:8.2f} {s['p99']:8.2f} {s['p100']:8.2f}")

    lines.append(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    lines.append(f"  {'TOTAL':<20} {f['p50']:8.2f} {f['p70']:8.2f} {f['p90']:8.2f} {f['p99']:8.2f} {f['p100']:8.2f}")

    lines += [
        "",
        "------------------------------------------------------------",
        "LANGUAGE BREAKDOWN",
        "------------------------------------------------------------",
        "",
        f"  {'Language':<16} {'P50':>8} {'P70':>8} {'P90':>8} {'P99':>8} {'P100':>8}",
        f"  {'-'*16} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}",
    ]

    for lg in ["Hindi", "Marathi", "English"]:
        if lg in r["by_lang_ms"]:
            s = r["by_lang_ms"][lg]
            lines.append(f"  {lg:<16} {s['p50']:8.2f} {s['p70']:8.2f} {s['p90']:8.2f} {s['p99']:8.2f} {s['p100']:8.2f}")

    lines += [
        "",
        "------------------------------------------------------------",
        "TASK #2 RESULT",
        "------------------------------------------------------------",
        "",
        f"  200 ms requirement : {status}",
        f"  Queries tested     : {r['n_queries']}",
        f"  Queries ≤ 200 ms   : {under_200}",
        f"  Pass rate          : {pass_pct:.1f}%",
        "",
        "============================================================",
    ]

    if "cold_first_query_ms" in r and "fast_path_ms" in r["cold_first_query_ms"]:
        lines.append(f"Note: Cold 1st query was {r['cold_first_query_ms']['fast_path_ms']:.2f} ms (excluded from steady-state percentiles).")

    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="pilot")
    ap.add_argument("--ensemble", nargs="+", default=DEFAULT_ENSEMBLE)
    ap.add_argument("--n-per-lang", type=int, default=100)
    ap.add_argument("--warmup-per-lang", type=int, default=10)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--label", default="fastpath")
    args = ap.parse_args()

    r = run(args.tag, args.ensemble, args.n_per_lang, args.warmup_per_lang, args.top_k)
    output = render(r)
    print(output)

    out = DATA_ROOT / "reports"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.tag}_{args.label}_latency.json").write_text(json.dumps(r, indent=2))
    (out / f"{args.tag}_{args.label}_latency.txt").write_text(output)


if __name__ == "__main__":
    main()
