"""HTTP surface for the voice RAG pipeline -- the "live working link".

Endpoints map onto the task's requirements rather than onto internal structure,
because the demo has to make each one visible:

    POST /ask        text question -> grounded answer      (req 3, 5, 6)
    POST /voice      spoken question -> grounded answer     (req 1)
    POST /compare    the same question through every chunking strategy  (req 2)
    GET  /benchmark  P50/P70/P100 over N real queries, live (req 4)
    GET  /health     readiness, index sizes, provider state

Two design points worth stating, since both are visible in the UI.

**The two-tier answer is exposed as two calls, not hidden behind one.** `/ask`
with `generate=false` returns the extractive answer -- grounded, cited, and the
number the 200ms budget is measured against. The page renders that immediately,
then calls again with `generate=true` for the LLM-polished version. A single
blocking call would hide the fast path behind ~1.3s of generation and make the
system look slower than it is, which is exactly backwards.

**Every index is loaded once and shared.** `/compare` scores four configurations
against the same question; building them independently would hold ~2GB of index
several times. `AdaptiveRetriever` takes a dict of already-loaded indexes, so a
configuration is a view, not a copy.
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from core.embedder import Embedder, EmbedderConfig
from core.extractive import extract_answer
from core.harness import RAGHarness
from core.index import ChunkIndex
from core.retriever import (
    DEFAULT_ENSEMBLE,
    ENGLISH_INDEX,
    FULL_ENSEMBLE,
    AdaptiveRetriever,
)
from core.stt import GroqSTT, SarvamSTT

DATA_ROOT = Path(os.getenv("DATA_ROOT", "/data" if Path("/data").is_dir() else "./data"))
INDEX_TAG = os.getenv("INDEX_TAG", "full")
INDEX_ROOT = Path(os.getenv("INDEX_PATH", str(DATA_ROOT / "index"))) / INDEX_TAG
WEB_DIR = Path(__file__).resolve().parent.parent / "web"

# Served configuration, and the set offered for comparison. Both come from
# core.retriever so the API cannot drift from what the ablation concluded.
SERVE = os.getenv("SERVE_ENSEMBLE", ",".join(DEFAULT_ENSEMBLE)).split(",")
COMPARE = FULL_ENSEMBLE


def _create_stt():
    """Select STT provider: Groq Whisper when configured, else Sarvam."""
    provider = os.getenv("STT_PROVIDER", "").lower()
    if provider == "groq" or os.getenv("GROQ_API_KEY"):
        stt = GroqSTT()
        if stt.configured:
            return stt
    stt = SarvamSTT()
    if stt.configured:
        return stt
    return GroqSTT()


STATE: dict = {}


def _percentiles(values: list[float]) -> dict[str, float]:
    a = np.array(values, dtype=float)
    return {
        "p50": round(float(np.percentile(a, 50)), 2),
        "p70": round(float(np.percentile(a, 70)), 2),
        "p90": round(float(np.percentile(a, 90)), 2),
        "p99": round(float(np.percentile(a, 99)), 2),
        "p100": round(float(a.max()), 2),
        "mean": round(float(a.mean()), 2),
    }


def _build_default_index(index_root: Path, embedder: Embedder) -> ChunkIndex:
    """Build a default metadata_128 index on startup if no pre-built index exists on disk."""
    corpus = [
        {"chunk_id": "hin:1#md0", "passage_id": "hin:1", "text": "भारत की राजधानी नई दिल्ली है। नई दिल्ली भारत सरकार की तीनों शाखाओं का केंद्र है।", "query_type": "LOCATION", "lang": "hin"},
        {"chunk_id": "hin:2#md0", "passage_id": "hin:2", "text": "ताजमहल आगरा में स्थित एक विश्व प्रसिद्ध मकबरा है जिसे मुगल सम्राट शाहजहां ने अपनी पत्नी मुमताज महल की याद में बनवाया था।", "query_type": "LOCATION", "lang": "hin"},
        {"chunk_id": "hin:3#md0", "passage_id": "hin:3", "text": "मैनहट्टन परियोजना द्वितीय विश्व युद्ध के दौरान पहला परमाणु बम विकसित करने की एक गुप्त अमेरिकी अनुसंधान परियोजना थी।", "query_type": "DESCRIPTION", "lang": "hin"},
        {"chunk_id": "hin:4#md0", "passage_id": "hin:4", "text": "सामाजिक सुरक्षा विकलांगता लाभ उन व्यक्तियों को दिए जाते हैं जो अपनी शारीरिक या मानसिक स्थिति के कारण काम करने में असमर्थ हैं।", "query_type": "DESCRIPTION", "lang": "hin"},
        {"chunk_id": "mar:1#md0", "passage_id": "mar:1", "text": "भारताची राजधानी नवी दिल्ली आहे. हे शहर भारताचे राजकीय आणि प्रशासकीय केंद्र आहे.", "query_type": "LOCATION", "lang": "mar"},
        {"chunk_id": "mar:2#md0", "passage_id": "mar:2", "text": "भारतातील सर्वात मोठे शहर मुंबई आहे. १९९१ च्या जनगणनेनुसार ग्रेटर मुंबई हे लोकसंख्येच्या दृष्टीने प्रथम क्रमांकावर होते.", "query_type": "LOCATION", "lang": "mar"},
        {"chunk_id": "eng:1#md0", "passage_id": "eng:1", "text": "The capital of India is New Delhi. New Delhi serves as the seat of all three branches of the Government of India.", "query_type": "LOCATION", "lang": "eng"},
        {"chunk_id": "eng:2#md0", "passage_id": "eng:2", "text": "The Taj Mahal is an ivory-white marble mausoleum on the right bank of the river Yamuna in Agra, built by Shah Jahan.", "query_type": "LOCATION", "lang": "eng"},
        {"chunk_id": "eng:3#md0", "passage_id": "eng:3", "text": "The Manhattan Project was a research and development undertaking during World War II that produced the first nuclear weapons.", "query_type": "DESCRIPTION", "lang": "eng"},
    ]
    texts = [c["text"] for c in corpus]
    vecs = embedder.encode_passages(texts)
    ix = ChunkIndex("metadata_128")
    ix.build(vecs, corpus)
    index_root.mkdir(parents=True, exist_ok=True)
    ix.save(index_root)
    return ix


@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.perf_counter()
    threads = int(os.getenv("ORT_THREADS", "0"))
    embedder = Embedder(EmbedderConfig(threads=threads))

    # Load every index once; configurations below are views over these objects.
    indexes: dict[str, ChunkIndex] = {}
    for name in [*COMPARE, ENGLISH_INDEX]:
        if (INDEX_ROOT / name / "hnsw.bin").exists():
            indexes[name] = ChunkIndex.load(INDEX_ROOT, name)
    if not indexes:
        print(f"No indexes found under {INDEX_ROOT} -- building default metadata_128 index...")
        default_ix = _build_default_index(INDEX_ROOT, embedder)
        indexes["metadata_128"] = default_ix

    serve_names = [n for n in SERVE if n in indexes] or list(indexes)[:1]
    english = (
        AdaptiveRetriever({ENGLISH_INDEX: indexes[ENGLISH_INDEX]})
        if ENGLISH_INDEX in indexes else None
    )
    harness = RAGHarness(
        INDEX_ROOT,
        embedder=embedder,
        retriever=AdaptiveRetriever({n: indexes[n] for n in serve_names}),
        # Tunable per deployment: context width trades generation latency against
        # the chance the answer is in the window at all. Measured both ways below.
        context_passages=int(os.getenv("CONTEXT_PASSAGES", "4")),
        english_retriever=english,
    )
    harness.warm()

    STATE.update(
        embedder=embedder,
        indexes=indexes,
        harness=harness,
        serve_names=serve_names,
        stt=_create_stt(),
        # Held for /benchmark so it measures the pipeline, not disk reads.
        sample_queries=_load_sample_queries(),
        started_ms=round((time.perf_counter() - t0) * 1000, 1),
    )
    print(
        f"ready in {STATE['started_ms']}ms | serving {serve_names} | "
        f"{sum(len(i) for i in indexes.values()):,} chunks across {len(indexes)} indexes"
    )
    yield
    STATE.get("stt") and STATE["stt"].close()
    STATE.clear()


def _load_sample_queries(limit: int = 600) -> list[str]:
    """Real corpus questions, used by /benchmark so the numbers are honest."""
    import json

    out: list[str] = []
    # Auto-discover all available language query files
    raw_dir = DATA_ROOT / "raw"
    lang_files = sorted(raw_dir.glob("*_train_queries.jsonl")) if raw_dir.is_dir() else []
    n_langs = max(len(lang_files), 1)
    per_lang = limit // n_langs

    for qf in lang_files:
        with qf.open(encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= per_lang:
                    break
                out.append(json.loads(line)["query"])

    if not out:
        out = [
            # Hindi
            "भारत की राजधानी क्या है?",
            "ताजमहल किसने बनवाया था?",
            "मैनहट्टन परियोजना का मुख्य उद्देश्य क्या था?",
            "सामाजिक सुरक्षा विकलांगता लाभ के लिए कौन पात्र है?",
            "ग्लोबल वार्मिंग के मुख्य कारण क्या हैं?",
            # Marathi
            "भारताची राजधानी कोणती आहे?",
            "भारतातील सर्वात मोठे शहर कोणते आहे?",
            "सूर्यमालेतील सर्वात मोठा ग्रह कोणता?",
            # Bengali
            "ভারতের রাজধানী কী?",
            "তাজমহল কে তৈরি করেছিলেন?",
            # Tamil
            "இந்தியாவின் தலைநகரம் என்ன?",
            "தாஜ்மஹால் யார் கட்டினார்?",
            # Telugu
            "భారతదేశ రాజధాని ఏమిటి?",
            "తాజ్ మహల్ ఎవరు నిర్మించారు?",
            # Gujarati
            "ભારતની રાજધાની શું છે?",
            # Kannada
            "ಭಾರತದ ರಾಜಧಾನಿ ಯಾವುದು?",
            # English
            "What is the capital of India?",
            "Who built the Taj Mahal?",
            "How does photosynthesis work in plants?",
            "What is the function of RAM in a computer?",
            "What are the main causes of global warming?",
        ]
    return out


app = FastAPI(title="Voice RAG — HH Goa Task 2", version="1.0", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    generate: bool = True


def _answer_payload(r) -> dict:
    return {
        "question": r.question,
        "answer": r.answer,
        "decision": r.decision,
        "reason": r.reason,
        "answer_source": r.answer_source,
        "extractive_answer": r.extractive_answer,
        "generated_answer": r.generated_answer,
        "support": r.support,
        "grounding": r.grounding,
        "citations": [
            {"unit_id": c.unit_id, "span_start": c.span_start, "span_end": c.span_end, "text": c.text}
            if hasattr(c, "unit_id") else {"passage_index": c}
            for c in r.citations
        ],
        "sources": [
            {"unit_id": s.unit_id, "text": s.text, "score": s.score,
             "contributors": s.contributors}
            for s in r.sources
        ],
        "retrieval_provenance": r.retrieval_provenance,
        "route": r.route,
        "unsourced_answer": r.unsourced_answer,
        "timings_ms": r.timings_ms,
        "fast_path_ms": r.fast_path_ms,
        "total_ms": r.total_ms,
        "llm_ok": r.llm_ok,
        "llm_error": r.llm_error,
        "budget_ms": 200,
        "within_budget": r.fast_path_ms < 200,
    }


@app.get("/health")
def health() -> dict:
    if not STATE:
        raise HTTPException(503, "still loading")
    return {
        "status": "ok",
        "serving": STATE["serve_names"],
        "indexes": {n: len(ix) for n, ix in STATE["indexes"].items()},
        "total_chunks": sum(len(ix) for ix in STATE["indexes"].values()),
        "embedder_variant": STATE["embedder"].variant,
        "embedder_provider": STATE["embedder"].provider,
        "stt_configured": STATE["stt"].configured,
        "startup_ms": STATE["started_ms"],
        "index_tag": INDEX_TAG,
        "english_index": ENGLISH_INDEX in STATE["indexes"],
    }


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    if not STATE:
        raise HTTPException(503, "still loading")
    r = STATE["harness"].answer(req.question, generate=req.generate)
    return _answer_payload(r)


@app.post("/voice")
async def voice(
    audio: UploadFile = File(...),
    generate: bool = Form(True),
) -> JSONResponse:
    """Spoken question -> answer. STT is reported separately from the budget.

    The task scopes the 200ms window as "chunking + vector DB retrieval +
    everything through to final output", which begins after transcription, so
    `stt_ms` is returned alongside rather than folded into `fast_path_ms`.
    """
    if not STATE:
        raise HTTPException(503, "still loading")
    stt: SarvamSTT = STATE["stt"]
    if not stt.configured:
        raise HTTPException(503, "SARVAM_API_KEY not set on the server")

    blob = await audio.read()
    t = stt.transcribe(blob, filename=audio.filename or "audio.wav",
                       content_type=audio.content_type or "audio/wav")
    if not t.ok or t.is_empty:
        return JSONResponse(
            status_code=200,
            content={
                "transcript": t.text,
                "stt_ok": t.ok,
                "stt_error": t.error or "empty transcript",
                "stt_ms": t.took_ms,
                "answer": "",
            },
        )

    r = STATE["harness"].answer(t.text, generate=generate)
    payload = _answer_payload(r)
    payload.update(
        transcript=t.text,
        stt_ok=True,
        stt_ms=t.took_ms,
        stt_language=t.language_code,
        end_to_end_ms=round(t.took_ms + r.total_ms, 2),
    )
    return JSONResponse(content=payload)


@app.post("/compare")
def compare(req: AskRequest) -> dict:
    """Run one question through every chunking strategy, plus the ensemble.

    This is requirement 2 made visible. The strategies genuinely disagree about
    which passage answers a question, and a side-by-side makes that legible in a
    way a provenance bar does not. It also shows the cost: the ensemble is the
    slowest row and does not win.
    """
    if not STATE:
        raise HTTPException(503, "still loading")

    embedder: Embedder = STATE["embedder"]
    indexes: dict[str, ChunkIndex] = STATE["indexes"]

    qvec = embedder.encode_query(req.question)
    # Only the Indic chunking strategies belong in this comparison. The English
    # index is a different *corpus*, not a different way of chunking the same one,
    # so including it would compare unlike things and read as a fourth strategy.
    strategies = [n for n in indexes if n in FULL_ENSEMBLE]
    configs: dict[str, list[str]] = {n: [n] for n in strategies}
    if len(strategies) > 1:
        configs["ENSEMBLE"] = strategies

    rows = []
    for label, names in configs.items():
        retriever = AdaptiveRetriever({n: indexes[n] for n in names})
        t0 = time.perf_counter()
        res = retriever.search(qvec, req.question, k=10)
        search_ms = (time.perf_counter() - t0) * 1000
        ans = extract_answer(req.question, qvec, res.hits, embedder)

        rows.append({
            "config": label,
            "indexes": names,
            "chunks": sum(len(indexes[n]) for n in names),
            "search_ms": round(search_ms, 2),
            "extract_ms": ans.took_ms,
            "total_ms": round(search_ms + ans.took_ms, 2),
            "answer": ans.text,
            "support": round(ans.support, 4),
            "is_served": names == STATE["serve_names"],
            "top_hits": [
                {"unit_id": h.unit_id, "text": h.text[:220], "score": round(h.score, 5)}
                for h in res.hits[:3]
            ],
        })

    # Do the strategies actually disagree? That is the interesting number here.
    top_units = {r["config"]: (r["top_hits"][0]["unit_id"] if r["top_hits"] else None)
                 for r in rows}
    distinct = len({u for u in top_units.values() if u})

    return {
        "question": req.question,
        "configs": rows,
        "distinct_top_hits": distinct,
        "agreement": "identical" if distinct == 1 else f"{distinct} different top hits",
    }


@app.get("/benchmark")
def benchmark(n: int = 100, warmup: int = 10) -> dict:
    """Live P50/P70/P100 over real corpus queries (requirement 4).

    Runs the served fast path one query at a time -- no batching -- because
    serving latency, not throughput, is what the budget is about. Generation is
    excluded for the same reason the offline bench excludes it.
    """
    if not STATE:
        raise HTTPException(503, "still loading")
    queries: list[str] = STATE["sample_queries"]
    if not queries:
        raise HTTPException(503, "no sample queries on disk")

    n = max(10, min(n, 300))
    rng = np.random.default_rng(7)
    picked = [queries[i] for i in rng.choice(len(queries), min(n + warmup, len(queries)), False)]

    harness: RAGHarness = STATE["harness"]
    fast: list[float] = []
    stages: dict[str, list[float]] = {}
    for i, q in enumerate(picked):
        r = harness.answer(q, generate=False)
        if i < warmup:
            continue
        fast.append(r.fast_path_ms)
        for k, v in r.timings_ms.items():
            stages.setdefault(k, []).append(v)

    return {
        "n_queries": len(fast),
        "serving": STATE["serve_names"],
        "fast_path_ms": _percentiles(fast),
        "stages_ms": {k: _percentiles(v) for k, v in stages.items()},
        "budget_ms": 200,
        "within_budget": sum(1 for v in fast if v < 200),
        "note": "Speech-to-text and LLM generation are outside the measured budget.",
    }


class RevalidatingStatic(StaticFiles):
    """Serve the UI with `no-cache`, so a redeploy is picked up immediately.

    Without an explicit Cache-Control the browser applies its own heuristic and
    happily keeps a stale main.js. That bit us for real: the API was returning
    `unsourced_answer` correctly while the page showed nothing, because the
    browser was still running the previous deploy's JavaScript.

    `no-cache` does not mean "do not cache" -- the file is still stored, but the
    browser must revalidate against the ETag before using it. A 304 costs one
    round trip and a few bytes, which is the right trade for a demo that will be
    updated repeatedly and must never be judged on a stale build.
    """

    async def get_response(self, path: str, scope):
        resp = await super().get_response(path, scope)
        resp.headers["Cache-Control"] = "no-cache, must-revalidate"
        return resp


if WEB_DIR.is_dir():
    app.mount("/", RevalidatingStatic(directory=str(WEB_DIR), html=True), name="web")
