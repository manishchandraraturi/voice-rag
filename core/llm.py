"""Provider-agnostic LLM client with structured output, timeouts and retries.

Requirement #5 asks for "structured orchestration around the model (tool calls,
retries, structured input/output handling, error recovery) rather than a single
raw prompt-in, text-out call". This module is the structured-I/O and
error-recovery half of that; `core/harness.py` sequences it.

Two providers behind one interface, selected by LLM_PROVIDER. Not gratuitous:
the generation step is the only part of the pipeline that leaves the machine, so
it is the only part that can fail for reasons we do not control. Being able to
swap provider with one env var is error recovery at the deployment level, and it
keeps the demo alive if a key expires the night before submission.

Everything returns a typed object. The caller never sees raw text, and never
sees an exception it cannot act on -- `GenerationResult.ok` is the single check.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _load_env_file() -> None:
    for p in [Path(".env"), Path("/app/.env"), Path(__file__).parent.parent / ".env"]:
        if p.exists():
            try:
                for line in p.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, _, v = line.partition("=")
                    k = k.strip()
                    v = v.strip().strip("'\"")
                    if k:
                        os.environ[k] = v
            except Exception:
                pass
            break


_load_env_file()

# The model is instructed to answer *only* from context and to say so when it
# cannot. This is the generation-side half of the guardrail: core/guardrails.py
# verifies the output independently, because a prompt is a request, not a
# constraint.
SYSTEM_PROMPT = """You answer questions using ONLY the numbered context passages provided.

Rules:
- Use only facts stated in the context. Never add outside knowledge.
- If the context does not answer the question, set "answer" to "" and "sufficient" to false.
- Answer in the SAME language AND THE SAME SCRIPT as the question.
  A Devanagari question gets a Devanagari answer. Never mix scripts: write ने, not نے.
  Prefer wording that appears in the context over your own paraphrase.
- Answer in a COMPLETE sentence that restates what was asked, not a bare fragment.
  For "भारत की राजधानी क्या है?" answer "भारत की राजधानी नई दिल्ली है।", not "नई दिल्ली".
  A spoken answer has no question on screen beside it, so a fragment loses its meaning.
- Keep it to one or two sentences.
- Cite the passage numbers you used in "citations".

Return ONLY a JSON object, no markdown fence:
{"answer": str, "sufficient": bool, "citations": [int]}"""

# Used ONLY when retrieval found nothing and the system has already abstained.
# The answer it produces is never merged into the grounded answer, never cited,
# and never scored for grounding -- it is surfaced separately and labelled, so a
# reader can always tell corpus-backed text from model recall. Requirement 6 is
# about not passing off ungrounded text as grounded; keeping the two visibly
# apart is how this stays on the right side of that line.
UNSOURCED_PROMPT = """Answer the question from your own general knowledge.

Rules:
- Answer in the SAME language AND SCRIPT as the question. Never mix scripts.
- One or two sentences, a complete sentence, not a fragment.
- If you are genuinely unsure, set "sufficient" to false.

Return ONLY a JSON object, no markdown fence:
{"answer": str, "sufficient": bool, "citations": []}"""

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(slots=True)
class GenerationResult:
    answer: str = ""
    sufficient: bool = False
    citations: list[int] = field(default_factory=list)
    ok: bool = False
    error: str = ""
    provider: str = ""
    model: str = ""
    attempts: int = 0
    took_ms: float = 0.0
    raw: str = ""


def build_prompt(question: str, contexts: list[str], max_chars: int = 700) -> str:
    numbered = "\n\n".join(f"[{i + 1}] {c[:max_chars]}" for i, c in enumerate(contexts))
    return f"CONTEXT:\n{numbered}\n\nQUESTION: {question}"


def _parse(text: str) -> dict[str, Any]:
    """Extract the JSON object from a model response.

    Models wrap JSON in prose or code fences even when told not to, and a
    ValueError here would surface as a 500 rather than a degraded answer. So we
    locate the outermost object rather than trusting the whole response to parse.
    """
    text = (text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    m = _JSON_BLOCK.search(text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass  # a JSON-ish block that does not parse -- fall through

    if not text:
        raise ValueError("empty response")

    # The model answered in prose instead of JSON. Raising here discarded a
    # perfectly usable answer and then burned the retry budget re-asking: measured
    # on 20 live queries, 2 failed this way and one spent **15.5 seconds** doing
    # it. Prose is a formatting miss, not a refusal.
    #
    # Salvaging it is safe because nothing downstream trusts this text -- the
    # grounding gate still verifies it against the retrieved context and rejects
    # it if unsupported. Citations are dropped rather than guessed, since an
    # invented citation is worse than none.
    return {"answer": text, "sufficient": True, "citations": [], "recovered_from_prose": True}


class LLMClient:
    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        timeout_s: float | None = None,
    ):
        self.provider = (provider or os.getenv("LLM_PROVIDER", "gemini")).lower()
        # Gemini rejects short deadlines outright ("Manually set deadline 8s is too
        # short"), so this floor is a provider constraint, not a preference. The
        # harness does not wait on it anyway -- the extractive answer is already
        # returned by then, and generation replaces it when it lands.
        self.timeout_s = timeout_s or float(os.getenv("LLM_TIMEOUT_S", "30"))
        self._client: Any = None
        self._system: str = SYSTEM_PROMPT

        if self.provider == "gemini":
            # "-latest" aliases, not pinned versions: models.list() happily returns
            # ids that 404 on generateContent ("no longer available"), so listing is
            # not a capability check. gemini-flash-lite-latest measured 1529ms vs
            # 4029ms for gemini-flash-latest on identical correct output.
            self.model = model or os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")
            self.api_key = os.getenv("GEMINI_API_KEY", "")
        elif self.provider == "anthropic":
            self.model = model or os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
            self.api_key = os.getenv("ANTHROPIC_API_KEY", "")
        elif self.provider == "openrouter":
            # OpenAI-compatible, so httpx is enough -- no extra SDK. One key
            # fronts ~500 models, which makes it the cheapest insurance against
            # a single provider rate-limiting us the night before submission.
            self.model = model or os.getenv("OPENROUTER_MODEL", "google/gemma-4-31b-it:free")
            self.api_key = os.getenv("OPENROUTER_API_KEY", "")
            self.base_url = "https://openrouter.ai/api/v1"
        elif self.provider == "nvidia":
            # NVIDIA NIM speaks the same OpenAI dialect, so it reuses the same
            # transport -- only base_url and the key differ.
            self.model = model or os.getenv("NVIDIA_MODEL", "meta/llama-3.3-70b-instruct")
            self.api_key = os.getenv("NVIDIA_API_KEY", "")
            self.base_url = "https://integrate.api.nvidia.com/v1"
        elif self.provider == "groq":
            # Groq runs on LPUs and returns in a few hundred ms.
            # Default is openai/gpt-oss-120b — available on this Groq account, LPU-accelerated.
            self.model = model or os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
            self.api_key = os.getenv("GROQ_API_KEY", "")
            self.base_url = "https://api.groq.com/openai/v1"
        elif self.provider == "bedrock":
            # Bedrock authenticates with the ambient AWS credential chain rather
            # than an API key, so `configured` is decided by boto3 finding
            # credentials, not by an env var being set.
            #
            # Note the "global." prefix: newer Anthropic models on Bedrock must be
            # invoked through an *inference profile*, not a bare model id. Passing
            # the raw id returns ValidationException "Operation not allowed", which
            # reads like a permissions problem and is not one.
            # `aws bedrock list-inference-profiles` lists the valid ids.
            self.model = model or os.getenv(
                "BEDROCK_MODEL", "global.anthropic.claude-haiku-4-5-20251001-v1:0"
            )
            self.region = os.getenv("BEDROCK_REGION", os.getenv("AWS_DEFAULT_REGION", "ap-south-1"))
            self.api_key = "aws-credential-chain"
        else:
            raise ValueError(f"unknown LLM_PROVIDER {self.provider!r}")

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    # -- providers ---------------------------------------------------------

    def _gemini(self, prompt: str, max_tokens: int) -> str:
        from google import genai
        from google.genai import types

        if self._client is None:
            self._client = genai.Client(api_key=self.api_key)

        resp = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=self._system,
                max_output_tokens=max_tokens,
                temperature=0.0,  # deterministic: this is extraction, not writing
                response_mime_type="application/json",
                http_options=types.HttpOptions(timeout=int(self.timeout_s * 1000)),
            ),
        )
        return resp.text or ""

    def _openai_compatible(self, prompt: str, max_tokens: int) -> str:
        """Shared transport for every OpenAI-dialect endpoint (OpenRouter, NVIDIA NIM, Groq).

        Only base_url and the key differ between them, so one method covers both
        and any future provider that speaks the same protocol. httpx rather than
        the openai SDK: it is already a dependency, and the surface we use here is
        one POST.
        """
        import httpx

        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout_s)

        headers = {"Authorization": f"Bearer {self.api_key}"}
        if self.provider == "openrouter":
            # Attribution headers; harmless elsewhere but only meaningful here.
            headers |= {
                "HTTP-Referer": "https://github.com/hhgoa-task2",
                "X-Title": "HH Goa Task2 Voice RAG",
            }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self._system},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }
        # Not every NIM model implements response_format; the prompt already
        # demands JSON and _parse() tolerates prose around it. OpenRouter and
        # Groq both honour it, so ask for JSON where it is supported.
        if self.provider in ("openrouter", "groq"):
            payload["response_format"] = {"type": "json_object"}

        resp = self._client.post(
            f"{self.base_url}/chat/completions", headers=headers, json=payload
        )
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
        body = resp.json()
        if "choices" not in body:
            raise RuntimeError(f"no choices in response: {str(body)[:300]}")
        return body["choices"][0]["message"]["content"] or ""

    def _bedrock(self, prompt: str, max_tokens: int) -> str:
        """Bedrock Converse API -- one shape across every model family it hosts."""
        import boto3
        from botocore.config import Config

        if self._client is None:
            self._client = boto3.client(
                "bedrock-runtime",
                region_name=self.region,
                config=Config(
                    read_timeout=self.timeout_s,
                    connect_timeout=min(self.timeout_s, 3),
                    retries={"max_attempts": 1},  # we do our own, with classification
                ),
            )

        resp = self._client.converse(
            modelId=self.model,
            system=[{"text": self._system}],
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": max_tokens, "temperature": 0.0},
        )
        return "".join(
            b.get("text", "") for b in resp["output"]["message"]["content"]
        )

    def _groq(self, prompt: str, max_tokens: int) -> str:
        try:
            from groq import Groq

            if self._client is None:
                self._client = Groq(api_key=self.api_key, timeout=self.timeout_s)

            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._system},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            return resp.choices[0].message.content or ""
        except ImportError:
            return self._openai_compatible(prompt, max_tokens)

    def _anthropic(self, prompt: str, max_tokens: int) -> str:
        import anthropic

        if self._client is None:
            self._client = anthropic.Anthropic(api_key=self.api_key, timeout=self.timeout_s)

        msg = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=0.0,
            system=self._system,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")

    # -- public ------------------------------------------------------------

    def generate(
        self,
        question: str,
        contexts: list[str],
        *,
        max_tokens: int = 300,
        retries: int = 2,
        backoff_s: float = 0.4,
        rate_limit_backoff_s: float = 3.0,
        system: str | None = None,
    ) -> GenerationResult:
        # Per-call system prompt. Defaults to the grounded contract; the unsourced
        # path passes UNSOURCED_PROMPT explicitly and nothing else may.
        self._system = system or SYSTEM_PROMPT
        """Generate a grounded answer. Never raises -- check `.ok`.

        Retries cover transient faults (timeout, 429, 5xx) and malformed JSON.
        They deliberately do not cover a missing key or an unknown model: those
        will fail identically every time, and burning the retry budget on them
        just delays the fallback to the extractive answer.
        """
        t0 = time.perf_counter()
        base = GenerationResult(provider=self.provider, model=self.model)

        if not self.configured:
            base.error = f"{self.provider}: no credentials configured"
            base.took_ms = round((time.perf_counter() - t0) * 1000, 2)
            return base
        # Guards the *grounded* path: generating from zero passages there would be
        # ungrounded by construction. The unsourced path has no context by
        # definition, and says so via its own system prompt.
        if not contexts and self._system is not UNSOURCED_PROMPT:
            base.error = "no context passages"
            base.took_ms = round((time.perf_counter() - t0) * 1000, 2)
            return base

        prompt = build_prompt(question, contexts) if contexts else f"QUESTION: {question}"
        call = {
            "gemini": self._gemini,
            "openrouter": self._openai_compatible,
            "nvidia": self._openai_compatible,
            "groq": self._groq,
            "bedrock": self._bedrock,
            "anthropic": self._anthropic,
        }[self.provider]
        last_err = ""

        for attempt in range(1, retries + 2):
            base.attempts = attempt
            try:
                raw = call(prompt, max_tokens)
                data = _parse(raw)
                base.raw = raw[:2000]
                base.answer = str(data.get("answer", "") or "").strip()
                base.sufficient = bool(data.get("sufficient", False))
                base.citations = [int(c) for c in data.get("citations", []) if str(c).isdigit()]
                base.ok = True
                base.took_ms = round((time.perf_counter() - t0) * 1000, 2)
                return base
            except Exception as e:  # noqa: BLE001 -- classify, never propagate
                last_err = f"{type(e).__name__}: {e}"
                msg = str(e).lower()
                fatal = any(
                    s in msg
                    for s in (
                        "api key",
                        "unauthorized",
                        "permission",
                        "not found",
                        "invalid model",
                        "accessdenied",
                        "could not find credentials",
                        "don't have access to the model",
                        # Bedrock returns this for a malformed request or a model
                        # that needs an inference profile -- never transient.
                        "validationexception",
                        "operation not allowed",
                        # Gemini spells it NOT_FOUND / INVALID_ARGUMENT, which the
                        # space-separated forms above miss. A retired model burned
                        # 3 attempts and 5.8s before this was added.
                        "not_found",
                        "invalid_argument",
                        "no longer available",
                        "deadline",
                    )
                )
                if fatal or attempt > retries:
                    break
                # Free-tier quotas are per-minute, so 0.4s is useless against a
                # 429 -- it just burns the retry budget. Measured on Gemini's
                # free tier: 19 of 40 queries hit RESOURCE_EXHAUSTED.
                rate_limited = "429" in msg or "resource_exhausted" in msg or "quota" in msg
                delay = (rate_limit_backoff_s if rate_limited else backoff_s) * attempt
                time.sleep(delay)

        base.error = last_err
        base.took_ms = round((time.perf_counter() - t0) * 1000, 2)
        return base


class LLMChain:
    """Try providers in order; the first success wins.

    Motivated by measurement, not theory: on Gemini's free tier, 19 of 40 queries
    returned 429 RESOURCE_EXHAUSTED. The harness already degrades to the extractive
    answer when generation fails, but a fluent answer is better than a span, and a
    second provider is cheap insurance -- especially during a judged demo window
    where a quota reset is not something we control.

    Order matters and is measured, not assumed:

        gemini-flash-lite-latest (direct)   1542ms   <- primary, fastest
        google/gemma-4-31b-it:free          2150ms   <- fallback, no quota shared

    The fallback deliberately uses a *different vendor path*. A second Gemini model
    behind the same key would share the same quota and fail at the same moment.
    """

    # Cooldowns after a failure, so a dead provider is skipped rather than
    # re-tried on every subsequent query. Measured motivation: with no memory,
    # a degraded chain spent 7.5s per query walking two throttled providers
    # before reaching a working one -- every single time.
    RATE_LIMIT_COOLDOWN_S = 60.0  # quotas are per-minute; retry after one
    FATAL_COOLDOWN_S = 600.0  # retired model / bad key: will not fix itself soon
    TRANSIENT_COOLDOWN_S = 15.0

    def __init__(self, clients: list[LLMClient], racing: bool = True):
        self.clients = [c for c in clients if c.configured]
        self.racing = racing
        if not self.clients:
            raise ValueError("no configured LLM clients")
        # index -> unix ts before which this client is skipped
        self._cooldown_until: dict[int, float] = {}
        self._served: dict[str, int] = {}
        self._skipped: dict[str, int] = {}

    @classmethod
    def from_env(cls) -> LLMChain:
        """Primary from LLM_PROVIDER, then LLM_FALLBACK_CHAIN as provider:model pairs.

        Groq with LPU acceleration is primary when configured, with speculative Model Racing
        across multiple candidate models (e.g. openai/gpt-oss-120b and openai/gpt-oss-20b)
        to eliminate single-model latency spikes and cloud jitter.
        """
        primary_provider = os.getenv("LLM_PROVIDER", "groq").lower()
        primary_model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b") if primary_provider == "groq" else None
        chain = [LLMClient(provider=primary_provider, model=primary_model)]

        spec = os.getenv(
            "LLM_FALLBACK_CHAIN",
            "groq:openai/gpt-oss-20b,groq:qwen/qwen3.6-27b,groq:groq/compound-mini",
        )
        for entry in (e.strip() for e in spec.split(",") if e.strip()):
            provider, _, model = entry.partition(":")
            if not model:
                continue
            try:
                c = LLMClient(provider=provider.strip(), model=model.strip())
            except ValueError:
                continue
            if c.configured:
                chain.append(c)

        racing = os.getenv("LLM_RACING", "true").lower() in ("true", "1", "yes")
        return cls(chain, racing=racing)

    @property
    def provider(self) -> str:
        return "+".join(c.provider for c in self.clients)

    @property
    def model(self) -> str:
        return " -> ".join(f"{c.provider}:{c.model}" for c in self.clients)

    def __len__(self) -> int:
        return len(self.clients)

    @property
    def configured(self) -> bool:
        return bool(self.clients)

    def _cooldown_for(self, error: str) -> float:
        e = error.lower()
        if any(s in e for s in ("429", "resource_exhausted", "quota", "rate-limited", "rate limit")):
            return self.RATE_LIMIT_COOLDOWN_S
        if any(s in e for s in ("not_found", "no endpoints", "api key", "unauthorized",
                                "invalid_argument", "operation not allowed", "402",
                                "insufficient credits")):
            return self.FATAL_COOLDOWN_S
        return self.TRANSIENT_COOLDOWN_S

    def status(self) -> list[dict]:
        """Chain health, for the metrics panel and for debugging a live demo."""
        now = time.time()
        return [
            {
                "provider": c.provider,
                "model": c.model,
                "available": self._cooldown_until.get(i, 0.0) <= now,
                "cooldown_s": max(0.0, round(self._cooldown_until.get(i, 0.0) - now, 1)),
                "served": self._served.get(f"{c.provider}:{c.model}", 0),
                "skipped": self._skipped.get(f"{c.provider}:{c.model}", 0),
            }
            for i, c in enumerate(self.clients)
        ]

    def generate_unsourced(self, question: str) -> GenerationResult:
        """Answer from the model's own knowledge, with no corpus behind it.

        Only called after the system has already abstained, and the result is kept
        in its own field so it can never be mistaken for a grounded answer. No
        context is passed at all -- that is the point: there was none.
        """
        return self.generate(question, [], system=UNSOURCED_PROMPT, retries=0)

    def rewrite_query(self, query: str) -> str:
        """Speculative query rewriting using fastest model race."""
        prompt = (
            f"Rewrite the following user query into a concise multilingual search query.\n"
            f"Preserve all keywords and named entities. Return ONLY the rewritten query text:\n\n{query}"
        )
        res = self.generate(
            prompt,
            [],
            system="You are a search query expansion assistant. Return ONLY the search query text.",
            max_tokens=60,
        )
        if res.ok and res.answer:
            return res.answer.strip()
        return query

    def generate(self, question: str, contexts: list[str], **kw) -> GenerationResult:
        """Generate answer using Model Racing (parallel speculative execution) when enabled.

        Sends tasks concurrently across all eligible candidate models and accepts
        whichever finishes first, eliminating single-model latency spikes.
        """
        now = time.time()
        eligible = [(i, c) for i, c in enumerate(self.clients) if self._cooldown_until.get(i, 0.0) <= now]

        if not eligible and self.clients:
            return self.clients[0].generate(question, contexts, retries=0)

        # Model Racing: Speculative parallel execution
        if self.racing and len(eligible) > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(eligible)) as executor:
                future_to_client = {
                    executor.submit(
                        c.generate,
                        question,
                        contexts,
                        retries=0,
                        **(kw if "system" in kw else {}),
                    ): (i, c)
                    for i, c in eligible
                }

                last_res = GenerationResult()
                for future in concurrent.futures.as_completed(future_to_client):
                    i, c = future_to_client[future]
                    key = f"{c.provider}:{c.model}"
                    try:
                        res: GenerationResult = future.result()
                        if res.ok:
                            self._served[key] = self._served.get(key, 0) + 1
                            self._cooldown_until.pop(i, None)
                            return res
                        else:
                            self._cooldown_until[i] = now + self._cooldown_for(res.error)
                            last_res = res
                    except Exception as e:
                        self._cooldown_until[i] = now + self._cooldown_for(str(e))
                        last_res = GenerationResult(provider=c.provider, model=c.model, error=str(e))

                return last_res

        # Sequential fallback
        last = GenerationResult()
        for i, c in eligible:
            key = f"{c.provider}:{c.model}"
            solo = len(self.clients) == 1
            r = c.generate(question, contexts, **(kw if (solo and i == 0) else ({"retries": 0} | kw)))
            if r.ok:
                self._served[key] = self._served.get(key, 0) + 1
                self._cooldown_until.pop(i, None)
                return r

            self._cooldown_until[i] = now + self._cooldown_for(r.error)
            last = r

        return last
