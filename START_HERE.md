# START HERE — what's done, what YOU need to do

⚠️ Your Sarvam key got exposed earlier in chat — revoke it at
dashboard.sarvam.ai and generate a new one before using this. This zip's
`.env` was deleted on purpose; recreate it fresh from `.env.example` with
NEW keys, never re-paste keys into any chat.

## What's already done (don't redo this)
- Full voice RAG pipeline: Sarvam STT → 4-strategy chunking → dense+sparse
  retrieval → extractive answer → LLM polish → guardrails.
- "Compare and pick optimal" work Aryan mentioned: already done. See
  README.md section "Chunking — 12 variants tested, 1 shipped" — 7 index
  configs were tested with statistical significance testing, best one
  (`metadata_128`) is what's shipped.
- Latency already fast: P50 64ms, P70 69ms, P100 115ms (budget is 200ms).
  Full breakdown in README.md "Latency analytics" section.
- Guardrails: input intent filter + grounding gate (abstains if support < 0.35)
  + output novel-fact check. See README.md "Guardrails" section.
- UI: rebranded and re-skinned (बोल / bol.sh, maroon+marigold theme).

## What YOU need to do (I cannot do these — need your accounts/laptop)

### 1. Get your API keys (free tier is enough)
- Sarvam: https://dashboard.sarvam.ai → get SARVAM_API_KEY
- LLM (pick ONE, Gemini free tier is easiest): https://aistudio.google.com/apikey → GEMINI_API_KEY
- Copy `.env.example` to `.env` and fill these two keys in. Set `LLM_PROVIDER=gemini`.

### 2. Install Docker Desktop if you don't have it
https://www.docker.com/products/docker-desktop/

### 3. Build the index (one-time, takes 2-3 hours, let it run in background)
```
docker compose run --rm ingest
```

### 4. Start the server
```
docker compose up api
```
Then open `web/index.html` in a browser, or check what port it's serving on
(should say `:8000` in the terminal).

### 5. Get YOUR OWN latency numbers (don't reuse the README's numbers as your own without running this — the task wants YOUR benchmark, not a copy-pasted one)
```
docker compose run --rm bench python -m bench.fastpath --tag full --n 300
```
This prints your P50/P70/P100. Screenshot or copy this output — you need it
for the submission form and to update the numbers in your own README if
they differ from what's shown now.

### 6. Deploy
- Backend → Render.com: connect this GitHub repo, it'll use the Dockerfile.
- Frontend → Vercel.com: connect same repo, set root directory to `web/`.
- Push this whole folder to a NEW GitHub repo under your team's account first
  (not siddharth's) — `git remote set-url origin <your-new-repo-url>` then
  `git push`. Update the "source ↗" link in `web/index.html` (currently a
  placeholder `#`) to point to your new repo.

### 7. Record videos, post with #RAGInGoa on Instagram/X/LinkedIn (every
member individually), then submit: https://forms.gle/MNvCjcv23Hn2Eeu58

## Honest note
I can't run Docker with real API keys, can't hit sarvam.ai or the LLM APIs,
and can't operate Antigravity — none of that is available in my environment.
Steps 1–6 above genuinely need to happen on your laptop with your accounts.
Everything I *could* prepare in advance (the code, the UI, understanding the
architecture) is done. Come back here once you've run step 5 and I'll help
you write up the numbers/README/submission text.
