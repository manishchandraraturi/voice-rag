/* ═══════════════════════════════════════════════════════════════
   बोल — client
   Three jobs: drive the dot-matrix field, talk to the API, and make
   the two-tier answer legible (fast grounded answer first, LLM polish
   second — the second can only replace the first, never remove it).
   ═══════════════════════════════════════════════════════════════ */

const $  = (s) => document.querySelector(s);
const esc = (s) => (s ?? '').replace(/[&<>"]/g, (c) => ({ '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;' }[c]));
const ms = (v) => `${(+v).toFixed(1)}ms`;

const API_BASE = (typeof window !== 'undefined' && window.location && window.location.protocol.startsWith('http')) ? '' : 'http://localhost:8000';

const api = async (path, opts) => {
  const r = await fetch(API_BASE + path, opts);
  if (!r.ok) throw new Error((await r.text()).slice(0, 240) || `HTTP ${r.status}`);
  return r.json();
};

/* ───────────────────────── dot-matrix field ─────────────────────────
   A grid of dots displaced by two travelling sine waves. `energy` is
   driven by live microphone amplitude while recording and by a decaying
   pulse while a query is in flight, so the background is a readout of
   system state rather than ambience.                                  */
(() => {
  const cv = document.getElementById('field');
  if (!cv || window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  const ctx = cv.getContext('2d', { alpha: true });

  let w = 0, h = 0, dpr = 1, cols = 0, rows = 0, GAP = 30;

  function size() {
    // DPR capped at 1.5: the field is soft by design, so the extra pixels cost
    // fill rate without being visible.
    dpr = Math.min(window.devicePixelRatio || 1, 1.5);
    w = cv.clientWidth; h = cv.clientHeight;
    cv.width = w * dpr; cv.height = h * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    // Hold the dot count near a budget rather than a fixed pitch, so a large
    // display doesn't quadratically increase per-frame work.
    GAP = Math.max(26, Math.sqrt((w * h) / 2600));
    cols = Math.ceil(w / GAP) + 1;
    rows = Math.ceil(h / GAP) + 1;
  }
  size();
  addEventListener('resize', size, { passive: true });

  const state = { energy: 0, target: 0, t: 0 };
  window.__field = state;

  // Pointer adds a soft local swell — rewards cursor movement without noise.
  let px = -999, py = -999;
  addEventListener('pointermove', (e) => { px = e.clientX; py = e.clientY; }, { passive: true });
  addEventListener('pointerleave', () => { px = py = -999; });

  // Square dots, drawn in alpha buckets. Setting fillStyle is the expensive
  // call, so instead of ~3,000 state changes per frame we make eight: every dot
  // is binned by opacity and each bin is filled in one pass. Squares also read
  // more like an LED matrix than circles, and fillRect is far cheaper than arc.
  const BINS = 8;
  const bins = Array.from({ length: BINS }, () => []);
  let last = 0;

  function frame(now) {
    requestAnimationFrame(frame);
    if (now - last < 32) return;            // ~30fps is plenty for this motion
    last = now;

    state.t += 0.02;
    state.energy += (state.target - state.energy) * 0.06;
    state.target *= 0.985;                   // decay toward calm

    ctx.clearRect(0, 0, w, h);
    const E = state.energy;
    const hot = E > 0.1;
    for (let b = 0; b < BINS; b++) bins[b].length = 0;

    for (let i = 0; i < cols; i++) {
      const wi = i * 0.22, si = Math.sin(wi + state.t * 2.1);
      const x = i * GAP;
      for (let j = 0; j < rows; j++) {
        const y = j * GAP;
        const wave = si * Math.cos(j * 0.19 - state.t * 1.5)
                   + Math.sin((i + j) * 0.11 + state.t * 1.2);

        // vertical falloff keeps the field quiet behind the headline
        const fall = y / h * 1.5 + 0.18;

        const dx = px - x, dy = py - y;
        const d2 = dx * dx + dy * dy;
        const near = d2 < 36100 ? 1 - Math.sqrt(d2) / 190 : 0;

        const a = (0.05 + wave * 0.05 + E * 0.2) * (fall > 1 ? 1 : fall) + near * 0.34;
        if (a <= 0.02) continue;

        const s = Math.min(2.6, (1.1 + wave * 0.8) * (0.5 + E * 1.4) + near * 2.4);
        if (s <= 0.35) continue;

        const b = Math.min(BINS - 1, (a * BINS / 0.5) | 0);
        bins[b].push(x, y + wave * 9 * E, s);
      }
    }

    for (let b = 0; b < BINS; b++) {
      const arr = bins[b];
      if (!arr.length) continue;
      const a = ((b + 0.5) / BINS) * 0.5;
      ctx.fillStyle = hot
        ? `rgba(255,${(107 + 110 * (1 - Math.min(1, E))) | 0},${(53 + 160 * (1 - Math.min(1, E))) | 0},${a})`
        : `rgba(255,255,255,${a})`;
      for (let k = 0; k < arr.length; k += 3) {
        const s = arr[k + 2];
        ctx.fillRect(arr[k], arr[k + 1], s, s);
      }
    }
  }
  requestAnimationFrame(frame);
})();

const pulse = (v) => { if (window.__field) window.__field.target = Math.max(window.__field.target, v); };

/* ───────────────────────── health ───────────────────────── */
let SERVING = [];
api('/health').then((h) => {
  SERVING = h.serving || [];
  const chip = $('#chipIndex');
  chip.classList.add('ready');
  chip.querySelector('span').textContent =
    `${h.total_chunks.toLocaleString()} chunks · ${h.serving.join('+')}`;
  $('#footHost').textContent = `${h.embedder_variant} · ${h.index_tag}`;
  if (!h.stt_configured) {
    $('#micBtn').disabled = true;
    $('#hint').textContent = 'Voice disabled — no STT key on this server. Typing works.';
  }
}).catch(() => {
  $('#chipIndex').querySelector('span').textContent = 'server unreachable';
});

/* ───────────────────────── Session Conversation Feed ───────────────────────── */
const convoFeed = $('#conversationFeed');
const convoItems = $('#convoItems');
let sessionTurns = [];

function clearSessionConvo() {
  sessionTurns = [];
  if (convoItems) convoItems.innerHTML = '';
  if (convoFeed) convoFeed.hidden = true;
  document.body.classList.remove('answered');
}

const clearBtn = $('#clearConvoBtn');
if (clearBtn) clearBtn.onclick = clearSessionConvo;

function formatTurnCard(d, tier) {
  const rewritten = !!d.generated_answer && d.generated_answer !== d.extractive_answer;
  const isMuted = ['abstain', 'refusal', 'greeting'].includes(d.answer_source);

  // Small detail chips
  const chips = [];
  const src = d.answer_source;
  if (src === 'refusal')            chips.push(`<span class="detail-chip bad">⛔ refused · ${esc(d.reason || 'unsafe intent')}</span>`);
  else if (src === 'greeting')      chips.push(`<span class="detail-chip">💬 conversational · 0ms retrieval</span>`);
  else if (src === 'general_knowledge') chips.push(`<span class="detail-chip warn">🌐 General Knowledge (Model)</span>`);
  else if (src === 'abstain')       chips.push(`<span class="detail-chip warn">⚠️ not found in corpus</span>`);
  else                              chips.push(`<span class="detail-chip good">🛡️ Grounded &amp; Cited</span>`);

  if (d.fast_path_ms != null) chips.push(`<span class="detail-chip">⚡ latency: ${ms(d.fast_path_ms)}</span>`);
  if (d.support   != null && src !== 'general_knowledge') chips.push(`<span class="detail-chip">support: ${d.support.toFixed(2)}</span>`);
  if (d.grounding != null && src !== 'general_knowledge') chips.push(`<span class="detail-chip">grounding: ${d.grounding.toFixed(2)}</span>`);
  if (d.citations?.length) chips.push(`<span class="detail-chip good">cited [${d.citations.join(', ')}]</span>`);
  
  if (tier === 'generated' && d.total_ms) {
    chips.push(`<span class="detail-chip good">✨ LLM: ${ms(d.total_ms)}</span>`);
  }
  if (d.stt_ms) chips.push(`<span class="detail-chip">🎙️ STT: ${ms(d.stt_ms)}</span>`);

  // 200ms Budget mini bar
  const fastMs = d.fast_path_ms || 0;
  const pct = Math.min(100, (fastMs / 200) * 100);
  const isOver = fastMs > 200;
  const budgetBarHtml = `
    <div class="budget-mini">
      <div class="budget-mini-bar">
        <div class="budget-mini-fill ${isOver ? 'over' : ''}" style="width: ${pct}%"></div>
      </div>
      <div class="budget-mini-legend">
        <span>${ms(fastMs)} fast path (${(100 - pct).toFixed(0)}% budget unused)</span>
        <span>200ms budget</span>
      </div>
    </div>`;

  // Collapsible sources (if any retrieved)
  const sourcesHtml = d.sources?.length
    ? `<details style="margin-top:8px; font-size:11px;"><summary style="cursor:pointer; color:var(--fg-dim); font-family:var(--mono);">Retrieved context (${d.sources.length} passages) ↘</summary>` +
      d.sources.map((s, i) => `
        <div class="src" style="margin:8px 0; font-size:12px;">[${i + 1}] ${esc(s.text.slice(0, 240))}…
          <div class="meta" style="font-size:9.5px;">${esc(s.unit_id)} · rrf ${s.score}</div>
        </div>`).join('') + `</details>`
    : '';

  return `
    <div class="main-answer ${isMuted ? 'muted' : ''}" lang="hi">${esc(d.answer || '(no answer)')}</div>
    ${budgetBarHtml}
    <div class="meta-details-row">
      ${chips.join('')}
    </div>
    ${sourcesHtml}
  `;
}

function renderAnswer(d, tier, questionText) {
  if (convoFeed) convoFeed.hidden = false;
  document.body.classList.add('answered');

  const currentTurnId = d._turnId || ('turn_' + Date.now());
  d._turnId = currentTurnId;

  let turnEl = document.getElementById(currentTurnId);
  if (!turnEl) {
    turnEl = document.createElement('div');
    turnEl.className = 'convo-turn';
    turnEl.id = currentTurnId;
    turnEl.innerHTML = `
      <div class="user-query-wrap">
        <div class="user-query-text">${esc(questionText || $('#q').value || '')}</div>
      </div>
      <div class="assistant-card" id="card_${currentTurnId}">
        ${formatTurnCard(d, tier)}
      </div>
    `;
    convoItems.appendChild(turnEl);
  } else {
    const cardEl = document.getElementById(`card_${currentTurnId}`);
    if (cardEl) {
      cardEl.innerHTML = formatTurnCard(d, tier);
    }
  }

  turnEl.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
}

/* ───────────────────────── ask ───────────────────────── */
let busy = false;

async function ask(question) {
  question = (question || '').trim();
  if (!question || busy) return;
  busy = true;
  pulse(0.9);
  $('#hint').classList.remove('err');
  $('#hint').textContent = 'retrieving…';

  const turnId = 'turn_' + Date.now();

  try {
    // Tier 1 — Extractive Fast Path
    const fast = await api('/ask', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, generate: false }),
    });
    fast._turnId = turnId;

    const willGenerate = !['refusal', 'greeting'].includes(fast.answer_source);
    renderAnswer(fast, willGenerate ? 'pending' : 'idle', question);
    $('#hint').textContent = `answered in ${ms(fast.fast_path_ms)} — fast path completed`;
    pulse(0.5);

    // Tier 2 — LLM Polish
    if (willGenerate) {
      try {
        const full = await api('/ask', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question, generate: true }),
        });
        full._turnId = turnId;
        renderAnswer(full, full.answer_source === 'generated' ? 'generated' : 'idle', question);
        $('#hint').textContent =
          full.answer_source === 'generated'
            ? `polished in ${ms(full.total_ms)} · fast answer stood at ${ms(full.fast_path_ms)}`
            : full.unsourced_answer
              ? `corpus could not answer — showing model's own knowledge, unverified`
              : `kept extracted answer in ${ms(full.fast_path_ms)}`;
      } catch {
        $('#hint').textContent = 'generation unavailable — extracted answer stands';
      }
    }
  } catch (e) {
    $('#hint').classList.add('err');
    $('#hint').textContent = e.message;
  } finally {
    busy = false;
  }
}

$('#askBtn').onclick = () => ask($('#q').value);
$('#q').addEventListener('keydown', (e) => { if (e.key === 'Enter') ask($('#q').value); });
$('#closeAns').onclick = () => {
  shell.hidden = true;
  document.body.classList.remove('answered');
};
document.querySelectorAll('.sample').forEach((b) => {
  b.onclick = () => { $('#q').value = b.dataset.q; ask(b.dataset.q); };
});

/* ───────────────────────── microphone ─────────────────────────
   Sarvam accepts wav/mp3; MediaRecorder emits webm/opus, so the blob is
   decoded and re-encoded to 16kHz mono PCM in the browser. Doing it here
   keeps ffmpeg off the server and matches Sarvam's recommended input.  */
let recorder = null, chunks = [], analyser = null, audioCtx = null, meterRAF = 0;

function encodeWav(samples, rate) {
  const buf = new ArrayBuffer(44 + samples.length * 2);
  const v = new DataView(buf);
  const str = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  str(0, 'RIFF'); v.setUint32(4, 36 + samples.length * 2, true); str(8, 'WAVE');
  str(12, 'fmt '); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, rate, true); v.setUint32(28, rate * 2, true); v.setUint16(32, 2, true);
  v.setUint16(34, 16, true); str(36, 'data'); v.setUint32(40, samples.length * 2, true);
  let o = 44;
  for (let i = 0; i < samples.length; i++, o += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    v.setInt16(o, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([buf], { type: 'audio/wav' });
}

async function toWav(blob) {
  const ac = new (window.AudioContext || window.webkitAudioContext)();
  const decoded = await ac.decodeAudioData(await blob.arrayBuffer());
  const rate = 16000;
  const off = new OfflineAudioContext(1, Math.ceil(decoded.duration * rate), rate);
  const src = off.createBufferSource();
  src.buffer = decoded; src.connect(off.destination); src.start();
  const out = await off.startRendering();
  ac.close();
  return encodeWav(out.getChannelData(0), rate);
}

/* ── Realtime Symmetrical Audio Waveform Visualizer ── */
const waveCanvas = document.getElementById('voiceWave');
const waveCtx = waveCanvas ? waveCanvas.getContext('2d') : null;
const voiceModal = document.getElementById('voiceModal');

function drawVoiceWave() {
  if (!analyser || !waveCtx || !waveCanvas) return;
  const bufferLength = analyser.frequencyBinCount;
  const dataArray = new Uint8Array(bufferLength);
  analyser.getByteFrequencyData(dataArray);

  const w = waveCanvas.width;
  const h = waveCanvas.height;
  waveCtx.clearRect(0, 0, w, h);

  const numBars = 36;
  const barWidth = 3.5;
  const gap = 5;
  const totalWidth = numBars * (barWidth + gap);
  const startX = (w - totalWidth) / 2;
  const centerY = h / 2;

  for (let i = 0; i < numBars; i++) {
    // Symmetrical distance from center
    const dist = Math.abs(i - numBars / 2) / (numBars / 2);
    const dataIdx = Math.floor((1 - dist * 0.7) * (bufferLength / 4));
    const val = dataArray[dataIdx] || 0;

    // Symmetrical vertical bar height with breathing minimum
    const minHeight = 4;
    const barHeight = Math.max(minHeight, (val / 255) * (h * 0.85) * (1 - dist * 0.35));

    const x = startX + i * (barWidth + gap);
    const y = centerY - barHeight / 2;

    const alpha = Math.max(0.35, Math.min(1, val / 180 + 0.2));
    waveCtx.fillStyle = `rgba(224, 167, 51, ${alpha})`;
    
    // Draw rounded vertical bar
    waveCtx.beginPath();
    if (waveCtx.roundRect) {
      waveCtx.roundRect(x, y, barWidth, barHeight, 2);
    } else {
      waveCtx.rect(x, y, barWidth, barHeight);
    }
    waveCtx.fill();
  }
}

function meter() {
  if (!analyser) return;
  const buf = new Uint8Array(analyser.frequencyBinCount);
  analyser.getByteTimeDomainData(buf);
  let peak = 0;
  for (let i = 0; i < buf.length; i++) peak = Math.max(peak, Math.abs(buf[i] - 128) / 128);
  if (window.__field) window.__field.target = Math.min(1.4, peak * 3.2);

  // Draw voice waveform in real time
  drawVoiceWave();

  meterRAF = requestAnimationFrame(meter);
}

function stopVoiceRecording() {
  if (recorder && recorder.state === 'recording') {
    recorder.stop();
  }
}

async function startVoiceRecording() {
  const btn = $('#micBtn');
  if (recorder && recorder.state === 'recording') {
    stopVoiceRecording();
    return;
  }

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    $('#hint').classList.add('err');
    $('#hint').textContent = 'Microphone requires HTTPS or localhost. Test via HTTPS or http://localhost:8000';
    return;
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    chunks = [];

    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    analyser = audioCtx.createAnalyser();
    analyser.fftSize = 256;
    audioCtx.createMediaStreamSource(stream).connect(analyser);
    meter();

    recorder = new MediaRecorder(stream);
    recorder.ondataavailable = (e) => chunks.push(e.data);

    recorder.onstop = async () => {
      stream.getTracks().forEach((t) => t.stop());
      cancelAnimationFrame(meterRAF);
      analyser = null;
      audioCtx?.close();
      audioCtx = null;
      btn.classList.remove('rec');
      if (voiceModal) voiceModal.hidden = true;
      $('#hint').textContent = 'transcribing…';

      try {
        const wav = await toWav(new Blob(chunks, { type: chunks[0]?.type || 'audio/webm' }));
        const fd = new FormData();
        fd.append('audio', wav, 'question.wav');
        fd.append('generate', 'true');
        const d = await api('/voice', { method: 'POST', body: fd });

        if (!d.stt_ok || !d.transcript) {
          $('#hint').classList.add('err');
          $('#hint').textContent = `couldn't hear that — ${esc(d.stt_error || 'empty transcript')}`;
          return;
        }
        $('#q').value = d.transcript;
        renderAnswer(d, d.answer_source === 'generated' ? 'generated' : 'idle', d.transcript);
        $('#hint').textContent =
          `heard “${d.transcript}” · STT ${ms(d.stt_ms)} (outside budget) · answer ${ms(d.fast_path_ms)}`;
      } catch (e) {
        $('#hint').classList.add('err');
        $('#hint').textContent = e.message;
      }
    };

    recorder.start();
    btn.classList.add('rec');
    if (voiceModal) voiceModal.hidden = false;
    $('#hint').classList.remove('err');
    $('#hint').textContent = 'listening — tap mic to finish & answer';
  } catch (e) {
    if (voiceModal) voiceModal.hidden = true;
    $('#hint').classList.add('err');
    $('#hint').textContent = `microphone blocked — ${esc(e.message)}`;
  }
}

$('#micBtn').onclick = startVoiceRecording;
const modalBtn = document.getElementById('voiceModalBtn');
if (modalBtn) modalBtn.onclick = stopVoiceRecording;

/* ───────────────────────── live benchmark ───────────────────────── */
function countTo(el, target, decimals = 0, dur = 1100) {
  const from = parseFloat(el.textContent) || 0;
  const t0 = performance.now();
  const set = (v) => { el.textContent = v.toFixed(decimals); };
  let settled = false;

  const step = (now) => {
    if (settled) return;
    const p = Math.min(1, (now - t0) / dur);
    set(from + (target - from) * (1 - Math.pow(1 - p, 3)));
    if (p < 1) requestAnimationFrame(step); else settled = true;
  };
  requestAnimationFrame(step);

  // rAF is throttled in background tabs, which would leave the tile showing a
  // stale placeholder while the API reported something else. The animation is
  // decoration; the value is not, so guarantee it lands either way.
  setTimeout(() => { if (!settled) { settled = true; set(target); } }, dur + 150);
}

$('#benchBtn').onclick = async () => {
  const btn = $('#benchBtn');
  if (btn.classList.contains('busy')) return;
  btn.classList.add('busy');
  btn.querySelector('span').textContent = 'running 100…';
  pulse(1.0);

  try {
    const d = await api('/benchmark?n=100');
    const p = d.fast_path_ms;
    countTo($('#mP50'), p.p50, 1);
    countTo($('#mP70'), p.p70, 1);
    countTo($('#mP100'), p.p100, 1);
    $('#mHit').textContent = `${d.within_budget}/${d.n_queries}`;
    btn.querySelector('span').textContent = `${d.n_queries} queries · live`;

    $('#stageRows').innerHTML = Object.entries(d.stages_ms).map(([k, s]) =>
      `<tr><td>${esc(k)}</td><td>${s.p50}</td><td>${s.p70}</td><td>${s.p90}</td><td>${s.p99}</td><td>${s.p100}</td></tr>`
    ).join('') +
      `<tr class="total"><td>fast path total</td><td>${p.p50}</td><td>${p.p70}</td><td>${p.p90}</td><td>${p.p99}</td><td>${p.p100}</td></tr>`;
  } catch (e) {
    btn.querySelector('span').textContent = 'failed — retry';
  } finally {
    btn.classList.remove('busy');
  }
};

/* ───────────────────────── strategy comparison ───────────────────────── */
$('#cmpBtn').onclick = async () => {
  const question = $('#q').value.trim();
  const btn = $('#cmpBtn');
  if (!question) { btn.textContent = 'ask something first ↑'; return; }
  btn.textContent = 'querying every index…';
  pulse(0.8);

  try {
    const d = await api('/compare', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
    });
    btn.textContent = `re-run · ${esc(d.agreement)}`;
    $('#compare').innerHTML = `
      <table class="grid-table">
        <thead><tr><th>Strategy</th><th>Chunks</th><th>Search</th><th>Extract</th><th>Support</th></tr></thead>
        <tbody>${d.configs.map((c) => `
          <tr class="${c.is_served ? 'served' : ''}">
            <td>${esc(c.config)}${c.is_served ? '<span class="tag">served</span>' : ''}</td>
            <td>${c.chunks.toLocaleString()}</td>
            <td>${c.search_ms}</td>
            <td>${c.extract_ms}</td>
            <td>${c.support}</td>
          </tr>`).join('')}
        </tbody>
      </table>`;
  } catch (e) {
    btn.textContent = 'comparison failed — retry';
  }
};
