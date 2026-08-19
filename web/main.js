/* ═══════════════════════════════════════════════════════════════
   बोल — client
   Three jobs: drive the dot-matrix field, talk to the API, and make
   the two-tier answer legible.
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

const pulse = (v) => {};

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
    $('#hint').textContent = 'Note: Set STT API key for voice transcription. Typing works.';
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
  else if (src === 'general_knowledge') {
    chips.push(`<span class="detail-chip warn">⚠️ Not in dataset · Model Fallback</span>`);
    if (d.total_ms) chips.push(`<span class="detail-chip warn">✨ LLM: ${ms(d.total_ms)}</span>`);
  }
  else if (src === 'abstain')       chips.push(`<span class="detail-chip warn">⚠️ not found in corpus</span>`);
  else                              chips.push(`<span class="detail-chip good">🛡️ Grounded &amp; Cited</span>`);

  if (d.fast_path_ms != null) chips.push(`<span class="detail-chip">⚡ latency: ${ms(d.fast_path_ms)}</span>`);
  if (d.support   != null && src !== 'general_knowledge') chips.push(`<span class="detail-chip">support: ${d.support.toFixed(2)}</span>`);
  if (d.grounding != null && src !== 'general_knowledge') chips.push(`<span class="detail-chip">grounding: ${d.grounding.toFixed(2)}</span>`);
  
  const citList = (Array.isArray(d.citations) ? d.citations : []).map(c => typeof c === 'object' ? (c.passage ?? c.id ?? Object.values(c)[0] ?? 1) : c).filter(Boolean);
  if (citList.length) chips.push(`<span class="detail-chip good">cited [${citList.join(', ')}]</span>`);
  
  if (d.stt_ms) chips.push(`<span class="detail-chip">🎙️ STT: ${ms(d.stt_ms)}</span>`);

  // 200ms Budget mini bar removed for clean UI
  const budgetBarHtml = '';

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
    convoItems.prepend(turnEl);
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
    // Fire BOTH calls in parallel — fast path shows immediately,
    // LLM generation runs simultaneously in background
    const fastPromise = api('/ask', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, generate: false }),
    });
    const fullPromise = api('/ask', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, generate: true }),
    });

    // Show fast answer as soon as it arrives
    const fast = await fastPromise;
    fast._turnId = turnId;

    const isAbstain = fast.answer_source === 'abstain';
    const isTerminal = ['refusal', 'greeting'].includes(fast.answer_source);

    if (isTerminal) {
      renderAnswer(fast, 'idle', question);
      $('#hint').textContent = `answered in ${ms(fast.fast_path_ms)}`;
      busy = false;
      return;
    }

    if (!isAbstain) {
      renderAnswer(fast, 'pending', question);
      $('#hint').textContent = `answered in ${ms(fast.fast_path_ms)} — polishing…`;
    } else {
      $('#hint').textContent = `searching deeper… (fast path ${ms(fast.fast_path_ms)})`;
    }
    pulse(0.5);

    // Now await the full/generated answer (already running in parallel)
    try {
      const full = await fullPromise;
      full._turnId = turnId;
      const src = full.answer_source;
      renderAnswer(full, (src === 'generated' || src === 'general_knowledge') ? 'generated' : 'idle', question);
      $('#hint').textContent =
        src === 'generated'
          ? `polished in ${ms(full.total_ms)} · fast path ${ms(full.fast_path_ms)}`
          : src === 'general_knowledge'
            ? `model knowledge · ${ms(full.total_ms)}`
            : `answered in ${ms(full.fast_path_ms)}`;
    } catch {
      if (isAbstain) renderAnswer(fast, 'idle', question);
      $('#hint').textContent = 'generation unavailable — extracted answer stands';
    }
  } catch (e) {
    $('#hint').classList.add('err');
    $('#hint').textContent = e.message;
  } finally {
    busy = false;
  }
}

const askBtnEl = $('#askBtn');
if (askBtnEl) askBtnEl.onclick = () => ask($('#q').value);
const qEl = $('#q');
if (qEl) qEl.addEventListener('keydown', (e) => { if (e.key === 'Enter') ask($('#q').value); });
const closeAnsBtn = $('#closeAns');
if (closeAnsBtn) closeAnsBtn.onclick = () => {
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
  if (!blob || blob.size < 50) {
    throw new Error('Recording was too short, please speak clearly');
  }
  const ac = new (window.AudioContext || window.webkitAudioContext)();
  const decoded = await ac.decodeAudioData(await blob.arrayBuffer());
  const rate = 16000;
  const off = new OfflineAudioContext(1, Math.max(1, Math.ceil(decoded.duration * rate)), rate);
  const src = off.createBufferSource();
  src.buffer = decoded; src.connect(off.destination); src.start();
  const out = await off.startRendering();
  ac.close();
  return encodeWav(out.getChannelData(0), rate);
}

/* ── Realtime Symmetrical Audio Waveform Visualizer ── */
let waveAnimFrame = 0;

function drawVoiceWave() {
  const canvas = document.getElementById('voiceWave');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  const numBars = 36;
  const barWidth = 3.5;
  const gap = 5;
  const totalWidth = numBars * (barWidth + gap);
  const startX = (w - totalWidth) / 2;
  const centerY = h / 2;

  let dataArray = null;
  let bufferLength = 0;
  if (analyser) {
    bufferLength = analyser.frequencyBinCount;
    dataArray = new Uint8Array(bufferLength);
    analyser.getByteFrequencyData(dataArray);
  }

  const t = performance.now() * 0.005;

  for (let i = 0; i < numBars; i++) {
    // Symmetrical distance from center
    const dist = Math.abs(i - numBars / 2) / (numBars / 2);
    let val = 0;
    if (dataArray && bufferLength > 0) {
      const dataIdx = Math.floor((1 - dist * 0.7) * (bufferLength / 4));
      val = dataArray[dataIdx] || 0;
    } else {
      // Idle undulating sine wave animation
      val = (Math.sin(t + i * 0.3) * 0.5 + 0.5) * 45;
    }

    // Symmetrical vertical bar height
    const minHeight = 5;
    const barHeight = Math.max(minHeight, (val / 255) * (h * 0.88) * (1 - dist * 0.35) + Math.sin(t * 2 + i) * 3);

    const x = startX + i * (barWidth + gap);
    const y = centerY - barHeight / 2;

    const alpha = Math.max(0.4, Math.min(1, val / 160 + 0.3));
    ctx.fillStyle = `rgba(224, 167, 51, ${alpha})`;
    
    // Draw rounded vertical bar
    ctx.beginPath();
    if (ctx.roundRect) {
      ctx.roundRect(x, y, barWidth, barHeight, 2);
    } else {
      ctx.rect(x, y, barWidth, barHeight);
    }
    ctx.fill();
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

function openVoiceModal() {
  const modal = document.getElementById('voiceModal');
  if (modal) {
    modal.style.display = 'flex';
    modal.classList.add('active');
    const vs = document.getElementById('voiceStatus'); if (vs) vs.textContent = 'Listening… speak now';
    const sub = document.getElementById('voiceSubtext'); if (sub) sub.textContent = 'Tap mic in center to finish & answer';
  }
  const btn = document.getElementById('micBtn');
  if (btn) btn.classList.add('rec');
  drawVoiceWave();
}

function closeVoiceModal() {
  if (recorder && recorder.state === 'recording') {
    try { recorder.stop(); } catch {}
  }
  const modal = document.getElementById('voiceModal');
  if (modal) {
    modal.classList.remove('active');
    modal.style.display = 'none';
  }
  const btn = document.getElementById('micBtn');
  if (btn) btn.classList.remove('rec');
}

function stopVoiceRecording() {
  if (recorder && recorder.state === 'recording') {
    try { recorder.stop(); } catch {}
  } else {
    closeVoiceModal();
  }
}

async function startVoiceRecording(e) {
  if (e && e.preventDefault) e.preventDefault();
  const btn = document.getElementById('micBtn');
  if (recorder && recorder.state === 'recording') {
    stopVoiceRecording();
    return;
  }

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    const hint = document.getElementById('hint');
    if (hint) {
      hint.classList.add('err');
      hint.textContent = 'Microphone requires HTTPS or http://localhost:8000';
    }
    return;
  }

  try {
    // Get mic stream FIRST (before showing modal) — avoids permission flash
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    // Only open modal AFTER permission is granted
    openVoiceModal();
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
      closeVoiceModal();
      $('#hint').textContent = 'transcribing…';

      try {
        const wav = await toWav(new Blob(chunks, { type: chunks[0]?.type || 'audio/webm' }));
        const fd = new FormData();
        fd.append('audio', wav, 'question.wav');
        fd.append('generate', 'false');
        const d = await api('/voice', { method: 'POST', body: fd });

        if (!d.stt_ok || !d.transcript) {
          $('#hint').classList.add('err');
          $('#hint').textContent = `couldn't hear that — ${esc(d.stt_error || 'empty transcript')}`;
          return;
        }
        $('#q').value = d.transcript;
        $('#q').focus();
        $('#hint').textContent =
          `heard "${esc(d.transcript)}" · STT ${ms(d.stt_ms)} — press Enter or ➜ to submit`;
      } catch (e) {
        $('#hint').classList.add('err');
        $('#hint').textContent = e.message;
      }
    };

    recorder.start();
    btn.classList.add('rec');
    $('#hint').classList.remove('err');
    $('#hint').textContent = 'listening — tap mic to finish';
  } catch (e) {
    const vs = $('#voiceStatus'); if (vs) vs.textContent = 'Microphone access blocked';
    const sub = $('#voiceSubtext'); if (sub) sub.textContent = 'Please allow microphone access';
    $('#hint').classList.add('err');
    $('#hint').textContent = `microphone blocked — ${esc(e.message)}`;
  }
}

const micBtn = $('#micBtn');
if (micBtn) micBtn.onclick = startVoiceRecording;

const modalBtn = document.getElementById('voiceModalBtn');
if (modalBtn) modalBtn.onclick = stopVoiceRecording;

const closeVoiceBtn = document.getElementById('closeVoiceModal');
if (closeVoiceBtn) closeVoiceBtn.onclick = closeVoiceModal;

{
  const vm = document.getElementById('voiceModal');
  if (vm) {
    vm.onclick = (e) => {
      if (e.target === vm) closeVoiceModal();
    };
  }
}

window.startVoiceRecording = startVoiceRecording;
window.stopVoiceRecording = stopVoiceRecording;
window.closeVoiceModal = closeVoiceModal;

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