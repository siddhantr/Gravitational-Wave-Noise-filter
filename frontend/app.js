/* ═══════════════════════════════════════════════════════════════════════════
   GravWave — app.js
   Handles: starfield, waveform charts, API calls, chatbot UI
════════════════════════════════════════════════════════════════════════════ */

const API_BASE = 'http://localhost:8000';

// ── State ───────────────────────────────────────────────────────────────────
const state = {
  currentData:    null,   // last generated signal data
  filteredData:   null,   // last filtered signal
  activeFilter:   'none',
  charts: {
    waveform:   null,
    frequency:  null,
  },
};

// ── Filter descriptions ──────────────────────────────────────────────────────
const FILTER_DESC = {
  none:      'No filter applied — showing raw noisy signal.',
  bandpass:  'Butterworth bandpass filter isolates the GW frequency band (20–2000 Hz) by attenuating low-frequency seismic noise and high-frequency shot noise.',
  whitening: 'Spectral whitening divides the signal by its noise ASD, normalizing noise power across all frequencies. Essential before statistical analysis.',
  matched:   '✦ Matched filtering cross-correlates data with the expected signal template — the optimal linear filter for known-waveform detection (used in LIGO pipelines).',
  wiener:    'Wiener filtering computes the optimal minimum-mean-squared-error linear estimate of the clean signal given the noisy observation and estimated noise PSD.',
  ml:        '🧠 Convolutional Autoencoder denoiser — a trained deep learning model that learns to reconstruct clean GW signals from noisy data without explicit noise models.',
};

// ── DOM refs ────────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);

const sliders = {
  m1:          { el: $('m1'),          val: $('m1-val'),          fmt: v => v },
  m2:          { el: $('m2'),          val: $('m2-val'),          fmt: v => v },
  distance:    { el: $('distance'),    val: $('distance-val'),    fmt: v => v },
  noiseLevel:  { el: $('noise-level'), val: $('noise-level-val'), fmt: v => `${v}×` },
  fLow:        { el: $('f-low'),       val: $('f-low-val'),       fmt: v => v },
  fHigh:       { el: $('f-high'),      val: $('f-high-val'),      fmt: v => v },
};

// ── Starfield ────────────────────────────────────────────────────────────────
(function initStarfield() {
  const canvas = $('starfield');
  const ctx = canvas.getContext('2d');
  let stars = [];
  let W, H;

  function resize() {
    W = canvas.width  = window.innerWidth;
    H = canvas.height = window.innerHeight;
    stars = Array.from({ length: 200 }, () => ({
      x: Math.random() * W,
      y: Math.random() * H,
      r: Math.random() * 1.2 + 0.2,
      a: Math.random(),
      speed: Math.random() * 0.5 + 0.1,
    }));
  }

  function draw() {
    ctx.clearRect(0, 0, W, H);
    stars.forEach(s => {
      s.a += s.speed * 0.005;
      const alpha = (Math.sin(s.a) + 1) / 2 * 0.7 + 0.1;
      ctx.beginPath();
      ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(180, 200, 255, ${alpha})`;
      ctx.fill();
    });
    requestAnimationFrame(draw);
  }

  resize();
  window.addEventListener('resize', resize);
  draw();
})();


// ── Navigation ───────────────────────────────────────────────────────────────
document.querySelectorAll('.nav-link').forEach(link => {
  link.addEventListener('click', e => {
    e.preventDefault();
    const target = link.getAttribute('href').slice(1);
    document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
    document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
    document.getElementById(target).classList.add('active');
    link.classList.add('active');
  });
});

document.querySelectorAll('.sq-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const q = btn.dataset.q;
    // Switch to chatbot view and send question
    document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
    document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
    $('chatbot').classList.add('active');
    $('nav-chatbot').classList.add('active');
    $('chat-input').value = q;
    sendChatMessage(q);
  });
});


// ── API Health Check ─────────────────────────────────────────────────────────
async function checkHealth() {
  try {
    const resp = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(4000) });
    if (resp.ok) {
      $('api-status-dot').className   = 'status-dot ok';
      $('api-status-label').textContent = 'API Online';
      return true;
    }
  } catch {
    /* ignore */
  }
  $('api-status-dot').className     = 'status-dot err';
  $('api-status-label').textContent = 'API Offline';
  return false;
}

checkHealth();
setInterval(checkHealth, 30000);


// ── Slider bindings ──────────────────────────────────────────────────────────
Object.values(sliders).forEach(({ el, val, fmt }) => {
  if (!el) return;
  el.addEventListener('input', () => { val.textContent = fmt(el.value); });
  val.textContent = fmt(el.value);
});


// ── Chart initialisation ─────────────────────────────────────────────────────
function initCharts() {
  const baseOpts = {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 600, easing: 'easeInOutQuart' },
    interaction: { mode: 'index', intersect: false },
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: 'rgba(6, 12, 30, 0.95)',
        borderColor: 'rgba(100, 130, 220, 0.4)',
        borderWidth: 1,
        titleColor: '#7b8fbf',
        bodyColor: '#e8eeff',
        titleFont: { family: 'JetBrains Mono', size: 11 },
        bodyFont:  { family: 'JetBrains Mono', size: 12 },
        padding: 10,
      },
      zoom: {
        zoom:  { wheel: { enabled: true }, pinch: { enabled: true }, mode: 'x' },
        pan:   { enabled: true, mode: 'x' },
      },
    },
    scales: {
      x: {
        type: 'linear',
        grid:  { color: 'rgba(100, 130, 220, 0.07)', drawBorder: false },
        ticks: { color: '#4a5580', font: { family: 'JetBrains Mono', size: 10 }, maxTicksLimit: 8 },
        title: { display: true, text: 'Time (s)', color: '#7b8fbf', font: { size: 11 } },
      },
      y: {
        grid:  { color: 'rgba(100, 130, 220, 0.07)', drawBorder: false },
        ticks: { color: '#4a5580', font: { family: 'JetBrains Mono', size: 10 }, maxTicksLimit: 6 },
        title: { display: true, text: 'Strain (×10⁻²¹)', color: '#7b8fbf', font: { size: 11 } },
      },
    },
  };

  // Waveform chart
  state.charts.waveform = new Chart($('waveform-chart'), {
    type: 'line',
    data: {
      datasets: [
        {
          label: 'Noisy',
          data: [],
          borderColor: '#00e5ff',
          borderWidth: 1,
          pointRadius: 0,
          tension: 0,
          fill: false,
        },
        {
          label: 'Clean / Filtered',
          data: [],
          borderColor: '#b388ff',
          borderWidth: 1.5,
          pointRadius: 0,
          tension: 0,
          fill: false,
        },
      ],
    },
    options: { ...baseOpts },
  });

  // Frequency chart
  state.charts.frequency = new Chart($('frequency-chart'), {
    type: 'line',
    data: {
      datasets: [
        {
          label: 'GW Frequency',
          data: [],
          borderColor: '#69ff47',
          backgroundColor: 'rgba(105, 255, 71, 0.08)',
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.4,
          fill: true,
        },
      ],
    },
    options: {
      ...baseOpts,
      scales: {
        ...baseOpts.scales,
        y: {
          ...baseOpts.scales.y,
          title: { display: true, text: 'Frequency (Hz)', color: '#7b8fbf', font: { size: 11 } },
        },
      },
    },
  });
}

initCharts();


// ── Update charts ────────────────────────────────────────────────────────────
function toXY(times, values) {
  return times.map((t, i) => ({ x: +t.toFixed(5), y: values[i] }));
}

function updateWaveformChart(times, noisy, cleanOrFiltered, label = 'Clean / Filtered') {
  const chart = state.charts.waveform;
  chart.data.datasets[0].data = toXY(times, noisy);
  chart.data.datasets[1].data = toXY(times, cleanOrFiltered);
  chart.data.datasets[1].label = label;
  chart.update();
}

function updateFrequencyChart(times, freq) {
  const chart = state.charts.frequency;
  // Only show non-zero frequency values (during chirp)
  const pts = times.map((t, i) => ({ x: +t.toFixed(5), y: freq[i] > 0 ? freq[i] : null }));
  chart.data.datasets[0].data = pts;
  chart.update();
}


// ── Generate Signal ──────────────────────────────────────────────────────────
async function generateSignal() {
  const btn   = $('generate-btn');
  const loader = $('btn-loader');

  btn.disabled = true;
  loader.classList.add('visible');

  const params = {
    m1_msun:      parseFloat($('m1').value),
    m2_msun:      parseFloat($('m2').value),
    distance_mpc: parseFloat($('distance').value),
    noise_type:   $('noise-type').value,
    noise_level:  parseFloat($('noise-level').value),
    sample_rate:  4096,
    duration:     4.0,
  };

  try {
    const resp = await fetch(`${API_BASE}/api/generate`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(params),
    });

    if (!resp.ok) {
      const err = await resp.json();
      throw new Error(err.detail || `HTTP ${resp.status}`);
    }

    const data = await resp.json();
    state.currentData  = data;
    state.filteredData = null;

    // Update stats
    $('stat-chirp').textContent   = data.metadata.chirp_mass_msun.toFixed(1);
    $('stat-samples').textContent = data.n_samples.toLocaleString();

    // Initial render: noisy + clean signal
    updateWaveformChart(data.times, data.h_noisy, data.h_clean, 'Clean GW Signal');
    updateFrequencyChart(data.times, data.frequency);

    // If a filter is active, apply it
    if (state.activeFilter !== 'none') {
      await applyFilter(state.activeFilter);
    } else {
      $('stat-snr').textContent = '—';
      $('snr-panel').style.display = 'none';
    }

  } catch (err) {
    console.error('Generate error:', err);
    showToast(`Error: ${err.message}`, 'error');
  } finally {
    btn.disabled = false;
    loader.classList.remove('visible');
  }
}

$('generate-btn').addEventListener('click', generateSignal);


// ── Apply Filter ─────────────────────────────────────────────────────────────
async function applyFilter(filterType) {
  if (!state.currentData) {
    showToast('Generate a signal first!', 'warn');
    return;
  }

  if (filterType === 'none') {
    updateWaveformChart(
      state.currentData.times,
      state.currentData.h_noisy,
      state.currentData.h_clean,
      'Clean GW Signal'
    );
    $('snr-panel').style.display = 'none';
    $('stat-snr').textContent = '—';
    return;
  }

  const payload = {
    h_noisy:     state.currentData.h_noisy,
    h_clean:     state.currentData.h_clean,
    sample_rate: state.currentData.metadata.sample_rate,
    filter_type: filterType,
    f_low:       parseFloat($('f-low').value),
    f_high:      parseFloat($('f-high').value),
  };

  try {
    const resp = await fetch(`${API_BASE}/api/filter`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(payload),
    });

    if (!resp.ok) {
      const err = await resp.json();
      throw new Error(err.detail || `HTTP ${resp.status}`);
    }

    const result = await resp.json();
    state.filteredData = result;

    updateWaveformChart(
      state.currentData.times,
      state.currentData.h_noisy,
      result.filtered,
      `Filtered (${result.filter_type})`
    );

    // Update SNR panel
    if (result.snr && Object.keys(result.snr).length > 0) {
      $('snr-panel').style.display = 'block';
      $('snr-before').textContent     = result.snr.snr_before_db != null ? `${result.snr.snr_before_db} dB` : '—';
      $('snr-after').textContent      = result.snr.snr_after_db  != null ? `${result.snr.snr_after_db} dB`  : '—';
      $('snr-improvement').textContent = result.snr.snr_improvement_db != null
        ? `+${result.snr.snr_improvement_db} dB` : '—';
      $('snr-method').textContent = result.description || filterType;

      if (result.snr.snr_after_db != null) {
        $('stat-snr').textContent = result.snr.snr_after_db;
      }
      if (result.snr.snr_max != null) {
        $('stat-snr').textContent = result.snr.snr_max.toFixed(1);
      }
    } else {
      $('snr-panel').style.display = 'none';
    }

  } catch (err) {
    console.error('Filter error:', err);
    showToast(`Filter error: ${err.message}`, 'error');
  }
}


// ── Filter button bindings ───────────────────────────────────────────────────
document.querySelectorAll('.filter-btn').forEach(btn => {
  btn.addEventListener('click', async () => {
    const filterType = btn.dataset.filter;
    state.activeFilter = filterType;

    // Update UI
    document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    $('filter-desc').textContent = FILTER_DESC[filterType] || '';

    // Show/hide frequency range controls for bandpass/wiener
    const freqRange = document.querySelector('.freq-range');
    if (['bandpass', 'wiener', 'matched'].includes(filterType)) {
      freqRange.classList.add('visible');
    } else {
      freqRange.classList.remove('visible');
    }

    await applyFilter(filterType);
  });
});


// ── Chatbot ───────────────────────────────────────────────────────────────────
const chatMessages = $('chat-messages');
const chatForm     = $('chat-form');
const chatInput    = $('chat-input');
const sendBtn      = $('chat-send-btn');

function appendMessage(role, content, sources = []) {
  // Remove welcome message on first message
  const welcome = chatMessages.querySelector('.chat-welcome');
  if (welcome) welcome.remove();

  const msg = document.createElement('div');
  msg.className = `chat-msg ${role}`;

  const avatar = document.createElement('div');
  avatar.className = `msg-avatar ${role === 'user' ? 'user-avatar' : 'bot-avatar'}`;
  avatar.textContent = role === 'user' ? 'U' : '🔭';

  const bubble = document.createElement('div');
  bubble.className = 'msg-bubble';

  // Format markdown-ish content
  const formatted = content
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.*?)\*/g,     '<em>$1</em>')
    .replace(/`([^`]+)`/g,     '<code style="font-family:JetBrains Mono;font-size:0.85em;color:#00e5ff;background:rgba(0,229,255,0.08);padding:1px 5px;border-radius:4px;">$1</code>')
    .replace(/\n/g, '<br>');

  bubble.innerHTML = formatted;

  if (sources.length > 0) {
    const srcDiv = document.createElement('div');
    srcDiv.className = 'msg-sources';
    sources.forEach(s => {
      const tag = document.createElement('span');
      tag.className = 'source-tag';
      tag.textContent = '📚 ' + s;
      srcDiv.appendChild(tag);
    });
    bubble.appendChild(srcDiv);
  }

  msg.appendChild(avatar);
  msg.appendChild(bubble);
  chatMessages.appendChild(msg);
  chatMessages.scrollTop = chatMessages.scrollHeight;
  return msg;
}

function appendTypingIndicator() {
  const msg = document.createElement('div');
  msg.className = 'chat-msg bot typing-indicator';
  msg.id = 'typing-indicator';

  const avatar = document.createElement('div');
  avatar.className = 'msg-avatar bot-avatar';
  avatar.textContent = '🔭';

  const bubble = document.createElement('div');
  bubble.className = 'msg-bubble';
  bubble.innerHTML = '<div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div>';

  msg.appendChild(avatar);
  msg.appendChild(bubble);
  chatMessages.appendChild(msg);
  chatMessages.scrollTop = chatMessages.scrollHeight;
  return msg;
}

async function sendChatMessage(question) {
  if (!question.trim()) return;

  appendMessage('user', question);
  chatInput.value = '';
  sendBtn.disabled = true;

  const typingEl = appendTypingIndicator();

  try {
    const resp = await fetch(`${API_BASE}/api/chat`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ question }),
    });

    typingEl.remove();

    if (!resp.ok) {
      const err = await resp.json();
      appendMessage('bot', `⚠️ ${err.detail || 'Server error'}`);
      return;
    }

    const result = await resp.json();
    appendMessage('bot', result.answer, result.sources || []);

  } catch (err) {
    typingEl.remove();
    appendMessage('bot',
      `⚠️ Could not connect to the API.\n\nMake sure the server is running:\n\`\`\`\npython -m uvicorn backend.api:app --reload\n\`\`\``
    );
  } finally {
    sendBtn.disabled = false;
  }
}

chatForm.addEventListener('submit', e => {
  e.preventDefault();
  sendChatMessage(chatInput.value.trim());
});


// ── Toast notifications ───────────────────────────────────────────────────────
function showToast(message, type = 'info') {
  const toast = document.createElement('div');
  toast.style.cssText = `
    position: fixed;
    bottom: 2rem;
    right: 2rem;
    z-index: 9999;
    background: ${type === 'error' ? 'rgba(255, 82, 82, 0.15)' : type === 'warn' ? 'rgba(255, 215, 64, 0.15)' : 'rgba(0, 229, 255, 0.15)'};
    border: 1px solid ${type === 'error' ? 'rgba(255, 82, 82, 0.5)' : type === 'warn' ? 'rgba(255, 215, 64, 0.5)' : 'rgba(0, 229, 255, 0.5)'};
    color: ${type === 'error' ? '#ff5252' : type === 'warn' ? '#ffd740' : '#00e5ff'};
    padding: 0.75rem 1.25rem;
    border-radius: 10px;
    font-size: 0.85rem;
    font-family: 'JetBrains Mono', monospace;
    backdrop-filter: blur(20px);
    animation: fadeInUp 0.3s ease;
    max-width: 360px;
  `;
  toast.textContent = message;
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 4000);
}


// ── Auto-generate on page load ────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', async () => {
  const alive = await checkHealth();
  if (alive) {
    await generateSignal();
  }
});
