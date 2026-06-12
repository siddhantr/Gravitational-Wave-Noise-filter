# 〜 Gravitational Wave Noise Filter

A full-stack ML project for simulating, filtering, and understanding gravitational wave (GW) signals from LIGO-like detectors. Features a beautiful dark-mode web interface with real-time waveform visualization and a GW physics RAG chatbot.

---

## Features

- **🌊 Signal Simulation** — Post-Newtonian binary merger chirp signals with realistic noise models (aLIGO, white, pink, seismic)
- **📡 Classical Filters** — Bandpass (Butterworth), Whitening, Matched Filtering, Wiener Filter
- **🧠 ML Denoiser** — 1D Convolutional Autoencoder trained on synthetic GW data
- **🔭 GravBot** — RAG chatbot answering questions about GW physics using a curated knowledge base
- **📊 Real-time Plots** — Interactive waveform comparison and chirp frequency evolution (Chart.js)

---

## Project Structure

```
gw_noise_filter/
├── backend/
│   ├── signal_gen.py       # GW chirp generator + noise models
│   ├── filters.py          # Bandpass, whitening, matched, Wiener filters
│   ├── ml_denoiser.py      # Conv Autoencoder denoiser
│   ├── ingest.py           # Corpus ingestion + chunking
│   ├── embed.py            # FAISS embedding index
│   ├── retriever.py        # Semantic search
│   ├── chain.py            # RAG chat chain (Ollama / OpenAI)
│   └── api.py              # FastAPI server
├── data/
│   └── gw_corpus/          # GW knowledge base (6 topics)
├── frontend/
│   ├── index.html          # Web UI
│   ├── style.css           # Deep-space dark theme
│   └── app.js              # Chart.js + API integration
├── requirements.txt
└── .env.example
```

---

## Quickstart

### 1. Set up environment

```bash
python -m venv .venv
source .venv/bin/activate       # macOS/Linux
.venv\Scripts\activate          # Windows

pip install -r requirements.txt
```

### 2. Build the knowledge base

```bash
python backend/ingest.py     # Load & chunk GW corpus
python backend/embed.py      # Generate FAISS embeddings
```

### 3. Start the API server

```bash
python -m uvicorn backend.api:app --reload --host 0.0.0.0 --port 8000
```

### 4. Open the app

```
http://localhost:8000
```

---

## Chatbot Setup

**Option A — Ollama (free, local):**
```bash
# Install from https://ollama.ai then:
ollama serve
ollama pull llama3
```

**Option B — OpenAI:**
```bash
cp .env.example .env
# Edit .env: set LLM_PROVIDER=openai and add your OPENAI_API_KEY
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/generate` | POST | Generate noisy GW chirp signal |
| `/api/filter` | POST | Apply noise filter |
| `/api/chat` | POST | GravBot RAG chatbot |
| `/api/noise-types` | GET | Available noise models |
| `/api/filter-types` | GET | Available filter types |
| `/health` | GET | Server status |

---

## Knowledge Base Topics

1. **GW Physics** — Chirp signals, quadrupole formula, strain, detection
2. **LIGO Detectors** — Interferometry, Fabry-Pérot cavities, noise budget
3. **Signal Processing** — FFT, PSD, whitening, matched filtering, Wiener filter
4. **Noise Types** — Seismic, thermal, shot, quantum, glitches
5. **ML Denoising** — Autoencoders, CNNs, LSTMs, BayesWave, transfer learning
6. **GW Discoveries** — GW150914, GW170817, GWTC catalogs, multi-messenger astronomy

---

## License

MIT License — see LICENSE file.
