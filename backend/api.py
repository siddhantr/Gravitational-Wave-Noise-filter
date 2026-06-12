"""
api.py — FastAPI server exposing GW signal generation, filtering, and RAG chatbot.
"""

import sys
import os
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional

from backend.signal_gen import generate_noisy_signal
from backend.filters import (
    bandpass_filter,
    whiten,
    matched_filter,
    wiener_filter,
    compute_snr,
    compute_snr_db,
    estimate_psd,
)

# ─── App Setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Gravitational Wave Noise Filter API",
    description=(
        "Generate synthetic GW signals, apply DSP/ML noise filters, "
        "and ask physics questions via RAG chatbot."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# ─── Request / Response Models ────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    m1_msun:      float = Field(default=30.0, ge=1.0,  le=150.0, description="Primary mass (solar masses)")
    m2_msun:      float = Field(default=30.0, ge=1.0,  le=150.0, description="Secondary mass (solar masses)")
    distance_mpc: float = Field(default=400.0, ge=10.0, le=5000.0, description="Distance in Megaparsecs")
    noise_type:   str   = Field(default="aligo",  description="Noise model: white | aligo | pink | seismic")
    noise_level:  float = Field(default=1.0, ge=0.1, le=5.0, description="Noise level multiplier")
    sample_rate:  int   = Field(default=4096, description="Sample rate in Hz")
    duration:     float = Field(default=4.0, ge=1.0, le=16.0, description="Segment duration in seconds")


class FilterRequest(BaseModel):
    h_noisy:      list  = Field(description="Noisy strain time series")
    h_clean:      Optional[list] = Field(default=None, description="Clean signal for SNR computation")
    sample_rate:  int   = Field(default=4096)
    filter_type:  str   = Field(default="bandpass", description="bandpass | whitening | matched | wiener | ml")
    f_low:        float = Field(default=20.0, description="Low frequency cutoff (Hz)")
    f_high:       float = Field(default=1000.0, description="High frequency cutoff (Hz)")


class ChatRequest(BaseModel):
    question: str = Field(description="Physics question about gravitational waves")


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    """Serve the frontend."""
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "GW Noise Filter API", "docs": "/docs"}


@app.get("/health")
async def health():
    return {"status": "ok", "service": "GW Noise Filter API"}


@app.post("/api/generate")
async def generate_signal(req: GenerateRequest):
    """
    Generate a synthetic binary merger GW chirp signal with noise.
    Returns time series for noisy, clean, and noise-only data.
    """
    try:
        # Clamp m2 <= m1
        m1 = max(req.m1_msun, req.m2_msun)
        m2 = min(req.m1_msun, req.m2_msun)

        data = generate_noisy_signal(
            sample_rate=req.sample_rate,
            duration=req.duration,
            m1_msun=m1,
            m2_msun=m2,
            distance_mpc=req.distance_mpc,
            noise_type=req.noise_type,
            noise_level=req.noise_level,
        )

        # Downsample for frontend (max 2048 points for performance)
        max_pts = 2048
        N = len(data["times"])
        step = max(1, N // max_pts)

        return {
            "times":        data["times"][::step],
            "h_clean":      data["h_clean"][::step],
            "h_noisy":      data["h_noisy"][::step],
            "h_noise_only": data["h_noise_only"][::step],
            "frequency":    data["frequency"][::step],
            "metadata":     data["metadata"],
            "n_samples":    len(data["times"][::step]),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/filter")
async def apply_filter(req: FilterRequest):
    """
    Apply a noise filter to the given noisy strain data.
    Returns the filtered signal and SNR metrics.
    """
    try:
        h_noisy = np.array(req.h_noisy, dtype=float)
        h_clean = np.array(req.h_clean, dtype=float) if req.h_clean else None
        sr = req.sample_rate

        filtered = None
        snr_info = {}
        method_description = ""

        if req.filter_type == "bandpass":
            filtered = bandpass_filter(h_noisy, sr, req.f_low, req.f_high)
            method_description = f"Butterworth bandpass filter [{req.f_low}–{req.f_high} Hz], order 4"

        elif req.filter_type == "whitening":
            filtered = whiten(h_noisy, sr, f_low=req.f_low)
            method_description = "PSD whitening (divides by √PSD, flattens noise spectrum)"

        elif req.filter_type == "matched":
            # Use clean signal as template if available, else use bandpassed signal
            if h_clean is not None:
                template = h_clean
            else:
                template = bandpass_filter(h_noisy, sr, req.f_low, req.f_high)
            mf_result = matched_filter(h_noisy, template, sr)
            filtered  = np.array(mf_result["filtered_signal"])
            snr_info  = {
                "snr_max":  mf_result["snr_max"],
                "snr_time": mf_result["snr_time"],
            }
            method_description = f"Matched filter (optimal linear filter, max SNR = {mf_result['snr_max']:.2f})"

        elif req.filter_type == "wiener":
            filtered = wiener_filter(h_noisy, sr, f_low=req.f_low, f_high=req.f_high)
            method_description = "Wiener filter (optimal MMSE linear estimator)"

        elif req.filter_type == "ml":
            try:
                from backend.ml_denoiser import ml_denoise, load_or_train_model
                model = load_or_train_model()
                filtered = ml_denoise(h_noisy, model=model)
                method_description = "ML Conv Autoencoder denoiser"
            except Exception as ml_err:
                # Graceful fallback
                filtered = wiener_filter(h_noisy, sr, f_low=req.f_low, f_high=req.f_high)
                method_description = f"Wiener fallback (ML error: {ml_err})"

        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown filter type: {req.filter_type}. "
                       f"Choose from: bandpass, whitening, matched, wiener, ml"
            )

        # Compute SNR metrics
        if h_clean is not None and filtered is not None:
            residual = h_noisy - filtered
            snr_before = compute_snr_db(h_clean, h_noisy - h_clean)
            snr_after  = compute_snr_db(h_clean, h_noisy - filtered)
            snr_info.update({
                "snr_before_db": round(snr_before, 2),
                "snr_after_db":  round(snr_after, 2),
                "snr_improvement_db": round(snr_after - snr_before, 2),
            })

        return {
            "filtered": filtered.tolist() if filtered is not None else [],
            "filter_type": req.filter_type,
            "description": method_description,
            "snr": snr_info,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """
    GW Physics RAG chatbot — answers questions using the knowledge base.
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    try:
        from backend.chain import answer
        result = answer(req.question.strip())
        return result
    except FileNotFoundError as e:
        return {
            "answer": (
                f"⚠️ Knowledge base not indexed yet.\n\n"
                f"Please run:\n"
                f"  python backend/ingest.py\n"
                f"  python backend/embed.py\n\n"
                f"Error: {e}"
            ),
            "sources": [],
            "chunks_used": 0,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/noise-types")
async def get_noise_types():
    return {
        "types": [
            {"id": "aligo",   "label": "aLIGO",    "description": "Advanced LIGO O3 noise curve (realistic)"},
            {"id": "white",   "label": "White",     "description": "Flat spectrum (equal noise at all frequencies)"},
            {"id": "pink",    "label": "Pink (1/f)", "description": "Pink noise — PSD ∝ 1/f"},
            {"id": "seismic", "label": "Seismic",   "description": "Seismic-dominated noise (PSD ∝ f⁻⁸ below 20 Hz)"},
        ]
    }


@app.get("/api/filter-types")
async def get_filter_types():
    return {
        "types": [
            {"id": "bandpass",  "label": "Bandpass",   "description": "Butterworth bandpass — isolates GW frequency band"},
            {"id": "whitening", "label": "Whitening",  "description": "Spectral whitening — flattens noise PSD"},
            {"id": "matched",   "label": "Matched Filter", "description": "Optimal SNR filter — cross-correlate with template"},
            {"id": "wiener",    "label": "Wiener",     "description": "Optimal MMSE linear estimator"},
            {"id": "ml",        "label": "ML Denoiser", "description": "Deep learning Conv Autoencoder"},
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.api:app", host="0.0.0.0", port=8000, reload=True)
