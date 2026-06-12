"""
ml_denoiser.py — Convolutional Autoencoder for GW signal denoising.
Falls back gracefully to scipy Wiener filter if PyTorch is unavailable.
"""

import numpy as np
import os
import pickle
from pathlib import Path

MODEL_PATH = Path(__file__).parent.parent / "data" / "models" / "gw_denoiser.pkl"

# ─── Check PyTorch availability ───────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ─── Conv1D Autoencoder Architecture ─────────────────────────────────────────

if TORCH_AVAILABLE:
    class GWDenoiserAE(nn.Module):
        """
        1D Convolutional Autoencoder for gravitational wave denoising.

        Encoder: progressively compress temporal signal
        Decoder: reconstruct clean signal from compressed representation
        Skip connections: preserve fine-grained GW features
        """

        def __init__(self, in_channels: int = 1, base_channels: int = 32):
            super().__init__()

            # ─ Encoder ─
            self.enc1 = nn.Sequential(
                nn.Conv1d(in_channels, base_channels, kernel_size=7, padding=3),
                nn.BatchNorm1d(base_channels),
                nn.ReLU(inplace=True),
                nn.Conv1d(base_channels, base_channels, kernel_size=7, padding=3),
                nn.BatchNorm1d(base_channels),
                nn.ReLU(inplace=True),
            )
            self.pool1 = nn.MaxPool1d(2, return_indices=True)

            self.enc2 = nn.Sequential(
                nn.Conv1d(base_channels, base_channels * 2, kernel_size=5, padding=2),
                nn.BatchNorm1d(base_channels * 2),
                nn.ReLU(inplace=True),
                nn.Conv1d(base_channels * 2, base_channels * 2, kernel_size=5, padding=2),
                nn.BatchNorm1d(base_channels * 2),
                nn.ReLU(inplace=True),
            )
            self.pool2 = nn.MaxPool1d(2, return_indices=True)

            self.enc3 = nn.Sequential(
                nn.Conv1d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1),
                nn.BatchNorm1d(base_channels * 4),
                nn.ReLU(inplace=True),
                nn.Conv1d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1),
                nn.BatchNorm1d(base_channels * 4),
                nn.ReLU(inplace=True),
            )
            self.pool3 = nn.MaxPool1d(2, return_indices=True)

            # ─ Bottleneck (dilated convolutions for larger receptive field) ─
            self.bottleneck = nn.Sequential(
                nn.Conv1d(base_channels * 4, base_channels * 4, kernel_size=3, padding=2, dilation=2),
                nn.BatchNorm1d(base_channels * 4),
                nn.ReLU(inplace=True),
                nn.Conv1d(base_channels * 4, base_channels * 4, kernel_size=3, padding=4, dilation=4),
                nn.BatchNorm1d(base_channels * 4),
                nn.ReLU(inplace=True),
            )

            # ─ Decoder ─
            self.unpool3 = nn.MaxUnpool1d(2)
            self.dec3 = nn.Sequential(
                nn.Conv1d(base_channels * 8, base_channels * 2, kernel_size=3, padding=1),
                nn.BatchNorm1d(base_channels * 2),
                nn.ReLU(inplace=True),
            )

            self.unpool2 = nn.MaxUnpool1d(2)
            self.dec2 = nn.Sequential(
                nn.Conv1d(base_channels * 4, base_channels, kernel_size=5, padding=2),
                nn.BatchNorm1d(base_channels),
                nn.ReLU(inplace=True),
            )

            self.unpool1 = nn.MaxUnpool1d(2)
            self.dec1 = nn.Sequential(
                nn.Conv1d(base_channels * 2, base_channels, kernel_size=7, padding=3),
                nn.BatchNorm1d(base_channels),
                nn.ReLU(inplace=True),
                nn.Conv1d(base_channels, in_channels, kernel_size=7, padding=3),
            )

        def forward(self, x):
            # Encode
            e1 = self.enc1(x)
            p1, idx1 = self.pool1(e1)

            e2 = self.enc2(p1)
            p2, idx2 = self.pool2(e2)

            e3 = self.enc3(p2)
            p3, idx3 = self.pool3(e3)

            # Bottleneck
            b = self.bottleneck(p3)

            # Decode with skip connections
            u3 = self.unpool3(b, idx3, output_size=e3.size())
            d3 = self.dec3(torch.cat([u3, e3], dim=1))

            u2 = self.unpool2(d3, idx2, output_size=e2.size())
            d2 = self.dec2(torch.cat([u2, e2], dim=1))

            u1 = self.unpool1(d2, idx1, output_size=e1.size())
            out = self.dec1(torch.cat([u1, e1], dim=1))

            return out


# ─── Training ─────────────────────────────────────────────────────────────────

def train_denoiser(
    n_samples: int = 2000,
    sample_rate: int = 4096,
    duration: float = 1.0,
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
) -> "GWDenoiserAE":
    """
    Train the autoencoder on synthetic GW + noise data.
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required. Install with: pip install torch")

    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from backend.signal_gen import generate_noisy_signal

    print(f"Generating {n_samples} training samples...")
    X_noisy, X_clean = [], []

    for i in range(n_samples):
        m1 = np.random.uniform(5, 80)
        m2 = np.random.uniform(5, m1)
        dist = np.random.uniform(100, 2000)
        noise_level = np.random.uniform(0.5, 2.0)

        try:
            data = generate_noisy_signal(
                sample_rate=sample_rate, duration=duration,
                m1_msun=m1, m2_msun=m2, distance_mpc=dist,
                noise_level=noise_level
            )
            noisy = np.array(data["h_noisy"])
            clean = np.array(data["h_clean"])

            X_noisy.append(noisy)
            X_clean.append(clean)
        except Exception:
            continue

        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{n_samples} samples generated")

    X_noisy = np.array(X_noisy, dtype=np.float32)
    X_clean = np.array(X_clean, dtype=np.float32)

    # Normalize
    scale = np.std(X_noisy, axis=1, keepdims=True) + 1e-30
    X_noisy_norm = X_noisy / scale
    X_clean_norm = X_clean / scale

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    model = GWDenoiserAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    X_noisy_t = torch.tensor(X_noisy_norm[:, np.newaxis, :]).to(device)
    X_clean_t = torch.tensor(X_clean_norm[:, np.newaxis, :]).to(device)

    dataset = torch.utils.data.TensorDataset(X_noisy_t, X_clean_t)
    loader  = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)
        scheduler.step()
        avg_loss = total_loss / len(dataset)
        print(f"  Epoch {epoch+1:3d}/{epochs}: loss = {avg_loss:.6f}")

    return model


# ─── Inference ────────────────────────────────────────────────────────────────

def ml_denoise(
    data: np.ndarray,
    model=None,
    chunk_size: int = 4096,
) -> np.ndarray:
    """
    Denoise a GW time series using the trained autoencoder.
    Processes in overlapping chunks if data is longer than chunk_size.

    Falls back to scipy Wiener filter if PyTorch is unavailable.
    """
    if not TORCH_AVAILABLE:
        from scipy.signal import wiener
        print("PyTorch not available; using scipy Wiener filter as fallback.")
        return wiener(data, mysize=21)

    if model is None:
        model = load_or_train_model()

    device = next(model.parameters()).device
    model.eval()

    N = len(data)
    if N <= chunk_size:
        # Single pass
        scale = np.std(data) + 1e-30
        x = torch.tensor(data[np.newaxis, np.newaxis, :] / scale, dtype=torch.float32).to(device)
        with torch.no_grad():
            out = model(x).squeeze().cpu().numpy()
        return out * scale
    else:
        # Overlap-add for longer signals
        hop = chunk_size // 2
        output = np.zeros(N)
        weight = np.zeros(N)
        window = np.hanning(chunk_size)

        for start in range(0, N - chunk_size + 1, hop):
            chunk = data[start:start + chunk_size]
            scale = np.std(chunk) + 1e-30
            x = torch.tensor(chunk[np.newaxis, np.newaxis, :] / scale, dtype=torch.float32).to(device)
            with torch.no_grad():
                out = model(x).squeeze().cpu().numpy()
            output[start:start + chunk_size] += out * scale * window
            weight[start:start + chunk_size] += window

        # Normalize by overlap window sum
        weight = np.maximum(weight, 1e-10)
        return output / weight


def load_or_train_model(force_retrain: bool = False):
    """
    Load the saved model, or train from scratch if not found.
    """
    if not TORCH_AVAILABLE:
        return None

    if MODEL_PATH.exists() and not force_retrain:
        print(f"Loading model from {MODEL_PATH}")
        with open(MODEL_PATH, "rb") as f:
            state_dict = pickle.load(f)
        model = GWDenoiserAE()
        model.load_state_dict(state_dict)
        model.eval()
        return model
    else:
        print("Training new denoiser model (this takes a few minutes)...")
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        model = train_denoiser(n_samples=1000, duration=1.0, epochs=20)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model.state_dict(), f)
        print(f"Model saved to {MODEL_PATH}")
        model.eval()
        return model


if __name__ == "__main__":
    print("ML Denoiser module loaded.")
    print(f"PyTorch available: {TORCH_AVAILABLE}")
    if TORCH_AVAILABLE:
        model = GWDenoiserAE()
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {total_params:,}")
