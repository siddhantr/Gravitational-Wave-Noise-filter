"""
filters.py — Classical signal processing filters for GW noise removal.
Implements: Bandpass, Whitening, Matched Filter, Wiener Filter.
"""

import numpy as np
from scipy import signal as scipy_signal
from typing import Optional


# ─── Bandpass Filter ──────────────────────────────────────────────────────────

def bandpass_filter(
    data: np.ndarray,
    sample_rate: int,
    f_low: float = 20.0,
    f_high: float = 1000.0,
    order: int = 4,
) -> np.ndarray:
    """
    Apply a zero-phase Butterworth bandpass filter to the data.

    Args:
        data: Input time series (strain).
        sample_rate: Sampling rate in Hz.
        f_low: Low-frequency cutoff (Hz).
        f_high: High-frequency cutoff (Hz).
        order: Filter order (higher = sharper rolloff).

    Returns:
        Filtered time series.
    """
    nyq = sample_rate / 2.0
    f_low  = max(f_low, 1.0)
    f_high = min(f_high, nyq * 0.99)

    sos = scipy_signal.butter(
        order,
        [f_low / nyq, f_high / nyq],
        btype='band',
        output='sos'
    )
    filtered = scipy_signal.sosfiltfilt(sos, data)
    return filtered


# ─── Whitening ────────────────────────────────────────────────────────────────

def estimate_psd(
    data: np.ndarray,
    sample_rate: int,
    nperseg: Optional[int] = None,
) -> tuple:
    """
    Estimate the noise Power Spectral Density using Welch's method.

    Returns:
        (freqs, psd): Frequency array and one-sided PSD in (strain)^2/Hz.
    """
    if nperseg is None:
        nperseg = min(len(data), sample_rate * 2)  # 2-second segments

    freqs, psd = scipy_signal.welch(
        data,
        fs=sample_rate,
        nperseg=nperseg,
        noverlap=nperseg // 2,
        window='hann',
    )
    return freqs, psd


def whiten(
    data: np.ndarray,
    sample_rate: int,
    psd: Optional[np.ndarray] = None,
    psd_freqs: Optional[np.ndarray] = None,
    f_low: float = 20.0,
) -> np.ndarray:
    """
    Whiten a GW data stream by dividing by the square root of the PSD.
    This normalizes the noise to unit variance across all frequencies.

    Args:
        data: Input strain time series.
        sample_rate: Sampling rate.
        psd: Optional pre-computed PSD (if None, estimated from data).
        psd_freqs: Frequencies corresponding to psd.
        f_low: Low frequency cutoff (below this, set to zero).

    Returns:
        Whitened time series with approximately unit-variance white noise.
    """
    N = len(data)
    dt = 1.0 / sample_rate

    # FFT of data
    data_f = np.fft.rfft(data)
    freqs = np.fft.rfftfreq(N, d=dt)

    # Estimate or interpolate PSD
    if psd is None:
        psd_freqs, psd = estimate_psd(data, sample_rate)

    # Interpolate PSD to match FFT frequencies
    psd_interp = np.interp(
        np.abs(freqs),
        psd_freqs,
        psd,
        left=psd[0],
        right=psd[-1]
    )
    psd_interp = np.maximum(psd_interp, 1e-60)  # avoid division by zero

    # Whiten: divide by sqrt(PSD) in frequency domain
    asd = np.sqrt(psd_interp)
    whitened_f = data_f / (asd * np.sqrt(sample_rate / 2))

    # Zero out below f_low
    whitened_f[np.abs(freqs) < f_low] = 0.0

    # IFFT back to time domain
    whitened = np.fft.irfft(whitened_f, n=N)

    # Taper to avoid edge ringing
    taper_len = min(int(0.1 * sample_rate), N // 10)
    if taper_len > 0:
        taper = np.hanning(2 * taper_len)
        whitened[:taper_len]  *= taper[:taper_len]
        whitened[-taper_len:] *= taper[taper_len:]

    return whitened


# ─── Matched Filter ───────────────────────────────────────────────────────────

def matched_filter(
    data: np.ndarray,
    template: np.ndarray,
    sample_rate: int,
    psd: Optional[np.ndarray] = None,
    psd_freqs: Optional[np.ndarray] = None,
) -> dict:
    """
    Compute the matched filter SNR time series.
    Optimal filter for detecting a known template in Gaussian noise.

    Args:
        data: Observed noisy strain data.
        template: Expected signal template (same length as data).
        sample_rate: Sampling rate in Hz.
        psd: Noise PSD (if None, estimated from data).
        psd_freqs: Frequencies for PSD.

    Returns:
        dict with 'snr', 'snr_max', 'snr_time', 'filtered_signal'.
    """
    N = len(data)
    dt = 1.0 / sample_rate
    df = 1.0 / (N * dt)

    # Estimate PSD if not provided
    if psd is None:
        psd_freqs, psd = estimate_psd(data, sample_rate)

    freqs = np.fft.rfftfreq(N, d=dt)

    # Interpolate PSD to FFT frequencies
    psd_interp = np.interp(
        np.abs(freqs),
        psd_freqs,
        psd,
        left=psd[0],
        right=psd[-1]
    )
    psd_interp = np.maximum(psd_interp, 1e-60)

    # FFT of data and template
    data_f     = np.fft.rfft(data)
    template_f = np.fft.rfft(template)

    # Matched filter in frequency domain: z(f) = d(f) * h*(f) / S_n(f)
    matched_f = data_f * np.conj(template_f) / psd_interp

    # Normalize: sigma² = 4 df * sum(|h(f)|²/S_n(f))
    sigma2 = 4 * df * np.sum(np.abs(template_f)**2 / psd_interp)
    sigma = np.sqrt(max(sigma2, 1e-60))

    # Inverse FFT to get SNR time series
    snr_complex = np.fft.irfft(matched_f, n=N) / sigma
    snr = np.abs(snr_complex)

    # Reconstruct filtered signal (matched filter output normalized)
    filtered = np.fft.irfft(matched_f, n=N).real

    return {
        "snr": snr.tolist(),
        "snr_max": float(np.max(snr)),
        "snr_time": float(np.argmax(snr) * dt),
        "filtered_signal": filtered.tolist(),
    }


# ─── Wiener Filter ────────────────────────────────────────────────────────────

def wiener_filter(
    data: np.ndarray,
    sample_rate: int,
    signal_psd: Optional[np.ndarray] = None,
    noise_psd: Optional[np.ndarray] = None,
    f_low: float = 20.0,
    f_high: float = 1500.0,
) -> np.ndarray:
    """
    Apply the Wiener filter for optimal noise reduction.
    H(f) = S_signal(f) / (S_signal(f) + S_noise(f))

    Args:
        data: Noisy strain time series.
        sample_rate: Sampling rate.
        signal_psd: Known signal PSD (approximated if None).
        noise_psd: Noise PSD (estimated from data if None).
        f_low: Low-frequency cutoff.
        f_high: High-frequency cutoff.

    Returns:
        Wiener-filtered time series.
    """
    N = len(data)
    dt = 1.0 / sample_rate
    freqs = np.fft.rfftfreq(N, d=dt)

    # Estimate noise PSD from data
    if noise_psd is None:
        psd_freqs, noise_psd_est = estimate_psd(data, sample_rate)
    else:
        psd_freqs, noise_psd_est = freqs, noise_psd

    # Interpolate noise PSD to FFT frequencies
    noise_psd_interp = np.interp(
        np.abs(freqs), psd_freqs, noise_psd_est,
        left=noise_psd_est[0], right=noise_psd_est[-1]
    )
    noise_psd_interp = np.maximum(noise_psd_interp, 1e-60)

    # Approximate signal PSD (flat in GW band, zero outside)
    if signal_psd is None:
        signal_psd_interp = np.where(
            (np.abs(freqs) >= f_low) & (np.abs(freqs) <= f_high),
            noise_psd_interp * 2.0,  # assume signal power similar to noise
            0.0
        )
    else:
        signal_psd_interp = np.interp(np.abs(freqs), psd_freqs, signal_psd)

    # Wiener transfer function: H(f) = Sss(f) / (Sss(f) + Snn(f))
    H_wiener = signal_psd_interp / (signal_psd_interp + noise_psd_interp + 1e-60)

    # Apply in frequency domain
    data_f = np.fft.rfft(data)
    filtered_f = data_f * H_wiener
    filtered = np.fft.irfft(filtered_f, n=N).real

    # Taper
    taper_len = min(int(0.05 * sample_rate), N // 20)
    if taper_len > 0:
        taper = np.hanning(2 * taper_len)
        filtered[:taper_len]  *= taper[:taper_len]
        filtered[-taper_len:] *= taper[taper_len:]

    return filtered


# ─── Compute SNR ──────────────────────────────────────────────────────────────

def compute_snr(signal: np.ndarray, noise: np.ndarray) -> float:
    """
    Compute simple time-domain SNR = RMS(signal) / RMS(noise).
    """
    rms_signal = np.sqrt(np.mean(signal**2))
    rms_noise  = np.sqrt(np.mean(noise**2))
    if rms_noise < 1e-60:
        return float('inf')
    return float(rms_signal / rms_noise)


def compute_snr_db(signal: np.ndarray, noise: np.ndarray) -> float:
    """
    Compute SNR in decibels.
    """
    snr_linear = compute_snr(signal, noise)
    if snr_linear <= 0:
        return float('-inf')
    return 20 * np.log10(snr_linear)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, __file__.rsplit("/", 2)[0])
    from backend.signal_gen import generate_noisy_signal

    print("Testing filters...")
    data = generate_noisy_signal(noise_level=1.0)
    h_noisy = np.array(data["h_noisy"])
    h_clean = np.array(data["h_clean"])
    sr = data["metadata"]["sample_rate"]

    # Bandpass
    bp = bandpass_filter(h_noisy, sr)
    print(f"Bandpass SNR: {compute_snr_db(h_clean, h_noisy - bp):.1f} dB")

    # Wiener
    w = wiener_filter(h_noisy, sr)
    print(f"Wiener SNR improvement: {compute_snr_db(h_clean, h_noisy - w):.1f} dB")
    print("Filters OK!")
