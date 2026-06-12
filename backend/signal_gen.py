"""
signal_gen.py — Generate synthetic gravitational wave chirp signals and realistic noise.
"""

import numpy as np
from scipy import signal


# ─── Constants ────────────────────────────────────────────────────────────────
G  = 6.674e-11   # m³/(kg·s²)
c  = 3e8         # m/s
M_sun = 1.989e30 # kg
PC   = 3.086e16  # metres per parsec


def solar_masses_to_kg(m: float) -> float:
    return m * M_sun


# ─── Chirp Signal Generator ───────────────────────────────────────────────────

def generate_chirp(
    sample_rate: int = 4096,
    duration: float = 4.0,
    m1_msun: float = 30.0,
    m2_msun: float = 30.0,
    distance_mpc: float = 400.0,
    f_lower: float = 20.0,
    t_merger_offset: float = 0.5,  # seconds before end of segment
) -> dict:
    """
    Generate a post-Newtonian binary merger GW chirp signal.

    Args:
        sample_rate: Samples per second (Hz).
        duration: Total segment duration in seconds.
        m1_msun: Primary mass in solar masses.
        m2_msun: Secondary mass in solar masses.
        distance_mpc: Luminosity distance in Megaparsecs.
        f_lower: Starting frequency in Hz.
        t_merger_offset: Time before end of segment where merger occurs.

    Returns:
        dict with 'times', 'h_plus', 'frequency', 'metadata'.
    """
    m1 = solar_masses_to_kg(m1_msun)
    m2 = solar_masses_to_kg(m2_msun)
    M  = m1 + m2
    mu = m1 * m2 / M           # reduced mass
    eta = (m1 * m2) / M**2     # symmetric mass ratio (0 < eta <= 0.25)
    Mc = M * eta**(3/5)         # chirp mass

    r = distance_mpc * 1e6 * PC  # convert Mpc to metres

    # ── Chirp time from f_lower to merger ──
    # t_chirp = (5/256) * (G*Mc/c^3)^(-5/3) * (pi*f_lower)^(-8/3)
    Mc_geom = G * Mc / c**3     # chirp mass in geometric units (seconds)
    t_chirp = (5 / 256) * Mc_geom**(-5/3) * (np.pi * f_lower)**(-8/3)
    t_chirp = min(t_chirp, duration - t_merger_offset - 0.1)

    # ── Time array ──
    dt = 1.0 / sample_rate
    times = np.arange(0, duration, dt)
    N = len(times)

    # Merger time in the array
    t_merger = duration - t_merger_offset
    t_start  = t_merger - t_chirp

    h_plus = np.zeros(N)

    # ── Compute GW strain using 0PN approximation ──
    for i, t in enumerate(times):
        tau = t - t_start
        if tau < 0 or tau >= t_chirp:
            continue

        tau_to_merger = t_chirp - tau
        if tau_to_merger <= 0:
            break

        # Orbital frequency (0PN)
        f_orb = (1 / (8 * np.pi)) * (tau_to_merger / (5 * Mc_geom))**(-3/8) * Mc_geom**(-1)
        f_gw  = 2 * f_orb
        if f_gw < f_lower or f_gw > 2000:
            continue

        # GW amplitude (leading order)
        A = (4 / r) * (G * Mc / c**2) * (np.pi * G * Mc * f_gw / c**3)**(2/3)

        # Phase (0PN)
        phi = -2 * (tau_to_merger / (5 * Mc_geom))**(5/8)

        h_plus[i] = A * np.cos(phi)

    # Taper edges to avoid edge effects
    taper_len = min(int(0.1 * sample_rate), N // 10)
    taper = np.hanning(2 * taper_len)
    h_plus[:taper_len]  *= taper[:taper_len]
    h_plus[-taper_len:] *= taper[taper_len:]

    # Instantaneous frequency track
    freq_track = np.zeros(N)
    for i, t in enumerate(times):
        tau = t - t_start
        tau_to_merger = t_chirp - tau
        if tau_to_merger > 0 and 0 <= tau < t_chirp:
            f_orb = (1 / (8 * np.pi)) * (tau_to_merger / (5 * Mc_geom))**(-3/8) * Mc_geom**(-1)
            freq_track[i] = 2 * f_orb

    return {
        "times": times.tolist(),
        "h_plus": h_plus.tolist(),
        "frequency": freq_track.tolist(),
        "metadata": {
            "sample_rate": sample_rate,
            "duration": duration,
            "m1_msun": m1_msun,
            "m2_msun": m2_msun,
            "distance_mpc": distance_mpc,
            "chirp_mass_msun": float(Mc / M_sun),
            "t_merger": t_merger,
            "f_lower": f_lower,
        }
    }


# ─── Noise Generators ─────────────────────────────────────────────────────────

def aLIGO_psd(freqs: np.ndarray) -> np.ndarray:
    """
    Approximate Advanced LIGO O3 sensitivity (noise PSD).
    Returns S_n(f) in units of (strain)^2 / Hz.
    Based on analytic fit to the design sensitivity curve.
    """
    f = np.asarray(freqs, dtype=float)
    S = np.zeros_like(f)

    mask = f > 0
    f0 = 215.0   # characteristic frequency

    # Simplified 4-parameter analytic fit
    seismic = (f0 / f[mask])**20 * 1e-47        # steep seismic wall
    thermal  = (f0 / f[mask])**4.0 * 5e-49      # thermal noise
    shot     = (f[mask] / f0)**2.0 * 3e-49      # shot noise
    S[mask]  = seismic + thermal + shot + 3e-49  # floor

    return S


def generate_noise(
    sample_rate: int = 4096,
    duration: float = 4.0,
    noise_type: str = "aligo",   # "white", "aligo", "pink", "seismic"
    noise_level: float = 1.0,
) -> np.ndarray:
    """
    Generate noise matching specified spectral characteristics.

    Args:
        noise_type: "white" | "aligo" | "pink" | "seismic"
        noise_level: Scalar multiplier (1.0 = nominal)
    """
    N = int(sample_rate * duration)
    dt = 1.0 / sample_rate
    freqs = np.fft.rfftfreq(N, d=dt)

    # White noise base
    white = np.random.randn(N) + 1j * np.random.randn(N)
    white_f = np.fft.rfft(white.real)

    if noise_type == "white":
        noise_f = white_f

    elif noise_type == "aligo":
        psd = aLIGO_psd(np.where(freqs == 0, 1e-10, freqs))
        psd[0] = psd[1]
        noise_f = white_f * np.sqrt(psd * sample_rate / 2)

    elif noise_type == "pink":
        # Pink noise: PSD ~ 1/f
        with np.errstate(divide='ignore', invalid='ignore'):
            pink_psd = np.where(freqs > 0, 1.0 / freqs, 1.0)
        noise_f = white_f * np.sqrt(pink_psd)

    elif noise_type == "seismic":
        # Seismic-like: PSD ~ f^-8 below 20 Hz, then flat
        with np.errstate(divide='ignore', invalid='ignore'):
            seismic_psd = np.where(freqs > 0, (20.0 / np.maximum(freqs, 1.0))**8, 1.0)
        noise_f = white_f * np.sqrt(seismic_psd)

    else:
        raise ValueError(f"Unknown noise_type: {noise_type}")

    noise = np.fft.irfft(noise_f, n=N).real

    # Normalize to unit variance, then scale
    std = np.std(noise)
    if std > 0:
        noise = noise / std

    return noise * noise_level


def generate_noisy_signal(
    sample_rate: int = 4096,
    duration: float = 4.0,
    m1_msun: float = 30.0,
    m2_msun: float = 30.0,
    distance_mpc: float = 400.0,
    noise_type: str = "aligo",
    noise_level: float = 1.0,
    f_lower: float = 20.0,
) -> dict:
    """
    Generate a full dataset: chirp signal + noise + combined noisy signal.
    """
    chirp_data = generate_chirp(
        sample_rate=sample_rate,
        duration=duration,
        m1_msun=m1_msun,
        m2_msun=m2_msun,
        distance_mpc=distance_mpc,
        f_lower=f_lower,
    )

    h = np.array(chirp_data["h_plus"])
    noise = generate_noise(sample_rate, duration, noise_type, noise_level)

    # Scale signal amplitude relative to noise
    h_max = np.max(np.abs(h))
    if h_max > 0:
        target_snr = 5.0 / noise_level   # nominal SNR
        h_scaled = h * (target_snr * np.std(noise) / h_max) * 1e-21
    else:
        h_scaled = h

    noisy = h_scaled + noise * 1e-21

    return {
        "times": chirp_data["times"],
        "h_clean": h_scaled.tolist(),
        "h_noisy": noisy.tolist(),
        "h_noise_only": (noise * 1e-21).tolist(),
        "frequency": chirp_data["frequency"],
        "metadata": chirp_data["metadata"],
    }


if __name__ == "__main__":
    data = generate_noisy_signal(m1_msun=36, m2_msun=29, distance_mpc=410,
                                  noise_type="aligo", noise_level=1.0)
    print("Signal generated!")
    print(f"  Samples: {len(data['times'])}")
    print(f"  Chirp mass: {data['metadata']['chirp_mass_msun']:.2f} M_sun")
    print(f"  Clean signal max: {max(abs(x) for x in data['h_clean']):.3e}")
    print(f"  Noisy signal std: {np.std(data['h_noisy']):.3e}")
