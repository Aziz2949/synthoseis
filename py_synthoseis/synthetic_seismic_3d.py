"""3-D synthetic seismic generator — synthoseis-style refactor of v6.1.

Same physics / feature menu as the original (Wu et al. 2019 reflectivity +
folding + channels + faults + reefs + salt + thin beds + unconformity +
gas chimney), but organised the way the rest of synthoseis is laid out:

    * One `@dataclass` (``GenConfig``) holds every knob — no module-level
      globals leaking into worker processes.
    * Each geological feature is a plain function ``f(r, rng, cfg) -> r``
      registered in a single ``PIPELINE`` table, so adding / disabling a
      feature is a one-line edit.
    * A small ``SyntheticVolumeBuilder`` class wraps the per-volume build
      (mirrors synthoseis ``SeismicVolume`` / ``Geomodel`` style).
    * Wavelet defaults to a zero-phase Ormsby 8 / 12 / 60 / 80 Hz (spectral
      centroid ≈ 40 Hz, matching synthoseis' clean-seismic defaults). Ricker
      is still available via ``cfg.wavelet_type = "ricker"``.
    * Output: one ``data_<i>.npy`` per cube, float32, shape from cfg.

Run as a script
---------------
    python py_synthoseis/synthetic_seismic_3d.py 100 ./out

Run from a Jupyter cell
-----------------------
    from py_synthoseis.synthetic_seismic_3d import GenConfig, run_dataset, build_volume

    cfg = GenConfig(
        n_volumes=4,
        out_dir="./synth_out",
        nz=256, nx=256, ny=256,
        n_workers=4,
    )
    run_dataset(cfg)

    # Or a single in-memory cube (no I/O):
    vol = build_volume(seed=10_000, cfg=GenConfig(n_volumes=1))
"""

# Thread caps MUST be set before NumPy / SciPy import — otherwise each worker
# spawns its own BLAS/OMP pool and we get n_workers * n_blas_threads contention.
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import time  # noqa: E402  (env vars must be set before numpy imports below)
from dataclasses import dataclass, field  # noqa: E402
from multiprocessing import get_context  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Callable, Dict, List, Optional, Tuple  # noqa: E402

import numpy as np  # noqa: E402
from scipy.ndimage import (  # noqa: E402
    gaussian_filter,
    gaussian_filter1d,
    map_coordinates,
    rotate,
)


# =====================================================================
#                              CONFIG
# =====================================================================

DEFAULT_FEATURE_PROBS: Dict[str, float] = {
    "unconformity": 0.30,
    "thin_beds": 0.35,
    "channels": 0.45,
    "reef": 0.25,
    "salt_body": 0.12,
    # post-convolution features:
    "main_faults": 0.70,
    "fine_faults": 0.55,
    "gas_chimney": 0.15,
}


@dataclass
class GenConfig:
    """Configuration for a synthetic seismic generation run.

    Attributes
    ----------
    nz, nx, ny : int
        Output cube shape (samples × inlines × crosslines).
    dt : float
        Vertical sample interval in seconds.
    wavelet_type : {"ormsby", "ricker"}
        Convolution wavelet. Ormsby = clean processed-seismic bandwidth
        (synthoseis default). Ricker = legacy v6.1 behaviour.
    ricker_freq_range : tuple of float
        Peak-frequency range when wavelet_type="ricker" (Hz, drawn uniform).
    ormsby_corners : tuple of float
        Four-corner Ormsby passband (Hz): f1 < f2 < f3 < f4.
    fold_strength, lateral_modulation : float
        Same meaning as v6.1: vertical-fold amplitude scale and lateral
        reflectivity modulation strength.
    feature_probs : dict[str, float]
        Independent Bernoulli probability per feature. Missing keys fall
        back to ``DEFAULT_FEATURE_PROBS``. Set to 0.0 to disable, 1.0 to
        always include.
    force_features : dict[str, bool] | None
        Optional per-feature override of the random roll. Wins over probs.
    n_volumes : int
        How many cubes to generate when running the batch entry point.
    start_index : int
        First volume index (used in the output filename and seed offset).
    out_dir : Path or str
        Output directory. Created if missing.
    seed_base : int
        Per-volume seed = ``seed_base + i`` for volume index i.
    n_workers : int or None
        ``None`` → ``os.cpu_count() - 1``. ``1`` → single-process (Jupyter-safe).
    final_smooth : bool
        Mild lateral Gaussian polish on the imaged volume.
    verbose : bool
        Print one progress line per volume.
    """

    # cube geometry
    nz: int = 256
    nx: int = 256
    ny: int = 256
    dt: float = 0.004

    # wavelet
    wavelet_type: str = "ormsby"
    ricker_freq_range: Tuple[float, float] = (28.0, 50.0)
    ormsby_corners: Tuple[float, float, float, float] = (8.0, 12.0, 60.0, 80.0)
    wavelet_length: int = 65  # samples; used by Ricker

    # reflectivity & folding
    fold_strength: float = 1.0
    lateral_modulation: float = 0.35

    # feature menu
    feature_probs: Dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_FEATURE_PROBS)
    )
    force_features: Optional[Dict[str, bool]] = None

    # batch
    n_volumes: int = 1
    start_index: int = 0
    out_dir: Path = Path(".")
    seed_base: int = 10_000
    n_workers: Optional[int] = None

    # cosmetics
    final_smooth: bool = False
    verbose: bool = True

    def __post_init__(self) -> None:
        self.out_dir = Path(self.out_dir)

    def resolve_n_workers(self) -> int:
        if self.n_workers is None:
            return max(1, (os.cpu_count() or 1) - 1)
        return max(1, int(self.n_workers))


# =====================================================================
#                              WAVELETS
# =====================================================================

def ricker(f_hz: float, dt: float, n: int = 65) -> np.ndarray:
    """Zero-phase Ricker wavelet, peak frequency ``f_hz``."""
    f = f_hz * dt
    t = np.arange(n, dtype=np.float32) - n // 2
    a = (np.pi * f * t) ** 2
    return ((1.0 - 2.0 * a) * np.exp(-a)).astype(np.float32)


def ormsby(f1: float, f2: float, f3: float, f4: float, dt: float,
           length_ms: float = 200.0) -> np.ndarray:
    """Zero-phase Ormsby wavelet via trapezoidal-spectrum IFFT.

    Parameters
    ----------
    f1, f2, f3, f4 : float
        Four passband corners in Hz, strictly increasing.
    dt : float
        Sample interval in seconds.
    length_ms : float
        Total wavelet length in milliseconds (an odd sample count is forced).
    """
    if not (f1 < f2 < f3 < f4):
        raise ValueError(f"Ormsby corners must be strictly increasing, got {f1},{f2},{f3},{f4}")
    nyq = 0.5 / dt
    if f4 >= nyq:
        raise ValueError(f"Ormsby f4={f4} Hz exceeds Nyquist={nyq:.1f} Hz for dt={dt}")

    dt_ms = dt * 1000.0
    n = int(round(length_ms / dt_ms))
    if n % 2 == 0:
        n += 1
    nfft = max(2048, 4 * n)
    freqs = np.fft.rfftfreq(nfft, d=dt)
    spec = np.zeros_like(freqs)
    plateau = (freqs >= f2) & (freqs <= f3)
    spec[plateau] = 1.0
    up = (freqs > f1) & (freqs < f2)
    spec[up] = (freqs[up] - f1) / (f2 - f1)
    dn = (freqs > f3) & (freqs < f4)
    spec[dn] = (f4 - freqs[dn]) / (f4 - f3)

    full = np.fft.fftshift(np.fft.irfft(spec, n=nfft))
    half = n // 2
    centre = nfft // 2
    w = full[centre - half : centre + half + 1].astype(np.float32)
    # Hann edge taper to suppress convolution ringing
    taper = np.hanning(n).astype(np.float32)
    w *= taper
    w /= np.max(np.abs(w)) + 1e-12
    return w


def _wavelet_for(cfg: GenConfig, rng: np.random.Generator) -> np.ndarray:
    if cfg.wavelet_type == "ricker":
        f_peak = float(rng.uniform(*cfg.ricker_freq_range))
        return ricker(f_peak, cfg.dt, n=cfg.wavelet_length)
    if cfg.wavelet_type == "ormsby":
        return ormsby(*cfg.ormsby_corners, dt=cfg.dt)
    raise ValueError(f"Unknown wavelet_type: {cfg.wavelet_type!r}")


def convolve_z(r: np.ndarray, w: np.ndarray) -> np.ndarray:
    """1-D z-axis convolution of a 3-D reflectivity cube with a wavelet."""
    nz = r.shape[0]
    nfft = nz + w.size - 1
    R = np.fft.rfft(r, n=nfft, axis=0)
    W = np.fft.rfft(w, n=nfft)[:, None, None]
    s = np.fft.irfft(R * W, n=nfft, axis=0)
    off = (w.size - 1) // 2
    return s[off : off + nz].astype(np.float32)


# =====================================================================
#                  REFLECTIVITY + STRUCTURAL DEFORMATION
# =====================================================================

def make_reflectivity(cfg: GenConfig, rng: np.random.Generator) -> np.ndarray:
    """Dense reflectivity à la Wu et al. (2019), with anisotropic modulation."""
    nz, nx, ny = cfg.nz, cfg.nx, cfg.ny
    r1d = rng.uniform(-1.0, 1.0, size=nz).astype(np.float32)
    r1d = gaussian_filter(r1d, sigma=float(rng.uniform(0.4, 1.0)))
    r1d /= np.max(np.abs(r1d)) + 1e-9
    r = np.broadcast_to(r1d[:, None, None], (nz, nx, ny)).copy()

    if cfg.lateral_modulation > 0:
        lat = rng.standard_normal((nz, nx, ny)).astype(np.float32)
        sig_long = float(rng.uniform(16.0, 30.0))
        sig_short = float(rng.uniform(6.0, 12.0))
        lat = gaussian_filter(lat, sigma=(2.0, sig_long, sig_short))
        ang = float(rng.uniform(0.0, 180.0))
        lat = rotate(lat, ang, axes=(1, 2), reshape=False, order=1,
                     mode="nearest").astype(np.float32)
        lat /= np.std(lat) + 1e-9
        r *= 1.0 + float(cfg.lateral_modulation) * lat
    return r


def folding_shifts(cfg: GenConfig, rng: np.random.Generator,
                   strength: float) -> np.ndarray:
    """Vertical displacement field for warping reflectivity."""
    nz, nx, ny = cfg.nz, cfg.nx, cfg.ny
    X, Y = np.meshgrid(
        np.arange(nx, dtype=np.float32),
        np.arange(ny, dtype=np.float32),
        indexing="ij",
    )
    N = int(rng.integers(5, 14))
    S2 = np.zeros((nx, ny), dtype=np.float32)
    max_amp = float(rng.uniform(6.0, 18.0)) * strength
    for _ in range(N):
        amp = float(rng.uniform(-max_amp, max_amp))
        cx = float(rng.uniform(-nx * 0.1, nx * 1.1))
        cy = float(rng.uniform(-ny * 0.1, ny * 1.1))
        s_short = float(rng.uniform(nx * 0.06, nx * 0.14))
        aspect = float(rng.uniform(3.0, 7.0))
        s_long = s_short * aspect
        theta = float(rng.uniform(0.0, np.pi))
        dx, dy = X - cx, Y - cy
        cos_t, sin_t = float(np.cos(theta)), float(np.sin(theta))
        u = cos_t * dx + sin_t * dy
        v = -sin_t * dx + cos_t * dy
        S2 += amp * np.exp(-0.5 * ((u / s_long) ** 2 + (v / s_short) ** 2))

    e = float(rng.uniform(-0.04, 0.04)) * strength
    f = float(rng.uniform(-0.04, 0.04)) * strength
    S1 = (e * X + f * Y).astype(np.float32)

    z = np.arange(nz, dtype=np.float32) / nz
    if rng.random() < 0.3:
        z = 1.0 - z
    return (z[:, None, None] * S2[None] + S1[None]).astype(np.float32)


def warp_z(r: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Apply a vertical-shift field to a 3-D volume by backward sampling."""
    nz, nx, ny = r.shape
    zz, xx, yy = np.meshgrid(
        np.arange(nz, dtype=np.float32),
        np.arange(nx, dtype=np.float32),
        np.arange(ny, dtype=np.float32),
        indexing="ij",
    )
    coords = np.stack([zz - S, xx, yy], axis=0)
    return map_coordinates(r, coords, order=1, mode="nearest",
                           prefilter=False).astype(np.float32)


# =====================================================================
#                       GEOLOGICAL FEATURE PIECES
# =====================================================================

def feat_unconformity(r: np.ndarray, rng: np.random.Generator,
                       cfg: GenConfig) -> np.ndarray:
    """Erosional unconformity: older deformed strata below, younger above."""
    nz, nx, ny = r.shape
    surf = rng.standard_normal((nx, ny)).astype(np.float32)
    surf = gaussian_filter(surf, sigma=float(rng.uniform(18.0, 35.0)))
    surf /= np.std(surf) + 1e-9
    relief = float(rng.uniform(4.0, 15.0))
    z_centre = float(rng.uniform(0.25 * nz, 0.60 * nz))
    z_unc = (z_centre + relief * surf).astype(np.float32)

    r_young = make_reflectivity(cfg, rng) * 1.0
    r_young = warp_z(
        r_young,
        folding_shifts(cfg, rng, cfg.fold_strength * float(rng.uniform(0.1, 0.4))),
    )

    zz = np.arange(nz, dtype=np.float32)[:, None, None]
    mask = 0.5 * (1.0 + np.tanh((zz - z_unc[None]) / 0.8)).astype(np.float32)
    r_out = mask * r + (1.0 - mask) * r_young

    bright = rng.choice([-1.0, 1.0]) * float(rng.uniform(0.4, 0.8))
    spike = bright * np.exp(-0.5 * ((zz - z_unc[None]) / 0.8) ** 2)
    return (r_out + spike).astype(np.float32)


def feat_thin_beds(r: np.ndarray, rng: np.random.Generator,
                   cfg: GenConfig) -> np.ndarray:
    """Localized package of high-frequency thin beds (e.g. turbidite)."""
    nz, nx, ny = r.shape
    x0 = float(rng.uniform(0.25 * nx, 0.75 * nx))
    y0 = float(rng.uniform(0.25 * ny, 0.75 * ny))
    z0 = float(rng.uniform(0.20 * nz, 0.80 * nz))
    rx = float(rng.uniform(nx * 0.12, nx * 0.30))
    ry = float(rng.uniform(ny * 0.12, ny * 0.30))
    hz = float(rng.uniform(nz * 0.08, nz * 0.20))

    zz, xx, yy = np.meshgrid(
        np.arange(nz, dtype=np.float32),
        np.arange(nx, dtype=np.float32),
        np.arange(ny, dtype=np.float32),
        indexing="ij",
    )
    xy_inside = 1.0 - (((xx - x0) / rx) ** 2 + ((yy - y0) / ry) ** 2)
    z_inside = 1.0 - ((zz - z0) / hz) ** 2
    inside = np.minimum(xy_inside, z_inside)
    mask = (0.5 * (1.0 + np.tanh(inside * 4.0))).astype(np.float32)

    thin = rng.uniform(-1.0, 1.0, size=nz).astype(np.float32)
    thin = gaussian_filter(thin, sigma=float(rng.uniform(0.15, 0.35)))
    thin /= np.max(np.abs(thin)) + 1e-9
    amp = float(rng.uniform(1.0, 1.5))
    thin_r = amp * np.broadcast_to(thin[:, None, None], (nz, nx, ny)).astype(np.float32)
    return ((1.0 - mask) * r + mask * thin_r).astype(np.float32)


def _stamp_one_channel(r: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Single sinuous meandering channel (preserved from v6.1)."""
    nz, nx, ny = r.shape

    theta = float(rng.uniform(0.0, 2.0 * np.pi))
    ca, sa = float(np.cos(theta)), float(np.sin(theta))
    x0 = float(rng.uniform(0.0, nx))
    y0 = float(rng.uniform(0.0, ny))
    z0 = float(rng.uniform(0.18 * nz, 0.85 * nz))

    lam1 = float(rng.uniform(nx * 0.18, nx * 0.55))
    amp1 = float(rng.uniform(nx * 0.02, nx * 0.10))
    ph1 = float(rng.uniform(0.0, 2.0 * np.pi))
    lam2 = lam1 * float(rng.uniform(0.25, 0.55))
    amp2 = amp1 * float(rng.uniform(0.20, 0.55))
    ph2 = float(rng.uniform(0.0, 2.0 * np.pi))

    W = float(rng.uniform(3.0, 12.0))
    D = float(rng.uniform(2.0, 7.0))

    zz, xx, yy = np.meshgrid(
        np.arange(nz, dtype=np.float32),
        np.arange(nx, dtype=np.float32),
        np.arange(ny, dtype=np.float32),
        indexing="ij",
    )
    dx, dy = xx - x0, yy - y0
    u = ca * dx + sa * dy
    v = -sa * dx + ca * dy
    v_axis = (
        amp1 * np.sin(2.0 * np.pi * u / lam1 + ph1)
        + amp2 * np.sin(2.0 * np.pi * u / lam2 + ph2)
    ).astype(np.float32)
    v_rel = v - v_axis
    norm_v = 2.0 * v_rel / W
    under = np.maximum(1.0 - norm_v * norm_v, 0.0)
    local_depth = (D * np.sqrt(under)).astype(np.float32)

    xy_inside = W * 0.5 - np.abs(v_rel)
    z_top_dist = zz - z0
    z_base_dist = (z0 + local_depth) - zz
    inside_sdf = np.minimum(np.minimum(xy_inside, z_top_dist), z_base_dist)
    mask = (0.5 * (1.0 + np.tanh(inside_sdf / 0.7))).astype(np.float32)

    style = rng.choice(["bright", "dim", "chaotic"])
    if style == "bright":
        base = rng.choice([-1.0, 1.0]) * float(rng.uniform(0.55, 0.90))
        fill = base + 0.12 * rng.standard_normal((nz, nx, ny)).astype(np.float32)
        fill = gaussian_filter(fill, sigma=(0.8, 0.4, 0.4))
    elif style == "dim":
        fill = 0.08 * rng.standard_normal((nz, nx, ny)).astype(np.float32)
        fill = gaussian_filter(fill, sigma=0.8)
    else:
        fill = 0.55 * rng.standard_normal((nz, nx, ny)).astype(np.float32)
        fill = gaussian_filter(fill, sigma=(0.5, 0.6, 0.6))
    fill = fill.astype(np.float32)

    r_out = ((1.0 - mask) * r + mask * fill).astype(np.float32)
    footprint = (0.5 * (1.0 + np.tanh(xy_inside / 0.7))).astype(np.float32)
    top_spike = (
        rng.choice([-1.0, 1.0]) * float(rng.uniform(0.3, 0.7))
        * np.exp(-0.5 * ((zz - z0) / 0.7) ** 2)
    ).astype(np.float32)
    return r_out + footprint * top_spike


def feat_channels(r: np.ndarray, rng: np.random.Generator,
                  cfg: GenConfig) -> np.ndarray:
    """1-3 stacked sinuous channels forming a 'system' (channel complex)."""
    for _ in range(int(rng.integers(1, 4))):
        r = _stamp_one_channel(r, rng)
    return r


def feat_reef(r: np.ndarray, rng: np.random.Generator,
              cfg: GenConfig) -> np.ndarray:
    """Carbonate reef / mound — chaotic internal character, mounded top."""
    nz, nx, ny = r.shape
    x0 = float(rng.uniform(0.25 * nx, 0.75 * nx))
    y0 = float(rng.uniform(0.25 * ny, 0.75 * ny))
    z0 = float(rng.uniform(0.25 * nz, 0.80 * nz))
    rx = float(rng.uniform(nx * 0.06, nx * 0.18))
    ry = float(rng.uniform(ny * 0.06, ny * 0.18))
    H = float(rng.uniform(6.0, 18.0))

    zz, xx, yy = np.meshgrid(
        np.arange(nz, dtype=np.float32),
        np.arange(nx, dtype=np.float32),
        np.arange(ny, dtype=np.float32),
        indexing="ij",
    )
    rho2 = ((xx - x0) / rx) ** 2 + ((yy - y0) / ry) ** 2
    height = H * np.maximum(1.0 - rho2, 0.0)
    inside = np.minimum(np.minimum(zz - (z0 - height), z0 - zz), 1.0 - rho2)
    mask = (0.5 * (1.0 + np.tanh(inside / 0.9))).astype(np.float32)

    reef = rng.uniform(0.15, 0.45) + 0.7 * rng.standard_normal((nz, nx, ny)).astype(np.float32)
    reef = gaussian_filter(reef, sigma=(0.4, 0.6, 0.6)).astype(np.float32)
    return ((1.0 - mask) * r + mask * reef).astype(np.float32)


def feat_salt_body(r: np.ndarray, rng: np.random.Generator,
                   cfg: GenConfig) -> np.ndarray:
    """Salt / shale piercement — vertical column with chaotic low-amp fill."""
    nz, nx, ny = r.shape
    x0 = float(rng.uniform(0.30 * nx, 0.70 * nx))
    y0 = float(rng.uniform(0.30 * ny, 0.70 * ny))
    z_top = float(rng.uniform(0.20 * nz, 0.55 * nz))
    r_top = float(rng.uniform(nx * 0.08, nx * 0.18))
    r_bot = float(rng.uniform(nx * 0.04, nx * 0.12))

    zz, xx, yy = np.meshgrid(
        np.arange(nz, dtype=np.float32),
        np.arange(nx, dtype=np.float32),
        np.arange(ny, dtype=np.float32),
        indexing="ij",
    )
    t = np.clip((zz - z_top) / max(nz - z_top, 1.0), 0.0, 1.0)
    radius = (r_top * (1.0 - t) + r_bot * t).astype(np.float32)
    dist = np.sqrt((xx - x0) ** 2 + (yy - y0) ** 2)
    radial = 0.5 * (1.0 - np.tanh((dist - radius) / 1.5)).astype(np.float32)
    top_fade = 0.5 * (1.0 + np.tanh((zz - z_top) / 2.0)).astype(np.float32)
    mask = (radial * top_fade).astype(np.float32)

    fill = 0.15 * rng.standard_normal((nz, nx, ny)).astype(np.float32)
    fill = gaussian_filter(fill, sigma=(1.2, 1.6, 1.6)).astype(np.float32)
    return ((1.0 - mask) * r + mask * fill).astype(np.float32)


def _apply_fault(vol: np.ndarray, rng: np.random.Generator,
                  scale: str) -> np.ndarray:
    """One planar fault. ``scale`` is "main" (5-18 vx throw) or "fine" (≤2.5)."""
    nz, nx, ny = vol.shape
    if scale == "fine":
        dip = np.deg2rad(rng.uniform(60.0, 88.0))
        throw_max = float(rng.uniform(0.5, 2.5))
        R_strike = float(rng.uniform(nx * 0.12, nx * 0.35))
        R_dip = float(rng.uniform(nx * 0.06, nx * 0.18))
        wave_amp = float(rng.uniform(0.4, 1.5))
        half_width = 0.7
        z0 = float(rng.uniform(nz * 0.15, nz * 0.85))
        x0 = float(rng.uniform(nx * 0.15, nx * 0.85))
        y0 = float(rng.uniform(ny * 0.15, ny * 0.85))
    elif scale == "main":
        dip = np.deg2rad(rng.uniform(55.0, 85.0))
        throw_max = float(rng.uniform(5.0, 18.0))
        R_strike = float(rng.uniform(nx * 0.45, nx * 0.80))
        R_dip = float(rng.uniform(nx * 0.20, nx * 0.40))
        wave_amp = float(rng.uniform(2.0, 5.0))
        half_width = 1.0
        z0 = float(rng.uniform(nz * 0.30, nz * 0.70))
        x0 = float(rng.uniform(nx * 0.25, nx * 0.75))
        y0 = float(rng.uniform(ny * 0.25, ny * 0.75))
    else:
        raise ValueError(f"scale must be 'main' or 'fine', got {scale!r}")
    strike = np.deg2rad(rng.uniform(0.0, 180.0))
    sense = float(rng.choice([-1.0, 1.0]))

    n_z = float(np.cos(dip))
    n_x = float(np.sin(dip) * np.cos(strike))
    n_y = float(np.sin(dip) * np.sin(strike))
    s_z = float(-np.sin(dip))
    s_x = float(np.cos(dip) * np.cos(strike))
    s_y = float(np.cos(dip) * np.sin(strike))

    zz, xx, yy = np.meshgrid(
        np.arange(nz, dtype=np.float32),
        np.arange(nx, dtype=np.float32),
        np.arange(ny, dtype=np.float32),
        indexing="ij",
    )
    d = n_z * (zz - z0) + n_x * (xx - x0) + n_y * (yy - y0)
    kz = float(rng.uniform(0.020, 0.060))
    kx = float(rng.uniform(0.020, 0.060))
    ky = float(rng.uniform(0.020, 0.060))
    ph = float(rng.uniform(0.0, 2.0 * np.pi))
    d = d + wave_amp * np.sin(kz * zz + kx * xx + ky * yy + ph).astype(np.float32)

    dzp = (zz - z0) - d * n_z
    dxp = (xx - x0) - d * n_x
    dyp = (yy - y0) - d * n_y
    d_slip = dzp * s_z + dxp * s_x + dyp * s_y
    rho_sq = dzp * dzp + dxp * dxp + dyp * dyp
    d_cross = np.sqrt(np.maximum(rho_sq - d_slip * d_slip, 0.0))
    taper = np.exp(
        -0.5 * ((d_cross / R_strike) ** 2 + (d_slip / R_dip) ** 2)
    ).astype(np.float32)

    full_slip = (sense * throw_max * taper).astype(np.float32)
    coords = np.stack(
        [zz - full_slip * s_z, xx - full_slip * s_x, yy - full_slip * s_y],
        axis=0,
    )
    hw = map_coordinates(vol, coords, order=1, mode="nearest",
                         prefilter=False).astype(np.float32)
    smooth = (0.5 * (1.0 + np.tanh(d / half_width))).astype(np.float32)
    return ((1.0 - smooth) * vol + smooth * hw).astype(np.float32)


def feat_main_faults(vol: np.ndarray, rng: np.random.Generator,
                     cfg: GenConfig) -> np.ndarray:
    for _ in range(int(rng.integers(1, 5))):
        vol = _apply_fault(vol, rng, scale="main")
    return vol


def feat_fine_faults(vol: np.ndarray, rng: np.random.Generator,
                     cfg: GenConfig) -> np.ndarray:
    for _ in range(int(rng.integers(2, 6))):
        vol = _apply_fault(vol, rng, scale="fine")
    return vol


def feat_gas_chimney(vol: np.ndarray, rng: np.random.Generator,
                     cfg: GenConfig) -> np.ndarray:
    """Vertical coherence-loss column (z-blur inside a narrow tube)."""
    nz, nx, ny = vol.shape
    x0 = float(rng.uniform(0.25 * nx, 0.75 * nx))
    y0 = float(rng.uniform(0.25 * ny, 0.75 * ny))
    radius = float(rng.uniform(3.0, 10.0))
    z_top = float(rng.uniform(0.10 * nz, 0.50 * nz))

    xx, yy = np.meshgrid(
        np.arange(nx, dtype=np.float32),
        np.arange(ny, dtype=np.float32),
        indexing="ij",
    )
    dist = np.sqrt((xx - x0) ** 2 + (yy - y0) ** 2)
    radial = 0.5 * (1.0 - np.tanh((dist - radius) / 1.2)).astype(np.float32)
    radial3d = np.broadcast_to(radial[None, :, :], (nz, nx, ny))
    zz = np.arange(nz, dtype=np.float32)[:, None, None]
    top_fade = 0.5 * (1.0 + np.tanh((zz - z_top) / 2.5)).astype(np.float32)
    mask = (radial3d * top_fade).astype(np.float32)

    smeared = gaussian_filter1d(vol, sigma=float(rng.uniform(2.5, 5.0)),
                                axis=0).astype(np.float32)
    return ((1.0 - mask) * vol + mask * smeared).astype(np.float32)


# =====================================================================
#                          PIPELINE REGISTRATION
# =====================================================================

# Each entry: (feature_name, function, stage). "pre" runs on reflectivity
# before wavelet convolution; "post" runs on the imaged seismic.
FeatureFn = Callable[[np.ndarray, np.random.Generator, GenConfig], np.ndarray]

PIPELINE: List[Tuple[str, FeatureFn, str]] = [
    ("unconformity", feat_unconformity, "pre"),
    ("thin_beds", feat_thin_beds, "pre"),
    ("channels", feat_channels, "pre"),
    ("reef", feat_reef, "pre"),
    ("salt_body", feat_salt_body, "pre"),
    ("main_faults", feat_main_faults, "post"),
    ("fine_faults", feat_fine_faults, "post"),
    ("gas_chimney", feat_gas_chimney, "post"),
]


# =====================================================================
#                        SINGLE-VOLUME BUILDER
# =====================================================================

class SyntheticVolumeBuilder:
    """Build one synthetic seismic cube using a ``GenConfig``."""

    def __init__(self, cfg: GenConfig) -> None:
        self.cfg = cfg

    def _roll_features(self, rng: np.random.Generator) -> Dict[str, bool]:
        probs = dict(DEFAULT_FEATURE_PROBS)
        probs.update(self.cfg.feature_probs)
        feats = {name: rng.random() < p for name, p in probs.items()}
        if self.cfg.force_features:
            feats.update({k: bool(v) for k, v in self.cfg.force_features.items()})
        return feats

    def build(self, seed: int) -> Tuple[np.ndarray, Dict[str, bool]]:
        """Generate a single cube. Returns (volume, feature-flags-dict)."""
        cfg = self.cfg
        rng = np.random.default_rng(seed)
        feats = self._roll_features(rng)

        # 1) reflectivity + folding
        r = make_reflectivity(cfg, rng)
        r = warp_z(r, folding_shifts(cfg, rng, cfg.fold_strength))
        if rng.random() < 0.5:
            r = warp_z(r, folding_shifts(cfg, rng, cfg.fold_strength) * 0.4)

        # 2) pre-convolution geological features
        for name, fn, stage in PIPELINE:
            if stage == "pre" and feats.get(name):
                r = fn(r, rng, cfg)

        # 3) wavelet convolution
        w = _wavelet_for(cfg, rng)
        vol = convolve_z(r, w)

        # 4) post-convolution features
        for name, fn, stage in PIPELINE:
            if stage == "post" and feats.get(name):
                vol = fn(vol, rng, cfg)

        # 5) optional lateral polish + normalize to ~[-1, 1]
        if cfg.final_smooth:
            vol = gaussian_filter(vol, sigma=(0.0, 0.6, 0.6)).astype(np.float32)
        clip = float(np.percentile(np.abs(vol), 99.5)) + 1e-12
        vol = (np.clip(vol, -clip, clip) / clip).astype(np.float32)
        return vol, feats


def build_volume(seed: int, cfg: GenConfig) -> np.ndarray:
    """Convenience: return just the cube (drops the feature-flag dict)."""
    return SyntheticVolumeBuilder(cfg).build(seed)[0]


# =====================================================================
#                             BATCH RUNNER
# =====================================================================

def _worker(payload):
    """Pool worker. Lives at module level so it pickles cleanly."""
    i, cfg = payload
    t0 = time.time()
    builder = SyntheticVolumeBuilder(cfg)
    vol, feats = builder.build(seed=cfg.seed_base + i)
    out_path = cfg.out_dir / f"data_{i}.npy"
    np.save(out_path, vol)
    active = sorted(k for k, v in feats.items() if v)
    return i, time.time() - t0, active


def run_dataset(cfg: GenConfig) -> None:
    """Generate ``cfg.n_volumes`` cubes into ``cfg.out_dir``.

    Saves only ``data_<i>.npy``. No noisy/clean pairs, no labels, no
    manifest — matching v6.1's data-only output policy.
    """
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    n_workers = cfg.resolve_n_workers()
    tasks = [
        (i, cfg) for i in range(cfg.start_index, cfg.start_index + cfg.n_volumes)
    ]
    if cfg.verbose:
        print(
            f"[synthetic_seismic_3d] {len(tasks)} cubes -> {cfg.out_dir} "
            f"(n_workers={n_workers}, shape={cfg.nz}x{cfg.nx}x{cfg.ny}, "
            f"wavelet={cfg.wavelet_type})",
            flush=True,
        )

    if n_workers == 1:
        for task in tasks:
            i, dt, active = _worker(task)
            if cfg.verbose:
                print(f"data_{i}.npy  dt={dt:.1f}s  feats={active}", flush=True)
        return

    ctx = get_context("fork")
    with ctx.Pool(processes=n_workers) as pool:
        for i, dt, active in pool.imap_unordered(_worker, tasks, chunksize=1):
            if cfg.verbose:
                print(f"data_{i}.npy  dt={dt:.1f}s  feats={active}", flush=True)


# =====================================================================
#                                CLI
# =====================================================================

def _parse_cli(argv):
    import argparse

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("n_volumes", nargs="?", type=int, default=2)
    p.add_argument("out_dir", nargs="?", default=".")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--seed-base", type=int, default=10_000)
    p.add_argument("--wavelet", choices=("ormsby", "ricker"), default="ormsby")
    p.add_argument("--shape", type=int, nargs=3, metavar=("NZ", "NX", "NY"),
                   default=(256, 256, 256))
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def _in_jupyter() -> bool:
    """True when this module is executing inside a Jupyter / IPython kernel.

    Used to skip CLI argument parsing when the whole file is copy-pasted
    into a notebook cell — otherwise argparse would try to consume the
    kernel's own argv (``-f /path/to/kernel.json``) and abort.
    """
    import sys
    # IPython injects get_ipython() into the user namespace of every cell.
    if "get_ipython" in globals():
        return True
    # The kernel launcher leaves "ipykernel" / "jupyter" in argv[0].
    argv0 = sys.argv[0] if sys.argv else ""
    if "ipykernel" in argv0 or "jupyter" in argv0:
        return True
    # Final fallback: check whether IPython has been imported & has a kernel.
    try:
        import IPython  # type: ignore
        ip = IPython.get_ipython()
        return ip is not None and "ipykernel" in type(ip).__module__
    except Exception:
        return False


def _run_default_cli() -> None:
    """Argparse-driven entry point used when invoked as a real script."""
    import sys

    args = _parse_cli(sys.argv[1:])
    cfg = GenConfig(
        nz=args.shape[0],
        nx=args.shape[1],
        ny=args.shape[2],
        n_volumes=args.n_volumes,
        start_index=args.start_index,
        out_dir=Path(args.out_dir),
        n_workers=args.workers,
        seed_base=args.seed_base,
        wavelet_type=args.wavelet,
        verbose=not args.quiet,
    )
    run_dataset(cfg)


if __name__ == "__main__":
    if _in_jupyter():
        # Pasted into a notebook cell — don't try to parse Jupyter's argv.
        # All classes/functions are now defined; call them from the next cell:
        #
        #     cfg = GenConfig(n_volumes=2, out_dir="./synth_out", n_workers=2)
        #     run_dataset(cfg)
        #
        #     # or a single in-memory cube:
        #     vol = build_volume(seed=10_000, cfg=GenConfig())
        print(
            "synthetic_seismic_3d loaded into the notebook. Example usage:\n"
            "    cfg = GenConfig(n_volumes=2, out_dir='./synth_out', n_workers=2)\n"
            "    run_dataset(cfg)\n"
            "    # or a single in-memory cube:\n"
            "    vol = build_volume(seed=10_000, cfg=GenConfig())"
        )
    else:
        _run_default_cli()
