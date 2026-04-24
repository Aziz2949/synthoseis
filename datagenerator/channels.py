"""Procedural channel-belt generator.

Produces geologically plausible sinuous channel bodies and stamps them into
the faulted lithology / net-to-gross cubes after faulting and before seismic
synthesis. This is a lightweight alternative to meanderpy's full meander-belt
simulator: it generates sinuous centerlines as a sum of a few harmonics with
decaying amplitudes (the first two harmonics dominate, matching real fluvial
planforms), then paints channel-fill sand into a thin stratigraphic layer.

The generator is deliberately dependency-free (NumPy only) so it works in
any Python environment that can run the rest of the pipeline.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class ChannelSpec:
    """Realised parameters of one channel for logging."""

    kind: str  # "fluvial" or "submarine"
    layer_index: int  # stratigraphic layer where the channel was placed
    width_samples: int
    depth_samples: int
    sinuosity_harmonics: int
    azimuth_deg: float


def _sinuous_centerline(length_pts: int, amplitude: float, harmonics: int, rng):
    """Return a 1D array of crossline offsets sampled at `length_pts` points.

    A channel centerline is modelled as the sum of a handful of harmonics with
    amplitudes decaying like 1/(k+1). This produces smooth, visually fluvial
    planforms without any PDE integration.
    """
    t = np.linspace(0.0, 2.0 * np.pi, length_pts)
    offset = np.zeros_like(t)
    for k in range(1, harmonics + 1):
        phase = rng.uniform(0, 2 * np.pi)
        amp = amplitude * rng.uniform(0.5, 1.0) / k
        offset += amp * np.sin(k * t + phase)
    return offset


def _stamp_channel_into_mask(mask, cx, cy, width_xy, cz, depth_z):
    """Paint an ellipsoidal channel section into a 3D mask volume in place."""
    nx, ny, nz = mask.shape
    # thickness of the cross-section — channel has an approximately U-shaped
    # profile in Z, so depth varies across width.
    rx = max(1, int(round(width_xy / 2)))
    rz = max(1, int(round(depth_z)))
    # Bounding box around (cx, cy)
    x0, x1 = max(0, cx - rx - 1), min(nx, cx + rx + 2)
    y0, y1 = max(0, cy - rx - 1), min(ny, cy + rx + 2)
    z0, z1 = max(0, cz - rz), min(nz, cz + rz + 1)
    for xi in range(x0, x1):
        for yi in range(y0, y1):
            dx = xi - cx
            dy = yi - cy
            r2 = dx * dx + dy * dy
            if r2 > rx * rx:
                continue
            # U-shaped depth: shallower at the edges, deepest on the thalweg
            depth_here = int(round(rz * np.sqrt(1.0 - r2 / (rx * rx))))
            zz0 = max(z0, cz - depth_here)
            zz1 = min(z1, cz + 1)
            if zz1 > zz0:
                mask[xi, yi, zz0:zz1] = 1.0


def generate_channel_overlays(
    cube_shape,
    age_volume,
    seed=None,
    n_channels_range=(1, 3),
    width_range=(8, 22),
    depth_range=(3, 8),
):
    """Generate a binary channel mask + per-channel metadata.

    Parameters
    ----------
    cube_shape : tuple (nx, ny, nz)
        Shape of the faulted lithology cube.
    age_volume : np.ndarray (nx, ny, nz)
        Integer-like geologic age cube used to pick a stratigraphic layer for
        each channel.
    seed : int or None
        Optional seed for reproducibility.
    n_channels_range : (int, int)
        Inclusive range for how many channels to generate.
    width_range, depth_range : (int, int)
        Channel dimensions in samples.

    Returns
    -------
    mask : np.ndarray of bool, shape cube_shape
        True where channel-fill sand should be placed.
    specs : list[ChannelSpec]
        Summary of each channel generated, for logging.
    """
    rng = np.random.default_rng(seed)
    nx, ny, nz = cube_shape
    mask = np.zeros(cube_shape, dtype=bool)
    specs = []

    n_channels = int(rng.integers(n_channels_range[0], n_channels_range[1] + 1))

    # Usable age range: exclude the shallowest water/seabed layers and the
    # deepest basement layers so channels sit in sensible stratigraphic slots.
    age_int = age_volume.astype(int)
    valid_ages = np.unique(age_int)
    valid_ages = valid_ages[(valid_ages > 2) & (valid_ages < valid_ages.max() - 2)]
    if valid_ages.size == 0:
        return mask, specs

    # Pre-pick per-channel stratigraphic layers spread across the column.
    chosen_ages = rng.choice(valid_ages, size=n_channels, replace=False) if len(valid_ages) >= n_channels else rng.choice(valid_ages, size=n_channels, replace=True)

    for layer_age in chosen_ages:
        # Find the mean Z-position for this age layer so the channel follows
        # the stratigraphy (rather than sitting at a fixed Z).
        zs = np.where(age_int == int(layer_age))
        if zs[0].size == 0:
            continue
        cz = int(np.median(zs[2]))

        # Build the centerline across the model.
        azimuth_deg = float(rng.uniform(-30.0, 30.0))
        use_inline_axis = rng.random() < 0.5  # run along X or Y
        length_axis = nx if use_inline_axis else ny
        cross_axis = ny if use_inline_axis else nx

        width_xy = int(rng.integers(width_range[0], width_range[1] + 1))
        depth_z = int(rng.integers(depth_range[0], depth_range[1] + 1))
        # Amplitude of meandering ~ 10-25% of cross-axis size.
        amplitude = rng.uniform(0.10, 0.25) * cross_axis
        harmonics = int(rng.integers(2, 5))

        offsets = _sinuous_centerline(length_axis, amplitude, harmonics, rng)
        # Tilt the whole belt by azimuth.
        tilt = np.tan(np.radians(azimuth_deg)) * (np.arange(length_axis) - length_axis / 2)
        y0 = cross_axis / 2 + rng.uniform(-cross_axis * 0.15, cross_axis * 0.15)
        centerline = y0 + offsets + tilt

        for i in range(length_axis):
            cy = int(round(centerline[i]))
            if cy < 0 or cy >= cross_axis:
                continue
            if use_inline_axis:
                _stamp_channel_into_mask(mask, i, cy, width_xy, cz, depth_z)
            else:
                _stamp_channel_into_mask(mask, cy, i, width_xy, cz, depth_z)

        kind = "submarine" if cz > nz * 0.55 else "fluvial"
        specs.append(
            ChannelSpec(
                kind=kind,
                layer_index=int(layer_age),
                width_samples=width_xy,
                depth_samples=depth_z,
                sinuosity_harmonics=harmonics,
                azimuth_deg=azimuth_deg,
            )
        )

    return mask, specs


def apply_channels_to_geomodel(faults, cfg):
    """Modify faults.faulted_lithology and faulted_net_to_gross to insert channels.

    Channels are painted as sand (lithology=1, net_to_gross=1.0) so the
    downstream elastic-property builder naturally assigns sand Vp/Vs/rho
    to the channel voxels and the resulting seismic shows a bright channel
    reflection.
    """
    if not getattr(cfg, "include_channels", False):
        return []

    lith = faults.faulted_lithology[:]
    n2g = faults.faulted_net_to_gross[:]
    age = faults.faulted_age_volume[:]

    mask, specs = generate_channel_overlays(lith.shape, age)

    if mask.any():
        faults.faulted_lithology[mask] = 1.0
        faults.faulted_net_to_gross[mask] = 1.0

    if cfg.verbose:
        for s in specs:
            print(
                f"\t... inserted {s.kind} channel in layer {s.layer_index}: "
                f"width={s.width_samples} samples, depth={s.depth_samples} samples, "
                f"harmonics={s.sinuosity_harmonics}, azimuth={s.azimuth_deg:.1f} deg"
            )
        print(f"\tChannel voxels painted: {int(mask.sum())}")

    cfg.write_to_logfile(
        f"channels_inserted: {len(specs)}",
        mainkey="model_parameters",
        subkey="channels_inserted",
        val=len(specs),
    )
    return specs
