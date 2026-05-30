# -----------------------------------------------------------------------------
# Simple 1D wave propagation code
# Author: Noah Day (University of Melbourne), May 2026
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd

try:
    from .attenuation_models import *
    from .parameters import GRAVITY
except ImportError:
    from attenuation_models import *
    from parameters import GRAVITY


def _as_freqs(freqs=None):
    """Return supplied frequencies or the default WW3 frequency grid."""
    if freqs is None:
        freqs, _ = get_ww3_freqs()
    return np.asarray(freqs, dtype=float)


def _attenuation(periods, thickness, floe_diameter, model="mbk_2014"):
    """Return wave-ice attenuation coefficients in m-1."""
    if model == "mbk_2014":
        return np.asarray(mbk_2014_attenuation(periods), dtype=float)
    if model == "meylan_2021":
        alpha, _ = meylan_2021_attenuation(periods, thickness, floe_diameter)
        return np.asarray(alpha, dtype=float)
    raise ValueError("model must be 'mbk_2014' or 'meylan_2021'")


def advect_spectrum_hs(
    freqs,
    S_om,
    aice=1.0,
    x_max=500e3,
    dx=1e3,
    alpha_units="m-1",
    floeDiameter=1000,
    thickness=0.5,
    attenuation_model="mbk_2014",
):
    """Advect a frequency spectrum through ice and return Hs along distance."""
    freqs = _as_freqs(freqs)
    omega = 2 * np.pi * freqs
    periods = 1.0 / freqs
    S_om = np.asarray(S_om, dtype=float)

    alpha = _attenuation(periods, thickness, floeDiameter, attenuation_model)

    if alpha_units == "km-1":
        alpha = alpha / 1000.0
        x = np.arange(0, x_max + dx, dx) * 1000.0
        x_out = x / 1000.0
    elif alpha_units == "m-1":
        x = np.arange(0, x_max + dx, dx)
        x_out = x
    else:
        raise ValueError("alpha_units must be 'm-1' or 'km-1'")

    S_x = S_om[:, None] * np.exp(-aice * alpha[:, None] * x[None, :])
    m0_x = np.trapz(S_x, omega, axis=0)
    hs_x = 4 * np.sqrt(m0_x)

    return x_out, hs_x


def distance_until_hs_below(
    hs0,
    tp,
    freqs=None,
    aice=1.0,
    floeDiameter=1000,
    thickness=0.5,
    hs_threshold=0.01,
    x_max=3000e3,
    dx=1e3,
    alpha_units="m-1",
    attenuation_model="mbk_2014",
):
    """Find the first distance where propagated Hs drops below a threshold."""
    freqs = _as_freqs(freqs)
    omega = 2 * np.pi * freqs
    S_om, _ = sdf_bretschneider(hs0, tp, omega)

    x, hs_x = advect_spectrum_hs(
        freqs=freqs,
        aice=aice,
        floeDiameter=floeDiameter,
        thickness=thickness,
        S_om=S_om,
        x_max=x_max,
        dx=dx,
        alpha_units=alpha_units,
        attenuation_model=attenuation_model,
    )

    below = np.where(hs_x < hs_threshold)[0]

    if len(below) == 0:
        return np.nan, x, hs_x

    return x[below[0]], x, hs_x


def threshold_distance_from_hs(
    hs0,
    tp=16,
    freqs=None,
    aice=1.0,
    floeDiameter=1000,
    thickness=0.5,
    attenuation_model="mbk_2014",
):
    """Return the distance in km where Hs first falls below 0.01 m."""
    if pd.isna(hs0):
        return np.nan

    distance, _, _ = distance_until_hs_below(
        hs0=hs0,
        tp=tp,
        freqs=freqs,
        aice=aice,
        floeDiameter=floeDiameter,
        thickness=thickness,
        hs_threshold=0.01,
        x_max=1000e3,
        dx=1e3,
        alpha_units="m-1",
        attenuation_model=attenuation_model,
    )

    return distance / 1000.0


def estimate_tp_swell_from_hs(
    hs,
    tp_min=8,
    tp_max=22.5,
    thickness=0.5,
    floeDiameter=1000,
):
    """Estimate swell peak period from Hs using a clipped empirical scaling."""
    hs = np.asarray(hs, dtype=float)
    tp = 6.0 * np.sqrt(hs)
    return np.clip(tp, tp_min, tp_max)


def hs_after_distance(
    hs0,
    distance_km,
    tp=10,
    freqs=None,
    direction_convention="from",
    use_southward_energy=False,
    mwd=0.0,
    aice=1.0,
    thickness=0.5,
    floeDiameter=1000,
    attenuation_model="mbk_2014",
):
    """Propagate Hs to a fixed distance through sea ice."""
    if pd.isna(hs0) or pd.isna(distance_km):
        return np.nan

    freqs = _as_freqs(freqs)
    distance_m = distance_km * 1000.0

    omega = 2 * np.pi * freqs
    periods = 1.0 / freqs
    S_om, _ = sdf_bretschneider(hs0, tp, omega)

    if use_southward_energy:
        if pd.isna(mwd):
            return np.nan

        south_frac = southward_energy_fraction_cos2(
            mwd,
            direction_convention=direction_convention,
        )

        if pd.isna(south_frac):
            return np.nan

        S_om = S_om * south_frac

    alpha = _attenuation(periods, thickness, floeDiameter, attenuation_model)
    S_at_x = S_om * np.exp(-aice * alpha * distance_m)

    m0_at_x = np.trapezoid(S_at_x, omega)
    hs_at_x = 4 * np.sqrt(m0_at_x)

    return hs_at_x


def distance_until_hs_fraction_below(
    hs0,
    tp,
    freqs=None,
    aice=1.0,
    thickness=0.5,
    floeDiameter=1000,
    fraction_threshold=0.08,
    x_max=10000e3,
    dx=1e4,
    alpha_units="m-1",
    attenuation_model="mbk_2014",
):
    """Return the distance in km where Hs/Hs0 first drops below a fraction."""
    if pd.isna(hs0) or pd.isna(tp):
        return np.nan

    freqs = _as_freqs(freqs)
    omega = 2 * np.pi * freqs
    S_om, _ = sdf_bretschneider(hs0, tp, omega)

    x, hs_x = advect_spectrum_hs(
        freqs=freqs,
        S_om=S_om,
        aice=aice,
        floeDiameter=floeDiameter,
        thickness=thickness,
        x_max=x_max,
        dx=dx,
        alpha_units=alpha_units,
        attenuation_model=attenuation_model,
    )

    hs_fraction = hs_x / hs0
    below = np.where(hs_fraction <= fraction_threshold)[0]

    if len(below) == 0:
        return np.nan

    if alpha_units == "km-1":
        return x[below[0]]
    return x[below[0]] / 1000.0


def southward_energy_fraction_cos2(
    mwd,
    s=2.5,
    direction_convention="from",
    n_dirs=720,
):
    """Estimate the fraction of directional wave energy travelling southward."""
    if pd.isna(mwd):
        return np.nan

    theta = np.linspace(0, 2 * np.pi, n_dirs, endpoint=False)
    dtheta = theta[1] - theta[0]

    mwd_rad = mwd % (2 * np.pi)

    if direction_convention == "from":
        theta0 = (mwd_rad + np.pi) % (2 * np.pi)
    elif direction_convention == "to":
        theta0 = mwd_rad
    else:
        raise ValueError("direction_convention must be 'from' or 'to'")

    delta = np.angle(np.exp(1j * (theta - theta0)))
    cos_delta = np.clip(np.cos(delta), 0, None)

    spread = np.where(
        np.abs(delta) <= np.pi / 2,
        cos_delta**s,
        0.0,
    )
    spread = spread / np.sum(spread * dtheta)

    southward = np.cos(theta) < 0
    return np.sum(spread[southward] * dtheta)


def distance_until_hs_below_target(
    hs0,
    target_hs,
    tp=10,
    freqs=None,
    mwd=None,
    direction_convention="from",
    use_southward_energy=False,
    aice=1.0,
    thickness=0.5,
    floeDiameter=1000,
    x_max=1000e3,
    dx=1e3,
    attenuation_model="mbk_2014",
):
    """Return the distance in km where propagated Hs reaches target_hs."""
    if pd.isna(hs0) or pd.isna(target_hs) or pd.isna(tp):
        return np.nan

    freqs = _as_freqs(freqs)
    omega = 2 * np.pi * freqs
    S_om, _ = sdf_bretschneider(hs0, tp, omega)

    if use_southward_energy:
        if pd.isna(mwd):
            return np.nan

        south_frac = southward_energy_fraction_cos2(
            mwd,
            direction_convention=direction_convention,
        )

        if pd.isna(south_frac):
            return np.nan

        S_om = S_om * south_frac
        hs0_eff = hs0 * np.sqrt(south_frac)
    else:
        hs0_eff = hs0

    if hs0_eff <= target_hs:
        return 0.0

    x, hs_x = advect_spectrum_hs(
        freqs=freqs,
        S_om=S_om,
        aice=aice,
        thickness=thickness,
        floeDiameter=floeDiameter,
        x_max=x_max,
        dx=dx,
        alpha_units="m-1",
        attenuation_model=attenuation_model,
    )

    below = np.where(hs_x <= target_hs)[0]

    if len(below) == 0:
        return np.nan

    return x[below[0]] / 1000.0


def sdf_bretschneider(hs, tm, omega, gravity=GRAVITY):
    """Compute a Bretschneider spectrum for Hs, mean period, and omega."""
    t = 2 * np.pi / omega
    om_m = 2 * np.pi / tm
    tau = 2 * np.pi / omega

    moment_no = 0
    f1 = (5 / 16) * (hs**2) * (om_m**4)
    f2 = omega ** (moment_no - 5)
    f3 = np.exp(-1.25 * ((tau / tm) ** 4))

    s = f1 * f2 * f3

    return s, t
