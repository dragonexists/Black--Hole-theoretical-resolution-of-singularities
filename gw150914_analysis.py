#!/usr/bin/env python3
"""
GW150914 LOSC analysis with GWpy.

Requirements:
  pip install gwpy numpy scipy matplotlib

This script:
  1) Downloads 32 s of H1/L1 strain centered on GW150914 merger time.
  2) Applies 15 Hz high-pass, 15-200 Hz band-pass, and 60 Hz notch.
  3) Computes Nair-Einstein logarithmic time delay.
  4) Searches for a ~20 Hz resonance plateau after merger.
  5) Cross-correlates H1/L1 and estimates SNR.
  6) Runs a Monte Carlo null test with 50 randomized non-event segments.

Example:
  python gw150914_analysis.py --gps 1185389807.3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries
from gwpy.table import EventTable
from scipy.signal import iirnotch, filtfilt
from scipy.special import erfcinv


GW150914_GPS = 1126259462.4  # Standard merger GPS time
PLANCK_LENGTH_M = 1.6e-35
SOLAR_MASS_KG = 1.98847e30
G = 6.67430e-11
C = 299_792_458.0

# Frozen pipeline defaults (do not change once frozen)
FROZEN = {
    "duration": 32.0,
    "bandpass_low": 15.0,
    "bandpass_high": 200.0,
    "notch_hz": 60.0,
    "echo_post_start": 0.05,
    "echo_post_dur": 0.2,
    "echo_search_start": 0.0,
    "echo_search_dur": 0.3,
    "echo_pre_start": -0.25,
    "echo_pre_dur": 0.2,
    "scan_fmin": 10.0,
    "scan_fmax": 50.0,
    "scan_step": 0.5,
    "scan_width": 2.0,
    "corr_center_offset": 0.054,
    "corr_window": 0.02,
    "lag_limit_s": 0.01,
    "null_trials": 10000,
    "offset_scan_count": 200,
    "offset_scan_jitter": 0.02,
    "fake_injections": 1000,
    "mass_min": 20.0,
    "mass_max": 40.0,
    "event_limit": 90,
}

# Pre-registered config (hash-locked). Do not edit unless you intend to re-freeze.
PREREG = {
    "version": 1,
    "frozen": FROZEN,
    "pipelines": [
        {"name": "whiten_bandpass_notch", "whiten": True, "bandpass": True, "notch": True},
        {"name": "bandpass_notch", "whiten": False, "bandpass": True, "notch": True},
        {"name": "whiten_only", "whiten": True, "bandpass": False, "notch": False},
    ],
    "look_elsewhere": {
        "fmin": 10.0,
        "fmax": 50.0,
        "fstep": 0.5,
        "band_width": 2.0,
        "tmin": 0.0,
        "tmax": 0.3,
        "tstep": 0.01,
    },
    "holdout": {"seed": 314159, "fraction": 0.3},
    "blind_injection": {"seed": 2026, "count": 1000},
}
PREREG_HASH = "91206619402d916c52fa215d2e9136e9a8b9357fecc460d3d9f4ebbd57f20366"


def prereg_hash() -> str:
    """Stable hash for the prereg config."""
    payload = json.dumps(PREREG, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class AnalysisResult:
    dt_log: float
    xcorr_peak: float
    xcorr_lag_s: float
    snr: float
    echo_plateau: float
    mc_pvalue: float
    mc_sigma: float


def nair_einstein_log_time_delay(mass_solar: float = 30.0) -> float:
    """Compute Δt = (2 r_s / c) * ln(r_s / Δr)."""
    m_kg = mass_solar * SOLAR_MASS_KG
    r_s = 2.0 * G * m_kg / (C ** 2)
    return (2.0 * r_s / C) * math.log(r_s / PLANCK_LENGTH_M)


def estimate_echo_time(
    ts: TimeSeries,
    t0: float,
    f_center: float = 20.0,
    width_hz: float = 2.0,
    search_start: float = 0.0,
    search_dur: float = 0.3,
    rms_window_s: float = 0.02,
) -> float:
    """
    Estimate echo time by finding max RMS of a narrow-band (f_center) signal
    within a post-merger search window. Returns offset (s) from merger.
    """
    start = t0 + search_start
    end = start + search_dur
    band = ts.crop(start, end).bandpass(f_center - width_hz / 2.0, f_center + width_hz / 2.0)
    data = band.value
    fs = band.sample_rate.value
    w = max(1, int(rms_window_s * fs))
    # Rolling RMS
    sq = data ** 2
    kernel = np.ones(w) / w
    rms = np.sqrt(np.convolve(sq, kernel, mode="same"))
    idx = int(np.argmax(rms))
    t_offset = (idx / fs) + search_start
    return float(t_offset)


def apply_notch_60hz(ts: TimeSeries, q: float = 30.0) -> TimeSeries:
    """Apply a 60 Hz notch filter using scipy."""
    fs = ts.sample_rate.value
    w0 = FROZEN["notch_hz"] / (fs / 2.0)
    b, a = iirnotch(w0, q)
    data = filtfilt(b, a, ts.value)
    return TimeSeries(data, sample_rate=ts.sample_rate, t0=ts.t0, unit=ts.unit)


def filter_strain(ts: TimeSeries) -> TimeSeries:
    """Whiten, band-pass 15-200 Hz, and 60 Hz notch."""
    # Whitening suppresses low-frequency rumble so ~20 Hz features are clearer.
    white = ts.whiten(fftlength=4, overlap=2)
    bp = white.bandpass(FROZEN["bandpass_low"], FROZEN["bandpass_high"])
    return apply_notch_60hz(bp)


def apply_pipeline(ts: TimeSeries, pipeline: dict) -> TimeSeries:
    """Apply a pre-registered pipeline variant."""
    out = ts
    if pipeline.get("whiten"):
        out = out.whiten(fftlength=4, overlap=2)
    if pipeline.get("bandpass"):
        out = out.bandpass(FROZEN["bandpass_low"], FROZEN["bandpass_high"])
    if pipeline.get("notch"):
        out = apply_notch_60hz(out)
    return out


def get_pipelines(selection: str) -> list[dict]:
    """Select pre-registered pipelines by name or return all."""
    if selection == "all":
        return PREREG["pipelines"]
    names = {n.strip() for n in selection.split(",") if n.strip()}
    return [p for p in PREREG["pipelines"] if p["name"] in names]


def cross_correlation_snr(
    h1: TimeSeries, l1: TimeSeries, lag_limit_s: float = 0.01
) -> tuple[float, float, float]:
    """
    Cross-correlate two time series and compute a simple SNR.
    Relative correlation SNR = peak / std of off-peak correlation.
    Returns (peak, lag_seconds, snr).
    """
    x = h1.value - np.mean(h1.value)
    y = l1.value - np.mean(l1.value)
    corr = np.correlate(x, y, mode="full")
    lags = np.arange(-len(x) + 1, len(x)) / h1.sample_rate.value
    if lag_limit_s is not None:
        mask = np.abs(lags) <= lag_limit_s
        corr = corr[mask]
        lags = lags[mask]
    peak_idx = int(np.argmax(np.abs(corr)))
    peak = float(corr[peak_idx])
    lag_s = float(lags[peak_idx])
    # Off-peak noise estimate: exclude +/- 10 ms around peak
    exclude = 0.01
    mask = np.abs(lags - lag_s) > exclude
    noise_std = float(np.std(corr[mask]))
    snr = peak / noise_std if noise_std > 0 else float("nan")
    return peak, lag_s, snr


def echo_resonance_plateau(ts: TimeSeries, t0: float) -> tuple[float, float, float]:
    """
    Search for ~20 Hz resonance plateau starting ~0.05 s post-merger.
    Returns (post_power, pre_power, ratio) for 18-22 Hz band.
    """
    post_start = t0 + FROZEN["echo_post_start"]
    post_end = post_start + FROZEN["echo_post_dur"]
    pre_start = t0 + FROZEN["echo_pre_start"]
    pre_end = pre_start + FROZEN["echo_pre_dur"]

    post = ts.crop(post_start, post_end).bandpass(18, 22)
    pre = ts.crop(pre_start, pre_end).bandpass(18, 22)

    post_power = float(np.mean(post.value ** 2))
    pre_power = float(np.mean(pre.value ** 2))
    ratio = post_power / pre_power if pre_power > 0 else float("nan")
    return post_power, pre_power, ratio


def monte_carlo_null_test(
    h1: TimeSeries,
    l1: TimeSeries,
    event_start: float,
    event_end: float,
    n_trials: int = 1000,
    rng_seed: int = 42,
) -> tuple[float, float]:
    """
    Compare event window correlation peak against randomized non-event segments.
    Returns (pvalue, sigma_equivalent).
    """
    rng = random.Random(rng_seed)
    event_h1 = h1.crop(event_start, event_end)
    event_l1 = l1.crop(event_start, event_end)
    event_peak, _, _ = cross_correlation_snr(event_h1, event_l1)
    event_peak = abs(event_peak)

    total_duration = h1.duration.value
    window = event_end - event_start

    peaks = []
    for _ in range(n_trials):
        # Choose a random start within the available data, excluding event window
        while True:
            offset = rng.uniform(0, total_duration - window)
            start = h1.t0.value + offset
            end = start + window
            if end < event_start or start > event_end:
                break
        seg_h1 = h1.crop(start, end)
        seg_l1 = l1.crop(start, end)
        peak, _, _ = cross_correlation_snr(seg_h1, seg_l1)
        peaks.append(abs(peak))

    peaks = np.array(peaks)
    # Use a finite-sample correction to avoid p=0 -> sigma=inf
    k = int(np.sum(peaks >= event_peak))
    pvalue = float((k + 1) / (n_trials + 1))
    # Convert p-value to sigma assuming normal tail
    sigma = math.sqrt(2) * erfcinv(2 * pvalue)
    return pvalue, sigma


def band_power(ts: TimeSeries, f_center: float, width_hz: float = 2.0) -> float:
    """Compute mean band power around f_center with total width width_hz."""
    half = width_hz / 2.0
    band = ts.bandpass(f_center - half, f_center + half)
    return float(np.mean(band.value ** 2))


def scan_band_power(
    ts: TimeSeries, fmin: float = 10.0, fmax: float = 50.0, step: float = 0.5, width_hz: float = 2.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Scan band power over frequency and return (freqs, powers)."""
    freqs = np.arange(fmin, fmax + 1e-6, step)
    powers = np.array([band_power(ts, f, width_hz=width_hz) for f in freqs])
    return freqs, powers


def fake_time_injection_test(
    h1: TimeSeries,
    l1: TimeSeries,
    gps: float,
    window: float,
    n_injections: int = 1000,
    rng_seed: int = 777,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Inject fake merger times and compute relative correlation SNR around each.
    Returns (offsets, snrs).
    """
    rng = random.Random(rng_seed)
    total_duration = h1.duration.value
    offsets = []
    snrs = []
    for _ in range(n_injections):
        # Pick a random center within available data
        offset = rng.uniform(-total_duration / 2.0 + 1.0, total_duration / 2.0 - 1.0)
        center = gps + offset
        start = center - window / 2.0
        end = center + window / 2.0
        h1_c = h1.crop(start, end)
        l1_c = l1.crop(start, end)
        _, _, snr = cross_correlation_snr(h1_c, l1_c, lag_limit_s=FROZEN["lag_limit_s"])
        offsets.append(offset)
        snrs.append(snr)
    return np.array(offsets), np.array(snrs)


def look_elsewhere_scan(
    ts: TimeSeries,
    t0: float,
    fmin: float,
    fmax: float,
    fstep: float,
    band_width: float,
    tmin: float,
    tmax: float,
    tstep: float,
) -> Tuple[float, Tuple[float, float]]:
    """
    Scan time-frequency grid and return (max_power, (t_offset, f_center)).
    """
    best = (-np.inf, (None, None))
    t_offsets = np.arange(tmin, tmax + 1e-9, tstep)
    f_centers = np.arange(fmin, fmax + 1e-9, fstep)
    for toff in t_offsets:
        seg = ts.crop(t0 + toff, t0 + toff + FROZEN["echo_post_dur"])
        for f in f_centers:
            p = band_power(seg, f_center=f, width_hz=band_width)
            if p > best[0]:
                best = (p, (toff, f))
    return float(best[0]), (float(best[1][0]), float(best[1][1]))


def look_elsewhere_pvalue(
    ts: TimeSeries,
    t0: float,
    n_trials: int,
    rng_seed: int = 999,
) -> Tuple[float, float, Tuple[float, float]]:
    """
    Compute look-elsewhere corrected p-value using max-statistic over grid.
    Returns (pvalue, sigma_equiv, (best_t, best_f)).
    """
    le = PREREG["look_elsewhere"]
    obs_max, best = look_elsewhere_scan(
        ts,
        t0,
        le["fmin"],
        le["fmax"],
        le["fstep"],
        le["band_width"],
        le["tmin"],
        le["tmax"],
        le["tstep"],
    )

    rng = random.Random(rng_seed)
    total_duration = ts.duration.value
    window = FROZEN["echo_post_dur"]
    null_max = []
    for _ in range(n_trials):
        offset = rng.uniform(0, total_duration - window)
        start = ts.t0.value + offset
        seg = ts.crop(start, start + window)
        # Use same frequency scan on random segment
        freqs = np.arange(le["fmin"], le["fmax"] + 1e-9, le["fstep"])
        powers = [band_power(seg, f_center=f, width_hz=le["band_width"]) for f in freqs]
        null_max.append(max(powers))

    null_max = np.array(null_max)
    k = int(np.sum(null_max >= obs_max))
    pvalue = float((k + 1) / (n_trials + 1))
    sigma = math.sqrt(2) * erfcinv(2 * pvalue)
    return pvalue, sigma, best


def blind_injection_times(
    t0: float, duration: float, count: int, rng_seed: int
) -> np.ndarray:
    """Generate blinded fake injection offsets."""
    rng = random.Random(rng_seed)
    offsets = [rng.uniform(-duration / 2 + 1.0, duration / 2 - 1.0) for _ in range(count)]
    return np.array(offsets)


def scan_offsets(
    h1: TimeSeries,
    l1: TimeSeries,
    gps: float,
    base_offset: float,
    window: float,
    lag_limit_s: float = 0.01,
    n_offsets: int = 200,
    jitter_s: float = 0.02,
    rng_seed: int = 123,
) -> Tuple[np.ndarray, np.ndarray]:
    """Scan random offsets around base_offset and return offsets vs relative correlation SNR."""
    rng = random.Random(rng_seed)
    offsets = np.array([base_offset + rng.uniform(-jitter_s, jitter_s) for _ in range(n_offsets)])
    snrs = []
    for off in offsets:
        center = gps + off
        start = center - window / 2.0
        end = center + window / 2.0
        h1_c = h1.crop(start, end)
        l1_c = l1.crop(start, end)
        _, _, snr = cross_correlation_snr(h1_c, l1_c, lag_limit_s=lag_limit_s)
        snrs.append(snr)
    return offsets, np.array(snrs)


def plot_offset_scan(offsets: np.ndarray, snrs: np.ndarray, out_path: str) -> None:
    """Plot relative correlation SNR vs time offsets."""
    plt.figure(figsize=(7, 4))
    plt.scatter(offsets, snrs, s=10)
    plt.axvline(0.054, color="r", linestyle="--", label="0.054 s")
    plt.xlabel("Offset from merger (s)")
    plt.ylabel("Relative correlation SNR")
    plt.title("Offset Scan (±20 ms)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)


def plot_psd(raw: TimeSeries, filt: TimeSeries, out_path: str) -> None:
    """Plot PSD before/after filtering."""
    raw_psd = raw.psd(fftlength=4)
    filt_psd = filt.psd(fftlength=4)
    plt.figure(figsize=(6, 4))
    plt.loglog(raw_psd.frequencies.value, raw_psd.value, label="Raw")
    plt.loglog(filt_psd.frequencies.value, filt_psd.value, label="Filtered")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD")
    plt.title("PSD Before/After Filtering")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)


def plot_strain(raw: TimeSeries, filt: TimeSeries, t0: float, out_path: str) -> None:
    """Plot raw vs filtered strain in a short window around merger."""
    start = t0 - 0.2
    end = t0 + 0.2
    r = raw.crop(start, end)
    f = filt.crop(start, end)
    t = r.times.value - t0
    plt.figure(figsize=(8, 4))
    plt.plot(t, r.value, label="Raw", alpha=0.7)
    plt.plot(t, f.value, label="Filtered", alpha=0.7)
    plt.xlabel("Time (s) relative to merger")
    plt.ylabel("Strain")
    plt.title("Raw vs Filtered Strain (±0.2 s)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)


def plot_frequency_scan(freqs: np.ndarray, powers: np.ndarray, out_path: str) -> None:
    """Plot band power vs frequency for the scan."""
    plt.figure(figsize=(8, 4))
    plt.plot(freqs, powers)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Band Power")
    plt.title("Band Power Scan (10–50 Hz)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)



def _safe_column(events: EventTable, name: str):
    return name if name in events.colnames else None


def select_events_by_mass(
    min_mass: float = FROZEN["mass_min"],
    max_mass: float = FROZEN["mass_max"],
    limit: int = FROZEN["event_limit"],
    catalogs: Iterable[str] = ("GWTC-1-confident", "GWTC-2", "GWTC-3-confident", "GWTC-4.0"),
) -> list[dict]:
    """
    Select events from GWTC with component masses in [min_mass, max_mass].
    Returns a list of dicts: {name, gps, m1, m2}.
    """
    all_rows: list[dict] = []
    for catalog in catalogs:
        events = EventTable.fetch_open_data(catalog)
        m1_col = _safe_column(events, "mass_1_source")
        m2_col = _safe_column(events, "mass_2_source")
        if m1_col is None or m2_col is None:
            print(f"Warning: catalog {catalog} missing mass_1_source/mass_2_source; skipping.")
            continue
        # Columns may contain units; access via .to_value for filtering.
        m1 = events[m1_col].to_value()
        m2 = events[m2_col].to_value()
        mask = (m1 >= min_mass) & (m1 <= max_mass) & (m2 >= min_mass) & (m2 <= max_mass)
        filtered = events[mask]
        for row in filtered:
            all_rows.append(
                {
                    "name": str(row["name"]),
                    "gps": float(row["GPS"]),
                    "m1": float(row[m1_col].value),
                    "m2": float(row[m2_col].value),
                    "catalog": catalog,
                }
            )

    # De-duplicate by event name, keep earliest GPS
    by_name = {}
    for row in all_rows:
        name = row["name"]
        if name not in by_name or row["gps"] < by_name[name]["gps"]:
            by_name[name] = row

    selected = sorted(by_name.values(), key=lambda r: r["gps"])[:limit]
    return list(selected)


def plot_noise_vs_signal(events: Iterable[dict], out_path: str, pipeline: dict) -> None:
    """
    Generate a plot comparing pre-event noise vs 20 Hz post-event power.
    """
    names = []
    pre_vals = []
    post_vals = []
    ratios = []

    for ev in events:
        gps = ev["gps"]
        start = gps - 16
        end = gps + 16
        h1 = TimeSeries.fetch_open_data("H1", start, end, cache=True)
        h1_f = apply_pipeline(h1, pipeline)
        post_p, pre_p, ratio = echo_resonance_plateau(h1_f, gps)
        names.append(ev["name"])
        pre_vals.append(pre_p)
        post_vals.append(post_p)
        ratios.append(ratio)

    x = np.arange(len(names))
    plt.figure(figsize=(12, 5))
    plt.plot(x, pre_vals, label="Pre-event noise (18-22 Hz)")
    plt.plot(x, post_vals, label="Post-event 20 Hz band power")
    plt.xticks(x, names, rotation=90, fontsize=6)
    plt.ylabel("Band power")
    plt.title("Noise vs 20 Hz Signal (H1, 0.05-0.25 s post-merger)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)

    plt.figure(figsize=(12, 3.5))
    plt.plot(x, ratios, label="Post/Pre Power Ratio")
    plt.xticks(x, names, rotation=90, fontsize=6)
    plt.ylabel("Ratio")
    plt.title("20 Hz Post/Pre Power Ratio")
    plt.legend()
    plt.tight_layout()
    ratio_path = out_path.replace(".png", "_ratio.png")
    plt.savefig(ratio_path, dpi=200)


def main() -> None:
    parser = argparse.ArgumentParser(description="GW150914 LOSC analysis with GWpy.")
    parser.add_argument("--gps", type=float, default=GW150914_GPS, help="Merger GPS time.")
    parser.add_argument("--duration", type=float, default=FROZEN["duration"], help="Total duration (s).")
    parser.add_argument("--null-trials", type=int, default=FROZEN["null_trials"], help="Null test trials.")
    parser.add_argument("--multi-event", action="store_true", help="Analyze 90 events and plot.")
    parser.add_argument("--plot-out", type=str, default="noise_vs_20hz.png", help="Plot output path.")
    parser.add_argument(
        "--multi-out",
        type=str,
        default="multi_event_summary.csv",
        help="CSV summary for multi-event analysis.",
    )
    parser.add_argument(
        "--catalogs",
        type=str,
        default="GWTC-1-confident,GWTC-2,GWTC-3-confident,GWTC-4.0",
        help="Comma-separated GWOSC catalogs (include GWTC-4.0 for O4).",
    )
    parser.add_argument(
        "--corr-center-offset",
        type=float,
        default=FROZEN["corr_center_offset"],
        help="Correlation window center offset from merger (s).",
    )
    parser.add_argument(
        "--corr-window",
        type=float,
        default=FROZEN["corr_window"],
        help="Correlation window duration (s).",
    )
    parser.add_argument(
        "--scan-out",
        type=str,
        default="band_scan_10_50hz.png",
        help="Output plot for band-power scan.",
    )
    parser.add_argument(
        "--scan-out-l1",
        type=str,
        default="band_scan_10_50hz_l1.png",
        help="Output plot for band-power scan (L1).",
    )
    parser.add_argument(
        "--psd-out",
        type=str,
        default="psd_before_after.png",
        help="Output plot for PSD before/after filtering.",
    )
    parser.add_argument(
        "--strain-out",
        type=str,
        default="strain_raw_filtered.png",
        help="Output plot for raw vs filtered strain.",
    )
    parser.add_argument(
        "--offset-scan-out",
        type=str,
        default="offset_scan.png",
        help="Output plot for random offset scan.",
    )
    parser.add_argument(
        "--offset-scan-count",
        type=int,
        default=FROZEN["offset_scan_count"],
        help="Number of random offsets to scan.",
    )
    parser.add_argument(
        "--offset-scan-jitter",
        type=float,
        default=FROZEN["offset_scan_jitter"],
        help="Random jitter range around base offset (s).",
    )
    parser.add_argument(
        "--nearest-event",
        action="store_true",
        help="Also run scans for the nearest-in-time event in the mass range.",
    )
    parser.add_argument(
        "--fake-injections",
        type=int,
        default=FROZEN["fake_injections"],
        help="Number of fake merger time injections.",
    )
    parser.add_argument(
        "--fake-out",
        type=str,
        default="fake_time_scan.png",
        help="Output plot for fake merger time injections.",
    )
    parser.add_argument(
        "--frozen",
        action="store_true",
        help="Freeze pipeline parameters to fixed values.",
    )
    parser.add_argument(
        "--strict-prereg",
        action="store_true",
        help="Require preregistered config hash to match before running.",
    )
    parser.add_argument(
        "--print-prereg-hash",
        action="store_true",
        help="Print preregistered config hash and exit.",
    )
    parser.add_argument(
        "--pipelines",
        type=str,
        default="all",
        help="Comma-separated pipeline names or 'all' (pre-registered only).",
    )
    parser.add_argument(
        "--holdout-only",
        action="store_true",
        help="When multi-event, run only on the holdout set.",
    )
    parser.add_argument(
        "--dev-only",
        action="store_true",
        help="When multi-event, run only on the dev set.",
    )
    parser.add_argument(
        "--look-elsewhere",
        action="store_true",
        help="Compute look-elsewhere corrected p-value on the scan grid.",
    )
    parser.add_argument(
        "--blind-injections",
        action="store_true",
        help="Run blinded fake injections using pre-registered count and seed.",
    )
    args = parser.parse_args()

    if args.print_prereg_hash:
        print(prereg_hash())
        return

    if args.strict_prereg:
        # Enforce prereg hash (freeze against edits)
        expected = PREREG_HASH
        if prereg_hash() != expected:
            raise RuntimeError("Preregistration hash mismatch.")

    if args.frozen:
        # Overwrite with frozen parameters
        args.duration = FROZEN["duration"]
        args.null_trials = FROZEN["null_trials"]
        args.corr_center_offset = FROZEN["corr_center_offset"]
        args.corr_window = FROZEN["corr_window"]
        args.offset_scan_count = FROZEN["offset_scan_count"]
        args.offset_scan_jitter = FROZEN["offset_scan_jitter"]
        args.fake_injections = FROZEN["fake_injections"]

    if args.multi_event:
        catalogs = tuple(c.strip() for c in args.catalogs.split(",") if c.strip())
        events = select_events_by_mass(catalogs=catalogs)
        if len(events) < 60:
            print(f"Warning: only found {len(events)} events in mass range 20-40.")
        # Holdout split
        rng = random.Random(PREREG["holdout"]["seed"])
        events_shuffled = list(events)
        rng.shuffle(events_shuffled)
        split_idx = int(len(events_shuffled) * (1 - PREREG["holdout"]["fraction"]))
        dev_events = events_shuffled[:split_idx]
        holdout_events = events_shuffled[split_idx:]
        if args.holdout_only:
            events = holdout_events
        elif args.dev_only:
            events = dev_events
        else:
            # Default to holdout only to minimize bias
            events = holdout_events

        pipelines = get_pipelines(args.pipelines)
        if not pipelines:
            raise RuntimeError("No valid pipelines selected.")
        print(f"Generating plot for {len(events)} events...")
        # Use first pipeline for summary plots to avoid duplication
        plot_noise_vs_signal(events, args.plot_out, pipelines[0])
        print(f"Wrote plot: {args.plot_out}")
        # Run null tests for each event and write a CSV summary
        import csv

        with open(args.multi_out, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["name", "gps", "m1", "m2", "pipeline", "pvalue", "sigma"])
            for ev in events:
                gps = ev["gps"]
                start = gps - 16
                end = gps + 16
                h1 = TimeSeries.fetch_open_data("H1", start, end, cache=True)
                l1 = TimeSeries.fetch_open_data("L1", start, end, cache=True)
                for p in pipelines:
                    h1_f = apply_pipeline(h1, p)
                    l1_f = apply_pipeline(l1, p)
                    pvalue, sigma = monte_carlo_null_test(
                        h1_f, l1_f, gps - 0.2, gps + 0.2, n_trials=args.null_trials
                    )
                    writer.writerow([ev["name"], gps, ev["m1"], ev["m2"], p["name"], pvalue, sigma])
        print(f"Wrote multi-event summary: {args.multi_out}")
        return

    half = args.duration / 2.0
    start = args.gps - half
    end = args.gps + half

    print(f"Fetching LOSC data: {start} to {end} (GPS)")
    h1 = TimeSeries.fetch_open_data("H1", start, end, cache=True)
    l1 = TimeSeries.fetch_open_data("L1", start, end, cache=True)

    pipelines = get_pipelines(args.pipelines)
    if not pipelines:
        raise RuntimeError("No valid pipelines selected.")

    for p in pipelines:
        print(f"Running pipeline: {p['name']}")
        h1_f = apply_pipeline(h1, p)
        l1_f = apply_pipeline(l1, p)

        print("Plotting PSD and strain (raw vs filtered)...")
        plot_psd(h1, h1_f, f"{p['name']}_{args.psd_out}")
        plot_strain(h1, h1_f, args.gps, f"{p['name']}_{args.strain_out}")

        dt_log = nair_einstein_log_time_delay(30.0)
        print(f"Logarithmic time delay Δt: {dt_log:.6e} s")

        post_p, pre_p, ratio = echo_resonance_plateau(h1_f, args.gps)
        print(
            "20 Hz plateau power (H1, 0.05-0.25 s post-merger): "
            f"{post_p:.6e}, pre: {pre_p:.6e}, ratio: {ratio:.3f}"
        )

        print("Scanning 10–50 Hz band power (post-merger window)...")
        scan_window = h1_f.crop(
            args.gps + FROZEN["echo_post_start"],
            args.gps + FROZEN["echo_post_start"] + FROZEN["echo_post_dur"],
        )
        freqs, powers = scan_band_power(
            scan_window,
            fmin=FROZEN["scan_fmin"],
            fmax=FROZEN["scan_fmax"],
            step=FROZEN["scan_step"],
            width_hz=FROZEN["scan_width"],
        )
        plot_frequency_scan(freqs, powers, f"{p['name']}_{args.scan_out}")
        print(f"Wrote band scan plot: {p['name']}_{args.scan_out}")
        scan_window_l1 = l1_f.crop(
            args.gps + FROZEN["echo_post_start"],
            args.gps + FROZEN["echo_post_start"] + FROZEN["echo_post_dur"],
        )
        freqs_l1, powers_l1 = scan_band_power(
            scan_window_l1,
            fmin=FROZEN["scan_fmin"],
            fmax=FROZEN["scan_fmax"],
            step=FROZEN["scan_step"],
            width_hz=FROZEN["scan_width"],
        )
        plot_frequency_scan(freqs_l1, powers_l1, f"{p['name']}_{args.scan_out_l1}")
        print(f"Wrote band scan plot (L1): {p['name']}_{args.scan_out_l1}")

        corr_center = args.gps + args.corr_center_offset
        corr_start = corr_center - args.corr_window / 2.0
        corr_end = corr_center + args.corr_window / 2.0
        h1_corr = h1_f.crop(corr_start, corr_end)
        l1_corr = l1_f.crop(corr_start, corr_end)
        peak, lag_s, snr = cross_correlation_snr(
            h1_corr, l1_corr, lag_limit_s=FROZEN["lag_limit_s"]
        )
        print(
            f"Cross-correlation peak: {peak:.6e}, lag: {lag_s:+.6f} s, "
            f"relative correlation SNR: {snr:.2f}"
        )

        offsets, snrs = scan_offsets(
            h1_f,
            l1_f,
            args.gps,
            base_offset=args.corr_center_offset,
            window=args.corr_window,
            lag_limit_s=FROZEN["lag_limit_s"],
            n_offsets=args.offset_scan_count,
            jitter_s=args.offset_scan_jitter,
        )
        plot_offset_scan(offsets, snrs, f"{p['name']}_{args.offset_scan_out}")
        print(f"Wrote offset scan plot: {p['name']}_{args.offset_scan_out}")

        if args.blind_injections:
            fake_offsets, fake_snrs = fake_time_injection_test(
                h1_f,
                l1_f,
                args.gps,
                window=args.corr_window,
                n_injections=args.fake_injections,
            )
            plot_offset_scan(
                fake_offsets + args.corr_center_offset,
                fake_snrs,
                f"{p['name']}_{args.fake_out}",
            )
            print(f"Wrote fake time injection plot: {p['name']}_{args.fake_out}")

        pvalue, sigma = monte_carlo_null_test(
            h1_f, l1_f, args.gps - 0.2, args.gps + 0.2, n_trials=args.null_trials
        )
        print(
            f"Monte Carlo null test (n={args.null_trials}): p={pvalue:.4f}, "
            f"sigma-equivalent under null≈{sigma:.2f}"
        )

        if args.look_elsewhere:
            le_p, le_sigma, (best_t, best_f) = look_elsewhere_pvalue(
                h1_f, args.gps, n_trials=args.null_trials
            )
            print(
                "Look-elsewhere correction: "
                f"p={le_p:.4f}, sigma≈{le_sigma:.2f}, best_t={best_t:.3f}s, best_f={best_f:.2f}Hz"
            )

    if args.nearest_event:
        catalogs = tuple(c.strip() for c in args.catalogs.split(",") if c.strip())
        events = select_events_by_mass(catalogs=catalogs)
        if events:
            # find nearest event in time excluding current gps
            events = [e for e in events if abs(e["gps"] - args.gps) > 1.0]
            if events:
                nearest = min(events, key=lambda e: abs(e["gps"] - args.gps))
                print(
                    f"Nearest event: {nearest['name']} at GPS {nearest['gps']} "
                    f"(m1={nearest['m1']:.1f}, m2={nearest['m2']:.1f})"
                )
                start = nearest["gps"] - 16
                end = nearest["gps"] + 16
                h1n = TimeSeries.fetch_open_data("H1", start, end, cache=True)
                l1n = TimeSeries.fetch_open_data("L1", start, end, cache=True)
                # Use first selected pipeline for nearest-event comparison
                p0 = get_pipelines(args.pipelines)[0]
                h1n_f = apply_pipeline(h1n, p0)
                l1n_f = apply_pipeline(l1n, p0)
                # Echo timing vs mass scaling
                total_mass = nearest["m1"] + nearest["m2"]
                predicted_dt = nair_einstein_log_time_delay(total_mass)
                echo_offset_h1 = estimate_echo_time(
                    h1n_f,
                    nearest["gps"],
                    f_center=20.0,
                    width_hz=FROZEN["scan_width"],
                    search_start=FROZEN["echo_search_start"],
                    search_dur=FROZEN["echo_search_dur"],
                )
                echo_offset_l1 = estimate_echo_time(
                    l1n_f,
                    nearest["gps"],
                    f_center=20.0,
                    width_hz=FROZEN["scan_width"],
                    search_start=FROZEN["echo_search_start"],
                    search_dur=FROZEN["echo_search_dur"],
                )
                print(
                    "Echo timing vs mass (nearest event): "
                    f"predicted Δt={predicted_dt:.6e}s, "
                    f"H1 echo≈{echo_offset_h1:.3f}s, L1 echo≈{echo_offset_l1:.3f}s"
                )
                scan_window_n = h1n_f.crop(
                    nearest["gps"] + FROZEN["echo_post_start"],
                    nearest["gps"] + FROZEN["echo_post_start"] + FROZEN["echo_post_dur"],
                )
                freqs_n, powers_n = scan_band_power(
                    scan_window_n,
                    FROZEN["scan_fmin"],
                    FROZEN["scan_fmax"],
                    FROZEN["scan_step"],
                    FROZEN["scan_width"],
                )
                plot_frequency_scan(freqs_n, powers_n, "band_scan_nearest_h1.png")
                scan_window_n_l1 = l1n_f.crop(
                    nearest["gps"] + FROZEN["echo_post_start"],
                    nearest["gps"] + FROZEN["echo_post_start"] + FROZEN["echo_post_dur"],
                )
                freqs_n_l1, powers_n_l1 = scan_band_power(
                    scan_window_n_l1,
                    FROZEN["scan_fmin"],
                    FROZEN["scan_fmax"],
                    FROZEN["scan_step"],
                    FROZEN["scan_width"],
                )
                plot_frequency_scan(freqs_n_l1, powers_n_l1, "band_scan_nearest_l1.png")
                print("Wrote nearest-event band scans: band_scan_nearest_h1.png, band_scan_nearest_l1.png")


if __name__ == "__main__":
    main()
