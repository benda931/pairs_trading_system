# -*- coding: utf-8 -*-
"""
core/cycle_detector.py — Wavelet & Fourier Cycle Detection
=============================================================

Frequency-domain analysis for pairs trading:

1. **FFT Spectral Analysis**
   - Power spectral density estimation
   - Dominant frequency/period detection
   - Spectral energy distribution

2. **Wavelet Decomposition**
   - Multi-resolution analysis (DWT)
   - Time-frequency energy maps
   - Scale-specific mean-reversion cycles

3. **Cycle-Based Trading Signals**
   - Dominant cycle period → optimal lookback window
   - Phase estimation → timing signals
   - Cycle strength → conviction scaling

4. **Regime-Cycle Interaction**
   - Cycle stability across regimes
   - Frequency shift detection
   - Cycle breakdown alerts

Usage:
    from core.cycle_detector import CycleDetector

    cd = CycleDetector()
    result = cd.analyze(spread_series)
    print(f"Dominant cycle: {result.dominant_period} days")
    print(f"Optimal lookback: {result.optimal_lookback}")
    print(f"Current phase: {result.current_phase}")
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SpectralPeak:
    """A peak in the power spectrum."""
    frequency: float                     # Cycles per day
    period_days: float                   # 1/frequency
    power: float                         # Spectral power
    power_pct: float                     # % of total power
    significance: float                  # Signal-to-noise ratio


@dataclass
class WaveletLevel:
    """Wavelet decomposition at one scale level."""
    level: int                           # Decomposition level (1=highest freq)
    period_range: Tuple[float, float]    # (min_period, max_period) in days
    energy: float                        # Energy at this level
    energy_pct: float                    # % of total energy
    mean_amplitude: float
    has_significant_cycle: bool


@dataclass
class CycleAnalysisResult:
    """Complete cycle analysis result."""
    n_observations: int

    # FFT results
    dominant_period: float = 0.0         # Days
    dominant_frequency: float = 0.0      # Cycles/day
    dominant_power_pct: float = 0.0      # % of total spectral power
    spectral_peaks: List[SpectralPeak] = field(default_factory=list)
    spectral_entropy: float = 0.0        # Higher = more random (less cyclical)

    # Wavelet results
    wavelet_levels: List[WaveletLevel] = field(default_factory=list)
    dominant_wavelet_scale: int = 0      # Most energetic scale
    wavelet_energy_ratio: float = 0.0    # Low-freq / high-freq energy

    # Trading signals
    optimal_lookback: int = 60           # Suggested lookback = dominant_period * 1.5
    current_phase: float = 0.0           # [-π, π] phase of dominant cycle
    phase_signal: str = "NEUTRAL"        # "NEAR_TROUGH" / "NEAR_PEAK" / "RISING" / "FALLING" / "NEUTRAL"
    cycle_strength: float = 0.0          # [0, 1] how strong the cycle is

    # Quality
    is_cyclical: bool = False            # Strong enough cycle to trade on
    confidence: float = 0.0              # [0, 1]
    warnings: List[str] = field(default_factory=list)


class CycleDetector:
    """
    Wavelet & Fourier cycle detection engine.

    Identifies dominant cycles in spread series and translates them
    into trading parameters (lookback, phase, conviction).
    """

    def __init__(
        self,
        min_cycle_days: int = 5,
        max_cycle_days: int = 126,
        n_fft_peaks: int = 5,
        significance_threshold: float = 2.0,   # SNR for peak detection
        wavelet_levels: int = 6,
    ):
        self.min_cycle = min_cycle_days
        self.max_cycle = max_cycle_days
        self.n_peaks = n_fft_peaks
        self.sig_threshold = significance_threshold
        self.n_wavelet_levels = wavelet_levels

    def analyze(self, series: pd.Series) -> CycleAnalysisResult:
        """Run complete cycle analysis on a spread or return series."""
        s = series.dropna().values
        n = len(s)

        result = CycleAnalysisResult(n_observations=n)

        if n < 64:
            result.warnings.append("Insufficient data for cycle analysis")
            return result

        # Detrend (remove linear trend)
        s_detrended = self._detrend(s)

        # FFT analysis
        peaks, entropy = self._fft_analysis(s_detrended)
        result.spectral_peaks = peaks
        result.spectral_entropy = entropy

        if peaks:
            result.dominant_period = peaks[0].period_days
            result.dominant_frequency = peaks[0].frequency
            result.dominant_power_pct = peaks[0].power_pct

        # Wavelet analysis
        levels = self._wavelet_analysis(s_detrended)
        result.wavelet_levels = levels
        if levels:
            dominant_level = max(levels, key=lambda l: l.energy)
            result.dominant_wavelet_scale = dominant_level.level
            low_freq_energy = sum(l.energy for l in levels if l.level >= len(levels) // 2)
            high_freq_energy = sum(l.energy for l in levels if l.level < len(levels) // 2)
            result.wavelet_energy_ratio = low_freq_energy / max(high_freq_energy, 1e-12)

        # Trading signals
        if peaks and peaks[0].significance > self.sig_threshold:
            period = peaks[0].period_days
            result.optimal_lookback = max(20, min(252, int(period * 1.5)))
            result.current_phase = self._estimate_phase(s_detrended, period)
            result.phase_signal = self._phase_to_signal(result.current_phase)
            result.cycle_strength = min(1.0, peaks[0].significance / (self.sig_threshold * 3))
            result.is_cyclical = True
            result.confidence = min(1.0, peaks[0].power_pct / 20 + result.cycle_strength * 0.5)
        else:
            result.optimal_lookback = 60  # Default
            result.is_cyclical = False
            result.warnings.append("No significant cycle detected")

        return result

    # ── FFT Analysis ──────────────────────────────────────────

    def _fft_analysis(
        self, s: np.ndarray,
    ) -> Tuple[List[SpectralPeak], float]:
        """FFT-based spectral analysis."""
        n = len(s)

        # Apply Hanning window to reduce spectral leakage
        window = np.hanning(n)
        s_windowed = s * window

        # FFT
        fft_vals = np.fft.rfft(s_windowed)
        freqs = np.fft.rfftfreq(n, d=1.0)  # d=1 day
        power = np.abs(fft_vals) ** 2

        # Normalize
        total_power = np.sum(power[1:])  # Exclude DC component
        if total_power < 1e-12:
            return [], 1.0

        # Filter to valid frequency range
        min_freq = 1.0 / self.max_cycle
        max_freq = 1.0 / self.min_cycle

        valid_mask = (freqs >= min_freq) & (freqs <= max_freq) & (freqs > 0)
        valid_freqs = freqs[valid_mask]
        valid_power = power[valid_mask]

        if len(valid_freqs) == 0:
            return [], 1.0

        # Find peaks (local maxima)
        peaks_list = []
        noise_floor = np.median(valid_power)

        for i in range(1, len(valid_power) - 1):
            if valid_power[i] > valid_power[i - 1] and valid_power[i] > valid_power[i + 1]:
                snr = valid_power[i] / max(noise_floor, 1e-12)
                period = 1.0 / valid_freqs[i] if valid_freqs[i] > 0 else 999
                power_pct = valid_power[i] / total_power * 100

                peaks_list.append(SpectralPeak(
                    frequency=round(float(valid_freqs[i]), 6),
                    period_days=round(float(period), 2),
                    power=round(float(valid_power[i]), 4),
                    power_pct=round(float(power_pct), 2),
                    significance=round(float(snr), 4),
                ))

        # Sort by power (strongest first)
        peaks_list.sort(key=lambda p: p.power, reverse=True)
        peaks_list = peaks_list[:self.n_peaks]

        # Spectral entropy (measure of spectral flatness)
        p_norm = valid_power / np.sum(valid_power)
        p_norm = p_norm[p_norm > 0]
        entropy = float(-np.sum(p_norm * np.log(p_norm)) / np.log(len(p_norm))) if len(p_norm) > 1 else 1.0

        return peaks_list, round(entropy, 4)

    # ── Wavelet Analysis ──────────────────────────────────────

    def _wavelet_analysis(self, s: np.ndarray) -> List[WaveletLevel]:
        """Multi-resolution wavelet decomposition (Haar basis)."""
        levels = []
        n = len(s)
        current = s.copy()
        total_energy = np.sum(s ** 2)
        if total_energy < 1e-12:
            return levels

        for level in range(1, self.n_wavelet_levels + 1):
            if len(current) < 4:
                break

            # Simple Haar wavelet: detail = high-pass, approx = low-pass
            n_c = len(current)
            n_half = n_c // 2

            approx = np.zeros(n_half)
            detail = np.zeros(n_half)

            for i in range(n_half):
                approx[i] = (current[2 * i] + current[2 * i + 1]) / np.sqrt(2)
                detail[i] = (current[2 * i] - current[2 * i + 1]) / np.sqrt(2)

            energy = float(np.sum(detail ** 2))
            energy_pct = energy / total_energy * 100

            # Period range for this level
            min_period = 2 ** level
            max_period = 2 ** (level + 1)

            mean_amp = float(np.mean(np.abs(detail)))

            # Significance: is energy at this level above noise?
            noise_energy = total_energy / self.n_wavelet_levels
            has_sig = energy > noise_energy * 1.5

            levels.append(WaveletLevel(
                level=level,
                period_range=(float(min_period), float(max_period)),
                energy=round(energy, 6),
                energy_pct=round(energy_pct, 2),
                mean_amplitude=round(mean_amp, 6),
                has_significant_cycle=has_sig,
            ))

            current = approx

        return levels

    # ── Phase Estimation ──────────────────────────────────────

    @staticmethod
    def _estimate_phase(s: np.ndarray, period: float) -> float:
        """Estimate current phase of the dominant cycle using Hilbert transform."""
        try:
            from scipy.signal import hilbert as scipy_hilbert

            # Bandpass around dominant frequency
            n = len(s)
            fft_vals = np.fft.rfft(s)
            freqs = np.fft.rfftfreq(n)

            target_freq = 1.0 / period
            bandwidth = target_freq * 0.3  # ±30% bandwidth

            # Apply bandpass
            mask = (np.abs(freqs - target_freq) > bandwidth)
            fft_filtered = fft_vals.copy()
            fft_filtered[mask] = 0
            s_filtered = np.fft.irfft(fft_filtered, n)

            # Hilbert transform for instantaneous phase
            analytic = scipy_hilbert(s_filtered)
            phase = np.angle(analytic)
            return float(phase[-1])
        except Exception:
            # Fallback: simple phase estimation
            half_period = int(period / 2)
            if len(s) > half_period:
                recent = s[-half_period:]
                mid = np.mean(recent)
                if s[-1] > mid:
                    return float(np.pi / 2)  # Near peak
                else:
                    return float(-np.pi / 2)  # Near trough
            return 0.0

    @staticmethod
    def _phase_to_signal(phase: float) -> str:
        """Convert phase angle to trading signal."""
        # phase ∈ [-π, π]
        # -π to -π/2: NEAR_TROUGH (buy zone)
        # -π/2 to 0: RISING
        # 0 to π/2: NEAR_PEAK (sell zone)
        # π/2 to π: FALLING
        if phase < -np.pi / 2:
            return "NEAR_TROUGH"
        elif phase < 0:
            return "RISING"
        elif phase < np.pi / 2:
            return "NEAR_PEAK"
        else:
            return "FALLING"

    @staticmethod
    def _detrend(s: np.ndarray) -> np.ndarray:
        """Remove linear trend from series."""
        n = len(s)
        x = np.arange(n)
        coeffs = np.polyfit(x, s, 1)
        return s - np.polyval(coeffs, x)
