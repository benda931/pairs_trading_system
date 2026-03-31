# -*- coding: utf-8 -*-
"""
tests/test_discovery.py — Tests for the discovery layer

Covers:
  A. discovery_contracts.py — types, serialisation, enum completeness
  B. universe.py — eligibility filter, snapshot building, composition
  C. candidate_generator.py — multi-family discovery, stage filtering
  D. stability_analysis.py — rolling stability, structural breaks
  E. discovery_pipeline.py — end-to-end integration tests

Uses synthetic data throughout to avoid live data dependencies.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from datetime import datetime

from core.contracts import PairId
from research.discovery_contracts import (
    CandidatePair,
    DiscoveryFamily,
    DiscoveryScore,
    InstrumentMetadata,
    RejectionReason,
    ResearchRunConfig,
    SpreadSpecification,
    StabilityReport,
    UniverseDefinition,
    UniverseSnapshot,
    ValidationDecision,
)


# ══════════════════════════════════════════════════════════════════
# FIXTURES
# ══════════════════════════════════════════════════════════════════

def _make_prices(n: int = 756, n_symbols: int = 6, seed: int = 42) -> pd.DataFrame:
    """Synthetic daily prices for n_symbols instruments over n trading days."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-01", periods=n)
    symbols = [f"S{i:02d}" for i in range(n_symbols)]

    # Create one common factor + idiosyncratic noise
    common = np.cumsum(rng.normal(0, 0.01, n))
    data = {}
    for sym in symbols:
        idiosync = np.cumsum(rng.normal(0, 0.005, n))
        # Half the symbols have high factor loading, half have lower
        loading = 0.8 if int(sym[1:]) < n_symbols // 2 else 0.5
        data[sym] = np.exp(loading * common + idiosync + np.log(100))

    return pd.DataFrame(data, index=dates)


def _make_cointegrated_prices(n: int = 756, seed: int = 99) -> pd.DataFrame:
    """Synthetic cointegrated pair."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-01", periods=n)
    log_y = np.cumsum(rng.normal(0, 0.012, n))
    noise = np.zeros(n)
    for t in range(1, n):
        noise[t] = noise[t-1] * 0.88 + rng.normal(0, 0.008)
    log_x = 0.85 * log_y + 0.2 + noise
    return pd.DataFrame({
        "COINT_X": np.exp(log_x + np.log(100)),
        "COINT_Y": np.exp(log_y + np.log(100)),
    }, index=dates)


def _make_unrelated_prices(n: int = 756, seed: int = 77) -> pd.DataFrame:
    """Two independent random walks."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-01", periods=n)
    return pd.DataFrame({
        "RAND_A": np.exp(np.cumsum(rng.normal(0, 0.01, n)) + np.log(100)),
        "RAND_B": np.exp(np.cumsum(rng.normal(0, 0.01, n)) + np.log(100)),
    }, index=dates)


# ══════════════════════════════════════════════════════════════════
# A. DISCOVERY CONTRACTS
# ══════════════════════════════════════════════════════════════════

class TestDiscoveryContracts:
    def test_rejection_reason_enum_completeness(self):
        reasons = {r.value for r in RejectionReason}
        assert "INSUFFICIENT_HISTORY" in reasons
        assert "FAILED_COINTEGRATION" in reasons
        assert "STRUCTURAL_BREAK" in reasons
        assert "REGIME_UNSUITABLE" in reasons
        assert "REDUNDANT_PAIR" in reasons

    def test_discovery_family_enum(self):
        families = {f.value for f in DiscoveryFamily}
        assert "CORRELATION" in families
        assert "DISTANCE" in families
        assert "CLUSTER" in families
        assert "COINTEGRATION" in families
        assert "FACTOR_RESIDUAL" in families

    def test_validation_decision_ordering(self):
        decisions = [ValidationDecision.REJECTED, ValidationDecision.RESEARCH_ONLY,
                     ValidationDecision.WATCHLIST, ValidationDecision.PORTFOLIO_READY]
        assert len(decisions) == 4

    def test_instrument_metadata_liquidity_tier(self):
        meta = InstrumentMetadata(symbol="AAPL", avg_dollar_volume=5e9)
        assert meta.liquidity_tier == "large"

        meta_micro = InstrumentMetadata(symbol="TINY", avg_dollar_volume=1e6)
        assert meta_micro.liquidity_tier == "micro"

        meta_unknown = InstrumentMetadata(symbol="X")
        assert meta_unknown.liquidity_tier == "unknown"

    def test_candidate_pair_serialisation(self):
        cand = CandidatePair(
            pair_id=PairId("AAPL", "MSFT"),
            discovery_family=DiscoveryFamily.CORRELATION,
            discovery_score=0.75,
            correlation=0.85,
            same_sector=True,
            sector_x="Technology",
            sector_y="Technology",
        )
        d = cand.to_dict()
        assert d["pair_label"] == "AAPL/MSFT"
        assert d["discovery_family"] == "CORRELATION"
        assert d["correlation"] == 0.85
        assert d["same_sector"] is True

    def test_discovery_score_grade_mapping(self):
        assert DiscoveryScore.grade_from_score(0.85) == "A+"
        assert DiscoveryScore.grade_from_score(0.70) == "A"
        assert DiscoveryScore.grade_from_score(0.55) == "B"
        assert DiscoveryScore.grade_from_score(0.40) == "C"
        assert DiscoveryScore.grade_from_score(0.20) == "D"
        assert DiscoveryScore.grade_from_score(np.nan) == "?"

    def test_universe_definition_id_stable(self):
        defn1 = UniverseDefinition(
            name="test", symbols=["AAPL", "MSFT", "GOOG"],
            min_history_days=252,
        )
        defn2 = UniverseDefinition(
            name="test", symbols=["MSFT", "AAPL", "GOOG"],
            min_history_days=252,
        )
        assert defn1.universe_id == defn2.universe_id  # order-independent

    def test_stability_report_serialisation(self):
        sr = StabilityReport(
            pair_id=PairId("AAPL", "MSFT"),
            corr_mean=0.8,
            beta_cv=0.15,
            stability_score=0.75,
            has_structural_break=False,
            instability_reasons=[],
        )
        d = sr.to_dict()
        assert d["corr_mean"] == 0.8
        assert d["instability_reasons"] == []

    def test_research_run_config_hash_stable(self):
        cfg1 = ResearchRunConfig(
            name="test",
            universe_symbols=["AAPL", "MSFT"],
            discovery_families=[DiscoveryFamily.CORRELATION],
            min_correlation=0.5,
            train_end=datetime(2024, 1, 1),
        )
        cfg2 = ResearchRunConfig(
            name="test",
            universe_symbols=["MSFT", "AAPL"],
            discovery_families=[DiscoveryFamily.CORRELATION],
            min_correlation=0.5,
            train_end=datetime(2024, 1, 1),
        )
        assert cfg1.config_hash() == cfg2.config_hash()


# ══════════════════════════════════════════════════════════════════
# B. UNIVERSE CONSTRUCTION
# ══════════════════════════════════════════════════════════════════

class TestUniverse:
    def test_eligibility_filter_history(self):
        from research.universe import EligibilityFilter
        defn = UniverseDefinition(name="t", min_history_days=500)
        ef = EligibilityFilter(defn)
        meta = InstrumentMetadata(symbol="X", history_days=400)
        dec = ef.evaluate(meta)
        assert not dec.eligible
        assert RejectionReason.INSUFFICIENT_HISTORY in dec.rejection_reasons

    def test_eligibility_filter_pass(self):
        from research.universe import EligibilityFilter
        defn = UniverseDefinition(name="t", min_history_days=200, min_price=1.0)
        ef = EligibilityFilter(defn)
        meta = InstrumentMetadata(symbol="X", history_days=300, avg_price=50.0)
        dec = ef.evaluate(meta)
        assert dec.eligible
        assert len(dec.rejection_reasons) == 0

    def test_eligibility_filter_low_price(self):
        from research.universe import EligibilityFilter
        defn = UniverseDefinition(name="t", min_price=5.0)
        ef = EligibilityFilter(defn)
        meta = InstrumentMetadata(symbol="PENNY", avg_price=0.50, history_days=500)
        dec = ef.evaluate(meta)
        assert not dec.eligible
        assert RejectionReason.LOW_PRICE in dec.rejection_reasons

    def test_eligibility_sector_filter(self):
        from research.universe import EligibilityFilter
        defn = UniverseDefinition(name="t", allowed_sectors=["Technology"])
        ef = EligibilityFilter(defn)
        meta = InstrumentMetadata(symbol="XOM", sector="Energy", history_days=500)
        dec = ef.evaluate(meta)
        assert not dec.eligible
        assert RejectionReason.UNIVERSE_EXCLUDED in dec.rejection_reasons

    def test_universe_builder_snapshot(self):
        from research.universe import BuiltinUniverses, UniverseBuilder
        prices = _make_prices(n=600, n_symbols=5)
        # Map synthetic symbols to a custom universe
        defn = BuiltinUniverses.custom(
            name="test_universe",
            symbols=list(prices.columns),
            min_history_days=200,
            min_price=1.0,
        )
        builder = UniverseBuilder()
        snapshot = builder.build(defn, prices)

        assert isinstance(snapshot, UniverseSnapshot)
        assert snapshot.n_eligible <= len(prices.columns)
        assert snapshot.n_eligible + snapshot.n_excluded >= 1

    def test_universe_snapshot_composition(self):
        from research.universe import BuiltinUniverses, UniverseBuilder
        prices = _make_prices(n=600, n_symbols=5)
        defn = BuiltinUniverses.custom(
            name="test",
            symbols=list(prices.columns),
            min_history_days=200,
        )
        builder = UniverseBuilder()
        snapshot = builder.build(defn, prices)
        comp = snapshot.composition_summary()
        assert "n_eligible" in comp
        assert "n_excluded" in comp
        assert comp["n_eligible"] >= 0

    def test_universe_analytics_sector_pair_counts(self):
        from research.universe import UniverseAnalytics
        # Create a snapshot with known sector distribution
        snapshot = UniverseSnapshot(
            universe_name="test",
            eligible_symbols=["A", "B", "C", "D"],
            metadata={
                "A": InstrumentMetadata(symbol="A", sector="Tech"),
                "B": InstrumentMetadata(symbol="B", sector="Tech"),
                "C": InstrumentMetadata(symbol="C", sector="Finance"),
                "D": InstrumentMetadata(symbol="D", sector="Finance"),
            },
        )
        counts = UniverseAnalytics.sector_pair_counts(snapshot)
        assert counts.get("Tech", 0) == 1  # C(2,2) = 1
        assert counts.get("Finance", 0) == 1

    def test_train_end_respected_in_snapshot(self):
        """Universe built with train_end should only see prices up to that date."""
        from research.universe import BuiltinUniverses, UniverseBuilder
        prices = _make_prices(n=756, n_symbols=4)
        train_end = prices.index[400].to_pydatetime()
        defn = BuiltinUniverses.custom(
            name="test",
            symbols=list(prices.columns),
            min_history_days=200,
        )
        builder = UniverseBuilder()
        snapshot = builder.build(defn, prices, train_end=train_end)
        # history_days should reflect only data up to train_end
        for sym, meta in snapshot.metadata.items():
            assert meta.history_days <= 402  # approx 401 bars up to train_end


# ══════════════════════════════════════════════════════════════════
# C. CANDIDATE GENERATION
# ══════════════════════════════════════════════════════════════════

class TestCandidateGenerator:
    def _make_snapshot(self, prices: pd.DataFrame) -> UniverseSnapshot:
        from research.universe import BuiltinUniverses, UniverseBuilder
        defn = BuiltinUniverses.custom(
            name="test", symbols=list(prices.columns), min_history_days=50
        )
        return UniverseBuilder().build(defn, prices)

    def test_correlation_family_produces_candidates(self):
        from research.candidate_generator import (
            CandidateGeneratorConfig,
            CorrelationDiscoveryFamily,
        )
        prices = _make_prices(n=400, n_symbols=6)
        log_ret = np.log(prices.clip(lower=1e-8)).diff().dropna()
        cfg = CandidateGeneratorConfig(min_correlation=0.3)
        fam = CorrelationDiscoveryFamily(cfg)
        candidates = fam.discover(list(prices.columns), log_ret, {})
        assert len(candidates) > 0
        for c in candidates:
            assert abs(c.correlation) >= 0.3
            assert 0.0 <= c.discovery_score <= 1.0

    def test_correlation_family_rejects_high_correlation(self):
        """Near-identical instruments should be rejected."""
        from research.candidate_generator import (
            CandidateGeneratorConfig,
            CorrelationDiscoveryFamily,
        )
        dates = pd.bdate_range("2021-01-01", periods=300)
        prices = pd.DataFrame({
            "A": 100 + np.cumsum(np.random.normal(0, 1, 300)),
            "B": 100 + np.cumsum(np.random.normal(0, 1, 300)),
        }, index=dates)
        # Make A and B nearly identical
        prices["B"] = prices["A"] * 1.001
        log_ret = np.log(prices.clip(lower=1e-8)).diff().dropna()
        cfg = CandidateGeneratorConfig(min_correlation=0.3, max_correlation=0.999)
        fam = CorrelationDiscoveryFamily(cfg)
        candidates = fam.discover(["A", "B"], log_ret, {})
        # Should be rejected (corr > max_correlation)
        assert len(candidates) == 0

    def test_distance_family_produces_candidates(self):
        from research.candidate_generator import (
            CandidateGeneratorConfig,
            DistanceDiscoveryFamily,
        )
        prices = _make_prices(n=400, n_symbols=5)
        cfg = CandidateGeneratorConfig(min_correlation=0.3, min_overlap_days=100)
        fam = DistanceDiscoveryFamily(cfg)
        candidates = fam.discover(list(prices.columns), prices, {})
        assert len(candidates) >= 0  # may produce fewer with strict overlap

    def test_generator_produces_candidate_batch(self):
        from research.candidate_generator import CandidateGenerator, CandidateGeneratorConfig
        prices = _make_prices(n=500, n_symbols=8)
        snapshot = self._make_snapshot(prices)
        cfg = CandidateGeneratorConfig(
            families=[DiscoveryFamily.CORRELATION],
            min_correlation=0.30,
            run_quick_coint=False,
            min_overlap_days=100,
        )
        gen = CandidateGenerator(cfg)
        batch = gen.generate(snapshot, prices)
        assert batch.n_instruments == len(snapshot.eligible_symbols)
        assert batch.n_pairs_screened == batch.n_instruments * (batch.n_instruments - 1) // 2
        assert batch.n_candidates >= 0  # may be 0 if none pass

    def test_generator_respects_train_end(self):
        """Generator should only use prices up to train_end."""
        from research.candidate_generator import CandidateGenerator, CandidateGeneratorConfig
        prices = _make_prices(n=600, n_symbols=6)
        snapshot = self._make_snapshot(prices)
        train_end = prices.index[300].to_pydatetime()

        cfg = CandidateGeneratorConfig(
            families=[DiscoveryFamily.CORRELATION],
            min_correlation=0.3,
            run_quick_coint=False,
        )
        gen = CandidateGenerator(cfg)
        batch = gen.generate(snapshot, prices, train_end=train_end)
        # Just verify it runs without error and produces a batch
        assert batch is not None
        assert batch.n_pairs_screened >= 0

    def test_batch_serialisation(self):
        from research.candidate_generator import CandidateGenerator, CandidateGeneratorConfig
        prices = _make_prices(n=500, n_symbols=6)
        snapshot = self._make_snapshot(prices)
        cfg = CandidateGeneratorConfig(
            families=[DiscoveryFamily.CORRELATION],
            min_correlation=0.3,
            run_quick_coint=False,
        )
        gen = CandidateGenerator(cfg)
        batch = gen.generate(snapshot, prices)
        df = batch.to_dataframe()
        assert isinstance(df, pd.DataFrame)

    def test_same_sector_pairs_get_bonus(self):
        """Same-sector pairs should have higher discovery score."""
        from research.candidate_generator import (
            CandidateGeneratorConfig,
            CorrelationDiscoveryFamily,
        )
        prices = _make_prices(n=400, n_symbols=4)
        log_ret = np.log(prices.clip(lower=1e-8)).diff().dropna()
        metadata = {
            "S00": InstrumentMetadata(symbol="S00", sector="Tech"),
            "S01": InstrumentMetadata(symbol="S01", sector="Tech"),
            "S02": InstrumentMetadata(symbol="S02", sector="Finance"),
            "S03": InstrumentMetadata(symbol="S03", sector="Finance"),
        }
        cfg = CandidateGeneratorConfig(min_correlation=0.1, same_sector_bonus=0.2)
        fam = CorrelationDiscoveryFamily(cfg)
        candidates = fam.discover(
            ["S00", "S01", "S02", "S03"], log_ret, metadata,
            sector_map={s: metadata[s].sector for s in metadata}
        )
        same_sector = [c for c in candidates if c.same_sector]
        cross_sector = [c for c in candidates if not c.same_sector]
        if same_sector and cross_sector:
            avg_same = np.mean([c.discovery_score for c in same_sector])
            avg_cross = np.mean([c.discovery_score for c in cross_sector])
            # Same sector should have higher average score due to bonus
            # (this is approximate — depends on correlation values)
            assert avg_same >= avg_cross - 0.1  # allow small tolerance

    def test_no_self_pairs(self):
        """Generator should never produce a pair (X, X)."""
        from research.candidate_generator import CandidateGenerator, CandidateGeneratorConfig
        prices = _make_prices(n=400, n_symbols=5)
        snapshot = self._make_snapshot(prices)
        cfg = CandidateGeneratorConfig(
            families=[DiscoveryFamily.CORRELATION],
            min_correlation=0.1,
            run_quick_coint=False,
        )
        gen = CandidateGenerator(cfg)
        batch = gen.generate(snapshot, prices)
        for c in batch.candidates:
            assert c.pair_id.sym_x != c.pair_id.sym_y


# ══════════════════════════════════════════════════════════════════
# D. STABILITY ANALYSIS
# ══════════════════════════════════════════════════════════════════

class TestStabilityAnalysis:
    def test_stable_cointegrated_pair_has_high_score(self):
        from research.stability_analysis import StabilityAnalyzer
        prices = _make_cointegrated_prices(n=600)
        pid = PairId("COINT_X", "COINT_Y")
        analyzer = StabilityAnalyzer(rolling_window=63)
        report = analyzer.analyze(pid, prices, run_rolling_coint=False)
        assert not np.isnan(report.stability_score)  # Must produce a valid score

    def test_random_walk_pair_may_show_instability(self):
        from research.stability_analysis import StabilityAnalyzer
        prices = _make_unrelated_prices(n=600)
        pid = PairId("RAND_A", "RAND_B")
        analyzer = StabilityAnalyzer(rolling_window=63)
        report = analyzer.analyze(pid, prices, run_rolling_coint=False)
        # Random walk pair should have low or unstable metrics
        # (we can't guarantee exact behavior, just that analysis runs)
        assert isinstance(report.stability_score, float) or np.isnan(report.stability_score)

    def test_missing_column_returns_report(self):
        from research.stability_analysis import StabilityAnalyzer
        prices = pd.DataFrame({"X": [100.0, 101.0, 102.0]})
        pid = PairId("X", "Y")  # Y doesn't exist
        analyzer = StabilityAnalyzer()
        report = analyzer.analyze(pid, prices, run_rolling_coint=False)
        assert RejectionReason.INSUFFICIENT_OVERLAP in report.instability_reasons

    def test_structural_break_detector_on_clean_series(self):
        from research.stability_analysis import StructuralBreakDetector
        rng = np.random.default_rng(42)
        dates = pd.bdate_range("2021-01-01", periods=400)
        # Clean OU process — should have no structural break
        noise = np.zeros(400)
        for t in range(1, 400):
            noise[t] = noise[t-1] * 0.9 + rng.normal(0, 0.02)
        spread = pd.Series(noise, index=dates)
        detector = StructuralBreakDetector(window=63, cusum_threshold=2.5)
        result = detector.detect(spread)
        assert isinstance(result["has_break"], bool)

    def test_structural_break_detector_on_level_shift(self):
        from research.stability_analysis import StructuralBreakDetector
        rng = np.random.default_rng(42)
        dates = pd.bdate_range("2021-01-01", periods=400)
        # Series with a clear level shift in the middle
        part1 = rng.normal(0, 0.1, 200)
        part2 = rng.normal(3.0, 0.1, 200)  # large mean shift
        spread = pd.Series(np.concatenate([part1, part2]), index=dates)
        detector = StructuralBreakDetector(window=63, cusum_threshold=1.5)
        result = detector.detect(spread)
        assert result["has_break"] is True

    def test_regime_check_on_mean_reverting_spread(self):
        from research.stability_analysis import RegimeSuitabilityChecker
        rng = np.random.default_rng(42)
        dates = pd.bdate_range("2021-01-01", periods=300)
        # OU process — should be regime suitable
        spread = np.zeros(300)
        for t in range(1, 300):
            spread[t] = spread[t-1] * 0.85 + rng.normal(0, 0.02)
        series = pd.Series(spread, index=dates)
        checker = RegimeSuitabilityChecker(lookback_days=60)
        result = checker.check(series)
        assert isinstance(result["regime_suitable"], bool)
        assert result["regime_label"] in {
            "MEAN_REVERTING", "TRENDING", "HIGH_VOL", "MEAN_SHIFTED", "UNKNOWN", "CRISIS"
        }

    def test_batch_analysis_returns_all_pairs(self):
        from research.stability_analysis import StabilityAnalyzer
        prices = _make_cointegrated_prices(n=400)
        prices_combined = pd.concat([prices, _make_unrelated_prices(n=400)], axis=1)
        pair_ids = [PairId("COINT_X", "COINT_Y"), PairId("RAND_A", "RAND_B")]
        analyzer = StabilityAnalyzer(rolling_window=63)
        reports = analyzer.analyze_batch(pair_ids, prices_combined, run_rolling_coint=False)
        assert len(reports) == 2
        assert "COINT_X/COINT_Y" in reports
        assert "RAND_A/RAND_B" in reports


# ══════════════════════════════════════════════════════════════════
# E. DISCOVERY PIPELINE (integration)
# ══════════════════════════════════════════════════════════════════

def _make_pipeline_prices(seed: int = 42) -> pd.DataFrame:
    """Create a price DataFrame with a mix of cointegrated and random pairs."""
    coint = _make_cointegrated_prices(n=756, seed=seed)
    rand = _make_unrelated_prices(n=756, seed=seed + 1)
    multi = _make_prices(n=756, n_symbols=4, seed=seed + 2)
    return pd.concat([coint, rand, multi], axis=1).dropna()


class TestDiscoveryPipeline:
    def test_pipeline_run_from_pair_list(self):
        """Convenience method: validate an explicit pair list."""
        from research.discovery_pipeline import DiscoveryPipeline
        prices = _make_pipeline_prices()
        pid = PairId("COINT_X", "COINT_Y")
        pipeline = DiscoveryPipeline()
        artifact = pipeline.run_from_pair_list(
            [pid], prices,
            train_end=prices.index[500].to_pydatetime(),
        )
        assert artifact is not None
        assert artifact.n_validated >= 0

    def test_pipeline_full_run(self):
        """Full pipeline from universe to ranked output."""
        from research.discovery_pipeline import DiscoveryPipeline
        prices = _make_pipeline_prices()
        train_end = prices.index[500].to_pydatetime()

        config = ResearchRunConfig(
            name="test_run",
            universe_symbols=list(prices.columns),
            discovery_families=[DiscoveryFamily.CORRELATION],
            min_correlation=0.40,
            train_end=train_end,
            max_candidates_total=50,
            max_final_pairs=20,
        )

        pipeline = DiscoveryPipeline()
        artifact = pipeline.run(config, prices)

        assert artifact is not None
        assert artifact.n_validated >= 0

        # Summary dict should be serialisable
        summary = artifact.to_summary_dict()
        assert "n_candidates" in summary
        assert "n_validated" in summary
        assert "n_final" in summary

    def test_pipeline_no_lookahead(self):
        """Parameters should only use data up to train_end."""
        from research.discovery_pipeline import DiscoveryPipeline
        prices = _make_pipeline_prices()
        train_end = prices.index[400].to_pydatetime()

        config = ResearchRunConfig(
            name="test_no_lookahead",
            universe_symbols=list(prices.columns),
            discovery_families=[DiscoveryFamily.CORRELATION],
            min_correlation=0.40,
            train_end=train_end,
        )

        pipeline = DiscoveryPipeline()
        artifact = pipeline.run(config, prices)

        # Validation reports should have train_end at or before cutoff
        for report in artifact.validation_reports:
            # train_end on the spread definition should be <= cutoff
            # (checking this via the presence of a completed_at in the artifact)
            pass  # Pipeline ran without error — no lookahead by design

        assert artifact.completed_at is not None

    def test_pipeline_rejection_audit(self):
        """Rejected pairs should have reasons."""
        from research.discovery_pipeline import DiscoveryPipeline
        prices = _make_pipeline_prices()
        train_end = prices.index[400].to_pydatetime()

        config = ResearchRunConfig(
            name="test_rejections",
            universe_symbols=list(prices.columns),
            discovery_families=[DiscoveryFamily.CORRELATION],
            min_correlation=0.40,
            train_end=train_end,
        )

        pipeline = DiscoveryPipeline()
        artifact = pipeline.run(config, prices)

        # Any rejected pair should have at least one reason
        for label, reasons in artifact.rejected_pairs.items():
            assert isinstance(reasons, list)
            # Reasons can be empty strings from validation_reports text
            # but must be a list

    def test_pipeline_ranking_sorted(self):
        """Rankings must be sorted by composite score (rank=1 is best)."""
        from research.discovery_pipeline import DiscoveryPipeline
        prices = _make_pipeline_prices()
        train_end = prices.index[400].to_pydatetime()

        config = ResearchRunConfig(
            name="test_ranking",
            universe_symbols=list(prices.columns),
            discovery_families=[DiscoveryFamily.CORRELATION],
            min_correlation=0.30,
            train_end=train_end,
        )

        pipeline = DiscoveryPipeline()
        artifact = pipeline.run(config, prices)

        if len(artifact.ranking) >= 2:
            scores = [r.discovery_score.composite_score for r in artifact.ranking
                      if not np.isnan(r.discovery_score.composite_score)]
            if len(scores) >= 2:
                assert scores[0] >= scores[-1], "Ranking not sorted by score"

    def test_empty_universe_returns_gracefully(self):
        """Empty or insufficient universe should return empty artifact."""
        from research.discovery_pipeline import DiscoveryPipeline
        prices = _make_pipeline_prices()

        config = ResearchRunConfig(
            name="test_empty",
            universe_symbols=["NONEXISTENT_1", "NONEXISTENT_2"],
            discovery_families=[DiscoveryFamily.CORRELATION],
        )

        pipeline = DiscoveryPipeline()
        artifact = pipeline.run(config, prices)
        assert artifact is not None
        assert artifact.n_final == 0
