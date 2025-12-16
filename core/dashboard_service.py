# core/dashboard_service.py
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence
import numpy as np
import pandas as pd

from .dashboard_models import (
    DashboardContext,
    DashboardSnapshot,
    MarketSnapshot,
    PortfolioSnapshot,
    PortfolioExposureBreakdown,
    PortfolioPnLBreakdown,
    PortfolioRiskMetrics,
    PositionSnapshot,
    PositionPnlBreakdown,
    PositionRiskMetrics,
    RiskSnapshot,
    SignalItem,
    SignalsSnapshot,
    SystemHealthSnapshot,
)
from .app_context import AppContext  # אצלך כבר קיים
from .sql_store import SqlStore      # קיים אצלך

logger = logging.getLogger(__name__)


class DashboardService:
    """
    DashboardService — מוח הדשבורד ברמת קרן גידור
    =============================================

    אחריות:
    --------
    1. איחוד כל מקורות המידע:
       • MarketData (VIX, מדדי שוק, פקטורים).
       • Broker / IBKR (פוזיציות, Equity, Cash).
       • Risk Engine (VaR / ES / Vol / Drawdown / Beta).
       • Signals / Pair Recommender / Optimization Results.
       • System Health (חיבור ברוקר, SQL, Latency, Errors).

    2. הפקת DashboardSnapshot אחד מרכזי:
       • MarketSnapshot
       • PortfolioSnapshot (HF-grade, כולל Exposure/PnL/Risk)
       • RiskSnapshot (מבוסס PortfolioRiskMetrics)
       • SignalsSnapshot (SignalItem עם Strategy/Regime/Model)
       • SystemHealthSnapshot

    3. Persistence ל-SQL (audit / היסטוריה / Monitoring).
    """

    def __init__(
        self,
        ctx: AppContext,
        sql_store: Optional[SqlStore] = None,
        enable_persistence: bool = True,
    ) -> None:
        self.ctx = ctx
        self.sql_store = sql_store
        self.enable_persistence = enable_persistence

        # Cache קליל של ה-snapshot האחרון
        self._last_snapshot: Optional[DashboardSnapshot] = None

    # ------------------------------------------------------------------
    # Helpers כלליים
    # ------------------------------------------------------------------
    @staticmethod
    def _now_utc() -> datetime:
        """
        זמן נוכחי ב-UTC עם timezone-aware datetime.

        זה עדיף ל-logging, ל-SQL, ולכל מקום שבו רוצים timestamps עקביים.
        """
        return datetime.now(timezone.utc)
    def _safe_call(self, desc: str, fn, default: Any = None) -> Any:
        """
        עטיפה כללית לקריאות שירותים אחרים — לא מפילה את הדשבורד,
        רק לוג + default.
        """
        try:
            return fn()
        except Exception as exc:  # pragma: no cover - הגנה קריטית לדשבורד
            logger.warning("[DashboardService] %s failed: %s", desc, exc, exc_info=True)
            return default

    # ------------------------------------------------------------------
    # Market snapshot
    # ------------------------------------------------------------------
    def _build_market_snapshot(self, dctx: DashboardContext) -> MarketSnapshot:
        """
        בניית MarketSnapshot:

        • רמת VIX + Regime בסיסי (Low / Normal / High / Stress).
        • תשואות Benchmark (1D / 5D / 30D).
        • Hook למאקרו/פקטורים (factor_returns / factor_zscores).
        """

        def _fetch_market() -> Dict[str, Any]:
            md = getattr(self.ctx, "market_data", None) or getattr(
                self.ctx, "market_data_router", None
            )

            result: Dict[str, Any] = {
                "vix_level": 0.0,
                "vix_regime": "unknown",
                "bench_ret_1d": 0.0,
                "bench_ret_5d": 0.0,
                "bench_ret_30d": 0.0,
                "factor_returns": {},
                "factor_zscores": {},
            }

            if md is None:
                return result

            # ---- VIX ----
            get_vix = getattr(md, "get_vix_snapshot", None)
            if callable(get_vix):
                try:
                    vix_snap = get_vix()
                    result["vix_level"] = float(getattr(vix_snap, "level", 0.0))
                    result["vix_regime"] = getattr(vix_snap, "regime", "unknown")
                except Exception as exc:
                    logger.warning("Failed to load VIX snapshot: %s", exc)

            # ---- Benchmark Returns ----
            benchmark_symbol = dctx.benchmark or "SPY"
            get_bench = getattr(md, "get_benchmark_returns", None)
            if callable(get_bench):
                try:
                    br = get_bench(benchmark_symbol)
                    result["bench_ret_1d"] = float(br.get("ret_1d", 0.0))
                    result["bench_ret_5d"] = float(br.get("ret_5d", 0.0))
                    result["bench_ret_30d"] = float(br.get("ret_30d", 0.0))
                except Exception as exc:
                    logger.warning("Failed to load benchmark returns: %s", exc)

            # ---- Factor returns (Hook) ----
            get_factors = getattr(md, "get_factor_snapshot", None)
            if callable(get_factors):
                try:
                    f = get_factors()
                    if isinstance(f, Mapping):
                        result["factor_returns"] = dict(f.get("returns", {}) or {})
                        result["factor_zscores"] = dict(f.get("zscores", {}) or {})
                except Exception as exc:
                    logger.warning("Failed to load factor snapshot: %s", exc)

            return result

        data = self._safe_call("build_market_snapshot", _fetch_market, default={}) or {}

        vix_level = float(data.get("vix_level", 0.0))
        vix_regime = str(data.get("vix_regime", "unknown"))
        bench_ret_1d = float(data.get("bench_ret_1d", 0.0))
        bench_ret_5d = float(data.get("bench_ret_5d", 0.0))
        bench_ret_30d = float(data.get("bench_ret_30d", 0.0))

        factor_returns = dict(data.get("factor_returns", {}) or {})
        factor_zscores = dict(data.get("factor_zscores", {}) or {})

        return MarketSnapshot(
            as_of=self._now_utc(),
            vix_level=vix_level,
            vix_regime=vix_regime,
            benchmark=dctx.benchmark,
            bench_ret_1d=bench_ret_1d,
            bench_ret_5d=bench_ret_5d,
            bench_ret_30d=bench_ret_30d,
            factor_returns=factor_returns,
            factor_zscores=factor_zscores,
            comment="macro_engine: TODO – plug core.macro_engine regimes",
        )

    def get_smart_scan_universe(self, ctx: "DashboardContext") -> pd.DataFrame:
        """
        מחזיר Universe של זוגות + פרמטרים ראשיים ל-Smart Scan.

        משתמש ב-SqlStore.load_pair_quality:
            - sym_x, sym_y → pair
            - z_entry, z_exit, lookback, hl_bars, corr_min (אם קיימים)
        """
        if self.sql_store is None:
            logger.info("get_smart_scan_universe: sql_store is None → returning empty df.")
            return pd.DataFrame()

        try:
            df = self.sql_store.load_pair_quality(
                env=getattr(ctx, "env", None),
                section="data_quality",
                latest_only=True,
            )
        except Exception as e:
            logger.warning("DashboardService.get_smart_scan_universe failed: %s", e)
            return pd.DataFrame()

        # לוודא שתמיד יש עמודת pair וטור אחד לפחות
        if df.empty:
            return df

        if "pair" not in df.columns:
            if {"sym_x", "sym_y"} <= set(df.columns):
                df["pair"] = df["sym_x"].astype(str) + "-" + df["sym_y"].astype(str)
            else:
                df["pair"] = df.index.astype(str)

        cols_preferred = [
            "pair",
            "sym_x",
            "sym_y",
            "z_entry",
            "z_exit",
            "lookback",
            "hl_bars",
            "corr_min",
        ]
        cols = [c for c in cols_preferred if c in df.columns]
        return df[cols].drop_duplicates("pair")

    def get_fair_value_universe(self, ctx: "DashboardContext") -> pd.DataFrame:
        """
        מחזיר DataFrame ברמת זוג:
            columns: 'pair', 'fv_pct_diff' (או fair_value_pct_diff שממופה)
        """
        if self.sql_store is None:
            logger.info("get_fair_value_universe: sql_store is None → returning empty df.")
            return pd.DataFrame()

        try:
            df = self.sql_store.load_fair_value_pairs(
                env=getattr(ctx, "env", None),
                latest_only=True,
            )
        except Exception as e:
            logger.warning("DashboardService.get_fair_value_universe failed: %s", e)
            return pd.DataFrame()

        if df.empty:
            return df

        # לוודא pair ועמודת fv_pct_diff קיימים
        if "pair" not in df.columns:
            if {"sym_x", "sym_y"} <= set(df.columns):
                df["pair"] = df["sym_x"].astype(str) + "-" + df["sym_y"].astype(str)
            else:
                df["pair"] = df.index.astype(str)

        if "fv_pct_diff" not in df.columns:
            for cand in ("fair_value_pct_diff", "fv_diff_pct"):
                if cand in df.columns:
                    df["fv_pct_diff"] = df[cand]
                    break

        return df[["pair", "fv_pct_diff"]].dropna(subset=["pair"]).drop_duplicates("pair")

    def get_fundamentals_universe(self, ctx: "DashboardContext") -> pd.DataFrame:
        """
        מחזיר DataFrame ברמת זוג:
            'pair', 'pe', 'roe', 'debt_to_equity', 'eps_growth_5y', ...

        מבוסס על SqlStore.load_pair_fundamentals.
        """
        if self.sql_store is None:
            logger.info("get_fundamentals_universe: sql_store is None → returning empty df.")
            return pd.DataFrame()

        try:
            df = self.sql_store.load_pair_fundamentals(
                env=getattr(ctx, "env", None),
                latest_only=True,
            )
        except Exception as e:
            logger.warning("DashboardService.get_fundamentals_universe failed: %s", e)
            return pd.DataFrame()

        if df.empty:
            return df

        if "pair" not in df.columns:
            if {"sym_x", "sym_y"} <= set(df.columns):
                df["pair"] = df["sym_x"].astype(str) + "-" + df["sym_y"].astype(str)
            else:
                df["pair"] = df.index.astype(str)

        return df.dropna(subset=["pair"]).drop_duplicates("pair")

    def get_pair_metrics_universe(self, ctx: "DashboardContext") -> pd.DataFrame:
        """
        מחזיר DataFrame ברמת זוג:
            'pair', 'Sharpe', 'Drawdown', 'WinRate', 'DSR', 'Trades', 'TailRisk', ...

        מבוסס על SqlStore.load_pair_backtest_metrics (טבלת bt_runs).
        """
        if self.sql_store is None:
            logger.info("get_pair_metrics_universe: sql_store is None → returning empty df.")
            return pd.DataFrame()

        try:
            df = self.sql_store.load_pair_backtest_metrics(
                env=getattr(ctx, "env", None),
                latest_only=True,
            )
        except Exception as e:
            logger.warning("DashboardService.get_pair_metrics_universe failed: %s", e)
            return pd.DataFrame()

        if df.empty:
            return df

        if "pair" not in df.columns:
            if {"sym_x", "sym_y"} <= set(df.columns):
                df["pair"] = df["sym_x"].astype(str) + "-" + df["sym_y"].astype(str)
            else:
                df["pair"] = df.index.astype(str)

        return df.dropna(subset=["pair"]).drop_duplicates("pair")

    # ------------------------------------------------------------------
    # Portfolio snapshot
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_exposure_by_key(
        positions: Sequence[Mapping[str, Any]],
        key: str,
        not_found_label: str = "OTHER",
    ) -> Dict[str, float]:
        """
        חישוב map של exposure ע"פ מפתח (asset_class / currency / sector וכו').

        ציפייה ל-positions כ-list של dict-like:
            { "symbol": ..., "market_value": ..., key: ... }
        """
        agg: Dict[str, float] = {}
        for pos in positions:
            try:
                mv = float(pos.get("market_value", 0.0))
            except Exception:
                continue
            label = str(pos.get(key, not_found_label) or not_found_label)
            agg[label] = agg.get(label, 0.0) + mv
        return agg

    def _build_portfolio_snapshot(self, dctx: DashboardContext) -> PortfolioSnapshot:
        """
        בניית PortfolioSnapshot HF-grade:

        • Equity / Cash / Margin / Leverage.
        • PnL יומי.
        • Gross / Net / Long / Short exposure.
        • Exposure breakdown (Asset class / Currency / Country / Sector).
        • Positions כ-PositionSnapshot (לפחות ברמת סמלים ו-MV).
        """

        def _fetch_portfolio() -> Dict[str, Any]:
            broker = getattr(self.ctx, "broker", None) or getattr(
                self.ctx, "ibkr", None
            )

            if broker is None:
                return {}

            # עדיפות לפונקציה עשירה אחת
            get_snapshot = getattr(broker, "get_portfolio_snapshot", None)
            if callable(get_snapshot):
                snap = get_snapshot(portfolio_id=dctx.portfolio_id)
                if isinstance(snap, Mapping):
                    return dict(snap)
                return dict(getattr(snap, "__dict__", {}))

            # fallback: get_positions + get_account_summary
            get_positions = getattr(broker, "get_positions", None)
            get_account = getattr(broker, "get_account_summary", None)

            result: Dict[str, Any] = {}
            positions: List[Mapping[str, Any]] = []

            if callable(get_positions):
                try:
                    raw_positions = get_positions()
                    for p in raw_positions:
                        if isinstance(p, Mapping):
                            positions.append(p)
                        else:
                            positions.append(
                                getattr(p, "__dict__", {}) or {"raw": repr(p)}
                            )
                except Exception as exc:
                    logger.warning("Failed to load positions: %s", exc)

            result["positions"] = positions

            if callable(get_account):
                try:
                    acct = get_account()
                    if isinstance(acct, Mapping):
                        result["account"] = dict(acct)
                    else:
                        result["account"] = dict(getattr(acct, "__dict__", {}))
                except Exception as exc:
                    logger.warning("Failed to load account summary: %s", exc)

            return result

        data = self._safe_call("build_portfolio_snapshot", _fetch_portfolio, default={}) or {}

        account: Mapping[str, Any] = data.get("account", {}) or {}
        positions_raw: Sequence[Mapping[str, Any]] = data.get("positions", []) or []

        # ===== 2. Equity / Cash / Margin =====
        nav = float(
            account.get("nav")  # אם יש שדה מפורש
            or account.get("total_equity")
            or account.get("net_liquidation", 0.0)
            or 0.0
        )
        nav_prev_close = account.get("nav_prev_close")

        total_equity = nav
        cash = float(account.get("cash", account.get("available_funds", 0.0) or 0.0))
        cash_available = float(account.get("cash_available", cash) or cash)
        margin_used = float(
            account.get("margin_used", account.get("maint_margin_req", 0.0) or 0.0)
        )
        margin_available = float(
            account.get("margin_available", account.get("excess_liquidity", 0.0) or 0.0)
        )

        # ===== 3. חשיפות (Gross / Net / Long / Short) =====
        gross_exposure = 0.0
        net_exposure = 0.0
        long_exposure = 0.0
        short_exposure = 0.0

        for pos in positions_raw:
            try:
                mv = float(pos.get("market_value", 0.0) or 0.0)
            except Exception:
                continue
            gross_exposure += abs(mv)
            net_exposure += mv
            if mv > 0:
                long_exposure += mv
            elif mv < 0:
                short_exposure += mv

        leverage = 0.0
        try:
            if total_equity > 0:
                leverage = gross_exposure / total_equity
        except Exception:
            leverage = 0.0

        # ===== 4. Exposure breakdowns =====
        by_asset_class = self._compute_exposure_by_key(
            positions_raw, key="asset_class", not_found_label="UNKNOWN"
        )
        by_currency = self._compute_exposure_by_key(
            positions_raw, key="currency", not_found_label="UNKNOWN"
        )
        by_sector = self._compute_exposure_by_key(
            positions_raw, key="sector", not_found_label="UNKNOWN"
        )
        by_country = self._compute_exposure_by_key(
            positions_raw, key="country", not_found_label="UNKNOWN"
        )

        exposure = PortfolioExposureBreakdown(
            by_asset_class=by_asset_class,
            by_sector=by_sector,
            by_industry={},
            by_currency=by_currency,
            by_country=by_country,
            by_factor={},
            by_curve_bucket={},
            gross_exposure=gross_exposure,
            net_exposure=net_exposure,
            long_exposure=long_exposure,
            short_exposure=short_exposure,
            # ריכוזיות – אפשר לשפר בהמשך מתוך positions
            concentration_top_1=0.0,
            concentration_top_5=0.0,
            concentration_top_10=0.0,
        )

        # ===== 5. Positions => PositionSnapshot =====
        positions: List[PositionSnapshot] = []
        for pos in positions_raw:
            try:
                mv = float(pos.get("market_value", 0.0) or 0.0)
                qty = float(pos.get("position", pos.get("qty", 0.0) or 0.0))
                last_price = float(pos.get("last_price", 0.0) or 0.0)
            except Exception:
                continue

            symbol = (
                pos.get("symbol")
                or pos.get("local_symbol")
                or pos.get("contract", {}).get("symbol")
                or "?"
            )

            side = "LONG" if qty >= 0 else "SHORT"
            weight = mv / total_equity if total_equity else 0.0
            entry_price = float(pos.get("avg_cost", pos.get("entry_price", 0.0)) or 0.0)

            pnl_breakdown = PositionPnlBreakdown(
                symbol=symbol,
                realized_today=float(pos.get("realized_pnl_today", 0.0) or 0.0),
                unrealized=float(pos.get("unrealized_pnl", 0.0) or 0.0),
            )

            # כרגע Risk per position לא ממומש → נשאיר None או default
            risk_metrics = PositionRiskMetrics(symbol=symbol)

            positions.append(
                PositionSnapshot(
                    symbol=symbol,
                    quantity=qty,
                    side=side,
                    last_price=last_price,
                    market_value=mv,
                    weight=weight,
                    entry_price=entry_price,
                    entry_ts=None,
                    realized_pnl=float(pos.get("realized_pnl", 0.0) or 0.0),
                    unrealized_pnl=float(pos.get("unrealized_pnl", 0.0) or 0.0),
                    pnl_breakdown=pnl_breakdown,
                    risk=risk_metrics,
                    tags=[],
                    metadata=dict(pos),
                )
            )

        num_positions = len(positions)
        num_long = sum(1 for p in positions if p.side.upper() == "LONG")
        num_short = sum(1 for p in positions if p.side.upper() == "SHORT")
        cash_pct = cash / total_equity if total_equity else 0.0

        # ===== 6. PnL & Risk placeholders =====
        pnl = PortfolioPnLBreakdown(
            realized_today=float(account.get("realized_pnl_today", 0.0) or 0.0),
            unrealized_today=float(account.get("unrealized_pnl", 0.0) or 0.0),
        )

        risk_metrics = PortfolioRiskMetrics(
            vol_annual=0.0,
            var_95=0.0,
            es_95=0.0,
            max_drawdown_1y=0.0,
        )

        return PortfolioSnapshot(
            as_of=self._now_utc(),
            portfolio_id=dctx.portfolio_id,
            nav=nav,
            nav_prev_close=nav_prev_close,
            total_equity=total_equity,
            cash=cash,
            cash_available=cash_available,
            margin_used=margin_used,
            margin_available=margin_available,
            leverage=leverage,
            gross_exposure=gross_exposure,
            net_exposure=net_exposure,
            long_exposure=long_exposure,
            short_exposure=short_exposure,
            exposure=exposure,
            positions=positions,
            num_positions=num_positions,
            num_long=num_long,
            num_short=num_short,
            cash_pct=cash_pct,
            pnl=pnl,
            risk=risk_metrics,
            turnover_1d=None,
            turnover_5d=None,
            turnover_20d=None,
            warnings=[],
            notes=[],
            metadata={},
        )

    # ------------------------------------------------------------------
    # Risk snapshot
    # ------------------------------------------------------------------
    def _build_risk_snapshot(self, dctx: DashboardContext) -> RiskSnapshot:
        """
        בניית RiskSnapshot HF-grade:

        • PortfolioRiskMetrics (Vol / VaR / ES / DD / TE / Beta / Corr).
        • risk_score (0–100) + traffic_light (green/yellow/red).
        • חריגות מלימיטים לפי DashboardContext (max_vol_annual / drawdown limits).
        """

        def _fetch_risk() -> Dict[str, Any]:
            risk_engine = getattr(self.ctx, "risk_engine", None)
            if risk_engine is None:
                return {}

            get_snapshot = getattr(risk_engine, "get_portfolio_risk_snapshot", None)
            if callable(get_snapshot):
                snap = get_snapshot(
                    portfolio_id=dctx.portfolio_id,
                    benchmark=dctx.benchmark,
                )
                if isinstance(snap, Mapping):
                    return dict(snap)
                return dict(getattr(snap, "__dict__", {}))

            # fallback – אם יש פונקציות פרטניות
            result: Dict[str, Any] = {}
            try:
                if hasattr(risk_engine, "compute_portfolio_vol"):
                    result["vol_annual"] = risk_engine.compute_portfolio_vol(
                        portfolio_id=dctx.portfolio_id
                    )
                if hasattr(risk_engine, "compute_var"):
                    result["var_95"] = risk_engine.compute_var(
                        portfolio_id=dctx.portfolio_id, level=0.95
                    )
                if hasattr(risk_engine, "compute_es"):
                    result["es_95"] = risk_engine.compute_es(
                        portfolio_id=dctx.portfolio_id, level=0.95
                    )
                if hasattr(risk_engine, "compute_max_drawdown"):
                    result["max_drawdown_1y"] = risk_engine.compute_max_drawdown(
                        portfolio_id=dctx.portfolio_id, horizon_days=252
                    )
                if hasattr(risk_engine, "compute_beta_corr"):
                    beta_corr = risk_engine.compute_beta_corr(
                        portfolio_id=dctx.portfolio_id, benchmark=dctx.benchmark
                    )
                    if isinstance(beta_corr, Mapping):
                        result["beta_vs_bench"] = beta_corr.get("beta", 0.0)
                        result["corr_vs_bench"] = beta_corr.get("corr", 0.0)
            except Exception as exc:
                logger.warning("RiskEngine fallback failed: %s", exc)

            return result

        data = self._safe_call("build_risk_snapshot", _fetch_risk, default={}) or {}

        vol_annual = float(
            data.get("vol_annual", data.get("volatility_annualized", 0.0)) or 0.0
        )
        var_95 = float(data.get("var_95", 0.0) or 0.0)
        es_95 = float(data.get("es_95", 0.0) or 0.0)
        max_dd_1y = float(data.get("max_drawdown_1y", 0.0) or 0.0)
        beta_vs_bench = float(
            data.get("beta_vs_bench", data.get("beta_vs_benchmark", 0.0)) or 0.0
        )
        corr_vs_bench = float(
            data.get("corr_vs_bench", data.get("corr_vs_benchmark", 0.0)) or 0.0
        )

        prisk = PortfolioRiskMetrics(
            vol_annual=vol_annual,
            var_95=var_95,
            es_95=es_95,
            max_drawdown_1y=max_dd_1y,
            beta_vs_bench=beta_vs_bench,
            corr_vs_bench=corr_vs_bench,
        )

        # ===== Risk score & traffic light =====
        warnings: List[str] = []
        breached_limits: List[str] = []
        limits_breached = False

        # מבוסס DashboardContext
        max_vol_allowed = getattr(dctx, "max_vol_annual", 0.25)
        soft_dd_limit = getattr(dctx, "drawdown_soft_limit", 0.10)
        hard_dd_limit = getattr(dctx, "drawdown_hard_limit", 0.20)

        # Max DD מניחים באחוזים (0.15 = 15%), אפשר להתאים לפי risk_engine
        dd_abs = abs(max_dd_1y)

        # בדיקות חריגה
        if vol_annual > max_vol_allowed:
            breached_limits.append(
                f"vol_annual {vol_annual:.2%} > max_vol_annual {max_vol_allowed:.2%}"
            )
        if dd_abs > hard_dd_limit:
            breached_limits.append(
                f"max_drawdown_1y {dd_abs:.2%} > hard_limit {hard_dd_limit:.2%}"
            )
        elif dd_abs > soft_dd_limit:
            breached_limits.append(
                f"max_drawdown_1y {dd_abs:.2%} > soft_limit {soft_dd_limit:.2%}"
            )

        limits_breached = len(breached_limits) > 0

        # traffic light
        traffic_light = "green"
        if dd_abs > hard_dd_limit or vol_annual > max_vol_allowed * 1.2:
            traffic_light = "red"
        elif dd_abs > soft_dd_limit or vol_annual > max_vol_allowed:
            traffic_light = "yellow"

        # risk_score בסיסי – אפשר לשפר בהמשך (0–100)
        risk_score = min(
            100.0,
            max(
                0.0,
                vol_annual * 100.0 + abs(var_95) * 1000.0 + dd_abs * 200.0,
            ),
        )

        # warnings טקסטואליים (משמשים גם כ-flags)
        if vol_annual > 0.5:
            warnings.append("⚠️ Annualized volatility > 50% – very high risk.")
        if beta_vs_bench > 1.5:
            warnings.append("⚠️ Beta vs benchmark > 1.5.")
        if limits_breached:
            warnings.append("⚠️ One or more risk limits have been breached.")

        return RiskSnapshot(
            as_of=self._now_utc(),
            portfolio_id=dctx.portfolio_id,
            portfolio_risk=prisk,
            risk_score=risk_score,
            traffic_light=traffic_light,
            limits_breached=limits_breached,
            breached_limits=breached_limits,
            flags=warnings,
        )

    # ------------------------------------------------------------------
    # Signals snapshot
    # ------------------------------------------------------------------
    def _build_signals_snapshot(self, dctx: DashboardContext) -> SignalsSnapshot:
        """
        בניית SignalsSnapshot HF-grade:

        • שילוב אותות מ:
          - core.signals_engine
          - core.pair_recommender
        • המרה ל-SignalItem (כולל strategy_family / regime / model).
        • ספירת:
          - n_new_today
          - n_conflicting (hook לעתיד)
          - פילוח לפי strategy/direction/regime.
        """

        def _fetch_signals() -> Dict[str, Any]:
            signals_engine = getattr(self.ctx, "signals_engine", None)
            pair_rec = getattr(self.ctx, "pair_recommender", None)

            items: List[Dict[str, Any]] = []

            # מקור ראשון: signals_engine
            if signals_engine is not None and hasattr(signals_engine, "get_signals"):
                try:
                    raw = signals_engine.get_signals(portfolio_id=dctx.portfolio_id)
                    for s in raw or []:
                        if isinstance(s, Mapping):
                            items.append(dict(s))
                        else:
                            items.append(getattr(s, "__dict__", {}) or {"raw": repr(s)})
                except Exception as exc:
                    logger.warning("Failed to load signals from signals_engine: %s", exc)

            # מקור שני: pair_recommender (רעיונות לזוגות)
            if pair_rec is not None and hasattr(pair_rec, "get_top_pairs"):
                try:
                    recs = pair_rec.get_top_pairs(limit=dctx.top_signals_limit)
                    for r in recs or []:
                        if isinstance(r, Mapping):
                            items.append(
                                {
                                    "type": "pair_recommendation",
                                    "pair_id": r.get("pair_id"),
                                    "symbol_1": r.get("symbol_1"),
                                    "symbol_2": r.get("symbol_2"),
                                    "score": r.get("score"),
                                    "edge": r.get("edge"),
                                    "strategy_family": "pairs_trading",
                                    "sub_strategy": "mean_reversion",
                                }
                            )
                except Exception as exc:
                    logger.warning("Failed to load pair recommendations: %s", exc)

            return {"items": items}

        data = self._safe_call("build_signals_snapshot", _fetch_signals, default={}) or {}
        items_raw: List[Dict[str, Any]] = data.get("items", []) or []

        now = self._now_utc()
        signal_items: List[SignalItem] = []
        n_new_today = 0
        n_conflicting = 0  # Hook לעתיד (כרגע לא מזהים קונפליקטים מורכבים)

        count_by_strategy: Dict[str, int] = {}
        count_by_direction: Dict[str, int] = {}
        count_by_regime: Dict[str, int] = {}

        for raw in items_raw:
            # סימבולים
            symbol_1 = (
                raw.get("symbol_1")
                or raw.get("symbol")
                or raw.get("underlying")
                or raw.get("pair_id")
                or "?"
            )
            symbol_2 = raw.get("symbol_2")

            # כיוון
            side = str(raw.get("side") or raw.get("direction") or "UNKNOWN").upper()
            side_map = {"BUY": "LONG", "SELL": "SHORT"}
            direction = side_map.get(side, side)

            # confidence / score / edge
            conf_raw = raw.get("confidence", raw.get("score", 0.0))
            try:
                confidence = float(conf_raw or 0.0)
            except Exception:
                confidence = 0.0

            z = raw.get("zscore", raw.get("z", raw.get("z_score")))
            try:
                zscore = float(z) if z is not None else None
            except Exception:
                zscore = None

            hl = raw.get("half_life", raw.get("hl"))
            try:
                half_life = float(hl) if hl is not None else None
            except Exception:
                half_life = None

            corr = raw.get("corr", raw.get("correlation"))
            try:
                corr_val = float(corr) if corr is not None else None
            except Exception:
                corr_val = None

            quality = raw.get("quality_score", raw.get("quality", raw.get("edge")))
            try:
                quality_score = float(quality) if quality is not None else None
            except Exception:
                quality_score = None

            regime = raw.get("regime")
            time_frame = raw.get("time_frame", raw.get("tf", "1D"))

            strategy_family = raw.get("strategy_family", "pairs_trading")
            sub_strategy = raw.get("sub_strategy", "mean_reversion")

            model_name = raw.get("model_name")
            model_version = raw.get("model_version")

            ts = raw.get("timestamp") or raw.get("as_of")
            created_at = ts if isinstance(ts, datetime) else None
            expires_at = raw.get("expires_at") if isinstance(raw.get("expires_at"), datetime) else None

            tags = list(raw.get("tags", []) or [])
            metadata = dict(raw)

            if created_at is not None and created_at.date() == now.date():
                n_new_today += 1

            item = SignalItem(
                symbol_1=symbol_1,
                symbol_2=symbol_2,
                direction=direction,
                confidence=confidence,
                edge=raw.get("edge"),
                zscore=zscore,
                half_life=half_life,
                corr=corr_val,
                quality_score=quality_score,
                regime=regime,
                time_frame=time_frame,
                strategy_family=strategy_family,
                sub_strategy=sub_strategy,
                model_name=model_name,
                model_version=model_version,
                created_at=created_at,
                expires_at=expires_at,
                tags=tags,
                metadata=metadata,
            )
            signal_items.append(item)

            # פילוחים
            count_by_strategy[strategy_family] = count_by_strategy.get(strategy_family, 0) + 1
            count_by_direction[direction] = count_by_direction.get(direction, 0) + 1
            if regime:
                count_by_regime[regime] = count_by_regime.get(regime, 0) + 1

        n_total = len(signal_items)

        return SignalsSnapshot(
            as_of=self._now_utc(),
            portfolio_id=dctx.portfolio_id,
            items=signal_items,
            n_new_today=n_new_today,
            n_conflicting=n_conflicting,
            n_total=n_total,
            count_by_strategy=count_by_strategy,
            count_by_direction=count_by_direction,
            count_by_regime=count_by_regime,
        )

    # ------------------------------------------------------------------
    # System health snapshot
    # ------------------------------------------------------------------
    def _build_system_health_snapshot(self, dctx: DashboardContext) -> SystemHealthSnapshot:
        """
        SystemHealthSnapshot:

        • חיבור ברוקר (IBKR).
        • מצב דאטה (fresh / stale).
        • Latency בסיסי.
        • SQL Status + Errors אחרונים.
        • אסטרטגיות פעילות + Agents Status (hook).
        """

        def _fetch_health() -> Dict[str, Any]:
            broker = getattr(self.ctx, "broker", None) or getattr(
                self.ctx, "ibkr", None
            )
            market_data = getattr(self.ctx, "market_data", None) or getattr(
                self.ctx, "market_data_router", None
            )

            broker_connected = False
            last_order_ts: Optional[datetime] = None
            last_price_update: Optional[datetime] = None
            data_latency_ms: float = 0.0
            running_strategies: List[str] = []
            recent_errors: List[str] = []
            data_fresh: bool = True

            # Broker connectivity
            if broker is not None:
                try:
                    if hasattr(broker, "is_connected"):
                        broker_connected = bool(broker.is_connected())
                except Exception as exc:
                    logger.warning("Failed to check broker connectivity: %s", exc)

                try:
                    if hasattr(broker, "get_last_order_timestamp"):
                        last_order_ts = broker.get_last_order_timestamp()
                except Exception:
                    pass

            # Market data health
            if market_data is not None:
                try:
                    if hasattr(market_data, "get_last_update_timestamp"):
                        last_price_update = market_data.get_last_update_timestamp()
                    if hasattr(market_data, "get_last_latency_ms"):
                        data_latency_ms = float(
                            market_data.get_last_latency_ms() or 0.0
                        )
                except Exception as exc:
                    logger.warning("Failed to load market data health: %s", exc)

            # Error bus / telemetry
            err_bus = getattr(self.ctx, "error_bus", None)
            if err_bus is not None and hasattr(err_bus, "get_recent_errors"):
                try:
                    recent_errors = list(err_bus.get_recent_errors(limit=20) or [])
                except Exception:
                    recent_errors = []

            # אסטרטגיות פעילות
            strat_mgr = getattr(self.ctx, "strategy_manager", None)
            if strat_mgr is not None and hasattr(strat_mgr, "list_running_strategies"):
                try:
                    running_strategies = list(
                        strat_mgr.list_running_strategies() or []
                    )
                except Exception:
                    running_strategies = []

            # SQL Status
            sql_ok = self.sql_store is not None
            sql_last_error: Optional[str] = None
            if self.sql_store is not None and hasattr(self.sql_store, "get_last_error"):
                try:
                    sql_last_error = self.sql_store.get_last_error()
                    if sql_last_error:
                        sql_ok = False
                except Exception:
                    pass

            # הערכה בסיסית האם הדאטה "טרייה"
            if last_price_update is not None:
                delta_seconds = (self._now_utc() - last_price_update).total_seconds()
                # לדוגמה: אם מעל שעה – נחשב כלא טרי
                data_fresh = delta_seconds <= 3600

            # Agents status (hook)
            agents_status: Dict[str, str] = {}
            agents_mgr = getattr(self.ctx, "agents_manager", None)
            if agents_mgr is not None and hasattr(agents_mgr, "get_status_dict"):
                try:
                    agents_status = dict(agents_mgr.get_status_dict() or {})
                except Exception:
                    agents_status = {}

            # אפשר לחבר פה למדדים מהמערכת (CPU, RAM) אם יש לך טלמטריה
            cpu_load_pct = None
            memory_used_pct = None

            return {
                "broker_connected": broker_connected,
                "data_fresh": data_fresh,
                "data_latency_ms": data_latency_ms,
                "last_price_update": last_price_update,
                "last_order_ts": last_order_ts,
                "sql_ok": sql_ok,
                "sql_last_error": sql_last_error,
                "running_strategies": running_strategies,
                "recent_errors": recent_errors,
                "agents_status": agents_status,
                "cpu_load_pct": cpu_load_pct,
                "memory_used_pct": memory_used_pct,
            }

        data = self._safe_call("build_system_health_snapshot", _fetch_health, default={}) or {}

        return SystemHealthSnapshot(
            as_of=self._now_utc(),
            broker_connected=bool(data.get("broker_connected", False)),
            data_fresh=bool(data.get("data_fresh", True)),
            data_latency_ms=float(data.get("data_latency_ms", 0.0) or 0.0),
            last_price_update=data.get("last_price_update"),
            last_order_ts=data.get("last_order_ts"),
            sql_ok=bool(data.get("sql_ok", self.sql_store is not None)),
            sql_last_error=data.get("sql_last_error"),
            running_strategies=list(data.get("running_strategies", []) or []),
            recent_errors=list(data.get("recent_errors", []) or []),
            cpu_load_pct=data.get("cpu_load_pct"),
            memory_used_pct=data.get("memory_used_pct"),
            agents_status=dict(data.get("agents_status", {}) or {}),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build_dashboard_snapshot(self, dctx: DashboardContext) -> DashboardSnapshot:
        """
        פונקציה ראשית: בונה Snapshot מלא לדשבורד.
        """
        market = self._build_market_snapshot(dctx)
        portfolio = self._build_portfolio_snapshot(dctx)
        risk = self._build_risk_snapshot(dctx)
        signals = self._build_signals_snapshot(dctx)
        system = self._build_system_health_snapshot(dctx)

        # חשוב: להשתמש ב-Risk metrics גם בתוך ה-PortfolioSnapshot
        try:
            portfolio.risk = risk.portfolio_risk
        except Exception:
            pass

        snapshot = DashboardSnapshot(
            ctx=dctx,
            as_of=self._now_utc(),
            market=market,
            portfolio=portfolio,
            risk=risk,
            signals=signals,
            system=system,
        )

        self._last_snapshot = snapshot

        if self.enable_persistence and self.sql_store is not None:
            try:
                self.sql_store.save_dashboard_snapshot(snapshot)
            except Exception as exc:
                logger.warning(
                    "[DashboardService] Failed to save dashboard snapshot: %s",
                    exc,
                    exc_info=True,
                )

        return snapshot

    # ------------------------------------------------------------------
    # Utilities נוספים לשימוש דשבורד/Agents
    # ------------------------------------------------------------------
    def get_last_snapshot(self) -> Optional[DashboardSnapshot]:
        """מחזיר את ה-snapshot האחרון מהזיכרון (אם קיים)."""
        return self._last_snapshot

    def load_last_snapshot_from_sql(
        self, ctx_key: Optional[str] = None
    ) -> Optional[DashboardSnapshot]:
        """
        טעינת snapshot אחרון מ-SQL (למשל בעת restart של האפליקציה).
        """
        if self.sql_store is None:
            return None

        try:
            snap = self.sql_store.load_last_dashboard_snapshot(ctx_key=ctx_key)
        except Exception as exc:
            logger.warning(
                "[DashboardService] Failed to load last snapshot from SQL: %s", exc
            )
            return None

        self._last_snapshot = snap
        return snap

    def diff_snapshots(
        self,
        old: Optional[DashboardSnapshot],
        new: DashboardSnapshot,
    ) -> Dict[str, Any]:
        """
        Diff בסיסי בין שני snapshotים — לציור חיצים / Alerts.
        """
        if old is None:
            return {
                "has_prev": False,
                "equity_change": None,
                "unrealized_pnl_change": None,
                "vix_change": None,
                "risk_change": None,
            }

        try:
            equity_change = new.portfolio.total_equity - old.portfolio.total_equity
        except Exception:
            equity_change = None

        try:
            unrealized_pnl_change = (
                new.portfolio.pnl.unrealized_today - old.portfolio.pnl.unrealized_today
            )
        except Exception:
            unrealized_pnl_change = None

        try:
            vix_change = new.market.vix_level - old.market.vix_level
        except Exception:
            vix_change = None

        try:
            risk_change = {
                "vol": new.risk.portfolio_risk.vol_annual
                - old.risk.portfolio_risk.vol_annual,
                "var_95": new.risk.portfolio_risk.var_95
                - old.risk.portfolio_risk.var_95,
                "es_95": new.risk.portfolio_risk.es_95
                - old.risk.portfolio_risk.es_95,
            }
        except Exception:
            risk_change = None

        return {
            "has_prev": True,
            "equity_change": equity_change,
            "unrealized_pnl_change": unrealized_pnl_change,
            "vix_change": vix_change,
            "risk_change": risk_change,
        }
