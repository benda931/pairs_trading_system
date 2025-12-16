# -*- coding: utf-8 -*-
"""
common/live_profiles.py ג€” Live Pair Profile Contract (HF-Grade)
================================================================

׳§׳•׳‘׳¥ ׳–׳” ׳׳’׳“׳™׳¨ ׳׳× ׳—׳•׳–׳” ׳”׳׳™׳™׳‘ ׳”׳¨׳©׳׳™ ׳׳–׳•׳’ ׳׳¡׳—׳¨ ׳׳—׳“ (Pair) ׳‘׳¨׳׳× ׳§׳¨׳ ׳’׳™׳“׳•׳¨.

׳”׳׳˜׳¨׳”:
    - ׳׳”׳™׳•׳× "׳©׳₪׳× ׳×׳•׳•׳" ׳׳—׳™׳“׳” ׳‘׳™׳:
        * Research / Optimization / ML / Macro
        * ׳׳ ׳•׳¢ ׳”׳׳¡׳—׳¨ ׳”׳—׳™ (Live Trading Engine)
        * ׳”-Dashboard (׳˜׳׳‘ ׳₪׳¨׳•׳˜׳₪׳•׳׳™׳• / ׳׳™׳™׳‘ / ׳׳ ׳׳™׳˜׳™׳§׳”)
    - ׳›׳ ׳׳•׳“׳•׳ ׳׳—׳§׳¨/׳׳•׳₪׳˜׳™׳׳™׳–׳¦׳™׳”/ML ׳›׳•׳×׳‘ ׳׳›׳׳ ׳׳× ׳”׳—׳׳˜׳•׳×׳™׳•.
    - ׳׳ ׳•׳¢ ׳”׳׳™׳™׳‘ *׳¨׳§ ׳§׳•׳¨׳* ׳׳× ׳”׳₪׳¨׳•׳₪׳™׳ ׳•׳׳‘׳¦׳¢ ׳׳₪׳™ ׳”׳—׳•׳§׳™׳ ׳•׳”׳₪׳¨׳׳˜׳¨׳™׳ ׳©׳‘׳•
      ׳×׳—׳× ׳׳’׳‘׳׳•׳× ׳”׳¡׳™׳›׳•׳ ׳”׳’׳׳•׳‘׳׳™׳•׳×.

׳¢׳§׳¨׳•׳ ׳•׳×:
    - ׳׳™׳ ׳׳•׳’׳™׳§׳” ׳¢׳¡׳§׳™׳× ׳›׳‘׳“׳” ׳‘׳×׳•׳ ׳”׳׳•׳“׳ ׳¢׳¦׳׳• (׳›׳׳¢׳˜ ׳ ׳˜׳• Data Contract).
    - ׳©׳“׳•׳× ׳׳—׳•׳׳§׳™׳ ׳׳§׳˜׳’׳•׳¨׳™׳•׳× ׳‘׳¨׳•׳¨׳•׳×:
        Identity, Trading Rules, Sizing & Risk, Quality & ML, Macro/Regime, Operational.
    - ׳¨׳•׳‘ ׳”׳©׳“׳•׳× ׳׳•׳₪׳¦׳™׳•׳ ׳׳™׳™׳ ׳›׳“׳™ ׳׳׳₪׳©׳¨ ׳‘׳ ׳™׳™׳” ׳”׳“׳¨׳’׳×׳™׳× (Partial Filling).
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional, Literal

from pydantic import BaseModel, Field, ConfigDict


class LivePairProfile(BaseModel):
    """
    ׳₪׳¨׳•׳₪׳™׳ ׳׳™׳™׳‘ ׳׳׳ ׳׳–׳•׳’ ׳׳—׳“.

    ׳–׳”׳• ׳”-"DNA" ׳©׳ ׳”׳–׳•׳’ ׳‘׳׳¡׳—׳¨ ׳—׳™:
        - ׳׳™׳ ׳׳–׳”׳•׳× ׳׳•׳×׳•
        - ׳׳™׳ ׳׳¡׳—׳•׳¨ ׳‘׳• (׳—׳•׳§׳™ ׳›׳ ׳™׳¡׳”/׳™׳¦׳™׳׳”/׳¡׳˜׳•׳₪׳™׳/׳–׳׳)
        - ׳׳™׳ ׳׳§׳‘׳•׳¢ ׳’׳•׳“׳ ׳₪׳•׳–׳™׳¦׳™׳” ׳•׳—׳©׳™׳₪׳”
        - ׳׳™׳–׳” ׳׳™׳›׳•׳×/Edge ׳™׳© ׳׳• ׳”׳™׳¡׳˜׳•׳¨׳™׳× ׳•-ML׳™׳×
        - ׳‘׳׳™׳–׳” ׳׳©׳˜׳¨׳™׳ (Regimes) ׳׳•׳×׳¨/׳׳¡׳•׳¨ ׳׳¡׳—׳•׳¨ ׳‘׳•
        - ׳”׳׳ ׳₪׳¢׳™׳, ׳׳•׳©׳”׳”, ׳•׳׳” ׳¡׳™׳‘׳× ׳”׳׳¦׳‘ ׳”׳ ׳•׳›׳—׳™

    ׳©׳“׳•׳× ׳¨׳‘׳™׳ ׳”׳ ׳׳•׳₪׳¦׳™׳•׳ ׳׳™׳™׳ ׳›׳“׳™ ׳©׳×׳•׳›׳ ׳׳”׳×׳—׳™׳ ׳₪׳©׳•׳˜ ׳•׳׳”׳•׳¡׳™׳£ ׳¢׳•׳׳§ ׳׳׳•׳¨׳ ׳”׳–׳׳.
    """

    # ׳₪׳™׳™׳“׳ ׳˜׳™׳§ v2 ג€“ ׳”׳’׳“׳¨׳•׳× ׳›׳׳׳™׳•׳×
    model_config = ConfigDict(
        extra="ignore",              # ׳”׳×׳¢׳׳׳•׳× ׳׳©׳“׳•׳× ׳׳ ׳׳•׳›׳¨׳™׳ ׳‘׳˜׳¢׳™׳ ׳”
        validate_assignment=True,    # ׳•׳׳™׳“׳¦׳™׳” ׳’׳ ׳‘׳©׳™׳ ׳•׳™ ׳׳׳—׳¨ ׳™׳¦׳™׳¨׳”
        arbitrary_types_allowed=True,
    )

    # ======================================================================
    # 1. Identity & Meta ג€” ׳׳™ ׳”׳–׳•׳’, ׳׳™׳₪׳” ׳”׳•׳ ׳ ׳¡׳—׳¨, ׳׳” ׳”׳”׳§׳©׳¨ ׳©׳׳•
    # ======================================================================
    pair_id: str = Field(
        ...,
        description="׳׳–׳”׳” ׳™׳™׳—׳•׳“׳™ ׳׳–׳•׳’, ׳׳“׳•׳’׳׳” 'QQQ_SOXX_US_EQ_1D' (׳׳©׳׳© ׳›׳׳₪׳×׳— ׳¨׳׳©׳™).",
    )
    sym_x: str = Field(
        ...,
        description="Leg X (׳‘׳“׳¨׳ ׳›׳׳ leg ׳”׳׳•׳ ׳’). ׳׳“׳•׳’׳׳”: 'QQQ'.",
    )
    sym_y: str = Field(
        ...,
        description="Leg Y (׳‘׳“׳¨׳ ׳›׳׳ leg ׳”׳©׳•׳¨׳˜). ׳׳“׳•׳’׳׳”: 'SOXX'.",
    )

    asset_class: str = Field(
        "EQUITY",
        description="׳׳—׳׳§׳× ׳ ׳›׳¡: EQUITY / ETF / FUTURES / FX / OPTIONS ׳•׳›׳•'.",
    )
    market: Optional[str] = Field(
        default=None,
        description="׳–׳™׳”׳•׳™ ׳©׳•׳§/׳‘׳•׳¨׳¡׳” ׳›׳׳׳™׳×, ׳׳“׳•׳’׳׳” 'US-STK', 'EU-STK', 'CME-FUT'.",
    )
    exchange_x: Optional[str] = Field(
        default=None,
        description="׳‘׳•׳¨׳¡׳” ׳-leg X, ׳׳“׳•׳’׳׳” 'NASDAQ', 'NYSE'.",
    )
    exchange_y: Optional[str] = Field(
        default=None,
        description="׳‘׳•׳¨׳¡׳” ׳-leg Y.",
    )

    base_currency: str = Field(
        "USD",
        description="׳׳˜׳‘׳¢ ׳‘׳¡׳™׳¡ ׳-PnL ׳•׳׳ ׳™׳”׳•׳ ׳¡׳™׳›׳•׳ (׳‘׳“׳¨׳ ׳›׳׳ USD).",
    )
    timezone: Optional[str] = Field(
        default=None,
        description="׳׳–׳•׳¨ ׳–׳׳ ׳¨׳׳©׳™ ׳׳׳¡׳—׳¨ ׳‘׳–׳•׳’ (׳׳׳©׳ 'America/New_York').",
    )

    timeframe: str = Field(
        "1D",
        description="׳˜׳™׳™׳-׳₪׳¨׳™׳™׳ ׳׳¨׳›׳–׳™ ׳׳—׳™׳©׳•׳‘׳™ Spread / Z / Backtest (׳׳׳©׳ '1D', '1H').",
    )
    data_source: str = Field(
        "IBKR",
        description="׳׳§׳•׳¨ ׳ ׳×׳•׳ ׳™׳ ׳׳₪׳•׳¢׳: IBKR / Yahoo / Mixed / DuckDB ׳•׳›׳•'.",
    )

    sector_label: Optional[str] = Field(
        default=None,
        description="׳¡׳§׳˜׳•׳¨/׳×׳׳” ׳›׳׳׳™׳× (׳׳׳©׳ 'Tech', 'Semiconductors', 'Growth').",
    )
    cluster_id: Optional[str] = Field(
        default=None,
        description="Cluster / ׳§׳׳׳¡׳˜׳¨ ׳§׳•׳¨׳׳¦׳™׳”/Factor (׳׳©׳•׳™׳ ׳-matrix_helpers / clustering).",
    )

    # ======================================================================
    # 2. Trading Rules ג€” ׳—׳•׳§׳™ ׳”׳׳¡׳—׳¨ ׳”׳‘׳¡׳™׳¡׳™׳™׳ ׳׳–׳•׳’ ׳”׳–׳”
    # ======================================================================
    direction_convention: str = Field(
        "long_x_short_y_on_positive_z",
        description=(
            "׳ convention ׳׳₪׳¨׳©׳ ׳•׳× ׳¡׳™׳׳ ׳”-Z: "
            "׳׳׳©׳ 'long_x_short_y_on_positive_z' ׳׳•׳׳¨: Z>0 ג†’ long X, short Y."
        ),
    )

    # --- ׳›׳ ׳™׳¡׳”/׳™׳¦׳™׳׳” ׳׳₪׳™ Z-Score ---
    z_entry: float = Field(
        2.0,
        description="׳¡׳£ Z ׳׳›׳ ׳™׳¡׳”: abs(Z) >= z_entry.",
    )
    z_exit: float = Field(
        0.5,
        description="׳¡׳£ Z ׳׳™׳¦׳™׳׳” ׳‘׳¡׳™׳¡׳™׳×: abs(Z) <= z_exit.",
    )
    z_take_profit: Optional[float] = Field(
        default=None,
        description="Z ׳׳¨׳•׳•׳— ׳™׳×׳¨ (TP). ׳׳ None ג€“ ׳׳™׳ TP ׳׳₪׳™ Z, ׳¨׳§ z_exit / ׳¡׳˜׳•׳₪׳™׳ ׳׳—׳¨׳™׳.",
    )
    z_hard_stop: Optional[float] = Field(
        default=None,
        description="Z ׳׳¡׳˜׳•׳₪ ׳§׳™׳¦׳•׳ ׳™ (Hard Stop). ׳׳ None ג€“ ׳׳™׳ Hard Stop ׳׳₪׳™ Z.",
    )

    # --- ׳׳’׳‘׳׳•׳× ׳–׳׳ / Re-entry ---
    min_holding_bars: int = Field(
        0,
        description="׳׳™׳ ׳™׳׳•׳ ׳‘׳¨׳™׳ ׳׳”׳—׳–׳§׳” ׳׳₪׳ ׳™ ׳©׳׳•׳×׳¨ ׳׳¡׳’׳•׳¨ (׳׳•׳ ׳¢ '׳ ׳™׳¢׳•׳¨' ׳׳”׳™׳¨ ׳׳“׳™).",
        ge=0,
    )
    max_holding_bars: int = Field(
        999_999,
        description="׳׳§׳¡׳™׳׳•׳ ׳‘׳¨׳™׳ ׳׳”׳—׳–׳§׳”. ׳׳—׳¨׳™ ׳–׳” ׳¡׳•׳’׳¨׳™׳ ׳‘׳›׳ ׳׳§׳¨׳”.",
        ge=1,
    )
    reentry_cooldown_bars: int = Field(
        0,
        description="׳›׳׳” ׳‘׳¨׳™׳ ׳—׳™׳™׳‘׳™׳ ׳׳¢׳‘׳•׳¨ ׳‘׳™׳ ׳¡׳’׳™׳¨׳” ׳׳₪׳×׳™׳—׳” ׳—׳“׳©׳” ׳‘׳׳•׳×׳• ׳›׳™׳•׳•׳ ׳‘׳׳¡׳—׳¨.",
        ge=0,
    )

    # --- ׳×׳ ׳׳™׳ ׳ ׳•׳¡׳₪׳™׳ (׳׳•׳₪׳¦׳™׳•׳ ׳׳™׳™׳) ---
    min_spread_std: Optional[float] = Field(
        default=None,
        description="׳¡׳˜׳™׳™׳× ׳×׳§׳ ׳׳™׳ ׳™׳׳׳™׳× ׳©׳ ׳”-Spread (׳›׳“׳™ ׳׳”׳™׳׳ ׳¢ ׳׳–׳•׳’׳•׳× '׳׳×׳™׳').",
    )
    min_corr_lookback: Optional[int] = Field(
        default=None,
        description="׳׳¡׳₪׳¨ ׳‘׳¨׳™׳ ׳׳™׳ ׳™׳׳׳™ ׳׳—׳™׳©׳•׳‘ ׳§׳•׳¨׳׳¦׳™׳”/׳§׳•׳׳™׳ ׳˜׳’׳¨׳¦׳™׳”.",
    )
    require_cointegration: bool = Field(
        True,
        description="׳”׳׳ ׳ ׳“׳¨׳© ׳©׳”׳–׳•׳’ ׳™׳¢׳‘׳•׳¨ ׳‘׳“׳™׳§׳× ׳§׳•׳׳™׳ ׳˜׳’׳¨׳¦׳™׳” (Engle-Granger/Johansen).",
    )
    allow_short_both_legs: bool = Field(
        False,
        description="׳”׳׳ ׳׳•׳×׳¨ ׳׳₪׳×׳•׳— ׳₪׳•׳–׳™׳¦׳™׳” ׳¢׳ ׳©׳ ׳™ ׳¨׳’׳׳™׳™׳ ׳‘׳©׳•׳¨׳˜ (׳׳׳©׳ ׳‘׳–׳•׳’׳™ ׳§׳¨׳™׳₪׳˜׳•/׳₪׳§׳˜׳•׳¨׳™׳).",
    )

    slippage_bp: float = Field(
        5.0,
        description="׳”׳—׳׳§׳” ׳¦׳₪׳•׳™׳” ׳‘-bps (0.01% = 1bp). ׳׳©׳₪׳™׳¢ ׳’׳ ׳¢׳ Backtest ׳•׳’׳ ׳¢׳ ׳”׳’׳“׳¨׳× LIMIT.",
        ge=0.0,
    )
    max_spread_bps_intraday: Optional[float] = Field(
        default=None,
        description="׳×׳§׳¨׳× Spread intraday ׳‘-bps (׳׳׳ ׳™׳¢׳× ׳›׳ ׳™׳¡׳” ׳‘׳×׳ ׳׳™ ׳׳¨׳•׳•׳— ׳©׳•׳§ ׳—׳¨׳™׳’).",
    )

    # ======================================================================
    # 3. Sizing & Local Risk ג€” ׳’׳•׳“׳ ׳₪׳•׳–׳™׳¦׳™׳” ׳•׳׳’׳‘׳׳•׳× ׳¡׳™׳›׳•׳ ׳‘׳¨׳׳× ׳”׳–׳•׳’
    # ======================================================================
    sizing_mode: Literal["fixed_notional", "vol_target", "risk_parity"] = Field(
        "fixed_notional",
        description="׳©׳™׳˜׳× ׳§׳‘׳™׳¢׳× ׳’׳•׳“׳ ׳”׳₪׳•׳–׳™׳¦׳™׳”: fixed_notional / vol_target / risk_parity.",
    )
    base_notional_usd: float = Field(
        5_000.0,
        description="׳ ׳•׳˜׳™׳•׳ ׳ ׳‘׳¡׳™׳¡׳™ ׳׳₪׳•׳–׳™׳¦׳™׳” ׳׳—׳× ׳‘׳–׳•׳’ (׳׳₪׳ ׳™ ׳”׳×׳׳׳•׳× ׳׳₪׳™ ML / Regime / Risk).",
        ge=0.0,
    )
    vol_target_annual: Optional[float] = Field(
        default=None,
        description="Vol annualized ׳¨׳¦׳•׳™ ׳׳–׳•׳’ ׳‘׳׳¡׳’׳¨׳× Vol Targeting (׳׳ None ג€“ ׳׳ ׳‘׳©׳™׳׳•׳©).",
    )
    risk_budget_fraction: Optional[float] = Field(
        default=None,
        description=(
            "׳׳—׳•׳– ׳׳×׳§׳¦׳™׳‘ ׳”׳¡׳™׳›׳•׳ ׳”׳›׳•׳׳ ׳©׳׳•׳§׳¦׳” ׳׳–׳•׳’ (0ג€“1). "
            "׳׳©׳׳© ׳‘׳¢׳×׳™׳“ ׳׳¨׳™׳¡׳§-׳₪׳¨׳™׳˜׳™/׳”׳§׳¦׳׳•׳× ׳—׳›׳׳•׳×."
        ),
    )

    leverage_max: float = Field(
        1.0,
        description="׳׳™׳ ׳•׳£ ׳׳§׳¡׳™׳׳׳™ ׳׳–׳•׳’ (׳™׳—׳¡׳™). ׳׳¨׳•׳‘ 1.0 ׳׳–׳•׳’׳™ ׳׳ ׳™׳•׳×/ETF.",
        ge=0.0,
    )
    pair_max_exposure_usd: Optional[float] = Field(
        default=None,
        description="׳×׳§׳¨׳× ׳—׳©׳™׳₪׳” ׳׳§׳•׳׳™׳× ׳׳–׳•׳’. ׳׳ None ג€“ ׳׳©׳×׳׳©׳™׳ ׳‘׳¢׳¨׳ ׳’׳׳•׳‘׳׳™ ׳׳”-Config.",
    )
    max_open_trades_per_pair: int = Field(
        1,
        description="׳›׳׳” ׳₪׳•׳–׳™׳¦׳™׳•׳× ׳©׳ ׳׳•׳×׳• ׳–׳•׳’ ׳׳•׳×׳¨ ׳׳₪׳×׳•׳— ׳‘׳• ׳–׳׳ ׳™׳×.",
        ge=1,
    )

    weight_in_portfolio: float = Field(
        0.0,
        description="׳׳©׳§׳ ׳׳•׳¢׳“׳£ ׳‘׳×׳™׳§ (0ג€“1), ׳׳©׳™׳׳•׳© Allocator ׳—׳›׳ (׳׳ ׳—׳•׳‘׳” ׳‘׳©׳׳‘ ׳¨׳׳©׳•׳).",
        ge=0.0,
        le=1.0,
    )
    min_trade_value_usd: Optional[float] = Field(
        default=None,
        description="׳¢׳¨׳ ׳׳™׳ ׳™׳׳׳™ ׳׳¢׳¡׳§׳” ׳›׳“׳™ ׳׳ ׳׳™׳™׳¦׳¨ ׳₪׳•׳–׳™׳¦׳™׳•׳× '׳–׳‘׳' ׳§׳˜׳ ׳•׳× ׳׳“׳™.",
    )

    # ======================================================================
    # 4. Quality, Stats & ML ג€” ׳׳™׳›׳•׳× ׳”׳–׳•׳’, ׳¡׳˜׳˜׳™׳¡׳˜׳™׳§׳•׳× ׳•-Edge ׳—׳›׳
    # ======================================================================
    # --- ׳¡׳˜׳˜׳™׳¡׳˜׳™׳§׳•׳× mean-reversion ׳•׳§׳•׳¨׳׳¦׳™׳” ---
    half_life_bars: Optional[float] = Field(
        default=None,
        description="Half-life ׳׳•׳¢׳¨׳ ׳©׳ ׳”-Spread (׳‘׳‘׳¨׳™׳). ׳§׳˜׳ ג‡’ mean reversion ׳׳”׳™׳¨.",
    )
    hurst_exponent: Optional[float] = Field(
        default=None,
        description="Hurst exponent ׳©׳ ׳”-Spread. <0.5 ׳׳¢׳™׳“ ׳¢׳ Mean Reversion.",
    )
    corr_lookback_bars: Optional[int] = Field(
        default=None,
        description="׳׳•׳¨׳ ׳—׳׳•׳ ׳׳—׳™׳©׳•׳‘ ׳§׳•׳¨׳׳¦׳™׳” (׳׳ ׳©׳•׳ ׳” ׳׳‘׳¨׳™׳¨׳× ׳׳—׳“׳ ׳’׳׳•׳‘׳׳™׳×).",
    )
    rolling_corr: Optional[float] = Field(
        default=None,
        description="׳§׳•׳¨׳׳¦׳™׳” ׳׳׳•׳¦׳¢׳×/׳׳—׳¨׳•׳ ׳” ׳©׳ ׳”׳–׳•׳’ ׳‘׳×׳§׳•׳₪׳× ׳”׳׳•׳₪׳˜׳™׳׳™׳–׳¦׳™׳”.",
    )

    cointegration_method: Optional[str] = Field(
        default=None,
        description="׳©׳™׳˜׳× ׳‘׳“׳™׳§׳× ׳§׳•׳׳™׳ ׳˜׳’׳¨׳¦׳™׳” (Engle-Granger, Johansen, CADF ׳•׳›׳•').",
    )
    cointegration_pvalue: Optional[float] = Field(
        default=None,
        description="p-value ׳©׳ ׳‘׳“׳™׳§׳× ׳”׳§׳•׳׳™׳ ׳˜׳’׳¨׳¦׳™׳” ׳׳–׳•׳’.",
    )
    adf_pvalue_spread: Optional[float] = Field(
        default=None,
        description="p-value ׳©׳ ADF ׳¢׳ ׳”-Spread (׳¡׳˜׳¦׳™׳•׳ ׳¨׳™׳•׳×).",
    )

    # --- ׳¦׳™׳•׳ ׳™׳ (Scores) ׳׳”-Recommender / Opt ---
    score_total: float = Field(
        0.0,
        description="׳¦׳™׳•׳ ׳׳©׳•׳§׳׳ ׳›׳׳׳™ (Composite Score) ׳׳”-Recommender/Opt.",
    )
    score_corr_stability: float = Field(
        0.0,
        description="׳׳“׳“ ׳™׳¦׳™׳‘׳•׳× ׳§׳•׳¨׳׳¦׳™׳” (0ג€“1 ׳׳• ׳¡׳§׳™׳™׳ ׳׳—׳¨ ׳׳₪׳™ ׳”׳׳¢׳¨׳›׳× ׳©׳׳).",
    )
    score_cointegration: float = Field(
        0.0,
        description="׳׳“׳“ ׳׳™׳›׳•׳× ׳§׳•׳׳™׳ ׳˜׳’׳¨׳¦׳™׳” (׳׳‘׳•׳¡׳¡ p-value/׳¡׳˜׳˜׳™׳¡׳˜׳™׳§׳”).",
    )
    score_mean_reversion_speed: float = Field(
        0.0,
        description="׳׳“׳“ ׳׳”׳™׳¨׳•׳× Mean Reversion (׳׳ ׳•׳¨׳׳ ׳-half-life/׳׳“׳“׳™׳ ׳׳—׳¨׳™׳).",
    )

    # --- ׳×׳•׳¦׳׳•׳× Backtest ׳׳¨׳›׳–׳™׳•׳× ---
    backtest_sharpe: float = Field(
        0.0,
        description="Sharpe Ratio ׳׳”-Backtest ׳©׳ ׳‘׳—׳¨ ׳׳׳•׳₪׳˜׳™׳׳™׳–׳¦׳™׳”.",
    )
    backtest_sortino: float = Field(
        0.0,
        description="Sortino Ratio ׳׳”-Backtest.",
    )
    backtest_max_drawdown: float = Field(
        0.0,
        description="Max Drawdown (׳™׳—׳¡׳™, ׳׳׳©׳ -0.15 = -15%).",
    )
    backtest_winrate: float = Field(
        0.0,
        description="׳׳—׳•׳– ׳¢׳¡׳§׳׳•׳× ׳¨׳•׳•׳—׳™׳•׳× (0ג€“1).",
    )
    backtest_trades_count: int = Field(
        0,
        description="׳׳¡׳₪׳¨ ׳¢׳¡׳§׳׳•׳× ׳‘-Backtest (׳׳“׳“ ׳™׳¦׳™׳‘׳•׳× ׳¡׳˜׳˜׳™׳¡׳˜׳™׳×).",
        ge=0,
    )

    # --- ׳×׳•׳¦׳¨׳™ ML/AutoML ---
    ml_edge_score: Optional[float] = Field(
        default=None,
        description="Edge ׳׳‘׳•׳¡׳¡ ML (׳׳׳©׳ scaled ׳‘׳™׳ -1 ׳-1 ׳׳• 0ג€“1).",
    )
    ml_confidence: Optional[float] = Field(
        default=None,
        description="׳¨׳׳× ׳‘׳™׳˜׳—׳•׳ (0ג€“1) ׳‘׳×׳—׳–׳™׳× ׳”׳׳•׳“׳ ׳¢׳‘׳•׳¨ ׳”׳–׳•׳’.",
    )
    ml_predicted_horizon_bars: Optional[int] = Field(
        default=None,
        description="׳׳•׳₪׳§ ׳–׳׳ (׳‘׳‘׳¨׳™׳) ׳©׳‘׳• ׳”׳׳•׳“׳ ׳¦׳•׳₪׳” ׳׳× ׳”׳׳™׳¡׳₪׳¨׳“/׳¨׳•׳•׳—.",
    )
    model_version: Optional[str] = Field(
        default=None,
        description="׳’׳¨׳¡׳× ׳”׳׳•׳“׳/׳₪׳™׳™׳₪׳׳™׳™׳ (׳׳“׳•׳’׳׳” 'pairs_ml_v3.1').",
    )

    # ======================================================================
    # 5. Macro & Regime ג€” ׳”׳§׳©׳¨ ׳׳׳§׳¨׳•/׳׳©׳˜׳¨ ׳©׳•׳§
    # ======================================================================
    regime_id: Optional[str] = Field(
        default=None,
        description="׳׳–׳”׳” ׳׳©׳˜׳¨/׳׳¦׳‘ (׳׳׳©׳ 'risk_on', 'risk_off', 'crash', 'range').",
    )
    macro_regime_id: Optional[str] = Field(
        default=None,
        description="׳׳©׳˜׳¨ ׳׳׳§׳¨׳• ׳’׳׳•׳‘׳׳™ (׳׳₪׳™ macro_tab, ׳׳•׳— ׳©׳ ׳” ׳›׳׳›׳׳™ ׳•׳›׳•').",
    )
    vol_regime_id: Optional[str] = Field(
        default=None,
        description="׳׳©׳˜׳¨ ׳×׳ ׳•׳“׳×׳™׳•׳× (׳׳׳©׳ 'low_vol', 'high_vol').",
    )
    macro_score: Optional[float] = Field(
        default=None,
        description="׳¦׳™׳•׳ ׳׳׳§׳¨׳• ׳›׳׳׳™ ׳¢׳‘׳•׳¨ ׳”׳–׳•׳’ (0ג€“1 ׳׳• -1ג€“1).",
    )

    # ======================================================================
    # 6. Operational / Control ג€” ׳©׳׳™׳˜׳”, ׳׳™׳×׳•׳’, ׳×׳™׳¢׳•׳“
    # ======================================================================
    is_active: bool = Field(
        False,
        description="׳”׳׳ ׳”׳–׳•׳’ ׳׳׳•׳©׳¨ ׳׳׳¡׳—׳¨ ׳—׳™ ׳›׳¨׳’׳¢ (׳×׳—׳× ׳׳’׳‘׳׳•׳× ׳’׳׳•׳‘׳׳™׳•׳×).",
    )
    is_suspended: bool = Field(
        False,
        description="׳”׳׳ ׳”׳–׳•׳’ ׳׳•׳©׳”׳” ׳–׳׳ ׳™׳× (׳¢׳•׳§׳£ ׳׳× is_active ׳׳¦׳•׳¨׳ ׳—׳™׳¨׳•׳/׳—׳“׳©׳•׳×).",
    )
    suspend_reason: Optional[str] = Field(
        default=None,
        description="׳¡׳™׳‘׳× ׳”׳”׳§׳₪׳׳” (׳׳׳©׳ 'Earnings', 'Macro event', 'Bug investigation').",
    )

    priority_rank: Optional[int] = Field(
        default=None,
        description="׳“׳™׳¨׳•׳’ ׳¢׳“׳™׳₪׳•׳× (1=׳’׳‘׳•׳” ׳‘׳™׳•׳×׳¨). ׳׳©׳׳© ׳׳¡׳™׳ ׳•׳ ׳‘׳₪׳•׳¢׳ ׳‘׳׳ ׳•׳¢ ׳”׳׳™׳™׳‘.",
    )
    tags: List[str] = Field(
        default_factory=list,
        description="׳¨׳©׳™׳׳× ׳×׳’׳™׳ (׳׳׳©׳ ['core', 'tech', 'experimental']).",
    )

    min_liquidity_usd: float = Field(
        0.0,
        description="׳ ׳–׳™׳׳•׳× ׳׳™׳ ׳™׳׳׳™׳× ׳ ׳“׳¨׳©׳× ׳׳™׳•׳/׳˜׳™׳™׳-׳₪׳¨׳™׳™׳ ׳›׳“׳™ ׳׳׳₪׳©׳¨ ׳₪׳×׳™׳—׳× ׳˜׳¨׳™׳™׳“׳™׳.",
        ge=0.0,
    )

    notes: Optional[str] = Field(
        default=None,
        description="׳”׳¢׳¨׳•׳× ׳—׳•׳₪׳©׳™׳•׳× ׳¢׳ ׳”׳–׳•׳’ (׳׳׳ ׳׳™׳¡׳˜/׳׳₪׳¢׳™׳ ׳”׳׳¢׳¨׳›׳×).",
    )

    last_optimized_at: Optional[datetime] = Field(
        default=None,
        description="׳׳•׳¢׳“ ׳”׳¨׳™׳¦׳” ׳”׳׳—׳¨׳•׳ ׳” ׳©׳ ׳”׳׳•׳₪׳˜׳™׳׳™׳–׳¦׳™׳” ׳¢׳‘׳•׳¨ ׳”׳–׳•׳’.",
    )
    last_backtest_at: Optional[datetime] = Field(
        default=None,
        description="׳׳•׳¢׳“ ׳”-Backtest ׳”׳׳—׳¨׳•׳ ׳¢׳‘׳•׳¨ ׳”׳–׳•׳’.",
    )
    last_ml_update_at: Optional[datetime] = Field(
        default=None,
        description="׳׳•׳¢׳“ ׳”׳¢׳“׳›׳•׳ ׳”׳׳—׳¨׳•׳ ׳©׳ ׳×׳•׳¦׳¨׳™ ׳”-ML ׳¢׳‘׳•׳¨ ׳”׳–׳•׳’.",
    )

    # ======================================================================
    # Helper methods ׳§׳׳™׳ (׳׳ ׳—׳•׳‘׳” ׳׳”׳©׳×׳׳©, ׳׳‘׳ ׳ ׳•׳—)
    # ======================================================================
    def is_tradeable_now(self) -> bool:
        """
        ׳₪׳•׳ ׳§׳¦׳™׳” ׳ ׳•׳—׳” ׳׳׳ ׳•׳¢ ׳”׳׳™׳™׳‘/׳“׳©׳‘׳•׳¨׳“:
            ׳׳—׳–׳™׳¨׳” True ׳¨׳§ ׳׳:
                - is_active == True
                - is_suspended == False

        ׳©׳׳¨ ׳”׳‘׳“׳™׳§׳•׳× (Risk ׳’׳׳•׳‘׳׳™, ׳׳’׳‘׳׳•׳× ׳—׳©׳™׳₪׳” ׳•׳›׳•') ׳ ׳¢׳©׳•׳× ׳׳—׳•׳¥ ׳׳׳•׳“׳.
        """
        return self.is_active and not self.is_suspended
