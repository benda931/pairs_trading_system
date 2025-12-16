# -*- coding: utf-8 -*-
"""
core/dashboard_models.py — HF-grade Dashboard Data Models
=========================================================

מודול זה מרכז את כל מודלי הנתונים המשמשים את הדשבורד הראשי
(גם ל-Web, גם ל-Desktop, גם ל-API וגם ל-SQL).

השכבה הזו צריכה להיות:
- נקייה מתלות ב-Streamlit / GUI.
- מתאימה לשימוש ב-Services, Agents, SQL store ודוחות.
- עשירה מספיק בשביל קרן גידור (Risk, PnL, Exposures, Signals, System health).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Dict, List, Optional, Any


# ======================================================================
# 1) DashboardContext — קונטקסט עשיר לדשבורד ברמת קרן גידור
# ======================================================================


@dataclass
class DashboardContext:
    """
    DashboardContext — קונטקסט עשיר לדשבורד ברמת קרן גידור
    ========================================================

    הקטגוריות המרכזיות:
    --------------------
    1. Environment & Session:
       - env: dev / paper / live
       - profile: monitoring / research / trading
       - timezone, run_id, user_id

    2. Time Horizon & Bar Settings:
       - start_date / end_date
       - intraday / bar_size (1D / 1H / 5min וכו')

    3. Portfolio & Account:
       - portfolio_id / account_id
       - base_currency, benchmark, benchmark_alt
       - capital_base, target_leverage, max_leverage

    4. Strategy & Universe:
       - universe_name / universe_size_limit
       - strategy_family (pairs_trading / stat_arb / macro_overlay ...)
       - sub_strategy (mean_reversion / momentum / carry / hedging ...)
       - tags: תגים חופשיים לסיווג (למשל "HF", "experiment", "prod")

    5. Risk Profile & Limits:
       - risk_profile: conservative / balanced / aggressive
       - target_vol_annual / max_vol_annual
       - VaR horizon / confidence
       - drawdown_soft_limit / drawdown_hard_limit
       - kill_switch_enabled, stress_mode

    6. Data & Infrastructure:
       - price_source (ibkr / yahoo / sql / blended)
       - fundamental_source / macro_source
       - use_sql_cache / use_live_data
       - data_quality_min_obs, data_staleness_max_minutes

    7. UI / UX Behaviour:
       - ui_mode: simple / pro
       - language: he / en
       - dark_mode, compact_layout
       - show_experimental_panels

    8. Research & Optimization Context:
       - optimization_profile, last_opt_run_id
       - prefer_optimized_params, show_research_overlays

    9. Macro / Regime Hints:
       - macro_regime_hint (growth / inflation / stagflation / recession ...)
       - vol_regime_hint (low / normal / high / crisis)
       - regime_confidence (0–1)

    10. Extra:
        - extra: מילון חופשי להרחבות עתידיות בלי לשבור API
    """

    # ---------- 1) Environment & Session ----------

    env: str = "dev"                 # "dev" / "paper" / "live"
    profile: str = "monitoring"      # "monitoring" / "research" / "trading"

    run_id: str = ""                 # מזהה ריצה/סשן (אפשר להשאיר ריק ולתת לשירות למלא)
    user_id: Optional[str] = None
    timezone: str = "Asia/Jerusalem"

    created_at_utc: datetime = field(default_factory=datetime.utcnow)

    # ---------- 2) Time Horizon & Bar Settings ----------

    start_date: date = field(default_factory=date.today)
    end_date: date = field(default_factory=date.today)

    intraday: bool = False
    # 1D, 1H, 30min, 5min, 1min וכו' — ליישר עם מה שאתה כבר משתמש בו ב-core
    bar_size: str = "1D"

    trading_calendar: str = "NYSE"   # או "TASE", "CME" וכו'
    include_prepost: bool = False

    # ---------- 3) Portfolio & Account ----------

    portfolio_id: str = "default"
    account_id: Optional[str] = None

    base_currency: str = "USD"
    benchmark: str = "SPY"
    benchmark_alt: Optional[str] = "QQQ"

    capital_base: float = 0.0        # הון נוכחי/מתוכנן (לצורך התייחסות Risk)
    target_leverage: float = 1.0
    max_leverage: float = 2.0
    allow_shorting: bool = True

    # שיוך ל-"Book"/Desk (אם תרצה להרחיב בעתיד לריבוי ספרים)
    book_name: Optional[str] = None

    # ---------- 4) Strategy & Universe ----------

    universe_name: str = "default"
    universe_size_limit: int = 500

    # משפחת אסטרטגיה כללית
    strategy_family: str = "pairs_trading"  # או "stat_arb", "macro_overlay" וכו'
    sub_strategy: str = "mean_reversion"    # "momentum", "carry", "hedging"...

    multi_strategy_mode: bool = False

    tags: List[str] = field(default_factory=list)

    # ---------- 5) Risk Profile & Limits ----------

    risk_profile: str = "balanced"   # "conservative" / "balanced" / "aggressive"

    # Vol targets – ברמת קרן
    target_vol_annual: float = 0.12
    max_vol_annual: float = 0.25

    # VaR / ES הגדרות בסיס
    var_horizon_days: int = 1
    var_confidence: float = 0.95

    # Drawdown limits (על ההון בפועל)
    drawdown_soft_limit: float = 0.10   # 10% — אפשר להתריע/לצמצם סיכון
    drawdown_hard_limit: float = 0.20   # 20% — Kill-switch/חירום

    kill_switch_enabled: bool = True
    stress_mode: bool = False           # מצב חירום/לחץ

    rebalance_frequency: str = "daily"  # "intraday" / "weekly" / "monthly"

    max_single_position_weight: float = 0.10   # 10%
    max_sector_exposure: float = 0.30          # 30%

    # ---------- 6) Data & Infrastructure ----------

    price_source: str = "ibkr"         # "ibkr" / "yahoo" / "sql" / "blended"
    fundamental_source: Optional[str] = "sql"
    macro_source: Optional[str] = "sql"

    use_sql_cache: bool = True
    use_live_data: bool = False        # אם False → לעבוד יותר עם snapshot/SQL/Backfill

    data_quality_min_obs: int = 252    # מינימום תצפיות לסדרה שמישה
    data_staleness_max_minutes: int = 15

    enable_data_quality_checks: bool = True

    # ---------- 7) UI / UX Behaviour ----------

    ui_mode: str = "simple"            # "simple" / "pro"
    language: str = "he"               # "he" / "en"
    dark_mode: bool = True

    show_experimental_panels: bool = False
    compact_layout: bool = False

    top_signals_limit: int = 20
    top_positions_limit: int = 20

    # ---------- 8) Research & Optimization Context ----------

    optimization_profile: Optional[str] = None   # שם פרופיל (למשל "HF_default")
    last_opt_run_id: Optional[str] = None        # מזהה ריצת Optuna אחרונה

    prefer_optimized_params: bool = True
    show_research_overlays: bool = True
    warn_if_off_optimal: bool = True

    # ---------- 9) Macro / Regime Hints ----------

    macro_regime_hint: Optional[str] = None      # "growth", "inflation", ...
    vol_regime_hint: Optional[str] = None        # "low", "normal", "high", "crisis"
    regime_confidence: float = 0.0               # 0–1

    macro_sensitivity_flag: bool = False

    # ---------- 10) Extra (Forward-compatible) ----------

    extra: Dict[str, Any] = field(default_factory=dict)


# ======================================================================
# 2) Portfolio-level data models
# ======================================================================


@dataclass
class PositionPnlBreakdown:
    """
    פירוק PnL ברמת פוזיציה בודדת – מאיפה הרווח/הפסד מגיע.
    """
    symbol: str
    realized_today: float = 0.0
    realized_mtd: float = 0.0
    realized_ytd: float = 0.0

    unrealized: float = 0.0

    fees: float = 0.0          # עמלות
    slippage: float = 0.0      # השפעת slippage
    funding: float = 0.0       # ריבית/מימון, borrow cost
    dividends: float = 0.0     # דיבידנדים/קופונים
    fx_pnl: float = 0.0        # רווח/הפסד משערי מט"ח

    def total_today(self) -> float:
        return self.realized_today + self.unrealized


@dataclass
class PositionRiskMetrics:
    """
    Risk metrics ברמת פוזיציה – תרומה ל-Vol/VaR ועוד.
    """
    symbol: str
    beta: float = 0.0
    vol_annual: float = 0.0
    var_95: float = 0.0
    es_95: float = 0.0

    contribution_to_var: float = 0.0
    contribution_to_vol: float = 0.0
    rho_to_portfolio: float = 0.0  # קורלציה לפורטפוליו

    # אופציונלי – לנכסי ריבית/אופציות
    duration: Optional[float] = None
    convexity: Optional[float] = None
    delta: Optional[float] = None
    gamma: Optional[float] = None
    vega: Optional[float] = None
    theta: Optional[float] = None


@dataclass
class PositionSnapshot:
    """
    תמונת מצב של פוזיציה בודדת:
    גודל, מחיר, משקל, PnL, Risk metrics, תגים, מטא דאטה.
    """
    symbol: str
    quantity: float
    side: str                 # "LONG", "SHORT", "FLAT"
    last_price: float
    market_value: float
    weight: float             # market_value / total_equity

    entry_price: float
    entry_ts: Optional[datetime] = None

    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0

    pnl_breakdown: Optional[PositionPnlBreakdown] = None
    risk: Optional[PositionRiskMetrics] = None

    tags: List[str] = field(default_factory=list)       # למשל: ["pair_leg", "hedge", "macro"]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PortfolioExposureBreakdown:
    """
    אקספוזר מפורק לפי ממדים שונים – Asset class, Sector, Country, Factor וכו'.
    כל הערכים הם יחסי (למשל אחוז מה-NAV) או Notional לפי החלטה שלך בלוגיקה.
    """
    by_asset_class: Dict[str, float] = field(default_factory=dict)   # Equities / Bonds / FX / Crypto ...
    by_sector: Dict[str, float] = field(default_factory=dict)        # Tech, Financials, ...
    by_industry: Dict[str, float] = field(default_factory=dict)
    by_currency: Dict[str, float] = field(default_factory=dict)
    by_country: Dict[str, float] = field(default_factory=dict)

    # פקטורים כמותיים – Value, Growth, Size, Quality, Momentum, Vol וכו'
    by_factor: Dict[str, float] = field(default_factory=dict)

    # לנכסי ריבית/Duration – buckets של yield curve
    by_curve_bucket: Dict[str, float] = field(default_factory=dict)  # "2Y", "5Y", "10Y", "30Y"

    # סיכום כולל:
    gross_exposure: float = 0.0
    net_exposure: float = 0.0
    long_exposure: float = 0.0
    short_exposure: float = 0.0

    # ריכוזיות – כמה מהפורטפוליו מרוכז ב-top names
    concentration_top_1: float = 0.0
    concentration_top_5: float = 0.0
    concentration_top_10: float = 0.0


@dataclass
class PortfolioPnLBreakdown:
    """
    פירוק PnL ברמת פורטפוליו – יומי, חודשי, שנתי, לפי Asset class/Strategy.
    """
    realized_today: float = 0.0
    unrealized_today: float = 0.0

    realized_mtd: float = 0.0
    realized_ytd: float = 0.0

    total_fees_today: float = 0.0
    total_funding_today: float = 0.0
    fx_pnl_today: float = 0.0

    pnl_by_asset_class: Dict[str, float] = field(default_factory=dict)
    pnl_by_strategy: Dict[str, float] = field(default_factory=dict)   # pairs / macro / hedging / ...
    pnl_by_desk: Dict[str, float] = field(default_factory=dict)       # למשל book / desk

    def total_today(self) -> float:
        return self.realized_today + self.unrealized_today


@dataclass
class PortfolioRiskMetrics:
    """
    מדדי סיכון ברמת פורטפוליו – Vol, VaR/ES, TE, Tail risk, Stress tests.
    """
    vol_annual: float = 0.0

    var_95: float = 0.0
    es_95: float = 0.0
    var_99: Optional[float] = None
    es_99: Optional[float] = None

    max_drawdown_1y: float = 0.0
    max_drawdown_itd: Optional[float] = None

    tracking_error_vs_bench: Optional[float] = None
    beta_vs_bench: Optional[float] = None
    corr_vs_bench: Optional[float] = None

    tail_risk_index: Optional[float] = None   # למשל מבוסס skew/excess kurtosis

    liquidity_stress_impact: Optional[float] = None  # אומדן הפסד בתרחיש לחץ נזילות

    # תרחישי stress שונים – למשל {"2008": -0.18, "covid_crash": -0.12}
    stress_test_scenarios: Dict[str, float] = field(default_factory=dict)


@dataclass
class PortfolioSnapshot:
    """
    PortfolioSnapshot — תמונת מצב מלאה של הפורטפוליו ברמת קרן גידור
    =================================================================

    משמשת ל:
    - דשבורד ראשי (Dashboard Home)
    - Risk Engine / Alerts
    - SQL / דוחות יומיים
    - Insights / ML (feature store)
    """

    # 1) Identification & Time
    as_of: datetime
    portfolio_id: str

    nav: float                        # NAV נוכחי
    nav_prev_close: Optional[float] = None

    # 2) Equity, Cash & Leverage
    total_equity: float = 0.0         # לעתים זהה ל-NAV, אבל נשאיר גמישות
    cash: float = 0.0
    cash_available: float = 0.0

    margin_used: float = 0.0
    margin_available: float = 0.0

    leverage: float = 1.0             # Gross Exposure / NAV (או כפי שתבחר להגדיר)

    # 3) Exposures & Concentration
    gross_exposure: float = 0.0       # Σ |position notional|
    net_exposure: float = 0.0         # Σ position notional (signed)
    long_exposure: float = 0.0
    short_exposure: float = 0.0

    exposure: PortfolioExposureBreakdown = field(
        default_factory=PortfolioExposureBreakdown
    )

    # 4) Positions
    positions: List[PositionSnapshot] = field(default_factory=list)

    num_positions: int = 0
    num_long: int = 0
    num_short: int = 0

    cash_pct: float = 0.0             # אחוז מזומן מה-NAV

    # 5) PnL
    pnl: PortfolioPnLBreakdown = field(default_factory=PortfolioPnLBreakdown)

    # 6) Risk
    risk: PortfolioRiskMetrics = field(default_factory=PortfolioRiskMetrics)

    # 7) Activity / Turnover
    turnover_1d: Optional[float] = None
    turnover_5d: Optional[float] = None
    turnover_20d: Optional[float] = None

    # 8) Flags & Notes
    warnings: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ======================================================================
# 3) Market, Risk, Signals & System Health snapshots
# ======================================================================


@dataclass
class MarketSnapshot:
    """
    תמונת מצב שוק ברמה גבוהה – מדדי תנודתיות, מדדי שוק, פקטורים עיקריים.
    """
    as_of: datetime

    # VIX / Volatility regime
    vix_level: float
    vix_regime: str                   # "low" / "normal" / "high" / "crisis"

    benchmark: str
    bench_ret_1d: float
    bench_ret_5d: float
    bench_ret_30d: float

    # אפשר להרחיב לפקטורים שונים (Equity / Credit / Rates / FX / Commodities)
    factor_returns: Dict[str, float] = field(default_factory=dict)
    factor_zscores: Dict[str, float] = field(default_factory=dict)

    comment: str = ""


@dataclass
class RiskSnapshot:
    """
    RiskSnapshot — מבט מרוכז על מצב הסיכון ברמת קרן.

    כולל:
    - PortfolioRiskMetrics מלא
    - ציוני סיכון / traffic-light
    - חריגות מלימיטים
    """
    as_of: datetime
    portfolio_id: str

    portfolio_risk: PortfolioRiskMetrics

    # ציון סיכון כללי (0–100, לדוגמה)
    risk_score: float = 0.0

    # traffic-light: "green" / "yellow" / "red"
    traffic_light: str = "green"

    # האם breached כלשהו (soft/hard)
    limits_breached: bool = False
    breached_limits: List[str] = field(default_factory=list)

    # Flags נוספים (למשל ריכוזיות, volatility spike וכו')
    flags: List[str] = field(default_factory=list)


@dataclass
class SignalItem:
    """
    ייצוג של אות/הזדמנות בודדת לדשבורד.

    תומך:
    - זוגות (symbol_1/symbol_2)
    - directional (symbol_1 בלבד)
    - מידע על איכות, Regime, אסטרטגיה, מודל.
    """
    symbol_1: str
    symbol_2: Optional[str]

    direction: str            # "LONG", "SHORT", "SPREAD", "FLAT"
    confidence: float         # 0–1 או 0–100 לפי החלטתך
    edge: Optional[float] = None      # אומדן Expected edge (למשל in bps)

    zscore: Optional[float] = None
    half_life: Optional[float] = None
    corr: Optional[float] = None

    quality_score: Optional[float] = None   # ציון כולל מבוסס פקטורים שונים
    regime: Optional[str] = None            # Regime label של האות

    time_frame: str = "1D"                 # "intraday_5m", "1H", "1D" וכו'

    strategy_family: str = "pairs_trading"
    sub_strategy: str = "mean_reversion"

    model_name: Optional[str] = None
    model_version: Optional[str] = None

    created_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None

    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SignalsSnapshot:
    """
    SignalsSnapshot — מצב האותות / ההזדמנויות ברמת דשבורד.
    """
    as_of: datetime
    portfolio_id: str

    items: List[SignalItem] = field(default_factory=list)

    n_new_today: int = 0
    n_conflicting: int = 0
    n_total: int = 0

    # פילוחים אפשריים לאנליזה מהירה בדשבורד
    count_by_strategy: Dict[str, int] = field(default_factory=dict)
    count_by_direction: Dict[str, int] = field(default_factory=dict)   # LONG/SHORT/SPREAD
    count_by_regime: Dict[str, int] = field(default_factory=dict)


@dataclass
class SystemHealthSnapshot:
    """
    SystemHealthSnapshot — בריאות מערכת, דאטה וברוקר.
    """
    as_of: datetime

    broker_connected: bool
    data_fresh: bool
    data_latency_ms: float

    last_price_update: Optional[datetime]
    last_order_ts: Optional[datetime]

    sql_ok: bool
    sql_last_error: Optional[str]

    running_strategies: List[str] = field(default_factory=list)
    recent_errors: List[str] = field(default_factory=list)

    cpu_load_pct: Optional[float] = None
    memory_used_pct: Optional[float] = None

    # מצב סוכנים/Agents (אם רלוונטי)
    agents_status: Dict[str, str] = field(default_factory=dict)  # {"SystemUpgrader": "idle", ...}


# ======================================================================
# 4) DashboardSnapshot — אובייקט עליון שמרכז את הכול
# ======================================================================


@dataclass
class DashboardSnapshot:
    """
    DashboardSnapshot — סיכום מרכזי של כל מה שהדשבורד צריך.

    משמש ל:
    - רינדור הדשבורד (Web/Desktop).
    - לוגים/SQL (שמירת snapshot יומי/תכוף).
    - חיבור ל-Insights / ML / Agents.
    """
    ctx: DashboardContext
    as_of: datetime

    market: MarketSnapshot
    portfolio: PortfolioSnapshot
    risk: RiskSnapshot
    signals: SignalsSnapshot
    system: SystemHealthSnapshot

    # מקום לאינפורמציה נוספת שתגיע מטאבים אחרים (למשל ResearchSnapshot, MacroSnapshot וכו')
    extra: Dict[str, Any] = field(default_factory=dict)
