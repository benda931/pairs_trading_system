# -*- coding: utf-8 -*-
"""
common/live_profiles.py — Live Pair Profile Contract (HF-Grade)
================================================================

קובץ זה מגדיר את חוזה הלייב הרשמי לזוג מסחר אחד (Pair) ברמת קרן גידור.

המטרה:
    - להיות "שפת תווך" אחידה בין:
        * Research / Optimization / ML / Macro
        * מנוע המסחר החי (Live Trading Engine)
        * ה-Dashboard (טאב פרוטפוליו / לייב / אנליטיקה)
    - כל מודול מחקר/אופטימיזציה/ML כותב לכאן את החלטותיו.
    - מנוע הלייב *רק קורא* את הפרופיל ומבצע לפי החוקים והפרמטרים שבו
      תחת מגבלות הסיכון הגלובליות.

עקרונות:
    - אין לוגיקה עסקית כבדה בתוך המודל עצמו (כמעט נטו Data Contract).
    - שדות מחולקים לקטגוריות ברורות:
        Identity, Trading Rules, Sizing & Risk, Quality & ML, Macro/Regime, Operational.
    - רוב השדות אופציונליים כדי לאפשר בנייה הדרגתית (Partial Filling).
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional, Literal

from pydantic import BaseModel, Field, ConfigDict


class LivePairProfile(BaseModel):
    """
    פרופיל לייב מלא לזוג אחד.

    זהו ה-"DNA" של הזוג במסחר חי:
        - איך לזהות אותו
        - איך לסחור בו (חוקי כניסה/יציאה/סטופים/זמן)
        - איך לקבוע גודל פוזיציה וחשיפה
        - איזה איכות/Edge יש לו היסטורית ו-MLית
        - באיזה משטרים (Regimes) מותר/אסור לסחור בו
        - האם פעיל, מושהה, ומה סיבת המצב הנוכחי

    שדות רבים הם אופציונליים כדי שתוכל להתחיל פשוט ולהוסיף עומק לאורך הזמן.
    """

    # פיידנטיק v2 – הגדרות כלליות
    model_config = ConfigDict(
        extra="ignore",              # התעלמות משדות לא מוכרים בטעינה
        validate_assignment=True,    # ולידציה גם בשינוי לאחר יצירה
        arbitrary_types_allowed=True,
    )

    # ======================================================================
    # 1. Identity & Meta — מי הזוג, איפה הוא נסחר, מה ההקשר שלו
    # ======================================================================
    pair_id: str = Field(
        ...,
        description="מזהה ייחודי לזוג, לדוגמה 'QQQ_SOXX_US_EQ_1D' (משמש כמפתח ראשי).",
    )
    sym_x: str = Field(
        ...,
        description="Leg X (בדרך כלל leg הלונג). לדוגמה: 'QQQ'.",
    )
    sym_y: str = Field(
        ...,
        description="Leg Y (בדרך כלל leg השורט). לדוגמה: 'SOXX'.",
    )

    asset_class: str = Field(
        "EQUITY",
        description="מחלקת נכס: EQUITY / ETF / FUTURES / FX / OPTIONS וכו'.",
    )
    market: Optional[str] = Field(
        default=None,
        description="זיהוי שוק/בורסה כללית, לדוגמה 'US-STK', 'EU-STK', 'CME-FUT'.",
    )
    exchange_x: Optional[str] = Field(
        default=None,
        description="בורסה ל-leg X, לדוגמה 'NASDAQ', 'NYSE'.",
    )
    exchange_y: Optional[str] = Field(
        default=None,
        description="בורסה ל-leg Y.",
    )

    base_currency: str = Field(
        "USD",
        description="מטבע בסיס ל-PnL ולניהול סיכון (בדרך כלל USD).",
    )
    timezone: Optional[str] = Field(
        default=None,
        description="אזור זמן ראשי למסחר בזוג (למשל 'America/New_York').",
    )

    timeframe: str = Field(
        "1D",
        description="טיים-פריים מרכזי לחישובי Spread / Z / Backtest (למשל '1D', '1H').",
    )
    data_source: str = Field(
        "IBKR",
        description="מקור נתונים לפועל: IBKR / Yahoo / Mixed / DuckDB וכו'.",
    )

    sector_label: Optional[str] = Field(
        default=None,
        description="סקטור/תמה כללית (למשל 'Tech', 'Semiconductors', 'Growth').",
    )
    cluster_id: Optional[str] = Field(
        default=None,
        description="Cluster / קלאסטר קורלציה/Factor (משויך ל-matrix_helpers / clustering).",
    )

    # ======================================================================
    # 2. Trading Rules — חוקי המסחר הבסיסיים לזוג הזה
    # ======================================================================
    direction_convention: str = Field(
        "long_x_short_y_on_positive_z",
        description=(
            "א convention לפרשנות סימן ה-Z: "
            "למשל 'long_x_short_y_on_positive_z' אומר: Z>0 → long X, short Y."
        ),
    )

    # --- כניסה/יציאה לפי Z-Score ---
    z_entry: float = Field(
        2.0,
        description="סף Z לכניסה: abs(Z) >= z_entry.",
    )
    z_exit: float = Field(
        0.5,
        description="סף Z ליציאה בסיסית: abs(Z) <= z_exit.",
    )
    z_take_profit: Optional[float] = Field(
        default=None,
        description="Z לרווח יתר (TP). אם None – אין TP לפי Z, רק z_exit / סטופים אחרים.",
    )
    z_hard_stop: Optional[float] = Field(
        default=None,
        description="Z לסטופ קיצוני (Hard Stop). אם None – אין Hard Stop לפי Z.",
    )

    # --- מגבלות זמן / Re-entry ---
    min_holding_bars: int = Field(
        0,
        description="מינימום ברים להחזקה לפני שמותר לסגור (מונע 'ניעור' מהיר מדי).",
        ge=0,
    )
    max_holding_bars: int = Field(
        999_999,
        description="מקסימום ברים להחזקה. אחרי זה סוגרים בכל מקרה.",
        ge=1,
    )
    reentry_cooldown_bars: int = Field(
        0,
        description="כמה ברים חייבים לעבור בין סגירה לפתיחה חדשה באותו כיוון במסחר.",
        ge=0,
    )

    # --- תנאים נוספים (אופציונליים) ---
    min_spread_std: Optional[float] = Field(
        default=None,
        description="סטיית תקן מינימלית של ה-Spread (כדי להימנע מזוגות 'מתים').",
    )
    min_corr_lookback: Optional[int] = Field(
        default=None,
        description="מספר ברים מינימלי לחישוב קורלציה/קואינטגרציה.",
    )
    require_cointegration: bool = Field(
        True,
        description="האם נדרש שהזוג יעבור בדיקת קואינטגרציה (Engle-Granger/Johansen).",
    )
    allow_short_both_legs: bool = Field(
        False,
        description="האם מותר לפתוח פוזיציה עם שני רגליים בשורט (למשל בזוגי קריפטו/פקטורים).",
    )

    slippage_bp: float = Field(
        5.0,
        description="החלקה צפויה ב-bps (0.01% = 1bp). משפיע גם על Backtest וגם על הגדרת LIMIT.",
        ge=0.0,
    )
    max_spread_bps_intraday: Optional[float] = Field(
        default=None,
        description="תקרת Spread intraday ב-bps (למניעת כניסה בתנאי מרווח שוק חריג).",
    )

    # ======================================================================
    # 3. Sizing & Local Risk — גודל פוזיציה ומגבלות סיכון ברמת הזוג
    # ======================================================================
    sizing_mode: Literal["fixed_notional", "vol_target", "risk_parity"] = Field(
        "fixed_notional",
        description="שיטת קביעת גודל הפוזיציה: fixed_notional / vol_target / risk_parity.",
    )
    base_notional_usd: float = Field(
        5_000.0,
        description="נוטיונל בסיסי לפוזיציה אחת בזוג (לפני התאמות לפי ML / Regime / Risk).",
        ge=0.0,
    )
    vol_target_annual: Optional[float] = Field(
        default=None,
        description="Vol annualized רצוי לזוג במסגרת Vol Targeting (אם None – לא בשימוש).",
    )
    risk_budget_fraction: Optional[float] = Field(
        default=None,
        description=(
            "אחוז מתקציב הסיכון הכולל שמוקצה לזוג (0–1). "
            "משמש בעתיד לריסק-פריטי/הקצאות חכמות."
        ),
    )

    leverage_max: float = Field(
        1.0,
        description="מינוף מקסימלי לזוג (יחסי). לרוב 1.0 לזוגי מניות/ETF.",
        ge=0.0,
    )
    pair_max_exposure_usd: Optional[float] = Field(
        default=None,
        description="תקרת חשיפה מקומית לזוג. אם None – משתמשים בערך גלובלי מה-Config.",
    )
    max_open_trades_per_pair: int = Field(
        1,
        description="כמה פוזיציות של אותו זוג מותר לפתוח בו זמנית.",
        ge=1,
    )

    weight_in_portfolio: float = Field(
        0.0,
        description="משקל מועדף בתיק (0–1), לשימוש Allocator חכם (לא חובה בשלב ראשון).",
        ge=0.0,
        le=1.0,
    )
    min_trade_value_usd: Optional[float] = Field(
        default=None,
        description="ערך מינימלי לעסקה כדי לא לייצר פוזיציות 'זבל' קטנות מדי.",
    )

    # ======================================================================
    # 4. Quality, Stats & ML — איכות הזוג, סטטיסטיקות ו-Edge חכם
    # ======================================================================
    # --- סטטיסטיקות mean-reversion וקורלציה ---
    half_life_bars: Optional[float] = Field(
        default=None,
        description="Half-life מוערך של ה-Spread (בברים). קטן ⇒ mean reversion מהיר.",
    )
    hurst_exponent: Optional[float] = Field(
        default=None,
        description="Hurst exponent של ה-Spread. <0.5 מעיד על Mean Reversion.",
    )
    corr_lookback_bars: Optional[int] = Field(
        default=None,
        description="אורך חלון לחישוב קורלציה (אם שונה מברירת מחדל גלובלית).",
    )
    rolling_corr: Optional[float] = Field(
        default=None,
        description="קורלציה ממוצעת/אחרונה של הזוג בתקופת האופטימיזציה.",
    )

    cointegration_method: Optional[str] = Field(
        default=None,
        description="שיטת בדיקת קואינטגרציה (Engle-Granger, Johansen, CADF וכו').",
    )
    cointegration_pvalue: Optional[float] = Field(
        default=None,
        description="p-value של בדיקת הקואינטגרציה לזוג.",
    )
    adf_pvalue_spread: Optional[float] = Field(
        default=None,
        description="p-value של ADF על ה-Spread (סטציונריות).",
    )

    # --- ציונים (Scores) מה-Recommender / Opt ---
    score_total: float = Field(
        0.0,
        description="ציון משוקלל כללי (Composite Score) מה-Recommender/Opt.",
    )
    score_corr_stability: float = Field(
        0.0,
        description="מדד יציבות קורלציה (0–1 או סקייל אחר לפי המערכת שלך).",
    )
    score_cointegration: float = Field(
        0.0,
        description="מדד איכות קואינטגרציה (מבוסס p-value/סטטיסטיקה).",
    )
    score_mean_reversion_speed: float = Field(
        0.0,
        description="מדד מהירות Mean Reversion (מנורמל מ-half-life/מדדים אחרים).",
    )

    # --- תוצאות Backtest מרכזיות ---
    backtest_sharpe: float = Field(
        0.0,
        description="Sharpe Ratio מה-Backtest שנבחר לאופטימיזציה.",
    )
    backtest_sortino: float = Field(
        0.0,
        description="Sortino Ratio מה-Backtest.",
    )
    backtest_max_drawdown: float = Field(
        0.0,
        description="Max Drawdown (יחסי, למשל -0.15 = -15%).",
    )
    backtest_winrate: float = Field(
        0.0,
        description="אחוז עסקאות רווחיות (0–1).",
    )
    backtest_trades_count: int = Field(
        0,
        description="מספר עסקאות ב-Backtest (מדד יציבות סטטיסטית).",
        ge=0,
    )

    # --- תוצרי ML/AutoML ---
    ml_edge_score: Optional[float] = Field(
        default=None,
        description="Edge מבוסס ML (למשל scaled בין -1 ל-1 או 0–1).",
    )
    ml_confidence: Optional[float] = Field(
        default=None,
        description="רמת ביטחון (0–1) בתחזית המודל עבור הזוג.",
    )
    ml_predicted_horizon_bars: Optional[int] = Field(
        default=None,
        description="אופק זמן (בברים) שבו המודל צופה את המיספרד/רווח.",
    )
    model_version: Optional[str] = Field(
        default=None,
        description="גרסת המודל/פייפליין (לדוגמה 'pairs_ml_v3.1').",
    )

    # ======================================================================
    # 5. Macro & Regime — הקשר מאקרו/משטר שוק
    # ======================================================================
    regime_id: Optional[str] = Field(
        default=None,
        description="מזהה משטר/מצב (למשל 'risk_on', 'risk_off', 'crash', 'range').",
    )
    macro_regime_id: Optional[str] = Field(
        default=None,
        description="משטר מאקרו גלובלי (לפי macro_tab, לוח שנה כלכלי וכו').",
    )
    vol_regime_id: Optional[str] = Field(
        default=None,
        description="משטר תנודתיות (למשל 'low_vol', 'high_vol').",
    )
    macro_score: Optional[float] = Field(
        default=None,
        description="ציון מאקרו כללי עבור הזוג (0–1 או -1–1).",
    )

    # ======================================================================
    # 6. Operational / Control — שליטה, מיתוג, תיעוד
    # ======================================================================
    is_active: bool = Field(
        False,
        description="האם הזוג מאושר למסחר חי כרגע (תחת מגבלות גלובליות).",
    )
    is_suspended: bool = Field(
        False,
        description="האם הזוג מושהה זמנית (עוקף את is_active לצורך חירום/חדשות).",
    )
    suspend_reason: Optional[str] = Field(
        default=None,
        description="סיבת ההקפאה (למשל 'Earnings', 'Macro event', 'Bug investigation').",
    )

    priority_rank: Optional[int] = Field(
        default=None,
        description="דירוג עדיפות (1=גבוה ביותר). משמש לסינון בפועל במנוע הלייב.",
    )
    tags: List[str] = Field(
        default_factory=list,
        description="רשימת תגים (למשל ['core', 'tech', 'experimental']).",
    )

    min_liquidity_usd: float = Field(
        0.0,
        description="נזילות מינימלית נדרשת ליום/טיים-פריים כדי לאפשר פתיחת טריידים.",
        ge=0.0,
    )

    notes: Optional[str] = Field(
        default=None,
        description="הערות חופשיות על הזוג (מאנליסט/מפעיל המערכת).",
    )

    last_optimized_at: Optional[datetime] = Field(
        default=None,
        description="מועד הריצה האחרונה של האופטימיזציה עבור הזוג.",
    )
    last_backtest_at: Optional[datetime] = Field(
        default=None,
        description="מועד ה-Backtest האחרון עבור הזוג.",
    )
    last_ml_update_at: Optional[datetime] = Field(
        default=None,
        description="מועד העדכון האחרון של תוצרי ה-ML עבור הזוג.",
    )

    # ======================================================================
    # Helper methods קלים (לא חובה להשתמש, אבל נוח)
    # ======================================================================
    def is_tradeable_now(self) -> bool:
        """
        פונקציה נוחה למנוע הלייב/דשבורד:
            מחזירה True רק אם:
                - is_active == True
                - is_suspended == False

        שאר הבדיקות (Risk גלובלי, מגבלות חשיפה וכו') נעשות מחוץ למודל.
        """
        return self.is_active and not self.is_suspended
