# -*- coding: utf-8 -*-
"""
generate_config.py — יצירת config.json עם 250 זוגות מדורגים

- בונה יקום סמלים לפי סקטורים (טכנולוגיה, פיננסים, אנרגיה, ETF, קריפטו, מדדים)
- יוצר את כל הקומבינציות האפשריות, מערבב ובוחר 250 זוגות
- לכל זוג נותן score בסיסי (0.5 + i * 0.005) רק כדי שיהיה דירוג ראשוני
- שומר לקובץ config.json בפורמט UTF-8
"""

import logging
import json
import random
from itertools import combinations
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# סמלים לפי תחום
tech = ["AAPL", "MSFT", "GOOG", "META", "NVDA", "AMZN", "TSLA", "CRM", "ADBE", "ORCL"]
finance = ["JPM", "BAC", "WFC", "C", "GS", "MS", "AXP", "USB", "TD", "RY"]
energy = ["XOM", "CVX", "COP", "EOG", "PSX", "MPC", "VLO", "OXY", "KMI", "HAL"]
etfs = ["XLY", "XLP", "XLC", "XLI", "XLB", "XLK", "XLV", "XBI", "XLE", "XLU"]
crypto = ["IBIT", "MSTR", "RIOT", "MARA", "COIN", "BTC", "ETH", "GBTC", "BITO", "SQ"]
indices = ["SPY", "QQQ", "DIA", "IWM", "VTI", "VOO", "IVV", "VT", "XLF", "XLK"]

# כל הסמלים ביקום אחד (ללא כפילויות)
all_symbols = list(set(tech + finance + energy + etfs + crypto + indices))
random.shuffle(all_symbols)

# ליציבות אפשרית של התוצאה (אם תרצה תמיד אותו config):
# random.seed(42)

# כל הקומבינציות האפשריות (ללא חזרות וללא כיוון)
all_combos = list(combinations(all_symbols, 2))
random.shuffle(all_combos)

# בחירת 250 זוגות
selected_pairs = all_combos[:250]

# יצירת רשימה עם דירוג לכל זוג
pairs = [
    {
        "symbols": [sym_x, sym_y],
        "score": round(0.5 + i * 0.005, 3)
    }
    for i, (sym_x, sym_y) in enumerate(selected_pairs)
]

# בניית ה-config המלא
config = {
    "strategy": {
        "z_open": 2.0,
        "z_close": 0.5,
        "max_exposure_per_trade": 0.10,
        "rolling_window": 60,
        "atr_window": 14,
        "use_volatility_adjustment": True,
        "auto_rebalance": True,
        "dry_run": False,
        "log_trades": True,
    },
    "filters": {
        "min_correlation": 0.7,
        "min_edge": 1.2,
        "min_half_life": 1,
        "max_half_life": 200,
        "min_volatility_zscore": -1.5,
        "max_volatility_zscore": 1.5,
    },
    "volatility_profiles": {
        "default": {"min_z": -1.5, "max_z": 1.5},
        "tech": {"min_z": -2.0, "max_z": 2.0},
        "etf": {"min_z": -1.0, "max_z": 1.0},
        "crypto": {"min_z": -2.5, "max_z": 2.5},
    },
    "optimization": {
        "enabled": True,
        "min_quality_score": 0.5,
        "max_drawdown_threshold": 0.15,
        "std_threshold": 0.01,
        "median_diff_threshold": 0.02,
        "z_of_atr": 1.0,
        "price_ratio_dev": 0.03,
        "volatility_spike": 0.2,
        "percentile_rank_spread": 85,
        "rolling_corr_drop": 0.1,
        "price_ratio_z": 1.5,
        "log_price_ratio_z": 1.2,
        "standardized_residual": 1.0,
        "kalman_residual": 1.1,
        "cointegration_deviation": 0.05,
    },
    "ib": {
        "host": "127.0.0.1",
        # שים לב: אם ibkr_connection מחובר ל־7497, או שתשנה כאן או שתיישר שם
        "port": 6083,
        "client_id": 1,
    },
    "pairs": pairs,
    "metadata": {
        "version": "3.2",
        "created": "2025-06-12",
        "author": "Omri",
        "note": "250 זוגות אוטומטיים מדורגים עם תחומים מגוונים",
    },
}

# שמירה ל-config.json באותה תיקייה של הסקריפט
output_path = Path(__file__).resolve().parent / "config.json"
with output_path.open("w", encoding="utf-8") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)

logger.info("✅ נוצר %s עם %d זוגות מדורגים.", output_path, len(pairs))
print(f"✅ נוצר {output_path.name} עם {len(pairs)} זוגות מדורגים.")
