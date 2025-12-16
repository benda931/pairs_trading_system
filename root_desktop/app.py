# -*- coding: utf-8 -*-
"""
root_desktop/app.py — Desktop Entry Point (HF-grade)
====================================================

נקודת הכניסה הרשמית לאפליקציית ה-Desktop של מערכת ה-Pairs Trading שלך.
לא WEB, לא דפדפן – אפליקציית Windows מקומית בלבד.

תפקידים עיקריים:
----------------
1. אתחול סביבת האפליקציה:
   - זיהוי PROJECT_ROOT, תיקיית logs, assets, וכו'.
   - טעינת פרופיל (APP_PROFILE) ו־Config בסיסי.

2. אתחול לוגים:
   - קובץ לוג מסתובב (RotatingFileHandler) + קונסול.
   - לוג ברמת INFO/DEBUG לפי דגלים.

3. הגדרות Qt גלובליות:
   - High-DPI scaling
   - שם האפליקציה, ארגון, וכו'.

4. UX בסיסי:
   - Splash Screen (לוגו אם קיים, אחרת טקסט).
   - טעינת Theme (קובץ .qss אופציונלי).
   - יצירת MainWindow ופתיחתו אחרי ה-Splash.

5. ניהול שגיאות גלובלי:
   - sys.excepthook → לוג + QMessageBox קריסתי.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Optional
from collections.abc import Sequence

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication, QMessageBox, QSplashScreen

# -------------------------
# נתיבי בסיס של הפרויקט
# -------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
ROOT_DESKTOP_DIR: Path = PROJECT_ROOT / "root_desktop"
LOGS_DIR: Path = PROJECT_ROOT / "logs"
ASSETS_DIR: Path = ROOT_DESKTOP_DIR / "assets"

LOGS_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

# קבצי עזר אופציונליים
DEFAULT_THEME_PATH: Path = ASSETS_DIR / "theme_dark.qss"
DEFAULT_LOGO_PATH: Path = ASSETS_DIR / "logo.png"

# פרופיל אפליקציה (יכול לשמש לקונפיגים שונים)
APP_PROFILE: str = (os.getenv("APP_PROFILE", "default") or "default").strip() or "default"

# לוג ראשי לאפליקציה
LOGGER = logging.getLogger("DesktopApp")


# =========================
#   Dataclasses – Context
# =========================

@dataclass
class AppConfig:
    """
    קונפיג בסיסי לאפליקציית ה-Desktop.

    בעתיד אפשר להרחיב:
    - שימוש במצב "safe_mode"
    - פרופילי סביבה (dev / prod)
    - קבצי config.json שונים לפי פרופיל
    """
    debug: bool = False
    profile: str = APP_PROFILE
    safe_mode: bool = False
    theme_path: Optional[Path] = None
    logo_path: Optional[Path] = None


@dataclass
class DesktopAppContext:
    """
    קונטקסט אפליקציה – עובר ל-MainWindow ולשאר השכבות.

    אפשר להרחיב עם:
    - config_data (מ-load_config)
    - חיבור ל-DB / DuckDB
    - pointers למודולי core חשובים, cache, וכו'
    """
    project_root: Path
    root_desktop_dir: Path
    logs_dir: Path
    assets_dir: Path
    started_at: datetime
    config: AppConfig
    config_data: Optional[dict[str, Any]] = None


# =========================
#   CLI / Args parsing
# =========================

def parse_cli_args(argv: Optional[Sequence[str]] = None) -> AppConfig:
    """
    פריסת ארגומנטים מה-CLI כדי לשלוט על מצב הרצה.
    לדוגמה:
        python -m root_desktop.app --debug --profile dev
    """
    parser = argparse.ArgumentParser(
        prog="OmriPairsDesktop",
        description="Desktop Pairs Trading Application (No-Web)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode (verbose logs, no splash delay)",
    )
    parser.add_argument(
        "--safe-mode",
        action="store_true",
        help="Run in safe mode (ללא פיצ'רים כבדים בעתיד)",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=APP_PROFILE,
        help=f"App profile name (default: {APP_PROFILE!r})",
    )
    parser.add_argument(
        "--theme",
        type=str,
        default="",
        help="Path to custom .qss theme file (optional)",
    )
    parser.add_argument(
        "--logo",
        type=str,
        default="",
        help="Path to custom logo .png for the splash screen (optional)",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    theme_path = Path(args.theme).resolve() if args.theme else DEFAULT_THEME_PATH
    logo_path = Path(args.logo).resolve() if args.logo else DEFAULT_LOGO_PATH

    if not theme_path.is_file():
        theme_path = None  # אין Theme – לא חובה
    if not logo_path.is_file():
        logo_path = None  # אין לוגו – נ fallback לטקסט

    return AppConfig(
        debug=bool(args.debug),
        profile=(args.profile or APP_PROFILE).strip() or "default",
        safe_mode=bool(args.safe_mode),
        theme_path=theme_path,
        logo_path=logo_path,
    )


# =========================
#   Logging setup
# =========================

def setup_logging(logs_dir: Path, debug: bool) -> None:
    """
    הגדרת לוגים: קובץ + קונסול.
    """
    level = logging.DEBUG if debug else logging.INFO
    LOGGER.setLevel(level)

    # פורמט אחיד
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # קונסול
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    LOGGER.addHandler(console_handler)

    # קובץ מסתובב
    log_file = logs_dir / "desktop_app.log"
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=5 * 1024 * 1024,  # 5MB
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)

    LOGGER.debug("Logging initialized (level=%s, file=%s)", logging.getLevelName(level), log_file)


# =========================
#   Config loading (optional)
# =========================

def load_main_config(profile: str) -> Optional[dict[str, Any]]:
    """
    מנסה לטעון config דרך common.config_manager אם קיים.
    לא חובה שיהיה – אם אין, חוזרים None.

    במערכת שלך יש כבר config_manager – אז בעתיד אפשר לשחק עם פרופילים.
    """
    try:
        from common.config_manager import load_config  # type: ignore
    except Exception:  # pragma: no cover - fallback
        LOGGER.warning("config_manager.load_config not available – running without external config")
        return None

    try:
        cfg = load_config()
        LOGGER.info("Loaded main config for profile=%s", profile)
        return cfg
    except Exception:
        LOGGER.exception("Failed to load main config")
        return None


# =========================
#   Global exception hook
# =========================

def _show_critical_message_box(title: str, text: str) -> None:
    """
    מנסה להציג QMessageBox קריסתי. אם אין QApplication – מדפיס לקונסול.
    """
    if QApplication.instance() is None:
        print(f"[CRITICAL] {title}: {text}", file=sys.stderr)
        return

    msg = QMessageBox()
    msg.setIcon(QMessageBox.Critical)
    msg.setWindowTitle(title)
    msg.setText(text)
    msg.setStandardButtons(QMessageBox.Ok)
    msg.exec()


def global_exception_hook(exc_type, exc_value, exc_traceback) -> None:  # type: ignore[override]
    """
    sys.excepthook → לוג + פופ-אפ.
    """
    if issubclass(exc_type, KeyboardInterrupt):
        # Ctrl+C – לתת לצאת בשקט
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    tb_str = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    LOGGER.critical("Uncaught exception:\n%s", tb_str)

    _show_critical_message_box(
        "Uncaught exception",
        f"An unexpected error occurred:\n{exc_value}\n\nSee logs for details.",
    )


# =========================
#   Qt helpers (theme, splash)
# =========================

def apply_theme_if_available(app: QApplication, theme_path: Optional[Path]) -> None:
    """
    מנסה לטעון ולהחיל קובץ .qss אם קיים.
    """
    if theme_path is None:
        LOGGER.info("No theme file found – using default Qt style")
        return

    try:
        content = theme_path.read_text(encoding="utf-8")
    except Exception:
        LOGGER.exception("Failed to read theme file: %s", theme_path)
        return

    app.setStyleSheet(content)
    LOGGER.info("Theme applied from %s", theme_path)


def create_splash_screen(config: AppConfig) -> Optional[QSplashScreen]:
    """
    יוצר SplashScreen עם לוגו אם קיים, אחרת טקסט פשוט.
    """
    logo_path = config.logo_path
    if logo_path is not None and logo_path.is_file():
        try:
            pixmap = QPixmap(str(logo_path))
        except Exception:
            pixmap = QPixmap()
    else:
        pixmap = QPixmap()

    if pixmap.isNull():
        # אין לוגו – ניצור Splash ריק עם טקסט בסיסי
        splash = QSplashScreen()
        splash.showMessage(
            "Omri Pairs Trading – Loading...",
            alignment=Qt.AlignCenter | Qt.AlignBottom,
            color=Qt.white,
        )
    else:
        splash = QSplashScreen(pixmap)
        splash.showMessage(
            "Loading Omri Pairs Trading Desktop...",
            alignment=Qt.AlignBottom | Qt.AlignCenter,
            color=Qt.white,
        )

    splash.setWindowFlag(Qt.WindowStaysOnTopHint, True)
    return splash


# =========================
#   Main run function
# =========================

def run(argv: Optional[Sequence[str]] = None) -> int:
    """
    הרצת האפליקציה (יכול לשמש גם לבדיקות יחידה).
    """
    # 1) CLI → AppConfig
    app_config = parse_cli_args(argv)

    # 2) Logging בזמן
    setup_logging(LOGS_DIR, debug=app_config.debug)
    LOGGER.info("===== Starting DesktopApp (profile=%s, debug=%s) =====", app_config.profile, app_config.debug)

    # 3) טעינת config ראשי (לא חובה)
    config_data = load_main_config(app_config.profile)

    # 4) יצירת קונטקסט
    app_ctx = DesktopAppContext(
        project_root=PROJECT_ROOT,
        root_desktop_dir=ROOT_DESKTOP_DIR,
        logs_dir=LOGS_DIR,
        assets_dir=ASSETS_DIR,
        started_at=datetime.now(timezone.utc)(),
        config=app_config,
        config_data=config_data,
    )

    # 5) Qt – הגדרות global
    # High-DPI awareness
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    qt_app = QApplication(list(argv) if argv is not None else sys.argv)
    qt_app.setApplicationName("Omri Pairs Trading – Desktop")
    qt_app.setOrganizationName("OmriHFDesk")

    # 6) Global exception hook
    sys.excepthook = global_exception_hook  # type: ignore[assignment]

    # 7) Theme
    apply_theme_if_available(qt_app, app_config.theme_path)

    # 8) Splash
    splash = create_splash_screen(app_config)
    if splash is not None:
        splash.show()
        qt_app.processEvents()

    # 9) יצירת MainWindow (עם הקונטקסט)
    #    שים לב: תצטרך לעדכן את MainWindow שיקבל app_ctx אם עוד לא עשית.
    from root_desktop.views.main_window import MainWindow  # type: ignore

    def _create_and_show_main_window() -> None:
        try:
            # נסה להעביר קונטקסט – אם החתימה של MainWindow היא MainWindow(app_ctx: DesktopAppContext)
            try:
                window = MainWindow(app_ctx)  # type: ignore[call-arg]
            except TypeError:
                # fallback: אין פרמטר – ניצור בלי, ואפשר אחר כך להכניס setter
                LOGGER.warning(
                    "MainWindow does not accept app_ctx in __init__, creating without context. "
                    "Consider updating MainWindow.__init__(self, app_ctx: DesktopAppContext | None = None)."
                )
                window = MainWindow()  # type: ignore[call-arg]

            window.show()
            if splash is not None:
                splash.finish(window)
            # לשמור רפרנס כדי ש-Python לא יאסוף את האובייקט.
            qt_app.main_window = window  # type: ignore[attr-defined]
        except Exception:
            LOGGER.exception("Failed to create/show MainWindow")
            _show_critical_message_box(
                "Startup error",
                "Failed to start the main window.\nSee logs for more details.",
            )
            qt_app.quit()

    # אם debug – אפשר לפתוח מהר יותר, אחרת לתת קצת זמן ל-Splash
    delay_ms = 100 if app_config.debug else 600
    QTimer.singleShot(delay_ms, _create_and_show_main_window)

    # 10) הרצת לולאת האירועים
    exit_code = qt_app.exec()
    LOGGER.info("DesktopApp exited with code %s", exit_code)
    return int(exit_code)


def main() -> None:
    """
    Entry-point סטנדרטי להרצה כמודול:
        python -m root_desktop.app
    """
    sys.exit(run())


if __name__ == "__main__":
    main()
