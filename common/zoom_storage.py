# -*- coding: utf-8 -*-
"""
common/zoom_storage.py — Shared Optuna Zoom storage contract (HF-grade)
======================================================================

תפקיד הקובץ:
------------

1. הגדרת "חוזה" אחד ברור לכל מה שקשור ל-Zoom Campaign:
   - איפה נשמרים ה-Studies של Optuna (storage_url).
   - איזה prefix משמש ל-Zoom (למשל "zoom").
   - איך נראה ה-study_name (zoom::PAIR::stage1).

2. מתן פונקציות עזר:
   - resolve_zoom_storage(project_root) -> ZoomStorageConfig
   - build_zoom_study_name(pair, stage, cfg) -> str
   - parse_zoom_study_name(name, expected_prefix) -> ZoomStudyMeta | None

3. עקרונות:
   ---------
   - גם CLI (scripts) וגם dashboard / optimization_tab משתמשים באותו חוזה.
   - אין תלות ב-Optuna כאן (רק string parsing) — פחות סיכוי ל-loops.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import logging
import os
import re

logger = logging.getLogger(__name__)

DEFAULT_ZOOM_DB_NAME = "zoom_studies.db"
DEFAULT_ZOOM_PREFIX = "zoom"


@dataclass(frozen=True)
class ZoomStorageConfig:
    """
    הגדרות בסיס ל-Zoom Storage.

    Attributes
    ----------
    project_root : Path
        שורש הפרויקט (root של pairs_trading_system).
    storage_url : str
        URL של Optuna storage, למשל:
        "sqlite:///C:/Users/omrib/AppData/Local/pairs_trading_system/zoom_studies.db"
    study_prefix : str
        prefix לשמות ה-studies, למשל "zoom".
    """

    project_root: Path
    storage_url: str
    study_prefix: str = DEFAULT_ZOOM_PREFIX


def resolve_zoom_storage(project_root: Path) -> ZoomStorageConfig:
    """
    מאתר את הגדרות ה-Zoom Storage לפי סדר:

    1. משתנה סביבה PAIRS_ZOOM_STORAGE_URL (אם קיים).
    2. ברירת מחדל: קובץ SQLite בשם data/zoom_studies.db מתחת ל-project_root.

    study_prefix:
    -------------
    1. משתנה סביבה PAIRS_ZOOM_STUDY_PREFIX (אם קיים).
    2. ברירת מחדל: "zoom".
    """
    project_root = project_root.resolve()

    env_url = os.getenv("PAIRS_ZOOM_STORAGE_URL")
    if env_url:
        storage_url = env_url
    else:
        db_path = project_root / "data" / DEFAULT_ZOOM_DB_NAME
        db_path.parent.mkdir(parents=True, exist_ok=True)
        # SQLite URL בפורמט ש-Optuna מבין
        storage_url = f"sqlite:///{db_path}"

    study_prefix = os.getenv("PAIRS_ZOOM_STUDY_PREFIX", DEFAULT_ZOOM_PREFIX)

    cfg = ZoomStorageConfig(
        project_root=project_root,
        storage_url=storage_url,
        study_prefix=study_prefix,
    )
    logger.debug(
        "Resolved ZoomStorageConfig: storage_url=%s, study_prefix=%s",
        cfg.storage_url,
        cfg.study_prefix,
    )
    return cfg


def _normalize_pair(pair: str) -> str:
    """
    נירמול שמות זוגות:
    - הורדת רווחים
    - upper
    """
    return pair.strip().upper().replace(" ", "")


def build_zoom_study_name(
    pair: str,
    *,
    cfg: ZoomStorageConfig,
    stage: Optional[int] = None,
) -> str:
    """
    יוצר study_name קונסיסטנטי ל-Zoom.

    פורמטים אפשריים:
    -----------------
    1. בלי stage:
       "{prefix}::{PAIR}"

    2. עם stage מספרי:
       "{prefix}::{PAIR}::stage{stage}"
    """
    norm_pair = _normalize_pair(pair)
    if stage is None:
        return f"{cfg.study_prefix}::{norm_pair}"
    return f"{cfg.study_prefix}::{norm_pair}::stage{int(stage)}"


@dataclass(frozen=True)
class ZoomStudyMeta:
    """
    Metadata בסיסי ל-study של Zoom שנפרסר משם ה-study.
    """

    raw_name: str
    prefix: str
    pair: str
    stage: Optional[int]


_STAGE_RE = re.compile(r"stage(\d+)", re.IGNORECASE)


def parse_zoom_study_name(
    name: str,
    *,
    expected_prefix: Optional[str] = None,
) -> Optional[ZoomStudyMeta]:
    """
    מנסה לפרסר שם study בפורמט Zoom.

    נתמך:
    -----
    - "zoom::BITO-BKCH::stage1"
    - "zoom::BITO-BKCH"
    - "ZOOM::QQQ-SPY::STAGE3" (לא רגיש לעולם ל-stage, אבל prefix כן).

    אם expected_prefix מוגדר:
        - אם החלק הראשון != expected_prefix → מחזיר None.
    """
    parts = name.split("::")
    if not parts:
        return None

    prefix = parts[0]
    if expected_prefix is not None and prefix != expected_prefix:
        return None

    if len(parts) == 1:
        # אין לנו מידע על pair — לא נחשב כ-Zoom תקין
        return None

    pair = parts[1]

    stage: Optional[int] = None
    if len(parts) >= 3:
        m = _STAGE_RE.search(parts[2])
        if m:
            try:
                stage = int(m.group(1))
            except ValueError:
                stage = None

    return ZoomStudyMeta(
        raw_name=name,
        prefix=prefix,
        pair=pair,
        stage=stage,
    )
