# -*- coding: utf-8 -*-
"""
root_desktop/views/main_window.py — Main Window & Tabs (HF-grade v2)
====================================================================

חלון ראשי משודרג לאפליקציית ה-Desktop:

- מקבל app_ctx (DesktopAppContext) מתוך root_desktop.app
- בונה חלון ראשי עם QTabWidget + StatusBar + Toolbar
- כולל:
    • טאב Backtest משודרג ברמת "קרן גידור":
        - בחירת טיקרים + טווח תאריכים
        - בחירת אסטרטגיה / פרמטרים (z-open / z-close / rolling window)
        - כפתורי Run / Clear / Export
        - טבלת תוצאות + תקציר ביצועים (Sharpe / CAGR / Max DD / Trades)
    • טאבים עתידיים (Placeholders):
        - Portfolio
        - Matrix Research
        - Insights
        - Risk Engine

תפקיד הקובץ:
-------------
- Desktop Shell (ללא WEB).
- נקודת החיבור ל-core/backtesting.run_backtest.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING, Optional

from datetime import date

from PySide6.QtCore import Qt, QDate
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QLabel,
    QLineEdit,
    QDateEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QMessageBox,
    QTabWidget,
    QComboBox,
    QSpinBox,
    QDoubleSpinBox,
    QGroupBox,
    QStatusBar,
    QToolBar,
    QHeaderView,
)

# טיפוס רק בשביל type hints, בלי יצירת תלות בזמן ריצה
if TYPE_CHECKING:
    from root_desktop.app import DesktopAppContext


# =========================
#   Backtest Tab (HF-grade)
# =========================

class BacktestTab(QWidget):
    """
    טאב Backtest משודרג:

    קלט:
    ----
    - Symbol 1 / Symbol 2
    - טווח תאריכים
    - אסטרטגיה (Strategy) – למשל:
        • Z-Score Mean Reversion
        • Cointegration
        • Vol Target Pairs
    - פרמטרים עיקריים:
        • rolling_window
        • z_open / z_close
        • capital_profile (Paper / Live)

    תצוגה:
    ------
    - כפתור Run Backtest
    - כפתור Clear
    - סיכום ביצועים (Sharpe / CAGR / Max DD / #Trades)
    - טבלת תוצאות (Date / PnL / Equity / Position)
    """

    def __init__(self, app_ctx: "DesktopAppContext | None" = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.app_ctx = app_ctx

        self._setup_ui()

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------
    def _setup_ui(self) -> None:
        main_layout = QVBoxLayout(self)

        # ====== Upper config area (inputs + strategy) ======
        top_layout = QHBoxLayout()
        main_layout.addLayout(top_layout)

        # ---- Left: Symbols + dates ----
        left_group = QGroupBox("Basic Inputs", self)
        left_form = QFormLayout(left_group)

        self.symbol1_edit = QLineEdit(self)
        self.symbol1_edit.setPlaceholderText("לדוגמה: SPY")
        left_form.addRow("Symbol 1:", self.symbol1_edit)

        self.symbol2_edit = QLineEdit(self)
        self.symbol2_edit.setPlaceholderText("לדוגמה: QQQ")
        left_form.addRow("Symbol 2:", self.symbol2_edit)

        self.start_date_edit = QDateEdit(self)
        self.start_date_edit.setCalendarPopup(True)
        self.start_date_edit.setDisplayFormat("yyyy-MM-dd")
        self.start_date_edit.setDate(QDate(2020, 1, 1))

        self.end_date_edit = QDateEdit(self)
        self.end_date_edit.setCalendarPopup(True)
        self.end_date_edit.setDisplayFormat("yyyy-MM-dd")
        self.end_date_edit.setDate(QDate.currentDate())

        dates_box = QHBoxLayout()
        dates_box.addWidget(QLabel("Start:", self))
        dates_box.addWidget(self.start_date_edit)
        dates_box.addWidget(QLabel("End:", self))
        dates_box.addWidget(self.end_date_edit)

        left_form.addRow("Dates:", QWidget(self))
        # נעשה טריק קטן – נוסיף Widget עטוף
        left_form.addRow(dates_box)

        top_layout.addWidget(left_group, stretch=2)

        # ---- Right: Strategy + params ----
        right_group = QGroupBox("Strategy & Parameters", self)
        right_form = QFormLayout(right_group)

        self.strategy_combo = QComboBox(self)
        self.strategy_combo.addItems(
            [
                "Z-Score Mean Reversion",
                "Cointegration (Pairs)",
                "Vol Target Pairs",
            ]
        )
        right_form.addRow("Strategy:", self.strategy_combo)

        self.rolling_window_spin = QSpinBox(self)
        self.rolling_window_spin.setRange(20, 1000)
        self.rolling_window_spin.setValue(120)
        right_form.addRow("Rolling window:", self.rolling_window_spin)

        self.z_open_spin = QDoubleSpinBox(self)
        self.z_open_spin.setRange(0.5, 10.0)
        self.z_open_spin.setSingleStep(0.1)
        self.z_open_spin.setValue(2.0)
        right_form.addRow("Z-open:", self.z_open_spin)

        self.z_close_spin = QDoubleSpinBox(self)
        self.z_close_spin.setRange(0.0, 10.0)
        self.z_close_spin.setSingleStep(0.1)
        self.z_close_spin.setValue(0.5)
        right_form.addRow("Z-close:", self.z_close_spin)

        self.capital_profile_combo = QComboBox(self)
        self.capital_profile_combo.addItems(["Paper", "Live (Preview)"])
        right_form.addRow("Capital profile:", self.capital_profile_combo)

        top_layout.addWidget(right_group, stretch=2)

        # ---- Actions row: Run / Clear / Export ----
        actions_layout = QHBoxLayout()
        main_layout.addLayout(actions_layout)

        self.run_btn = QPushButton("Run Backtest", self)
        self.run_btn.clicked.connect(self.on_run_clicked)  # type: ignore[arg-type]
        actions_layout.addWidget(self.run_btn)

        self.clear_btn = QPushButton("Clear Results", self)
        self.clear_btn.clicked.connect(self.on_clear_clicked)  # type: ignore[arg-type]
        actions_layout.addWidget(self.clear_btn)

        self.export_btn = QPushButton("Export CSV", self)
        self.export_btn.clicked.connect(self.on_export_clicked)  # type: ignore[arg-type]
        self.export_btn.setEnabled(False)  # עד שיהיו תוצאות
        actions_layout.addWidget(self.export_btn)

        actions_layout.addStretch(1)

        # ====== Summary bar (KPIs) ======
        self.summary_group = QGroupBox("Backtest Summary", self)
        summary_layout = QHBoxLayout(self.summary_group)

        self.sharpe_label = QLabel("Sharpe: N/A", self.summary_group)
        self.cagr_label = QLabel("CAGR: N/A", self.summary_group)
        self.maxdd_label = QLabel("Max DD: N/A", self.summary_group)
        self.trades_label = QLabel("#Trades: N/A", self.summary_group)

        for lbl in (self.sharpe_label, self.cagr_label, self.maxdd_label, self.trades_label):
            lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            summary_layout.addWidget(lbl)

        summary_layout.addStretch(1)
        main_layout.addWidget(self.summary_group)

        # ====== Results table ======
        self.results_table = QTableWidget(self)
        self.results_table.setColumnCount(4)
        self.results_table.setHorizontalHeaderLabels(["Date", "PnL", "Equity", "Position"])
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        main_layout.addWidget(self.results_table)

        # Internal cache of last result DF (for export)
        self._last_df = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _get_safe_value(row: Any, key: str) -> str:
        """
        מנסה להוציא ערך לעמודה מהשורה (Series / dict / אובייקט) ולהציג יפה.
        """
        try:
            if hasattr(row, "__getitem__") and key in row:  # type: ignore[operator]
                val = row[key]
            elif hasattr(row, key):
                val = getattr(row, key)
            else:
                val = ""
        except Exception:
            val = ""

        try:
            return f"{float(val):.4f}"
        except Exception:
            return str(val)

    def _populate_table_from_df(self, df) -> None:
        """
        ממלא את הטבלה מתוך DataFrame.
        מצופה שיהיו בו עמודות: 'pnl', 'equity', 'position' (או subset).
        """
        from root_desktop.app import LOGGER

        # נשמור לעתיד (Export CSV)
        self._last_df = df
        self.export_btn.setEnabled(True)

        try:
            import pandas as pd  # type: ignore
            if not isinstance(df, pd.DataFrame):  # type: ignore[arg-type]
                LOGGER.warning("Backtest result is not a pandas DataFrame, got: %r", type(df))
        except Exception:
            # אם pandas לא זמין – לא נבדוק
            pass

        try:
            index_values = list(df.index)  # type: ignore[attr-defined]
        except Exception:
            index_values = list(range(len(getattr(df, "values", []))))

        n_rows = len(index_values)
        self.results_table.setRowCount(n_rows)
        self.results_table.setColumnCount(4)
        self.results_table.setHorizontalHeaderLabels(["Date", "PnL", "Equity", "Position"])

        for row_idx, idx in enumerate(index_values):
            try:
                row = df.loc[idx] if hasattr(df, "loc") else df[row_idx]  # type: ignore[index]
            except Exception:
                row = {}

            # Date
            self.results_table.setItem(row_idx, 0, QTableWidgetItem(str(idx)))

            # PnL
            pnl_val = self._get_safe_value(row, "pnl")
            self.results_table.setItem(row_idx, 1, QTableWidgetItem(pnl_val))

            # Equity
            eq_val = self._get_safe_value(row, "equity")
            self.results_table.setItem(row_idx, 2, QTableWidgetItem(eq_val))

            # Position
            pos_val = self._get_safe_value(row, "position")
            self.results_table.setItem(row_idx, 3, QTableWidgetItem(pos_val))

        self.results_table.resizeColumnsToContents()
        self.results_table.resizeRowsToContents()

    def _update_summary_from_result(self, bt_result: Any) -> None:
        """
        מנסה למשוך מדדי ביצוע מהתוצאה (Sharpe / CAGR / MaxDD / #Trades) ולעדכן כותרת עליונה.
        """
        sharpe = getattr(bt_result, "sharpe", None)
        cagr = getattr(bt_result, "cagr", None)
        max_dd = getattr(bt_result, "max_dd", None)
        n_trades = getattr(bt_result, "n_trades", None)

        def fmt(x: Optional[float], pct: bool = False) -> str:
            try:
                xf = float(x)
                if pct:
                    return f"{xf * 100:.2f}%"
                return f"{xf:.2f}"
            except Exception:
                return "N/A"

        self.sharpe_label.setText(f"Sharpe: {fmt(sharpe)}")
        self.cagr_label.setText(f"CAGR: {fmt(cagr, pct=True)}")
        self.maxdd_label.setText(f"Max DD: {fmt(max_dd, pct=True)}")
        self.trades_label.setText(f"#Trades: {int(n_trades):d}" if isinstance(n_trades, (int, float)) else "#Trades: N/A")

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------
    def on_run_clicked(self) -> None:
        from root_desktop.app import LOGGER  # שימוש בלוגר המרכזי

        symbol_1 = self.symbol1_edit.text().strip()
        symbol_2 = self.symbol2_edit.text().strip()

        if not symbol_1 or not symbol_2:
            QMessageBox.warning(self, "חסר קלט", "נא למלא שני טיקרים.")
            return

        start_dt = self.start_date_edit.date().toPython()
        end_dt = self.end_date_edit.date().toPython()

        if start_dt >= end_dt:
            QMessageBox.warning(self, "טווח שגוי", "תאריך ההתחלה חייב להיות לפני תאריך הסיום.")
            return

        strategy = self.strategy_combo.currentText()
        rolling = int(self.rolling_window_spin.value())
        z_open = float(self.z_open_spin.value())
        z_close = float(self.z_close_spin.value())
        capital_profile = self.capital_profile_combo.currentText()

        LOGGER.info(
            "Running backtest: %s vs %s (%s → %s) | strat=%s, roll=%s, z=(%.2f, %.2f), profile=%s",
            symbol_1,
            symbol_2,
            start_dt,
            end_dt,
            strategy,
            rolling,
            z_open,
            z_close,
            capital_profile,
        )

        try:
            # נקודת החיבור ל-core/backtesting
            from core.backtesting import run_backtest  # type: ignore

            # לפי הצורך – אפשר להעביר את הפרמטרים הנוספים אם הפונקציה תומכת בכך
            try:
                bt_result: Any = run_backtest(
                    symbol_1=symbol_1,
                    symbol_2=symbol_2,
                    start_date=start_dt,
                    end_date=end_dt,
                    strategy=strategy,
                    rolling_window=rolling,
                    z_open=z_open,
                    z_close=z_close,
                    profile=capital_profile,
                )
            except TypeError:
                # תאימות לאחור – חתימה ישנה שלא מכירה את כל הפרמטרים
                bt_result = run_backtest(
                    symbol_1=symbol_1,
                    symbol_2=symbol_2,
                    start_date=start_dt,
                    end_date=end_dt,
                )

            # נניח שיש ל-Result שלך פונקציה שמחזירה DataFrame של ה-PnL
            if hasattr(bt_result, "to_pnl_df"):
                df = bt_result.to_pnl_df()
            elif hasattr(bt_result, "pnl_df"):
                df = bt_result.pnl_df  # type: ignore[assignment]
            else:
                # אם אין – מניחים שזה כבר DataFrame
                df = bt_result

            self._populate_table_from_df(df)
            self._update_summary_from_result(bt_result)

        except ImportError:
            LOGGER.exception("core.backtesting.run_backtest לא זמין")
            QMessageBox.critical(
                self,
                "Backtest unavailable",
                "לא הצלחתי לייבא core.backtesting.run_backtest.\n"
                "תבדוק שקיים core/backtesting.py עם הפונקציה run_backtest.",
            )
        except Exception as exc:
            LOGGER.exception("Backtest failed")
            QMessageBox.critical(
                self,
                "Backtest failed",
                f"קרתה שגיאה בזמן הרצת ה-Backtest:\n{exc}",
            )

    def on_clear_clicked(self) -> None:
        """ניקוי טבלה ו-KPIs."""
        self.results_table.clearContents()
        self.results_table.setRowCount(0)
        self._last_df = None
        self.export_btn.setEnabled(False)

        self.sharpe_label.setText("Sharpe: N/A")
        self.cagr_label.setText("CAGR: N/A")
        self.maxdd_label.setText("Max DD: N/A")
        self.trades_label.setText("#Trades: N/A")

    def on_export_clicked(self) -> None:
        """ייצוא ל-CSV – לעתיד אפשר להרחיב עם QFileDialog."""
        if self._last_df is None:
            QMessageBox.information(self, "אין נתונים", "אין תוצאות לייצוא כרגע.")
            return

        try:
            import pandas as pd  # type: ignore

            if not hasattr(self._last_df, "to_csv"):
                QMessageBox.warning(self, "Export", "אובייקט התוצאות לא תומך ב-to_csv.")
                return

            # ייצוא פשוט לקובץ זמני ליד האפליקציה
            from pathlib import Path
            out_path = Path.cwd() / "backtest_result.csv"
            self._last_df.to_csv(out_path, index=True)  # type: ignore[call-arg]
            QMessageBox.information(self, "Export", f"נשמר קובץ: {out_path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", f"כשל בייצוא CSV:\n{exc}")


# =========================
#   Placeholder Tabs
# =========================

class PortfolioTab(QWidget):
    """טאב פורטפוליו – Placeholder מקצועי לעתיד."""
    def __init__(self, app_ctx: "DesktopAppContext | None" = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Portfolio View (coming soon) — חשיפות, PnL, Allocations.", self))


class MatrixResearchTab(QWidget):
    """טאב Matrix Research – Placeholder מקצועי לעתיד."""
    def __init__(self, app_ctx: "DesktopAppContext | None" = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Matrix Research (coming soon) — קורלציות, מרחקים, Clusters.", self))


class InsightsTab(QWidget):
    """טאב Insights – Placeholder מקצועי לעתיד."""
    def __init__(self, app_ctx: "DesktopAppContext | None" = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Insights (coming soon) — תובנות אוטומטיות, חריגים, Alerts.", self))


class RiskEngineTab(QWidget):
    """טאב Risk Engine – Placeholder מקצועי לעתיד."""
    def __init__(self, app_ctx: "DesktopAppContext | None" = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Risk Engine (coming soon) — Limits, VaR/ES, Kill-switches.", self))


# =========================
#   Main Window
# =========================

class MainWindow(QMainWindow):
    """
    החלון הראשי של האפליקציה (Desktop Shell HF-grade).

    כרגע:
    -----
    - Toolbar עליון עם פעולות עיקריות.
    - StatusBar תחתון עם פרופיל / זמן התחלה.
    - טאבים:
        • Backtest (מורחב)
        • Portfolio (placeholder)
        • Matrix Research (placeholder)
        • Insights (placeholder)
        • Risk Engine (placeholder)
    """

    def __init__(self, app_ctx: "DesktopAppContext | None" = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.app_ctx = app_ctx

        self._setup_window()
        self._setup_tabs()
        self._setup_toolbar()
        self._setup_statusbar()

    def _setup_window(self) -> None:
        profile = getattr(self.app_ctx.config, "profile", "default") if self.app_ctx else "default"
        self.setWindowTitle(f"Omri Pairs Trading – Desktop (profile={profile})")
        self.resize(1400, 900)
        # אפשר להוסיף כאן אייקון בעתיד (setWindowIcon)

    def _setup_tabs(self) -> None:
        central = QWidget(self)
        layout = QVBoxLayout(central)

        self.tabs = QTabWidget(central)
        layout.addWidget(self.tabs)

        # ------- Tab 1: Backtest -------
        self.backtest_tab = BacktestTab(app_ctx=self.app_ctx, parent=self)
        self.tabs.addTab(self.backtest_tab, "Backtest")

        # ------- Tab 2: Portfolio (placeholder) -------
        self.portfolio_tab = PortfolioTab(app_ctx=self.app_ctx, parent=self)
        self.tabs.addTab(self.portfolio_tab, "Portfolio")

        # ------- Tab 3: Matrix Research (placeholder) -------
        self.matrix_tab = MatrixResearchTab(app_ctx=self.app_ctx, parent=self)
        self.tabs.addTab(self.matrix_tab, "Matrix Research")

        # ------- Tab 4: Insights (placeholder) -------
        self.insights_tab = InsightsTab(app_ctx=self.app_ctx, parent=self)
        self.tabs.addTab(self.insights_tab, "Insights")

        # ------- Tab 5: Risk Engine (placeholder) -------
        self.risk_tab = RiskEngineTab(app_ctx=self.app_ctx, parent=self)
        self.tabs.addTab(self.risk_tab, "Risk Engine")

        self.setCentralWidget(central)

    def _setup_toolbar(self) -> None:
        toolbar = QToolBar("Main Toolbar", self)
        toolbar.setMovable(False)
        self.addToolBar(Qt.TopToolBarArea, toolbar)

        # Run Backtest
        act_run_bt = QAction("Run Backtest", self)
        act_run_bt.triggered.connect(self._toolbar_run_backtest)  # type: ignore[arg-type]
        toolbar.addAction(act_run_bt)

        # Clear Backtest
        act_clear_bt = QAction("Clear", self)
        act_clear_bt.triggered.connect(self._toolbar_clear_backtest)  # type: ignore[arg-type]
        toolbar.addAction(act_clear_bt)

        toolbar.addSeparator()

        # Refresh Data (עתידי)
        act_refresh = QAction("Refresh Data", self)
        act_refresh.triggered.connect(self._toolbar_refresh_data)  # type: ignore[arg-type]
        toolbar.addAction(act_refresh)

        toolbar.addSeparator()

        # Exit
        act_exit = QAction("Exit", self)
        act_exit.triggered.connect(self.close)  # type: ignore[arg-type]
        toolbar.addAction(act_exit)

    def _setup_statusbar(self) -> None:
        status = QStatusBar(self)
        self.setStatusBar(status)

        if self.app_ctx is not None:
            started_at = getattr(self.app_ctx, "started_at", None)
            profile = getattr(self.app_ctx.config, "profile", "default")
            root = getattr(self.app_ctx, "project_root", None)
            status.showMessage(
                f"Profile: {profile} | Started: {started_at} | Root: {root}",
            )
        else:
            status.showMessage("No DesktopAppContext attached (running in basic mode).")

    # ------------------------------------------------------------------
    # Toolbar actions
    # ------------------------------------------------------------------
    def _toolbar_run_backtest(self) -> None:
        """מריץ Backtest מה-toolbar (מפעיל את כפתור הטאב)."""
        self.tabs.setCurrentWidget(self.backtest_tab)
        self.backtest_tab.on_run_clicked()

    def _toolbar_clear_backtest(self) -> None:
        """מנקה את תוצאות ה-Backtest מה-toolbar."""
        self.tabs.setCurrentWidget(self.backtest_tab)
        self.backtest_tab.on_clear_clicked()

    def _toolbar_refresh_data(self) -> None:
        """Placeholder לרענון דאטה (בהמשך יחובר ל-Core/IBKR/Data Client)."""
        QMessageBox.information(
            self,
            "Refresh Data",
            "בגרסאות הבאות כפתור זה ירענן דאטה חי (IBKR / DuckDB / SQL) עבור כל הטאבים.",
        )
