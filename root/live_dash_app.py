# -*- coding: utf-8 -*-
"""
live_dash_app.py — לוח בקרה חי ל‑IBKR + סוכן ויזואליזציה
------------------------------------------------------
• אפליקציית Dash נקייה בעברית (UTF‑8), עם חוויית משתמש מקצועית.
• טעינה בטוחה של נתוני זוגות, גרף משודרג, וטיפול שגיאות אלגנטי.
• שדרוגים: אימות קלט, שמירת מצב, רענון אוטומטי אופציונלי, ולוגים.
"""

from __future__ import annotations

import traceback
from typing import Optional

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import pandas as pd

# סוכן הוויזואליזציה (עם נפילת חסד אם אינו קיים)
try:
    from visualization_agent import VisualizationAgent  # type: ignore
except Exception:  # fallback דליל לשימוש בסיסי
    class VisualizationAgent:  # type: ignore
        def __init__(self, dark_mode: bool = True) -> None:
            self.dark_mode = dark_mode
        def load_live_data(self, sym1: str, sym2: str) -> pd.DataFrame:
            return pd.DataFrame()
        def generate_enhanced_plot(self, df: pd.DataFrame, pair_name: str = ""):
            return go.Figure(layout=dict(title=f"{pair_name} — אין נתונים"))

# כלי עזר מ‑ib_insync (לא חובה לרוץ)
try:
    from ib_insync import util  # type: ignore
except Exception:
    util = None  # לא חובה לשימוש כאן

# ------------------------- אפליקציית Dash -------------------------
external_scripts: list[str] = []
external_stylesheets: list[str] = [
    "https://cdnjs.cloudflare.com/ajax/libs/normalize/8.0.1/normalize.min.css",
]

app = dash.Dash(
    __name__,
    suppress_callback_exceptions=True,
    external_scripts=external_scripts,
    external_stylesheets=external_stylesheets,
)
server = app.server
agent = VisualizationAgent(dark_mode=True)

# --------------------------- Layout -------------------------------
app.layout = html.Div(
    [
        html.H1("📈 לוח מסחר חי — IBKR", style={"textAlign": "center", "margin": "8px 0 16px"}),
        dcc.Tabs(
            id="tabs",
            value="tab-graph",
            children=[
                dcc.Tab(label="📊 גרף חי מ‑IBKR", value="tab-graph"),
                dcc.Tab(label="🤖 סוכן קוד (קריאה בלבד)", value="tab-code-agent"),
            ],
        ),
        html.Div(id="tab-content"),
        dcc.Store(id="store-data"),  # שמירת דאטה בזיכרון דפדפן
        dcc.Interval(id="auto-refresh", interval=60_000, disabled=True),  # רענון כל דקה (מושבת כברירת מחדל)
    ],
    style={"maxWidth": "1100px", "margin": "0 auto", "padding": "12px"},
)

# ------------------------ Tab Renderer ----------------------------
@app.callback(Output("tab-content", "children"), Input("tabs", "value"))
def render_tab_content(tab: str):
    if tab == "tab-graph":
        return html.Div(
            [
                html.Div(
                    [
                        dcc.Input(
                            id="symbol1",
                            type="text",
                            placeholder="נכס 1 (למשל: XLC)",
                            value="XLC",
                            persistence=True,
                            style={"marginRight": "8px", "width": "160px"},
                        ),
                        dcc.Input(
                            id="symbol2",
                            type="text",
                            placeholder="נכס 2 (למשל: XLY)",
                            value="XLY",
                            persistence=True,
                            style={"marginRight": "8px", "width": "160px"},
                        ),
                        html.Button("משוך נתונים", id="fetch-button", n_clicks=0, style={"marginRight": "8px"}),
                        dcc.Checklist(
                            id="autorefresh-toggle",
                            options=[{"label": "רענון אוטומטי (כל דקה)", "value": "on"}],
                            value=[],
                            style={"display": "inline-block", "marginRight": "8px"},
                        ),
                    ],
                    style={"marginBottom": "14px", "display": "flex", "alignItems": "center"},
                ),
                dcc.Loading(id="loading", type="circle", children=[dcc.Graph(id="live-graph")]),
                html.Div(id="status-line", style={"marginTop": "6px", "color": "#888"}),
            ]
        )
    else:  # tab-code-agent
        return html.Div(
            [
                html.P(
                    "סוכן הקוד מיועד להדגמות ותצפיות בלבד בסביבה זו. אינטגרציית אוטומציה מלאה תרוץ מאובטחת בליבה.",
                    style={"margin": "8px 0"},
                )
            ]
        )

# הפעלה/כיבוי רענון אוטומטי
@app.callback(
    Output("auto-refresh", "disabled"),
    Input("autorefresh-toggle", "value"),
)
def toggle_autorefresh(val):
    return not (val and "on" in val)

# לוגיקת משיכת הנתונים — מופעלת בלחיצה או באינטרוול
@app.callback(
    Output("store-data", "data"),
    Output("status-line", "children"),
    Input("fetch-button", "n_clicks"),
    Input("auto-refresh", "n_intervals"),
    State("symbol1", "value"),
    State("symbol2", "value"),
    prevent_initial_call=True,
)
def fetch_data(n_clicks, n_intervals, symbol1: Optional[str], symbol2: Optional[str]):
    try:
        s1 = (symbol1 or "").strip().upper()
        s2 = (symbol2 or "").strip().upper()
        if not s1 or not s2 or s1 == s2:
            return dash.no_update, "⚠️ הזן שני נכסים שונים (לדוגמה: XLC, XLY)."

        # טעינת דאטה מסוכן הוויזואליזציה
        df = agent.load_live_data(s1, s2)
        if df is None or df.empty:
            return dash.no_update, f"ℹ️ אין נתונים זמינים עבור {s1}-{s2}."

        # אחידות שמות עמודות (Date/Close חובה לתצוגה)
        df = df.copy()
        if "Date" not in df.columns:
            if not isinstance(df.index, pd.DatetimeIndex):
                return dash.no_update, "⚠️ פורמט תאריך לא נתמך."
            df = df.reset_index().rename(columns={"index": "Date"})
        if "Close" not in df.columns:
            # ננסה לקחת את העמודה המספרית הראשונה
            num_cols = df.select_dtypes(include="number").columns
            if len(num_cols) == 0:
                return dash.no_update, "⚠️ לא נמצאה עמודת מחיר מספרית."
            df = df.rename(columns={num_cols[0]: "Close"})

        # שמירה ב‑Store כ‑records (ידידותי ל‑Dash)
        return df.to_dict("records"), f"✅ נטענו {len(df)} שורות עבור {s1}-{s2}."
    except Exception:
        traceback.print_exc()
        return dash.no_update, "❌ שגיאה בעת משיכת הנתונים. ראה לוג קונסול."

# רינדור הגרף
@app.callback(
    Output("live-graph", "figure"),
    Input("store-data", "data"),
    State("symbol1", "value"),
    State("symbol2", "value"),
)
def update_graph(records, symbol1, symbol2):
    try:
        if not records:
            return go.Figure()
        df = pd.DataFrame.from_records(records)
        pair = f"{(symbol1 or '').upper()}-{(symbol2 or '').upper()}"
        # גרף משודרג מסוכן אם קיים
        try:
            fig = agent.generate_enhanced_plot(df, pair_name=pair)
            # אם הסוכן מחזיר Figure — נחזיר כפי שהוא
            if isinstance(fig, go.Figure):
                return fig
        except Exception:
            pass
        # גרף בסיסי (fallback)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], mode="lines", name=pair))
        fig.update_layout(title=f"{pair}", margin=dict(l=10, r=10, t=40, b=10))
        return fig
    except Exception:
        traceback.print_exc()
        return go.Figure(layout=dict(title="❌ שגיאה בעת ציור הגרף"))


if __name__ == "__main__":
    # הפעלה מקומית
    app.run_server(debug=True, port=6083)  # שמרנו על אותו פורט
