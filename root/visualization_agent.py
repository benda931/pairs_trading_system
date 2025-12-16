# נ“ visualization_agent.py ג€“ ׳›׳•׳׳ ׳׳× ׳›׳ ׳”׳™׳›׳•׳׳•׳× ׳”׳׳×׳§׳“׳׳•׳× + ׳”׳˜׳׳¢׳” ׳׳•׳˜׳•׳׳˜׳™׳× ׳‘׳׳׳©׳§ ׳”׳¨׳׳©׳™

import os
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from datetime import datetime
from statsmodels.tsa.stattools import coint

def connect_to_ibkr(self, host='127.0.0.1', port=6083, client_id=1):
    try:
        from ib_insync import IB
        ib = IB()
        ib.connect(host, port, clientId=client_id)
        print(f"נ” Connected to IBKR at {host}:{port} with client ID {client_id}")
        return ib
    except Exception as e:
        print("ג Failed to connect to IBKR:", e)
        return None


class VisualizationAgent:
    def __init__(self, dark_mode=False, insights_log='insights_log.csv'):
        self.dark_mode = dark_mode
        self.insights_log = insights_log

    def log_insight(self, message):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(self.insights_log, 'a', encoding='utf-8') as f:
            f.write(f"{timestamp},{message}\n")

    def fix_dataframe(self, df):
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
            df = df.sort_values(by='datetime')
        df = df.fillna(method='ffill').fillna(method='bfill')
        return df

    def calculate_pair_metrics(self, df):
        results = {}
        if {'price1', 'price2'}.issubset(df.columns):
            X = df['price2'].values.reshape(-1, 1)
            y = df['price1'].values
            beta = np.linalg.lstsq(X, y, rcond=None)[0][0]
            results['beta'] = round(beta, 4)
            results['rolling_corr'] = df['price1'].rolling(30).corr(df['price2']).iloc[-1]
            score, pvalue, _ = coint(df['price1'], df['price2'])
            results['cointegration_pvalue'] = round(pvalue, 5)
            z = df['zscore'].dropna()
            reversion_strength = -np.polyfit(np.arange(len(z)), np.abs(z), 1)[0]
            results['mean_reversion_score'] = round(reversion_strength, 3)
            std_beta = df['price1'].std() / df['price2'].std()
            stability = (1 - abs(std_beta - beta)) * results['rolling_corr']
            results['pair_stability_index'] = round(stability, 3)
            noise = df['spread'].diff().std() / df['spread'].std()
            results['noise_ratio'] = round(noise, 3)
            if 'zscore' in df.columns:
                edge = abs(df['zscore'].iloc[-1]) / (1 + noise)
                confidence = edge * (1 - pvalue)
                results['entry_confidence'] = round(confidence, 3)
            distance = np.abs(df['price1'] - df['price2'])
            ddr = np.mean(distance.tail(5)) / np.mean(distance.head(5))
            results['dynamic_distance_ratio'] = round(ddr, 3)
            svr = df['spread'].rolling(10).std().iloc[-1] / df['spread'].rolling(30).std().iloc[-1]
            results['spread_volatility_ratio'] = round(svr, 3)
            z_crosses = ((df['zscore'].shift(1) * df['zscore']) < 0).sum()
            results['zscore_reversals'] = int(z_crosses)
        return results

    def auto_color_scheme(self, df):
        if 'price' in df.columns:
            trend = df['price'].iloc[-1] - df['price'].iloc[0]
            if trend > 0:
                return dict(color='green')
            elif trend < 0:
                return dict(color='red')
            else:
                return dict(color='gray')
        return dict(color='blue')

    def auto_layout_config(self, df):
        layout = dict(
            height=600,
            autosize=True,
            hovermode='x unified',
            margin=dict(l=10, r=10, t=40, b=10),
            xaxis=dict(rangeslider_visible=True)
        )
        return layout

    def select_chart_type(self, df):
        if 'price' in df.columns and df['price'].diff().abs().mean() > 0.5:
            return 'candlestick'
        elif 'zscore' in df.columns:
            return 'lines+markers'
        else:
            return 'line'

    def generate_enhanced_plot(self, df, pair_name="׳¦׳׳“", chart_type=None):
        df = self.fix_dataframe(df)
        fig = go.Figure()
        template = 'plotly_dark' if self.dark_mode else 'plotly_white'
        annotations = []
        metrics = self.calculate_pair_metrics(df)
        for key, val in metrics.items():
            text = f"נ“ {key}: {val}"
            annotations.append(dict(xref='paper', yref='paper', x=0.01, y=1.15 - 0.05 * len(annotations), showarrow=False,
                                    text=text, font=dict(color='orange', size=12)))
            self.log_insight(f"{pair_name} ג€“ {text}")
        chart_type = chart_type or self.select_chart_type(df)
        line_style = self.auto_color_scheme(df)
        if chart_type == 'candlestick' and {'open', 'high', 'low', 'close'}.issubset(df.columns):
            fig.add_trace(go.Candlestick(x=df['datetime'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Candlestick'))
        elif chart_type == 'lines+markers' and 'zscore' in df.columns:
            fig.add_trace(go.Scatter(x=df['datetime'], y=df['zscore'], mode='lines+markers', name='Z-Score', line=line_style))
        elif 'price' in df.columns:
            fig.add_trace(go.Scatter(x=df['datetime'], y=df['price'], mode='lines', name='Price', line=line_style))
        layout_config = self.auto_layout_config(df)
        layout_config.update({
            'title': f"{pair_name} ג€“ ׳’׳¨׳£ ׳׳©׳•׳“׳¨׳’",
            'xaxis_title': '׳–׳׳',
            'yaxis_title': '׳¢׳¨׳',
            'template': template,
            'annotations': annotations
        })
        fig.update_layout(**layout_config)
        return fig

    def generate_comparison_plot(self, df_short, df_long, pair_name="׳¦׳׳“"):
        df_short = self.fix_dataframe(df_short)
        df_long = self.fix_dataframe(df_long)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_short['datetime'], y=df_short['zscore'], mode='lines', name='Z-Score ג€“ ׳˜׳•׳•׳— ׳§׳¦׳¨'))
        fig.add_trace(go.Scatter(x=df_long['datetime'], y=df_long['zscore'], mode='lines', name='Z-Score ג€“ ׳˜׳•׳•׳— ׳׳¨׳•׳'))
        fig.update_layout(
            title=f"{pair_name} ג€“ ׳”׳©׳•׳•׳׳× Z-Score ׳‘׳˜׳•׳•׳—׳™׳ ׳©׳•׳ ׳™׳",
            xaxis_title='׳–׳׳',
            yaxis_title='Z-Score',
            template='plotly_dark' if self.dark_mode else 'plotly_white',
            hovermode='x unified',
            height=600,
            margin=dict(l=10, r=10, t=40, b=10)
        )
        return fig

    def generate_metric_table(self, df):
        metrics = self.calculate_pair_metrics(df)
        table_data = [{'׳׳“׳“': key, '׳¢׳¨׳': val} for key, val in metrics.items()]
        return pd.DataFrame(table_data)


    def upgrade_dashboard(self, dashboard_path='dashboard.py'):
        import re
        tab_code = '''
# נ”§ ׳˜׳׳‘ ׳’׳¨׳₪׳™ ׳׳©׳•׳“׳¨׳’ ג€“ ׳ ׳•׳¦׳¨ ׳׳•׳˜׳•׳׳˜׳™׳× ׳¢׳ ׳™׳“׳™ VisualizationAgent
import pandas as pd
from visualization_agent import VisualizationAgent
import streamlit as st

def render_visuals_tab():
    st.header("נ“ ׳’׳¨׳£ ׳׳©׳•׳“׳¨׳’ ׳׳”׳¡׳•׳›׳")
    agent = VisualizationAgent(dark_mode=True)
    try:
        df = pd.read_csv("logs/XLC-XLY_log.csv")
        fig = agent.generate_enhanced_plot(df, pair_name="XLC-XLY")
        st.plotly_chart(fig, width="stretch")
        st.subheader("נ“‹ ׳׳“׳“׳™׳ ׳˜׳›׳ ׳™׳™׳")
        st.dataframe(agent.generate_metric_table(df))
    except Exception as e:
        st.error(f"׳©׳’׳™׳׳” ׳‘׳˜׳¢׳™׳ ׳× ׳ ׳×׳•׳ ׳™׳: {e}")
'''

        if not os.path.exists(dashboard_path):
            print("ג ׳§׳•׳‘׳¥ dashboard.py ׳׳ ׳ ׳׳¦׳")
            return

        with open(dashboard_path, "r", encoding="utf-8") as f:
            content = f.read()

        if not re.search(r'def\s+render_visuals_tab\s*\(', content):
            with open(dashboard_path, "a", encoding="utf-8") as f:
                f.write(tab_code)
            print("ג… ׳ ׳•׳¡׳₪׳” ׳₪׳•׳ ׳§׳¦׳™׳™׳× render_visuals_tab ׳-dashboard.py")
        else:
            print("ג„¹ן¸ ׳₪׳•׳ ׳§׳¦׳™׳™׳× render_visuals_tab ׳›׳‘׳¨ ׳§׳™׳™׳׳× ׳‘׳§׳•׳‘׳¥")
        tab_code = '''\n\n# נ”§ ׳˜׳׳‘ ׳’׳¨׳₪׳™ ׳׳©׳•׳“׳¨׳’ ג€“ ׳ ׳•׳¦׳¨ ׳׳•׳˜׳•׳׳˜׳™׳× ׳¢׳ ׳™׳“׳™ VisualizationAgent\nimport pandas as pd\nfrom visualization_agent import VisualizationAgent\nimport streamlit as st\n\ndef render_visuals_tab():\n    st.header("נ“ ׳’׳¨׳£ ׳׳©׳•׳“׳¨׳’ ׳׳”׳¡׳•׳›׳")\n    agent = VisualizationAgent(dark_mode=True)\n    try:\n        df = pd.read_csv("logs/XLC-XLY_log.csv")\n        fig = agent.generate_enhanced_plot(df, pair_name="XLC-XLY")\n        st.plotly_chart(fig, width="stretch")\n        st.subheader("נ“‹ ׳׳“׳“׳™׳ ׳˜׳›׳ ׳™׳™׳")\n        st.dataframe(agent.generate_metric_table(df))\n    except Exception as e:\n        st.error(f"׳©׳’׳™׳׳” ׳‘׳˜׳¢׳™׳ ׳× ׳ ׳×׳•׳ ׳™׳: {e}")\n'''

        if not os.path.exists(dashboard_path):
            print("ג ׳§׳•׳‘׳¥ dashboard.py ׳׳ ׳ ׳׳¦׳")
            return

        with open(dashboard_path, "r", encoding="utf-8") as f:
            content = f.read()

        if "render_visuals_tab" not in content:
            with open(dashboard_path, "a", encoding="utf-8") as f:
                f.write(tab_code)
            print("ג… ׳ ׳•׳¡׳₪׳” ׳₪׳•׳ ׳§׳¦׳™׳™׳× render_visuals_tab ׳-dashboard.py")
        else:
            print("ג„¹ן¸ ׳₪׳•׳ ׳§׳¦׳™׳™׳× render_visuals_tab ׳›׳‘׳¨ ׳§׳™׳™׳׳× ׳‘-dashboard.py")

