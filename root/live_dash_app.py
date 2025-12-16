# נ“ live_dash_app.py (׳׳¢׳•׳“׳›׳) ג€“ ׳׳•׳— ׳‘׳§׳¨׳” ׳׳—׳•׳“ ׳׳׳¢׳¨׳›׳× + ׳¡׳•׳›׳ ׳ ׳¨׳׳•׳× + ׳¡׳•׳›׳ ׳׳”׳ ׳“׳¡ ׳§׳•׳“

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import pandas as pd
from visualization_agent import VisualizationAgent
import traceback

# ׳”׳•׳¡׳₪׳× ׳×׳׳™׳›׳” ׳‘-util ׳׳×׳•׳ ib_insync (׳“׳¨׳•׳© ׳׳’׳¨׳£ ׳—׳™)
from ib_insync import util

app = dash.Dash(__name__)
agent = VisualizationAgent(dark_mode=True)

app.layout = html.Div([
    html.H1("נ₪– ׳׳¢׳¨׳›׳× ׳׳¡׳—׳¨ ׳—׳›׳׳” ג€“ ׳׳•׳— ׳‘׳§׳¨׳” ׳׳—׳•׳“", style={'textAlign': 'center'}),

    dcc.Tabs(id='tabs', value='tab-graph', children=[
        dcc.Tab(label='נ“¡ ׳’׳¨׳£ ׳—׳™ ׳-IBKR', value='tab-graph'),
        dcc.Tab(label='נ› ן¸ ׳¡׳•׳›׳ ׳׳”׳ ׳“׳¡ ׳§׳•׳“ (׳‘׳§׳¨׳•׳‘)', value='tab-code-agent'),
    ]),

    html.Div(id='tab-content')
])

@app.callback(
    Output('tab-content', 'children'),
    Input('tabs', 'value')
)
def render_tab_content(tab):
    if tab == 'tab-graph':
        return html.Div([
            html.Div([
                dcc.Input(id='symbol1', type='text', placeholder='׳ ׳›׳¡ 1 (׳׳׳©׳: XLC)', value='XLC', style={'marginRight': '10px'}),
                dcc.Input(id='symbol2', type='text', placeholder='׳ ׳›׳¡ 2 (׳׳׳©׳: XLY)', value='XLY', style={'marginRight': '10px'}),
                html.Button('נ€ ׳׳©׳•׳ ׳“׳׳˜׳” ׳—׳™׳”', id='fetch-button', n_clicks=0)
            ], style={'marginBottom': '20px', 'display': 'flex'}),

            dcc.Loading(
                id="loading",
                children=[dcc.Graph(id='live-graph')],
                type="circle"
            )
        ])

    elif tab == 'tab-code-agent':
        return html.Div([
            html.P("ג™ן¸ ׳‘׳§׳¨׳•׳‘ ג€“ ׳¡׳•׳›׳ ׳׳”׳ ׳“׳¡ ׳§׳•׳“ ׳™׳©׳•׳׳‘ ׳›׳׳ ׳׳•׳˜׳•׳׳˜׳™׳× ׳¢׳ ׳×׳™׳§׳•׳ ׳§׳•׳“, ׳‘׳“׳™׳§׳•׳× ׳•׳”׳¦׳¢׳•׳× ׳©׳“׳¨׳•׳’׳™׳ ׳׳׳¢׳¨׳›׳×.")
        ])

@app.callback(
    Output('live-graph', 'figure'),
    Input('fetch-button', 'n_clicks'),
    State('symbol1', 'value'),
    State('symbol2', 'value')
)
def update_graph(n_clicks, symbol1, symbol2):
    if n_clicks == 0:
        return go.Figure()

    try:
        print(f"נ”„ ׳׳ ׳¡׳” ׳׳׳©׳•׳ ׳“׳׳˜׳” ׳—׳™׳” ׳¢׳‘׳•׳¨: {symbol1}-{symbol2}")
        df = agent.load_live_data(symbol1, symbol2)
        if df is None or df.empty:
            print(f"ג ן¸ ׳׳ ׳”׳×׳§׳‘׳׳• ׳ ׳×׳•׳ ׳™׳ ׳-IBKR ׳¢׳‘׳•׳¨ {symbol1}-{symbol2}")
            return go.Figure(layout=dict(title=f"ג ׳©׳’׳™׳׳”: ׳׳ ׳”׳×׳§׳‘׳׳• ׳ ׳×׳•׳ ׳™׳ ׳-IBKR ׳¢׳‘׳•׳¨ {symbol1}-{symbol2}"))

        fig = agent.generate_enhanced_plot(df, pair_name=f"{symbol1}-{symbol2}")
        return fig
    except Exception as e:
        print("נ”¥ ׳©׳’׳™׳׳” ׳—׳¨׳™׳’׳”:")
        traceback.print_exc()
        return go.Figure(layout=dict(title=f"נ”¥ ׳©׳’׳™׳׳” ׳—׳¨׳™׳’׳”: {str(e)}"))

if __name__ == '__main__':
    app.run(debug=True, port=6083)

