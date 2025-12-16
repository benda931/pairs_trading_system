# root/visualization.py
"""
Visualization toolkit for pairs-trading optimization system.
Heatmaps, scatter plots, feature importance, rolling metrics, and outlier visualization.
All functions return Plotly figures (ready for st.plotly_chart).
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def plot_heatmap(df, x='z_entry', y='lookback', z='sharpe', title='Parameter Heatmap'):
    """
    Create a heatmap of parameter search results.
    """
    if df.empty or x not in df or y not in df or z not in df:
        return go.Figure()
    pivot = df.pivot_table(index=y, columns=x, values=z, aggfunc='mean')
    fig = px.imshow(pivot, aspect='auto', color_continuous_scale='RdBu_r', title=title)
    fig.update_layout(xaxis_title=x, yaxis_title=y)
    return fig

def plot_cluster_scatter(df, x='feature1', y='feature2', color='cluster', hover='cluster_info', title='Cluster Scatter (PCA)'):
    """
    Plot clusters in 2D PCA space.
    """
    if df.empty or x not in df or y not in df or color not in df:
        return go.Figure()
    fig = px.scatter(df, x=x, y=y, color=df[color].astype(str), hover_data=[hover], title=title, template='plotly_white')
    fig.update_traces(marker=dict(size=10, opacity=0.7), selector=dict(mode='markers'))
    fig.update_layout(legend_title_text='Cluster')
    return fig

def plot_feature_importance(imp_df, x='feature', y='importance', title='Feature Importance'):
    """
    Plot bar chart of feature importances.
    """
    if imp_df is None or imp_df.empty or x not in imp_df or y not in imp_df:
        return go.Figure()
    fig = px.bar(imp_df, x=x, y=y, title=title, template='plotly_white')
    fig.update_layout(xaxis_title='Feature', yaxis_title='Importance')
    return fig

def plot_rolling_sharpe(df, window=60, returns_col='returns', title='Rolling Sharpe Ratio'):
    """
    Plot rolling Sharpe ratio.
    """
    if returns_col not in df:
        return go.Figure()
    rolling_sharpe = df[returns_col].rolling(window).mean() / (df[returns_col].rolling(window).std() + 1e-8) * (window ** 0.5)
    fig = go.Figure(go.Scatter(
        x=df.index,
        y=rolling_sharpe,
        mode='lines',
        name='Rolling Sharpe'
    ))
    fig.update_layout(title=title, xaxis_title='Date', yaxis_title='Sharpe Ratio')
    return fig

def plot_return_histogram(df, returns_col='returns', title='Return Distribution'):
    """
    Plot histogram of strategy returns.
    """
    if returns_col not in df:
        return go.Figure()
    fig = px.histogram(df, x=returns_col, nbins=50, title=title, template='plotly_white')
    fig.update_layout(xaxis_title='Returns', yaxis_title='Frequency')
    return fig

def plot_outlier_scatter(df, x='feature1', y='feature2', z='sharpe', threshold=None, title='Outlier Scatter'):
    """
    Highlight outlier configs in scatter, e.g., by Sharpe threshold.
    """
    if df.empty or x not in df or y not in df or z not in df:
        return go.Figure()
    if threshold is None:
        threshold = df[z].quantile(0.98)
    df['outlier'] = df[z] > threshold
    fig = px.scatter(df, x=x, y=y, color='outlier', hover_data=[z], title=title)
    fig.update_traces(marker=dict(size=11, opacity=0.8))
    fig.update_layout(legend_title_text='Outlier')
    return fig

# Optional: Utility for dark mode
def set_dark_mode(fig):
    fig.update_layout(template='plotly_dark')
    return fig

