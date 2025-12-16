# core/clustering.py
"""
Clustering for Pairs Optimization ג€“ Advanced Professional Module

- Unsupervised clustering (KMeans/DBSCAN) of optimal configurations.
- Auto selection of n_clusters (Elbow, Silhouette, user config).
- Dimensionality reduction for visualization (PCA/t-SNE).
- Cluster analytics: composition, sharpe/correlation profile, cluster scoring.
- Returns labeled DataFrame, suitable for visualization and meta-optimization.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

def auto_n_clusters(X, max_k=10):
    """Auto-select n_clusters via Silhouette method."""
    best_k, best_score = 2, -1
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        if score > best_score:
            best_k, best_score = k, score
    return best_k

def cluster_pairs(selected: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Cluster configs/pairs using KMeans (default) or DBSCAN.
    Adds PCA features for visualization. Calculates cluster-level analytics.
    Args:
        selected: DataFrame, output of select_features (top configs).
        config: dict, pipeline config (supports clustering method/n_clusters).
    Returns:
        DataFrame with cluster labels, PCA dims, and analytics per row.
    """
    if selected is None or selected.empty:
        return pd.DataFrame()
    df = selected.copy()
    # Select features for clustering (exclude metrics/ids)
    non_features = {'sharpe', 'return', 'drawdown', 'win_rate', 'trial_id'}
    features = [c for c in df.columns if c not in non_features and pd.api.types.is_numeric_dtype(df[c])]
    if len(features) < 2:
        raise ValueError("Not enough features for clustering")
    X = df[features].fillna(0).values
    X_scaled = StandardScaler().fit_transform(X)

    # Select clustering algorithm
    method = config.get('clustering', {}).get('method', 'kmeans')
    if method == 'dbscan':
        clusterer = DBSCAN(eps=0.8, min_samples=3)
        labels = clusterer.fit_predict(X_scaled)
    else:  # kmeans default
        n_clusters = config.get('clustering', {}).get('n_clusters', None)
        if not n_clusters:
            n_clusters = auto_n_clusters(X_scaled, max_k=8)
        clusterer = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        labels = clusterer.fit_predict(X_scaled)

    df['cluster'] = labels

    # Dimensionality reduction for visualization
    pca = PCA(n_components=2, random_state=42)
    pca_proj = pca.fit_transform(X_scaled)
    df['feature1'] = pca_proj[:, 0]
    df['feature2'] = pca_proj[:, 1]

    # Cluster analytics (mean Sharpe, drawdown, etc. per cluster)
    cluster_summary = df.groupby('cluster').agg(
        count=('sharpe', 'size'),
        mean_sharpe=('sharpe', 'mean'),
        min_drawdown=('drawdown', 'min'),
        max_return=('return', 'max'),
        avg_win_rate=('win_rate', 'mean')
    ).reset_index()
    # Merge analytics into main df for Streamlit popups
    df = df.merge(cluster_summary, on='cluster', suffixes=('', '_cluster'))

    # For plotly hover: Add cluster profile summary
    df['cluster_info'] = df.apply(
        lambda r: f"Cluster: {r['cluster']}<br>Size: {r['count']}<br>Sharpe: {r['mean_sharpe']:.2f}<br>Drawdown: {r['min_drawdown']:.2f}<br>Return: {r['max_return']:.2f}",
        axis=1,
    )
    return df

# If needed, you can also add plot_cluster_scatter for advanced Streamlit visualization:
def plot_cluster_scatter(df):
    import plotly.express as px
    fig = px.scatter(
        df,
        x='feature1',
        y='feature2',
        color='cluster',
        hover_data=['mean_sharpe', 'min_drawdown', 'count'],
        template='plotly_white',
        symbol='cluster'
    )
    fig.update_layout(title='Clustered Configurations (PCA)')
    return fig


