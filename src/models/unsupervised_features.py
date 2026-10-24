from __future__ import annotations

from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

from src.etl.feature_pipeline import add_engineered_features


class DomainFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Transforms raw banking features by computing domain-specific risk,
    friction, and customer health indices.
    """

    def fit(self, X: pd.DataFrame | np.ndarray, y: Optional[pd.Series] = None) -> DomainFeatureEngineer:
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            # Fallback if an array was passed
            df = pd.DataFrame(X)
        else:
            df = X.copy()
        return add_engineered_features(df)


class UnsupervisedFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Unsupervised feature extractor utilizing:
    1. Isolation Forest: Computes customer anomaly score and binary outlier flags.
    2. K-Means: Clusters customer behavioral patterns, outputting cluster distances and assignments.
    """

    def __init__(
        self,
        n_clusters: int = 4,
        contamination: float = 0.05,
        random_state: int = 42,
    ) -> None:
        self.n_clusters = n_clusters
        self.contamination = contamination
        self.random_state = random_state
        self.iso_forest_: Optional[IsolationForest] = None
        self.kmeans_: Optional[KMeans] = None

    def fit(self, X: np.ndarray | pd.DataFrame, y: Optional[np.ndarray | pd.Series] = None) -> UnsupervisedFeatureEngineer:
        X_arr = np.asarray(X, dtype=np.float64)
        self.iso_forest_ = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=100,
        )
        self.iso_forest_.fit(X_arr)

        self.kmeans_ = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,
        )
        self.kmeans_.fit(X_arr)
        return self

    def transform(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        if self.iso_forest_ is None or self.kmeans_ is None:
            raise RuntimeError("UnsupervisedFeatureEngineer must be fitted before transform.")

        X_arr = np.asarray(X, dtype=np.float64)
        n_samples = X_arr.shape[0]

        # Isolation Forest outputs
        # score_samples returns opposite of anomaly score (lower is more anomalous),
        # so invert it so higher = more anomalous
        anomaly_score = -self.iso_forest_.score_samples(X_arr).reshape(-1, 1)
        is_outlier = (self.iso_forest_.predict(X_arr) == -1).astype(np.float64).reshape(-1, 1)

        # K-Means outputs
        cluster_distances = self.kmeans_.transform(X_arr)
        min_cluster_dist = np.min(cluster_distances, axis=1, keepdims=True)
        cluster_labels = self.kmeans_.predict(X_arr)

        cluster_dummies = np.zeros((n_samples, self.n_clusters), dtype=np.float64)
        for i in range(self.n_clusters):
            cluster_dummies[:, i] = (cluster_labels == i).astype(np.float64)

        return np.hstack(
            [
                X_arr,
                anomaly_score,
                is_outlier,
                min_cluster_dist,
                cluster_distances,
                cluster_dummies,
            ]
        )

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> np.ndarray:
        base_names = list(input_features) if input_features is not None else []
        unsup_names = [
            "unsup_iso_anomaly_score",
            "unsup_iso_is_outlier",
            "unsup_kmeans_min_dist",
        ]
        unsup_names.extend([f"unsup_kmeans_dist_c{i}" for i in range(self.n_clusters)])
        unsup_names.extend([f"unsup_kmeans_cluster_{i}" for i in range(self.n_clusters)])
        return np.array(base_names + unsup_names, dtype=object)
