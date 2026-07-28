import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

class CosmeticsClusteringEngine:
    """
    Unsupervised ML Clustering Engine with standardized persona mapping.
    """
    def __init__(self, feature_path=config.FEATURE_DATA_PATH):
        self.feature_path = Path(feature_path)
        self.scaler = StandardScaler()
        self.pca_2d = PCA(n_components=2, random_state=config.RANDOM_STATE)
        self.pca_3d = PCA(n_components=3, random_state=config.RANDOM_STATE)

    def prepare_clustering_features(self, df):
        feature_cols = [
            'Age', 'Annual_Income', 'Purchase_Frequency', 
            'Total_Spending', 'Average_Order_Value', 
            'Discount_Usage_Rate', 'Days_Since_Last_Purchase'
        ]
        X = df[feature_cols].copy().fillna(0)
        X_scaled = self.scaler.fit_transform(X)
        return X, X_scaled, feature_cols

    def evaluate_kmeans_elbow(self, X_scaled, max_k=8):
        k_results = []
        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=config.RANDOM_STATE, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            k_results.append({
                'k': k,
                'Inertia': round(kmeans.inertia_, 2),
                'Silhouette': round(silhouette_score(X_scaled, labels), 4),
                'Davies_Bouldin': round(davies_bouldin_score(X_scaled, labels), 4),
                'Calinski_Harabasz': round(calinski_harabasz_score(X_scaled, labels), 2)
            })
        return pd.DataFrame(k_results)

    def run_all_clustering_algorithms(self, n_clusters=4):
        print("\n================ STARTING ML CLUSTERING PIPELINE ================")
        if not self.feature_path.exists():
            from src.feature_engineering import CosmeticsFeatureEngineer
            df = CosmeticsFeatureEngineer().extract_customer_features()
        else:
            df = pd.read_csv(self.feature_path)

        X, X_scaled, feature_cols = self.prepare_clustering_features(df)

        # 1. K-Means
        kmeans = KMeans(n_clusters=n_clusters, random_state=config.RANDOM_STATE, n_init=10)
        kmeans_labels = kmeans.fit_predict(X_scaled)

        # 2. Agglomerative
        agg = AgglomerativeClustering(n_clusters=n_clusters)
        agg_labels = agg.fit_predict(X_scaled)

        # 3. DBSCAN
        dbscan = DBSCAN(eps=1.2, min_samples=10)
        dbscan_labels = dbscan.fit_predict(X_scaled)

        # PCA Transformations
        pca_2d_coords = self.pca_2d.fit_transform(X_scaled)
        pca_3d_coords = self.pca_3d.fit_transform(X_scaled)

        df['Cluster_KMeans'] = kmeans_labels
        df['Cluster_Agglomerative'] = agg_labels
        df['Cluster_DBSCAN'] = dbscan_labels
        
        df['PCA1'] = pca_2d_coords[:, 0]
        df['PCA2'] = pca_2d_coords[:, 1]
        df['PCA3_1'] = pca_3d_coords[:, 0]
        df['PCA3_2'] = pca_3d_coords[:, 1]
        df['PCA3_3'] = pca_3d_coords[:, 2]

        # Standardized Persona Assignment Based on Cluster Ranked Spending
        cluster_means = df.groupby('Cluster_KMeans')['Total_Spending'].mean().sort_values(ascending=False)
        ranked_clusters = cluster_means.index.tolist()

        persona_map = {}
        if len(ranked_clusters) >= 4:
            persona_map[ranked_clusters[0]] = "VIP Cosmetics Enthusiasts ✨"
            persona_map[ranked_clusters[1]] = "Frequent Buyers 🛍️"
            persona_map[ranked_clusters[2]] = "Budget Conscious 💄"
            persona_map[ranked_clusters[3]] = "At-Risk Customers ⚠️"
        else:
            labels_preset = ["VIP Cosmetics Enthusiasts ✨", "Frequent Buyers 🛍️", "Budget Conscious 💄", "At-Risk Customers ⚠️"]
            for idx, cl in enumerate(ranked_clusters):
                persona_map[cl] = labels_preset[idx % len(labels_preset)]

        df['Customer_Persona'] = df['Cluster_KMeans'].map(persona_map)

        # Evaluation Matrix
        models_eval = [
            {'Algorithm': 'K-Means', 'Silhouette': round(silhouette_score(X_scaled, kmeans_labels), 4), 'Davies-Bouldin': round(davies_bouldin_score(X_scaled, kmeans_labels), 4), 'Calinski-Harabasz': round(calinski_harabasz_score(X_scaled, kmeans_labels), 2)},
            {'Algorithm': 'Agglomerative', 'Silhouette': round(silhouette_score(X_scaled, agg_labels), 4), 'Davies-Bouldin': round(davies_bouldin_score(X_scaled, agg_labels), 4), 'Calinski-Harabasz': round(calinski_harabasz_score(X_scaled, agg_labels), 2)},
            {'Algorithm': 'DBSCAN', 'Silhouette': round(silhouette_score(X_scaled, dbscan_labels) if len(set(dbscan_labels)) > 1 else -1, 4), 'Davies-Bouldin': round(davies_bouldin_score(X_scaled, dbscan_labels) if len(set(dbscan_labels)) > 1 else -1, 4), 'Calinski-Harabasz': round(calinski_harabasz_score(X_scaled, dbscan_labels) if len(set(dbscan_labels)) > 1 else -1, 2)}
        ]
        eval_df = pd.DataFrame(models_eval)

        # Merge RFM & CLV
        from src.rfm import RFMAnalyzer
        from src.clv import CLVCalculator

        rfm_df = RFMAnalyzer().calculate_rfm(df)
        clv_df = CLVCalculator().calculate_clv(df)

        df['RFM_Segment'] = rfm_df['RFM_Segment']
        df['RFM_Score'] = rfm_df['RFM_Score']
        df['Predicted_CLV_3Yr'] = clv_df['Predicted_CLV_3Yr']
        df['CLV_Tier'] = clv_df['CLV_Tier']

        df.to_csv(config.FINAL_SEGMENTED_PATH, index=False)
        print(f"✅ Final Segmented Dataset saved to '{config.FINAL_SEGMENTED_PATH}'")
        return df, eval_df

if __name__ == "__main__":
    CosmeticsClusteringEngine().run_all_clustering_algorithms()
