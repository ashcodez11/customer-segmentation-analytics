import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from src.rfm import RFMAnalyzer
from src.clv import CLVCalculator
import config

class CosmeticsClusteringEngine:
    def prepare_clustering_features(self, df):
        features = ['Age', 'Annual_Income', 'Total_Spending', 'Average_Order_Value', 'Purchase_Frequency', 'Days_Since_Last_Purchase']
        X = df[features]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X, X_scaled, scaler

    def run_all_clustering_algorithms(self, df_input=None):
        from src.feature_engineering import CosmeticsFeatureEngineer
        if df_input is None:
            engineer = CosmeticsFeatureEngineer()
            df = engineer.extract_customer_features()
        else:
            df = df_input.copy()

        df = RFMAnalyzer.compute_rfm(df)
        df = CLVCalculator.calculate_3yr_clv(df)

        X, X_scaled, _ = self.prepare_clustering_features(df)

        kmeans = KMeans(n_clusters=config.N_CLUSTERS, random_state=config.RANDOM_STATE, n_init=10)
        df['Cluster'] = kmeans.fit_predict(X_scaled)

        agg = AgglomerativeClustering(n_clusters=config.N_CLUSTERS)
        agg_labels = agg.fit_predict(X_scaled)

        dbscan = DBSCAN(eps=1.2, min_samples=5)
        dbscan_labels = dbscan.fit_predict(X_scaled)

        eval_metrics = pd.DataFrame({
            'Algorithm': ['K-Means', 'Agglomerative', 'DBSCAN'],
            'Silhouette Score': [
                round(silhouette_score(X_scaled, df['Cluster']), 3),
                round(silhouette_score(X_scaled, agg_labels), 3),
                round(silhouette_score(X_scaled, dbscan_labels), 3) if len(set(dbscan_labels)) > 1 else 0.0
            ],
            'Davies-Bouldin Index': [
                round(davies_bouldin_score(X_scaled, df['Cluster']), 3),
                round(davies_bouldin_score(X_scaled, agg_labels), 3),
                round(davies_bouldin_score(X_scaled, dbscan_labels), 3) if len(set(dbscan_labels)) > 1 else 0.0
            ],
            'Calinski-Harabasz Index': [
                round(calinski_harabasz_score(X_scaled, df['Cluster']), 1),
                round(calinski_harabasz_score(X_scaled, agg_labels), 1),
                round(calinski_harabasz_score(X_scaled, dbscan_labels), 1) if len(set(dbscan_labels)) > 1 else 0.0
            ]
        })

        cluster_means = df.groupby('Cluster')['Total_Spending'].mean()
        sorted_clusters = cluster_means.sort_values(ascending=False).index.tolist()

        persona_map = {
            sorted_clusters[0]: 'VIP Cosmetics Enthusiasts',
            sorted_clusters[1]: 'Frequent Buyers',
            sorted_clusters[2]: 'Budget Conscious',
            sorted_clusters[3]: 'At-Risk Customers'
        }
        df['Customer_Persona'] = df['Cluster'].map(persona_map)

        df.to_csv(config.FINAL_SEGMENTED_PATH, index=False)
        return df, eval_metrics

    def evaluate_kmeans_elbow(self, X_scaled):
        ks = range(2, 9)
        inertias, silhouettes = [], []
        for k in ks:
            km = KMeans(n_clusters=k, random_state=config.RANDOM_STATE, n_init=10)
            labels = km.fit_predict(X_scaled)
            inertias.append(km.inertia_)
            silhouettes.append(silhouette_score(X_scaled, labels))
        return pd.DataFrame({'K': ks, 'Inertia': inertias, 'Silhouette': silhouettes})
