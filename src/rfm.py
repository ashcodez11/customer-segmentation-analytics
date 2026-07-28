import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

class RFMAnalyzer:
    """
    Computes Recency, Frequency, Monetary (RFM) metrics and 
    assigns customer behavioral segments using rank-based binning.
    """
    def __init__(self, feature_path=config.FEATURE_DATA_PATH):
        self.feature_path = Path(feature_path)

    def calculate_rfm(self, df=None):
        print("\n================ STARTING RFM ANALYSIS ================")
        if df is None:
            if not self.feature_path.exists():
                from src.feature_engineering import CosmeticsFeatureEngineer
                df = CosmeticsFeatureEngineer().extract_customer_features()
            else:
                df = pd.read_csv(self.feature_path)

        rfm_df = df.copy()

        # Rank-based quantile cut to guarantee equal bins even with duplicates
        # Recency: Lower Days_Since_Last_Purchase = Higher Score (5 is best)
        rfm_df['R_Score'] = pd.qcut(
            rfm_df['Days_Since_Last_Purchase'].rank(method='first', ascending=True),
            q=5,
            labels=[5, 4, 3, 2, 1]
        ).astype(int)
        
        # Frequency: Higher Purchase_Frequency = Higher Score (5 is best)
        rfm_df['F_Score'] = pd.qcut(
            rfm_df['Purchase_Frequency'].rank(method='first', ascending=True),
            q=5,
            labels=[1, 2, 3, 4, 5]
        ).astype(int)
        
        # Monetary: Higher Total_Spending = Higher Score (5 is best)
        rfm_df['M_Score'] = pd.qcut(
            rfm_df['Total_Spending'].rank(method='first', ascending=True),
            q=5,
            labels=[1, 2, 3, 4, 5]
        ).astype(int)

        # Combined Scores
        rfm_df['RFM_Score'] = rfm_df['R_Score'].astype(str) + rfm_df['F_Score'].astype(str) + rfm_df['M_Score'].astype(str)
        rfm_df['RFM_Sum'] = rfm_df['R_Score'] + rfm_df['F_Score'] + rfm_df['M_Score']

        # Industry Standard RFM Persona Assignment
        def assign_rfm_segment(row):
            r, f, m = row['R_Score'], row['F_Score'], row['M_Score']
            
            if r >= 4 and f >= 4 and m >= 4:
                return "Champions"
            elif r >= 3 and f >= 3 and m >= 3:
                return "Loyal Customers"
            elif r >= 3 and f <= 2 and m >= 3:
                return "Potential Loyalists"
            elif r >= 3 and f <= 2 and m <= 2:
                return "New / Promising"
            elif r == 2 and f >= 2:
                return "Need Attention"
            elif r <= 2 and f >= 3 and m >= 3:
                return "At Risk"
            elif r <= 2 and f <= 2 and m >= 4:
                return "Can't Lose Them"
            else:
                return "Lost Customers"

        rfm_df['RFM_Segment'] = rfm_df.apply(assign_rfm_segment, axis=1)

        print("📊 RFM Segmentation Breakdown:")
        print(rfm_df['RFM_Segment'].value_counts())
        print("================ RFM ANALYSIS COMPLETE ================\n")

        return rfm_df

if __name__ == "__main__":
    analyzer = RFMAnalyzer()
    rfm_res = analyzer.calculate_rfm()
    print(rfm_res[['Customer_ID', 'Days_Since_Last_Purchase', 'Purchase_Frequency', 'Total_Spending', 'RFM_Score', 'RFM_Segment']].head())
