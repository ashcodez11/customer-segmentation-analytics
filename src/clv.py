import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

class CLVCalculator:
    """
    Estimates Customer Lifetime Value (CLV) and ranks customer monetary tier.
    """
    def __init__(self, feature_path=config.FEATURE_DATA_PATH):
        self.feature_path = Path(feature_path)

    def calculate_clv(self, df=None):
        print("\n================ STARTING CLV CALCULATION ================")
        if df is None:
            if not self.feature_path.exists():
                from src.feature_engineering import CosmeticsFeatureEngineer
                df = CosmeticsFeatureEngineer().extract_customer_features()
            else:
                df = pd.read_csv(self.feature_path)

        clv_df = df.copy()

        # Historical CLV = Aggregate Total Spending
        clv_df['Historical_CLV'] = clv_df['Total_Spending']

        # Predictive CLV Formula Model
        # Predictive CLV = AOV * Annual Purchase Frequency * Estimated Customer Lifespan Years * Margin (0.60)
        customer_lifespan_years = 3.0
        profit_margin = 0.60
        
        # Estimate Annual Frequency
        lifespan_days = clv_df['Customer_Lifetime_Days'].replace(0, 1)
        annual_freq = (clv_df['Purchase_Frequency'] / lifespan_days) * 365.0
        
        clv_df['Predicted_CLV_3Yr'] = (
            clv_df['Average_Order_Value'] * annual_freq * customer_lifespan_years * profit_margin
        ).round(2)

        # Cap unrealistic projections
        max_clv_cap = clv_df['Historical_CLV'].max() * 5.0
        clv_df['Predicted_CLV_3Yr'] = np.clip(clv_df['Predicted_CLV_3Yr'], clv_df['Historical_CLV'], max_clv_cap)

        # Tier Ranking (Platinum, Gold, Silver, Bronze)
        clv_quantiles = clv_df['Predicted_CLV_3Yr'].quantile([0.50, 0.80, 0.95]).values
        
        def assign_tier(clv):
            if clv >= clv_quantiles[2]:
                return "Platinum Tier 💎"
            elif clv >= clv_quantiles[1]:
                return "Gold Tier 🥇"
            elif clv >= clv_quantiles[0]:
                return "Silver Tier 🥈"
            else:
                return "Bronze Tier 🥉"

        clv_df['CLV_Tier'] = clv_df['Predicted_CLV_3Yr'].apply(assign_tier)

        print("💎 Customer CLV Tier Breakdown:")
        print(clv_df['CLV_Tier'].value_counts())
        print("================ CLV CALCULATION COMPLETE ================\n")

        return clv_df

if __name__ == "__main__":
    calculator = CLVCalculator()
    clv_res = calculator.calculate_clv()
    print(clv_res[['Customer_ID', 'Total_Spending', 'Predicted_CLV_3Yr', 'CLV_Tier']].head())
