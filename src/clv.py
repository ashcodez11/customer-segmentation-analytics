import pandas as pd
import numpy as np

class CLVCalculator:
    @staticmethod
    def calculate_3yr_clv(df):
        gross_margin = 0.65
        retention_rate = 0.82
        
        df['Predicted_CLV_3Yr'] = (
            df['Monthly_Spend_Velocity'] * 36 * gross_margin * retention_rate
        ).round(2)

        quantiles = df['Predicted_CLV_3Yr'].quantile([0.6, 0.8, 0.9]).values
        def clv_tier(val):
            if val >= quantiles[2]:
                return 'Platinum Tier 💎'
            elif val >= quantiles[1]:
                return 'Gold Tier 🥇'
            elif val >= quantiles[0]:
                return 'Silver Tier 🥈'
            else:
                return 'Bronze Tier 🥉'

        df['CLV_Tier'] = df['Predicted_CLV_3Yr'].apply(clv_tier)
        return df
