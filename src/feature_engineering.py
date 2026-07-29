import pandas as pd
import numpy as np
import config

class CosmeticsFeatureEngineer:
    def __init__(self, filepath=config.CLEAN_DATA_PATH):
        self.filepath = filepath

    def extract_customer_features(self):
        df = pd.read_csv(self.filepath)

        # Derived real features
        df['Spending_To_Income_Ratio'] = (df['Total_Spending'] / df['Annual_Income']).round(4)
        df['Monthly_Spend_Velocity'] = (df['Total_Spending'] / df['Customer_Lifetime_Months'].clip(1)).round(2)
        df['Discount_Sensitivity'] = pd.qcut(df['Discount_Usage'], q=3, labels=['Low', 'Moderate', 'High'])
        df['Recency_Score'] = pd.qcut(df['Days_Since_Last_Purchase'], q=5, labels=[5, 4, 3, 2, 1]).astype(int)
        df['Frequency_Score'] = pd.qcut(df['Purchase_Frequency'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
        df['Monetary_Score'] = pd.qcut(df['Total_Spending'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5]).astype(int)

        return df
