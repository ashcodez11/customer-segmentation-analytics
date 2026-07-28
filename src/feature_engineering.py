import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

class CosmeticsFeatureEngineer:
    """
    Transforms transaction-level cosmetics data into customer-aggregated profile metrics.
    """
    def __init__(self, cleaned_path=config.CLEANED_DATA_PATH):
        self.cleaned_path = Path(cleaned_path)
        self.featured_path = config.FEATURE_DATA_PATH

    def extract_customer_features(self):
        print("\n================ STARTING FEATURE ENGINEERING ================")
        if not self.cleaned_path.exists():
            raise FileNotFoundError(f"Cleaned dataset not found at '{self.cleaned_path}'")

        df = pd.read_csv(self.cleaned_path)
        df['Transaction_Date'] = pd.to_datetime(df['Transaction_Date'])

        # Reference snapshot date for recency calculation (1 day after latest transaction)
        snapshot_date = df['Transaction_Date'].max() + pd.Timedelta(days=1)
        print(f"📅 Dataset Snapshot Reference Date: {snapshot_date.strftime('%Y-%m-%d')}")

        # Helper function for mode
        def get_mode(x):
            m = x.mode()
            return m.iloc[0] if not m.empty else "Unknown"

        # Customer Aggregations
        print("⚙️ Aggregating transaction metrics per Customer_ID...")
        customer_df = df.groupby('Customer_ID').agg(
            Age=('Age', 'first'),
            Gender=('Gender', 'first'),
            City=('City', 'first'),
            Annual_Income=('Annual_Income', 'first'),
            
            # Recency & Frequency
            First_Purchase_Date=('Transaction_Date', 'min'),
            Last_Purchase_Date=('Transaction_Date', 'max'),
            Purchase_Frequency=('Transaction_Date', 'count'),
            
            # Monetary Features
            Total_Spending=('Purchase_Amount', 'sum'),
            Average_Order_Value=('Purchase_Amount', 'mean'),
            Max_Single_Spend=('Purchase_Amount', 'max'),
            
            # Behavioral Features
            Total_Items_Count=('Items_Count', 'sum'),
            Discount_Usage_Rate=('Discount_Rate', 'mean'),
            Preferred_Category=('Product_Category', get_mode)
        ).reset_index()

        # Calculated Temporal Features
        customer_df['Days_Since_Last_Purchase'] = (snapshot_date - customer_df['Last_Purchase_Date']).dt.days
        customer_df['Customer_Lifetime_Days'] = (customer_df['Last_Purchase_Date'] - customer_df['First_Purchase_Date']).dt.days
        customer_df['Customer_Lifetime_Days'] = customer_df['Customer_Lifetime_Days'].apply(lambda x: max(x, 1))

        # Monthly Purchase Velocity
        customer_df['Monthly_Purchase_Rate'] = (customer_df['Purchase_Frequency'] / (customer_df['Customer_Lifetime_Days'] / 30.0)).round(2)
        
        # Round continuous features
        customer_df['Total_Spending'] = customer_df['Total_Spending'].round(2)
        customer_df['Average_Order_Value'] = customer_df['Average_Order_Value'].round(2)
        customer_df['Discount_Usage_Rate'] = customer_df['Discount_Usage_Rate'].round(4)
        customer_df['Annual_Income'] = customer_df['Annual_Income'].round(2)

        # Drop temporary date columns
        customer_df.drop(columns=['First_Purchase_Date', 'Last_Purchase_Date'], inplace=True)

        # Save featured dataset
        customer_df.to_csv(self.featured_path, index=False)
        print(f"✅ Extracted features for {len(customer_df)} unique customers.")
        print(f"💾 Saved featured customer dataset to '{self.featured_path}'")
        print("================ FEATURE ENGINEERING COMPLETE ================\n")

        return customer_df

if __name__ == "__main__":
    engineer = CosmeticsFeatureEngineer()
    df_featured = engineer.extract_customer_features()
    print(df_featured.head())
