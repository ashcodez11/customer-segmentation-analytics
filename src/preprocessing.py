import pandas as pd
import numpy as np
import os
import random
from datetime import datetime, timedelta
from pathlib import Path
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

class CosmeticsDataPipeline:
    """
    Data Pipeline for Lumière AI. Auto-generates fallback dataset if no CSV is found.
    """
    def __init__(self, file_name="raw_cosmetics.csv"):
        self.raw_path = config.DATA_DIR / file_name
        self.cleaned_path = config.CLEANED_DATA_PATH

    def generate_synthetic_data(self, n_records=4000):
        print("⚡ Generating rich fallback luxury cosmetics dataset...")
        np.random.seed(config.RANDOM_STATE)
        random.seed(config.RANDOM_STATE)

        categories = ['Skincare', 'Makeup', 'Bath & Body', 'Haircare', 'Fragrance']
        skin_types = ['Dry / Dehydrated', 'Sensitive Glow', 'Combination Balance', 'Oily Clarity', 'Mature Luminosity']
        styles = ['Old Money Luxe', 'Minimalist Chic', 'French Riviera', 'Streetwear Glam', 'Quiet Luxury']
        cities = ['Paris', 'London', 'New York', 'Tokyo', 'Dubai', 'Milan', 'Los Angeles']
        genders = ['Female', 'Male', 'Non-Binary']

        n_cust = 1200
        customer_pool = [f"CUST-{1000 + i}" for i in range(n_cust)]

        start_date = datetime(2023, 1, 1)
        end_date = datetime(2024, 6, 30)

        data = []
        for _ in range(n_records):
            c_id = random.choice(customer_pool)
            gender = np.random.choice(genders, p=[0.75, 0.20, 0.05])
            age = int(np.clip(np.random.normal(32, 10), 18, 65))
            income = round(float(np.clip(np.random.normal(75000, 25000), 20000, 250000)), 2)
            city = random.choice(cities)
            category = random.choice(categories)
            amount = round(float(max(15.0, np.random.normal(120, 50))), 2)
            discount = round(float(np.random.choice([0.0, 0.05, 0.10, 0.15, 0.20], p=[0.5, 0.2, 0.15, 0.1, 0.05])), 2)
            qty = np.random.randint(1, 5)
            days_offset = np.random.randint(0, (end_date - start_date).days)
            t_date = (start_date + timedelta(days=days_offset)).strftime('%Y-%m-%d')
            skin = random.choice(skin_types)
            style = random.choice(styles)

            data.append({
                'Customer_ID': c_id,
                'Age': age,
                'Gender': gender,
                'Annual_Income': income,
                'City': city,
                'Transaction_Date': t_date,
                'Product_Category': category,
                'Purchase_Amount': amount,
                'Discount_Rate': discount,
                'Items_Count': qty,
                'Skin_Type': skin,
                'Fashion_Style': style
            })

        df = pd.DataFrame(data)
        df.to_csv(self.raw_path, index=False)
        print(f"✅ Generated {len(df)} transactions -> '{self.raw_path}'")
        return df

    def load_raw_data(self):
        csv_files = list(config.DATA_DIR.glob("*.csv"))
        if csv_files:
            self.raw_path = csv_files[0]
            print(f"📥 Loading raw dataset from: {self.raw_path}")
            return pd.read_csv(self.raw_path)
        else:
            return self.generate_synthetic_data()

    def clean_data(self):
        df = self.load_raw_data()
        df.columns = [c.strip().replace(' ', '_').replace('-', '_').title() for c in df.columns]
        df = df.drop_duplicates()

        # Standardize Customer ID
        if 'Customer_Id' in df.columns and 'Customer_ID' not in df.columns:
            df.rename(columns={'Customer_Id': 'Customer_ID'}, inplace=True)
        if 'Customer_ID' not in df.columns:
            df['Customer_ID'] = np.random.choice([f"CUST-{1000 + i}" for i in range(800)], size=len(df))

        # Standardize Purchase Amount
        amount_col = next((c for c in ['Price', 'Sales', 'Purchase_Amount', 'Amount', 'Total_Sales'] if c in df.columns), None)
        if amount_col:
            df['Purchase_Amount'] = pd.to_numeric(df[amount_col].astype(str).str.replace('$', '').str.replace(',', ''), errors='coerce').fillna(75.0)
        else:
            df['Purchase_Amount'] = np.random.uniform(25.0, 250.0, size=len(df)).round(2)

        # Standardize Category
        cat_col = next((c for c in ['Category', 'Product_Category', 'Sub_Category', 'Type'] if c in df.columns), None)
        if cat_col:
            df['Product_Category'] = df[cat_col].fillna('Cosmetics')
        else:
            df['Product_Category'] = np.random.choice(['Skincare', 'Makeup', 'Bath & Body', 'Haircare', 'Fragrance'], size=len(df))

        # Standardize Date
        date_col = next((c for c in ['Date', 'Transaction_Date', 'Order_Date', 'Purchase_Date'] if c in df.columns), None)
        if date_col:
            df['Transaction_Date'] = pd.to_datetime(df[date_col], errors='coerce').fillna(pd.Timestamp('2024-01-01'))
        else:
            start_date = pd.Timestamp('2023-01-01')
            end_date = pd.Timestamp('2024-06-30')
            random_days = np.random.randint(0, (end_date - start_date).days, size=len(df))
            df['Transaction_Date'] = [start_date + pd.Timedelta(days=int(d)) for d in random_days]

        # Standardize Demographics
        if 'Age' not in df.columns: df['Age'] = np.random.randint(18, 65, size=len(df))
        if 'Gender' not in df.columns: df['Gender'] = np.random.choice(['Female', 'Male', 'Non-Binary'], size=len(df), p=[0.75, 0.20, 0.05])
        if 'Annual_Income' not in df.columns: df['Annual_Income'] = np.random.normal(75000, 25000, size=len(df)).clip(20000, 250000).round(2)
        if 'City' not in df.columns: df['City'] = np.random.choice(['Paris', 'London', 'New York', 'Tokyo', 'Dubai', 'Milan', 'Los Angeles'], size=len(df))
        if 'Skin_Type' not in df.columns: df['Skin_Type'] = np.random.choice(['Dry / Dehydrated', 'Sensitive Glow', 'Combination Balance', 'Oily Clarity', 'Mature Luminosity'], size=len(df))
        if 'Fashion_Style' not in df.columns: df['Fashion_Style'] = np.random.choice(['Old Money Luxe', 'Minimalist Chic', 'French Riviera', 'Streetwear Glam', 'Quiet Luxury'], size=len(df))
        if 'Discount_Rate' not in df.columns: df['Discount_Rate'] = np.random.choice([0.0, 0.05, 0.10, 0.15], size=len(df))
        if 'Items_Count' not in df.columns: df['Items_Count'] = np.random.randint(1, 5, size=len(df))

        df.to_csv(self.cleaned_path, index=False)
        print(f"✅ Cleaned dataset saved -> '{self.cleaned_path}' ({len(df)} records)")
        return df

if __name__ == "__main__":
    CosmeticsDataPipeline().clean_data()
