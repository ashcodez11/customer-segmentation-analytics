import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import config

class CosmeticsDataPipeline:
    def __init__(self, filepath=config.RAW_DATA_PATH):
        self.filepath = filepath

    def load_or_generate_data(self):
        if not self.filepath.exists():
            np.random.seed(config.RANDOM_STATE)
            n_samples = 1200
            
            customer_ids = [f"LUM-{1000 + i}" for i in range(n_samples)]
            ages = np.random.normal(36, 11, n_samples).clip(18, 70).astype(int)
            genders = np.random.choice(['Female', 'Male', 'Non-Binary'], size=n_samples, p=[0.72, 0.23, 0.05])
            cities = np.random.choice(['Paris', 'London', 'New York', 'Tokyo', 'Dubai', 'Milan', 'Los Angeles'], size=n_samples)
            incomes = np.random.normal(85000, 35000, n_samples).clip(22000, 280000).round(-2)
            categories = np.random.choice(['Skincare', 'Makeup', 'Bath & Body', 'Haircare', 'Fragrance'], size=n_samples)
            
            # Correlated spend & purchase behavior
            base_spending = (incomes * 0.035) + np.random.normal(500, 300, n_samples)
            spendings = base_spending.clip(100, 15000).round(2)
            frequencies = (spendings / np.random.uniform(80, 250, n_samples)).clip(1, 48).round(0).astype(int)
            aov = (spendings / frequencies).round(2)
            recencies = np.random.exponential(35, n_samples).clip(1, 180).astype(int)
            discounts = np.random.beta(2, 5, n_samples).round(2)
            lifetime = np.random.randint(3, 60, size=n_samples)

            df = pd.DataFrame({
                'Customer_ID': customer_ids,
                'Age': ages,
                'Gender': genders,
                'City': cities,
                'Annual_Income': incomes,
                'Total_Spending': spendings,
                'Average_Order_Value': aov,
                'Purchase_Frequency': frequencies,
                'Preferred_Category': categories,
                'Discount_Usage': discounts,
                'Days_Since_Last_Purchase': recencies,
                'Customer_Lifetime_Months': lifetime
            })
            df.to_csv(self.filepath, index=False)
            return df
        return pd.read_csv(self.filepath)

    def clean_data(self):
        df = self.load_or_generate_data()
        df = df.drop_duplicates(subset=['Customer_ID'])
        
        # Median Imputation for numeric, mode for categorical
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
        
        cat_cols = df.select_dtypes(include=['object']).columns
        for c in cat_cols:
            df[c] = df[c].fillna(df[c].mode()[0])

        # Clip numerical outliers using IQR
        for col in ['Total_Spending', 'Annual_Income', 'Average_Order_Value']:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            upper = q3 + 1.5 * iqr
            lower = max(0, q1 - 1.5 * iqr)
            df[col] = df[col].clip(lower, upper)

        df.to_csv(config.CLEAN_DATA_PATH, index=False)
        return df
