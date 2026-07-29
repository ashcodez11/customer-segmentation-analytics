import pandas as pd

class AutomatedInsightGenerator:
    @staticmethod
    def generate_insights_from_df(df):
        if df.empty:
            return ["No data available for selected filters."]

        total_rev = df['Total_Spending'].sum()
        total_cust = len(df)
        
        # 1. Pareto Top 10% Contribution
        top10_cust_count = max(1, int(total_cust * 0.10))
        top10_rev = df.nlargest(top10_cust_count, 'Total_Spending')['Total_Spending'].sum()
        pareto_pct = (top10_rev / total_rev * 100) if total_rev > 0 else 0

        # 2. Income Spend Ratio
        high_inc_spend = df[df['Annual_Income'] >= df['Annual_Income'].median()]['Total_Spending'].mean()
        low_inc_spend = df[df['Annual_Income'] < df['Annual_Income'].median()]['Total_Spending'].mean()
        ratio = (high_inc_spend / low_inc_spend) if low_inc_spend > 0 else 1.0

        # 3. Top Category
        top_cat = df.groupby('Preferred_Category')['Total_Spending'].sum().idxmax()
        top_cat_rev = df.groupby('Preferred_Category')['Total_Spending'].sum().max()
        top_cat_pct = (top_cat_rev / total_rev * 100) if total_rev > 0 else 0

        # 4. Highest Value Persona
        top_persona = df.groupby('Customer_Persona')['Total_Spending'].mean().idxmax()

        return [
            f"Top 10% of customers generate {pareto_pct:.1f}% of total filtered platform revenue (${top10_rev:,.0f}).",
            f"High-income customers spend {ratio:.1f}× more per order than low-income segments.",
            f"The '{top_cat}' category commands {top_cat_pct:.1f}% of gross customer revenue.",
            f"The '{top_persona}' persona yields the highest average spending per customer.",
            f"Average order value across filtered users stands at ${df['Average_Order_Value'].mean():,.2f}."
        ]
