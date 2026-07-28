import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

class AutomatedInsightGenerator:
    """
    Analyzes segmented customer data and generates natural-language business insights.
    """
    def __init__(self, data_path=config.FINAL_SEGMENTED_PATH):
        self.data_path = Path(data_path)

    def generate_all_insights(self):
        if not self.data_path.exists():
            from src.clustering import CosmeticsClusteringEngine
            df, _ = CosmeticsClusteringEngine().run_all_clustering_algorithms()
        else:
            df = pd.read_csv(self.data_path)

        insights = []

        total_customers = len(df)
        total_revenue = df['Total_Spending'].sum()
        avg_spend = df['Total_Spending'].mean()

        # 1. Pareto Concentration Insight
        top_10_percent_count = int(total_customers * 0.10)
        top_10_revenue = df.nlargest(top_10_percent_count, 'Total_Spending')['Total_Spending'].sum()
        pareto_pct = (top_10_revenue / total_revenue) * 100
        insights.append(
            f"💡 **Pareto Revenue Concentration**: The top 10% of customers ({top_10_percent_count:,} users) generate "
            f"**{pareto_pct:.1f}%** of total historical revenue (${top_10_revenue:,.2f})."
        )

        # 2. Persona Insights
        if 'Customer_Persona' in df.columns:
            vip_persona = df[df['Customer_Persona'].str.contains('VIP', na=False)]
            if not vip_persona.empty:
                vip_count = len(vip_persona)
                vip_rev_share = (vip_persona['Total_Spending'].sum() / total_revenue) * 100
                vip_avg_spend = vip_persona['Total_Spending'].mean()
                insights.append(
                    f"✨ **VIP Persona Dominance**: **{vip_count} VIP Enthusiasts** represent "
                    f"**{(vip_count/total_customers)*100:.1f}%** of the user base but produce "
                    f"**{vip_rev_share:.1f}%** of revenue with an average spend of **${vip_avg_spend:,.2f}** per customer."
                )

        # 3. Category Preference Insight
        top_cat = df['Preferred_Category'].mode()[0]
        cat_share = (df['Preferred_Category'] == top_cat).mean() * 100
        insights.append(
            f"💄 **Top Product Category**: **{top_cat}** is the primary driver for **{cat_share:.1f}%** of repeat buyers."
        )

        # 4. Age Bracket & Income Correlation Insight
        df['Age_Group'] = pd.cut(df['Age'], bins=[17, 25, 35, 50, 100], labels=['18-25', '26-35', '36-50', '50+'])
        highest_spending_age = df.groupby('Age_Group', observed=False)['Total_Spending'].mean().idxmax()
        highest_age_val = df.groupby('Age_Group', observed=False)['Total_Spending'].mean().max()
        insights.append(
            f"🎯 **High-Value Demographics**: Customers in the **{highest_spending_age} age bracket** display the highest average customer spend at **${highest_age_val:,.2f}**."
        )

        # 5. Churn Risk Warning
        if 'RFM_Segment' in df.columns:
            at_risk = df[df['RFM_Segment'].isin(['At Risk', 'Lost Customers', "Can't Lose Them"])]
            at_risk_count = len(at_risk)
            at_risk_revenue = at_risk['Total_Spending'].sum()
            insights.append(
                f"⚠️ **Churn Exposure**: **{at_risk_count} customers** ({(at_risk_count/total_customers)*100:.1f}%) are currently inactive or at-risk, representing **${at_risk_revenue:,.2f}** in potential lost annual value."
            )

        # 6. CLV Tier Projection
        if 'CLV_Tier' in df.columns:
            plat_tier = df[df['CLV_Tier'].str.contains('Platinum', na=False)]
            if not plat_tier.empty:
                plat_3yr = plat_tier['Predicted_CLV_3Yr'].sum()
                insights.append(
                    f"💎 **3-Year CLV Forecast**: Platinum Tier customers are projected to generate **${plat_3yr:,.2f}** in high-margin revenue over the next 36 months."
                )

        return insights

if __name__ == "__main__":
    generator = AutomatedInsightGenerator()
    insights = generator.generate_all_insights()
    print("\n--- AUTOMATED BUSINESS INSIGHTS ---")
    for i, ins in enumerate(insights, 1):
        print(f"\n{i}. {ins}")
