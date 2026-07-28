import plotly.express as px
import plotly.graph_objects as gg
import pandas as pd
import numpy as np

class CosmeticsVisualizer:
    """
    Renders soft luxury Plotly visualizations matching the Lumière AI Figma specification.
    """
    
    PALETTE = ['#D9A5A5', '#C9A86A', '#D8D2F0', '#C9D8C5', '#E8CFCF', '#B8929A']
    BG_TRANSPARENT = "rgba(0,0,0,0)"
    TEXT_COLOR = "#2E2A28"
    GRID_COLOR = "rgba(201, 168, 106, 0.15)"
    FONT_FAMILY = "Plus Jakarta Sans, sans-serif"

    @staticmethod
    def _apply_luxury_layout(fig, title=""):
        fig.update_layout(
            paper_bgcolor=CosmeticsVisualizer.BG_TRANSPARENT,
            plot_bgcolor=CosmeticsVisualizer.BG_TRANSPARENT,
            font=dict(family=CosmeticsVisualizer.FONT_FAMILY, color=CosmeticsVisualizer.TEXT_COLOR, size=11),
            title=dict(
                text=f"<b>{title}</b>",
                font=dict(size=14, color="#2E2A28", family="Cormorant Garamond")
            ),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        return fig

    # --- EDA CHARTS ---
    @staticmethod
    def plot_age_dist(df):
        fig = px.histogram(df, x='Age', nbins=20, color_discrete_sequence=['#D9A5A5'], title="Age Distribution")
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=True, gridcolor=CosmeticsVisualizer.GRID_COLOR)
        return CosmeticsVisualizer._apply_luxury_layout(fig, "Age Distribution")

    @staticmethod
    def plot_income_dist(df):
        fig = px.histogram(df, x='Annual_Income', nbins=20, color_discrete_sequence=['#C9A86A'], title="Annual Income Distribution")
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=True, gridcolor=CosmeticsVisualizer.GRID_COLOR)
        return CosmeticsVisualizer._apply_luxury_layout(fig, "Annual Income Distribution")

    @staticmethod
    def plot_spending_dist(df):
        fig = px.violin(df, y='Total_Spending', box=True, color_discrete_sequence=['#D8D2F0'], title="Spending Score Distribution")
        fig.update_yaxes(showgrid=True, gridcolor=CosmeticsVisualizer.GRID_COLOR)
        return CosmeticsVisualizer._apply_luxury_layout(fig, "Spending Score Distribution")

    @staticmethod
    def plot_gender_dist(df):
        gender_counts = df['Gender'].value_counts().reset_index()
        gender_counts.columns = ['Gender', 'Count']
        fig = px.pie(gender_counts, names='Gender', values='Count', hole=0.6, color_discrete_sequence=['#D8D2F0', '#C9A86A', '#D9A5A5'])
        return CosmeticsVisualizer._apply_luxury_layout(fig, "Gender Distribution")

    @staticmethod
    def plot_freq_dist(df):
        fig = px.histogram(df, x='Purchase_Frequency', color_discrete_sequence=['#C9D8C5'], title="Purchase Frequency")
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=True, gridcolor=CosmeticsVisualizer.GRID_COLOR)
        return CosmeticsVisualizer._apply_luxury_layout(fig, "Purchase Frequency")

    @staticmethod
    def plot_revenue_dist(df):
        fig = px.area(df.sort_values('Total_Spending'), y='Total_Spending', color_discrete_sequence=['#E8CFCF'], title="Revenue Distribution")
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=True, gridcolor=CosmeticsVisualizer.GRID_COLOR)
        return CosmeticsVisualizer._apply_luxury_layout(fig, "Revenue Distribution Flow")

    @staticmethod
    def plot_correlation_heatmap(df):
        num_cols = ['Age', 'Annual_Income', 'Purchase_Frequency', 'Total_Spending', 'Average_Order_Value']
        corr = df[num_cols].corr()
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale=['#FAF8F5', '#E8CFCF', '#D9A5A5', '#C9A86A'], title="Correlation Heatmap")
        return CosmeticsVisualizer._apply_luxury_layout(fig, "Correlation Matrix")

    # --- CLUSTERING & RFM CHARTS ---
    @staticmethod
    def plot_pca_2d(df):
        fig = px.scatter(
            df, x='PCA1', y='PCA2', color='Customer_Persona',
            hover_data=['Customer_ID', 'Total_Spending'],
            color_discrete_sequence=CosmeticsVisualizer.PALETTE, template='plotly_white'
        )
        fig.update_traces(marker=dict(size=9, opacity=0.85, line=dict(width=1, color='#FFFFFF')))
        fig.update_xaxes(showgrid=True, gridcolor=CosmeticsVisualizer.GRID_COLOR)
        fig.update_yaxes(showgrid=True, gridcolor=CosmeticsVisualizer.GRID_COLOR)
        return CosmeticsVisualizer._apply_luxury_layout(fig, "PCA VISUALIZATION (2D)")

    @staticmethod
    def plot_pca_3d(df):
        fig = px.scatter_3d(
            df, x='PCA3_1', y='PCA3_2', z='PCA3_3', color='Customer_Persona',
            color_discrete_sequence=CosmeticsVisualizer.PALETTE, opacity=0.85
        )
        fig.update_layout(paper_bgcolor=CosmeticsVisualizer.BG_TRANSPARENT, height=480)
        return fig

    @staticmethod
    def plot_rfm_treemap(df):
        rfm_counts = df['RFM_Segment'].value_counts().reset_index()
        rfm_counts.columns = ['RFM_Segment', 'Count']
        fig = px.treemap(
            rfm_counts, path=['RFM_Segment'], values='Count', color='Count',
            color_continuous_scale=['#FAF8F5', '#E8CFCF', '#D9A5A5', '#C9A86A']
        )
        return CosmeticsVisualizer._apply_luxury_layout(fig, "RFM Behavioral Client Portfolio")

    @staticmethod
    def plot_clv_distribution(df):
        fig = px.histogram(
            df, x='Predicted_CLV_3Yr', color='CLV_Tier', nbins=30,
            color_discrete_sequence=['#C9A86A', '#D9A5A5', '#D8D2F0', '#C9D8C5'], template='plotly_white'
        )
        return CosmeticsVisualizer._apply_luxury_layout(fig, "Predicted 3-Year Customer Lifetime Value")

    @staticmethod
    def plot_category_spend(df):
        cat_df = df.groupby('Preferred_Category')['Total_Spending'].sum().reset_index()
        fig = px.pie(cat_df, names='Preferred_Category', values='Total_Spending', hole=0.5, color_discrete_sequence=CosmeticsVisualizer.PALETTE)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        return CosmeticsVisualizer._apply_luxury_layout(fig, "Revenue Contribution by Category")
