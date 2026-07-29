import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class CosmeticsVisualizer:
    # Editorial Soft Luxury Color Palette
    LUXE_PALETTE = ['#C9A86A', '#D9A5A5', '#8C707A', '#2E2A28', '#E8CFCF', '#B8A9C9']
    HEATMAP_GRADIENT = [[0.0, '#FAF8F5'], [0.5, '#D9A5A5'], [1.0, '#2E2A28']]

    @staticmethod
    def _apply_luxe_layout(fig, title=""):
        fig.update_layout(
            title={
                'text': f"<b>{title.upper()}</b>",
                'font': {'family': 'Cormorant Garamond, serif', 'size': 16, 'color': '#2E2A28'},
                'x': 0.0,
                'xanchor': 'left'
            },
            font={'family': 'Plus Jakarta Sans, sans-serif', 'size': 11, 'color': '#2E2A28'},
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='#FFFFFF',
            margin=dict(l=20, r=20, t=50, b=20),
            xaxis=dict(showgrid=True, gridcolor='#F7F1EC', zeroline=False),
            yaxis=dict(showgrid=True, gridcolor='#F7F1EC', zeroline=False),
            legend=dict(
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='#F7F1EC',
                borderwidth=1,
                font=dict(size=10)
            )
        )
        return fig

    @staticmethod
    def plot_pca_2d(df):
        features = ['Age', 'Annual_Income', 'Total_Spending', 'Average_Order_Value', 'Purchase_Frequency', 'Days_Since_Last_Purchase']
        X = df[features]
        X_scaled = StandardScaler().fit_transform(X)
        pca = PCA(n_components=2)
        pcs = pca.fit_transform(X_scaled)
        
        pca_df = df.copy()
        pca_df['PCA1'] = pcs[:, 0]
        pca_df['PCA2'] = pcs[:, 1]

        fig = px.scatter(
            pca_df, x='PCA1', y='PCA2', color='Customer_Persona',
            hover_data=['Customer_ID', 'Total_Spending', 'Annual_Income'],
            color_discrete_sequence=CosmeticsVisualizer.LUXE_PALETTE,
            opacity=0.88
        )
        fig.update_traces(marker=dict(size=9, line=dict(width=1, color='#FFFFFF')))
        return CosmeticsVisualizer._apply_luxe_layout(
            fig, 
            f"2D PCA Cluster Map ({pca.explained_variance_ratio_.sum()*100:.1f}% Variance Retained)"
        )

    @staticmethod
    def plot_age_dist(df):
        fig = px.histogram(
            df, x='Age', nbins=18, 
            color_discrete_sequence=['#D9A5A5'],
            opacity=0.85
        )
        fig.update_traces(marker=dict(line=dict(width=1, color='#FFFFFF')))
        return CosmeticsVisualizer._apply_luxe_layout(fig, "Customer Age Demographics")

    @staticmethod
    def plot_income_dist(df):
        fig = px.box(
            df, y='Annual_Income', x='Gender', 
            color='Gender',
            color_discrete_sequence=CosmeticsVisualizer.LUXE_PALETTE
        )
        return CosmeticsVisualizer._apply_luxe_layout(fig, "Income Distribution by Gender ($)")

    @staticmethod
    def plot_spending_dist(df):
        fig = px.violin(
            df, y='Total_Spending', x='Preferred_Category', 
            color='Preferred_Category', box=True, points=False,
            color_discrete_sequence=CosmeticsVisualizer.LUXE_PALETTE
        )
        return CosmeticsVisualizer._apply_luxe_layout(fig, "Spending Velocity by Product Category")

    @staticmethod
    def plot_gender_dist(df):
        fig = px.pie(
            df, names='Gender', 
            color_discrete_sequence=['#D9A5A5', '#C9A86A', '#2E2A28'], 
            hole=0.55
        )
        fig.update_traces(textinfo='percent+label', marker=dict(line=dict(color='#FFFFFF', width=2)))
        return CosmeticsVisualizer._apply_luxe_layout(fig, "Gender Share Distribution")

    @staticmethod
    def plot_freq_dist(df):
        fig = px.histogram(
            df, x='Purchase_Frequency', nbins=12,
            color_discrete_sequence=['#C9A86A'],
            opacity=0.85
        )
        fig.update_traces(marker=dict(line=dict(width=1, color='#FFFFFF')))
        return CosmeticsVisualizer._apply_luxe_layout(fig, "Annual Order Frequency")

    @staticmethod
    def plot_revenue_dist(df):
        cat_rev = df.groupby('Preferred_Category')['Total_Spending'].sum().reset_index()
        fig = px.bar(
            cat_rev, x='Preferred_Category', y='Total_Spending', 
            color='Preferred_Category',
            color_discrete_sequence=CosmeticsVisualizer.LUXE_PALETTE,
            text_auto='.2s'
        )
        fig.update_traces(marker=dict(line=dict(width=1, color='#FFFFFF')))
        return CosmeticsVisualizer._apply_luxe_layout(fig, "Gross Revenue Contribution ($)")

    @staticmethod
    def plot_correlation_heatmap(df):
        num_cols = ['Age', 'Annual_Income', 'Total_Spending', 'Average_Order_Value', 'Purchase_Frequency', 'Days_Since_Last_Purchase']
        corr = df[num_cols].corr()
        
        # FIXED: Continuous gradient using valid luxury color scale
        fig = px.imshow(
            corr, 
            text_auto=".2f", 
            color_continuous_scale=CosmeticsVisualizer.HEATMAP_GRADIENT,
            aspect="auto"
        )
        fig.update_coloraxes(showscale=False)
        return CosmeticsVisualizer._apply_luxe_layout(fig, "Feature Correlation Matrix")

    @staticmethod
    def plot_pair_sample(df):
        fig = px.scatter(
            df, x='Annual_Income', y='Total_Spending', 
            color='Customer_Persona', 
            color_discrete_sequence=CosmeticsVisualizer.LUXE_PALETTE,
            opacity=0.85
        )
        fig.update_traces(marker=dict(size=8, line=dict(width=0.8, color='#FFFFFF')))
        return CosmeticsVisualizer._apply_luxe_layout(fig, "Income vs Spending Segmentation")

    @staticmethod
    def plot_category_spend(df):
        fig = px.treemap(
            df, path=['City', 'Preferred_Category'], values='Total_Spending', 
            color_discrete_sequence=CosmeticsVisualizer.LUXE_PALETTE
        )
        return CosmeticsVisualizer._apply_luxe_layout(fig, "Geographic Category Spend Treemap")
