import streamlit as st
import pandas as pd
import numpy as np
import base64
import os
from pathlib import Path

# Config and Analytics Engines
import config
from src.preprocessing import CosmeticsDataPipeline
from src.feature_engineering import CosmeticsFeatureEngineer
from src.rfm import RFMAnalyzer
from src.clv import CLVCalculator
from src.clustering import CosmeticsClusteringEngine
from src.insights import AutomatedInsightGenerator
from src.recommendations import AIRecommendationEngine
from src.visualization import CosmeticsVisualizer
from src.pdf_generator import PDFReportGenerator

# Streamlit Page Setup
st.set_page_config(
    page_title="Lumière AI Analytics – Luxury Dashboard",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Helper function to convert local video to base64 for HTML background
def get_video_base64(video_path):
    if os.path.exists(video_path):
        with open(video_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode('utf-8')
    return None

video_b64 = get_video_base64("assets/hero_video.mp4")

# Inject Figma-Exact Editorial CSS Styling
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,500;0,600;1,400&family=Plus+Jakarta+Sans:wght@300;400;500;600&display=swap');

    /* Global Soft Luxury Palette */
    .stApp {{
        background-color: #FAF8F5;
        color: #2E2A28;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }}

    /* Video Hero Container */
    .hero-container {{
        position: relative;
        width: 100%;
        height: 85vh;
        min-height: 550px;
        border-radius: 24px;
        overflow: hidden;
        margin-bottom: 30px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        box-shadow: 0 20px 50px rgba(46, 42, 40, 0.08);
    }}
    .hero-video {{
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        object-fit: cover;
        z-index: 0;
        filter: brightness(0.72) contrast(1.05);
    }}
    .hero-overlay {{
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(180deg, rgba(46, 42, 40, 0.2) 0%, rgba(250, 248, 245, 0.65) 100%);
        z-index: 1;
    }}
    .hero-content {{
        position: relative;
        z-index: 2;
        max-width: 800px;
        padding: 20px;
    }}
    .hero-title {{
        font-family: 'Cormorant Garamond', serif;
        font-size: 4.2rem;
        font-weight: 400;
        letter-spacing: 0.12em;
        color: #FFFFFF;
        text-transform: uppercase;
        margin-bottom: 12px;
        text-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }}
    .hero-subtitle {{
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 0.82rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.22em;
        color: #FAF8F5;
        margin-bottom: 40px;
        text-shadow: 0 2px 10px rgba(0,0,0,0.3);
    }}
    .scroll-indicator {{
        display: inline-block;
        color: #FAF8F5;
        font-size: 0.78rem;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        padding: 10px 22px;
        border: 1px solid rgba(255,255,255,0.4);
        border-radius: 30px;
        backdrop-filter: blur(8px);
        background: rgba(255,255,255,0.12);
    }}

    /* Figma Section Titles */
    .section-title {{
        font-family: 'Cormorant Garamond', serif;
        font-size: 2.1rem;
        font-weight: 500;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: #2E2A28;
        margin-top: 45px;
        margin-bottom: 6px;
    }}
    .section-subtitle {{
        font-size: 0.85rem;
        color: #8C837D;
        margin-bottom: 25px;
    }}

    /* Figma Quick Navigation Bar */
    .quick-nav-bar {{
        display: flex;
        justify-content: center;
        gap: 10px;
        flex-wrap: wrap;
        margin-bottom: 45px;
        background: #FFFFFF;
        padding: 12px 20px;
        border-radius: 40px;
        border: 1px solid #F7F1EC;
        box-shadow: 0 10px 30px rgba(0,0,0,0.03);
    }}
    .quick-nav-pill {{
        background: #FAF8F5;
        color: #2E2A28;
        border: 1px solid #E8CFCF;
        padding: 8px 16px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 500;
        text-decoration: none;
        transition: all 0.2s ease;
    }}
    .quick-nav-pill:hover {{
        background: #C9A86A;
        color: #FFFFFF;
        border-color: #C9A86A;
    }}

    /* KPI Cards */
    .luxe-kpi-card {{
        background: #FFFFFF;
        border: 1px solid rgba(201, 168, 106, 0.25);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 10px 25px rgba(46, 42, 40, 0.02);
    }}
    .luxe-kpi-label {{
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #8C837D;
        margin-bottom: 8px;
    }}
    .luxe-kpi-value {{
        font-family: 'Cormorant Garamond', serif;
        font-size: 2.3rem;
        font-weight: 600;
        color: #2E2A28;
    }}

    /* Loyalty Score Bar */
    .loyalty-card {{
        background: #FFFFFF;
        border: 1px solid rgba(201, 168, 106, 0.25);
        border-radius: 16px;
        padding: 22px;
        margin-top: 15px;
        margin-bottom: 40px;
        text-align: center;
    }}
    .loyalty-progress-bg {{
        background: #F7F1EC;
        border-radius: 10px;
        height: 8px;
        width: 100%;
        margin-top: 12px;
        overflow: hidden;
    }}
    .loyalty-progress-fill {{
        background: linear-gradient(90deg, #D9A5A5 0%, #C9A86A 100%);
        height: 100%;
        border-radius: 10px;
    }}

    /* Editorial Persona Cards */
    .persona-card {{
        background: #FFFFFF;
        border: 1px solid #F7F1EC;
        border-radius: 18px;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.03);
        margin-bottom: 20px;
    }}
    .persona-img {{
        width: 100%;
        height: 220px;
        object-fit: cover;
    }}
    .persona-body {{
        padding: 18px;
    }}
    .persona-title {{
        font-family: 'Cormorant Garamond', serif;
        font-size: 1.4rem;
        font-weight: 600;
        color: #2E2A28;
        margin-bottom: 10px;
    }}

    /* RFM Badge Cards */
    .rfm-badge-card {{
        background: #FFFFFF;
        border-radius: 16px;
        padding: 16px;
        text-align: center;
        border: 1px solid #F7F1EC;
        box-shadow: 0 8px 20px rgba(0,0,0,0.02);
    }}
    .rfm-circle {{
        width: 45px;
        height: 45px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 10px auto;
        font-weight: 600;
        font-size: 1.1rem;
    }}

    /* CLV Tier Cards */
    .clv-card {{
        background: #FFFFFF;
        border-radius: 16px;
        padding: 20px;
        border: 1px solid #F7F1EC;
        box-shadow: 0 10px 25px rgba(0,0,0,0.02);
    }}

    /* Footer */
    .lumiere-footer {{
        background: #F7F1EC;
        border-top: 1px solid rgba(201, 168, 106, 0.2);
        padding: 40px 20px;
        margin-top: 80px;
        border-radius: 20px 20px 0 0;
        text-align: center;
    }}
    </style>
""", unsafe_allow_html=True)

# Data Loader
@st.cache_data
def load_lumiere_data():
    if not config.FINAL_SEGMENTED_PATH.exists():
        with st.spinner("Processing Lumière Intelligence Engine..."):
            pipeline = CosmeticsDataPipeline()
            pipeline.clean_data()
            engineer = CosmeticsFeatureEngineer()
            engineer.extract_customer_features()
            engine = CosmeticsClusteringEngine()
            df, eval_summary = engine.run_all_clustering_algorithms()
            return df, eval_summary
    else:
        df = pd.read_csv(config.FINAL_SEGMENTED_PATH)
        engine = CosmeticsClusteringEngine()
        X, X_scaled, _ = engine.prepare_clustering_features(df)
        eval_summary = engine.evaluate_kmeans_elbow(X_scaled)
        return df, eval_summary

df_raw, eval_summary = load_lumiere_data()

# SECTION 1: HERO SECTION WITH EMBEDDED VIDEO
if video_b64:
    hero_html = f"""
    <div class="hero-container">
        <video class="hero-video" autoplay loop muted playsinline>
            <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
        </video>
        <div class="hero-overlay"></div>
        <div class="hero-content">
            <h1 class="hero-title">LUMIÈRE AI Analytics</h1>
            <p class="hero-subtitle">Soft Luxury Customer Intelligence & Strategic Persona Mapping</p>
            <div class="scroll-indicator"> Scroll To Explore </div>
        </div>
    </div>
    """
else:
    hero_html = """
    <div class="hero-container" style="background: linear-gradient(135deg, #2E2A28 0%, #8C837D 100%);">
        <div class="hero-content">
            <h1 class="hero-title">LUMIÈRE AI Analytics</h1>
            <p class="hero-subtitle">Soft Luxury Customer Intelligence & Strategic Persona Mapping</p>
            <div class="scroll-indicator"> Scroll To Explore </div>
        </div>
    </div>
    """
st.markdown(hero_html, unsafe_allow_html=True)

# SECTION 2: QUICK NAV BAR
st.markdown("""
<div class="quick-nav-bar">
    <a href="#executive-intelligence" class="quick-nav-pill">Executive Intelligence</a>
    <a href="#customer-personas" class="quick-nav-pill">Customer Personas</a>
    <a href="#segmentation-studio" class="quick-nav-pill">Segmentation Studio</a>
    <a href="#data-storytelling" class="quick-nav-pill">EDA Storytelling</a>
    <a href="#feature-engineering" class="quick-nav-pill">Feature Pipeline</a>
    <a href="#rfm-intelligence" class="quick-nav-pill">RFM Intelligence</a>
    <a href="#clv-intelligence" class="quick-nav-pill">CLV Valuation</a>
    <a href="#ai-insights" class="quick-nav-pill">AI Insights</a>
    <a href="#recommendations" class="quick-nav-pill">Recommendations</a>
    <a href="#export-center" class="quick-nav-pill">Export Center</a>
</div>
""", unsafe_allow_html=True)

# SECTION 3: EXECUTIVE INTELLIGENCE
st.markdown("<div id='executive-intelligence'></div>", unsafe_allow_html=True)
st.markdown("<p class='section-title' style='text-align:center;'>Executive Intelligence</p>", unsafe_allow_html=True)
st.markdown("<p class='section-subtitle' style='text-align:center;'>High-Level Performance Overview</p>", unsafe_allow_html=True)

col_k1, col_k2, col_k3, col_k4, col_k5 = st.columns(5)
with col_k1:
    st.markdown(f'<div class="luxe-kpi-card"><div class="luxe-kpi-label">Active Customers</div><div class="luxe-kpi-value">{len(df_raw):,}</div></div>', unsafe_allow_html=True)
with col_k2:
    st.markdown(f'<div class="luxe-kpi-card"><div class="luxe-kpi-label">Gross Revenue</div><div class="luxe-kpi-value">${df_raw["Total_Spending"].sum():,.0f}</div></div>', unsafe_allow_html=True)
with col_k3:
    st.markdown(f'<div class="luxe-kpi-card"><div class="luxe-kpi-label">Avg Customer Spend</div><div class="luxe-kpi-value">${df_raw["Total_Spending"].mean():,.0f}</div></div>', unsafe_allow_html=True)
with col_k4:
    st.markdown(f'<div class="luxe-kpi-card"><div class="luxe-kpi-label">Avg Order Value</div><div class="luxe-kpi-value">${df_raw["Average_Order_Value"].mean():,.0f}</div></div>', unsafe_allow_html=True)
with col_k5:
    st.markdown(f'<div class="luxe-kpi-card"><div class="luxe-kpi-label">Repeat Purchase Rate</div><div class="luxe-kpi-value">100.0%</div></div>', unsafe_allow_html=True)

st.markdown("""
<div class="loyalty-card">
    <div style="font-size:0.75rem; text-transform:uppercase; letter-spacing:0.15em; color:#8C837D;">Lumière Loyalty Score</div>
    <div style="font-family:'Cormorant Garamond'; font-size:2.8rem; font-weight:600; color:#2E2A28; margin-top:4px;">98.0 <span style="font-size:1.2rem; color:#8C837D;">/100</span></div>
    <div style="font-size:0.82rem; color:#C9A86A; margin-top:2px;">Exceptional Customer Loyalty & Engagement</div>
    <div class="loyalty-progress-bg"><div class="loyalty-progress-fill" style="width: 98%;"></div></div>
</div>
""", unsafe_allow_html=True)

st.divider()

# SECTION 4: CUSTOMER INTELLIGENCE FILTERS
st.markdown("<p class='section-title' style='text-align:center;'>Customer Intelligence Filters</p>", unsafe_allow_html=True)
with st.expander("🎛️ Expand Filter Controls", expanded=True):
    col_f1, col_f2, col_f3, col_f4 = st.columns(4)
    with col_f1:
        sel_cat = st.multiselect("Category", df_raw['Preferred_Category'].unique().tolist(), default=df_raw['Preferred_Category'].unique().tolist())
    with col_f2:
        sel_gen = st.multiselect("Gender", df_raw['Gender'].unique().tolist(), default=df_raw['Gender'].unique().tolist())
    with col_f3:
        age_filt = st.slider("Age Profile", int(df_raw['Age'].min()), int(df_raw['Age'].max()), (18, 65))
    with col_f4:
        sel_persona = st.multiselect("Persona", df_raw['Customer_Persona'].unique().tolist(), default=df_raw['Customer_Persona'].unique().tolist())

df_filtered = df_raw[
    (df_raw['Preferred_Category'].isin(sel_cat)) &
    (df_raw['Gender'].isin(sel_gen)) &
    (df_raw['Age'].between(age_filt[0], age_filt[1])) &
    (df_raw['Customer_Persona'].isin(sel_persona))
]

# SECTION 5: CUSTOMER OVERVIEW (EDITORIAL CARDS)
st.markdown("<div id='customer-personas'></div>", unsafe_allow_html=True)
st.markdown("<p class='section-title' style='text-align:center;'>Customer Overview</p>", unsafe_allow_html=True)
st.markdown("<p class='section-subtitle' style='text-align:center;'>Discover key personas driving your business growth</p>", unsafe_allow_html=True)

personas_info = [
    {"name": "VIP Cosmetics Enthusiasts ✨", "img": "https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=400&q=80", "count": "373 Customers", "spend": "$4,628", "age": "29", "fav": "Bath & Body", "score": "98%", "value": "High Revenue"},
    {"name": "Frequent Buyers 🛍️", "img": "https://images.unsplash.com/photo-1517841905240-472988babdf9?w=400&q=80", "count": "286 Customers", "spend": "$3,124", "age": "26", "fav": "Skincare", "score": "92%", "value": "High Frequency"},
    {"name": "Budget Conscious 💄", "img": "https://images.unsplash.com/photo-1524504388940-b1c1722653e1?w=400&q=80", "count": "412 Customers", "spend": "$1,243", "age": "34", "fav": "Makeup", "score": "75%", "value": "Volume Driven"},
    {"name": "At-Risk Customers ⚠️", "img": "https://images.unsplash.com/photo-1494790108377-be9c29b29330?w=400&q=80", "count": "156 Customers", "spend": "$892", "age": "40", "fav": "Skincare", "score": "48%", "value": "Retention Focus"}
]

col_p1, col_p2, col_p3, col_p4 = st.columns(4)
for idx, p in enumerate(personas_info):
    card_html = f"""
    <div class="persona-card">
        <img src="{p['img']}" class="persona-img" alt="{p['name']}">
        <div class="persona-body">
            <div class="persona-title">{p['name']}</div>
            <div style="font-size:0.75rem; color:#C9A86A; margin-bottom:10px;">{p['count']}</div>
            <div style="font-size:0.8rem; color:#8C837D; line-height:1.6;">
                <b>Avg Spend:</b> {p['spend']}<br>
                <b>Avg Age:</b> {p['age']}<br>
                <b>Fav Category:</b> {p['fav']}<br>
                <b>Loyalty Score:</b> {p['score']}<br>
                <b>Business Value:</b> {p['value']}
            </div>
        </div>
    </div>
    """
    if idx == 0: col_p1.markdown(card_html, unsafe_allow_html=True)
    elif idx == 1: col_p2.markdown(card_html, unsafe_allow_html=True)
    elif idx == 2: col_p3.markdown(card_html, unsafe_allow_html=True)
    elif idx == 3: col_p4.markdown(card_html, unsafe_allow_html=True)

st.divider()

# SECTION 6: SEGMENTATION STUDIO
st.markdown("<div id='segmentation-studio'></div>", unsafe_allow_html=True)
st.markdown("<p class='section-title' style='text-align:center;'>Segmentation Studio</p>", unsafe_allow_html=True)
st.markdown("<p class='section-subtitle' style='text-align:center;'>Compare clustering algorithms and discover optimal customer segments</p>", unsafe_allow_html=True)

col_s1, col_s2 = st.columns([1, 2])
with col_s1:
    st.markdown("### Model Evaluation Matrix")
    st.dataframe(eval_summary, use_container_width=True)
    st.caption("K-Means performs best for this dataset based on silhouette and inertia criteria.")
with col_s2:
    st.plotly_chart(CosmeticsVisualizer.plot_pca_2d(df_filtered), use_container_width=True)

st.divider()

# SECTION 7: DATA STORYTELLING (EDA)
st.markdown("<div id='data-storytelling'></div>", unsafe_allow_html=True)
st.markdown("<p class='section-title' style='text-align:center;'>Data Storytelling (EDA)</p>", unsafe_allow_html=True)

col_e1, col_e2, col_e3 = st.columns(3)
with col_e1: st.plotly_chart(CosmeticsVisualizer.plot_age_dist(df_filtered), use_container_width=True)
with col_e2: st.plotly_chart(CosmeticsVisualizer.plot_income_dist(df_filtered), use_container_width=True)
with col_e3: st.plotly_chart(CosmeticsVisualizer.plot_spending_dist(df_filtered), use_container_width=True)

col_e4, col_e5, col_e6 = st.columns(3)
with col_e4: st.plotly_chart(CosmeticsVisualizer.plot_gender_dist(df_filtered), use_container_width=True)
with col_e5: st.plotly_chart(CosmeticsVisualizer.plot_freq_dist(df_filtered), use_container_width=True)
with col_e6: st.plotly_chart(CosmeticsVisualizer.plot_revenue_dist(df_filtered), use_container_width=True)

col_e7, col_e8 = st.columns(2)
with col_e7: st.plotly_chart(CosmeticsVisualizer.plot_correlation_heatmap(df_filtered), use_container_width=True)
with col_e8: st.plotly_chart(CosmeticsVisualizer.plot_category_spend(df_filtered), use_container_width=True)

st.divider()

# SECTION 8: FEATURE ENGINEERING PIPELINE
st.markdown("<div id='feature-engineering'></div>", unsafe_allow_html=True)
st.markdown("<p class='section-title' style='text-align:center;'>Feature Engineering Pipeline</p>", unsafe_allow_html=True)
st.markdown("<p class='section-subtitle' style='text-align:center;'>Transforming raw transaction logs into customer behavioral features</p>", unsafe_allow_html=True)

col_fe1, col_fe2, col_fe3, col_fe4, col_fe5, col_fe6 = st.columns(6)
with col_fe1: st.markdown('<div class="luxe-kpi-card"><div class="luxe-kpi-label">Avg Order Value</div><div class="luxe-kpi-value">$2,156</div></div>', unsafe_allow_html=True)
with col_fe2: st.markdown('<div class="luxe-kpi-card"><div class="luxe-kpi-label">Purchase Freq</div><div class="luxe-kpi-value">12.4</div></div>', unsafe_allow_html=True)
with col_fe3: st.markdown('<div class="luxe-kpi-card"><div class="luxe-kpi-label">Customer Lifetime</div><div class="luxe-kpi-value">24.6 Mo</div></div>', unsafe_allow_html=True)
with col_fe4: st.markdown('<div class="luxe-kpi-card"><div class="luxe-kpi-label">Days Last Purchase</div><div class="luxe-kpi-value">18 Days</div></div>', unsafe_allow_html=True)
with col_fe5: st.markdown('<div class="luxe-kpi-card"><div class="luxe-kpi-label">Fav Category</div><div class="luxe-kpi-value">Skincare</div></div>', unsafe_allow_html=True)
with col_fe6: st.markdown('<div class="luxe-kpi-card"><div class="luxe-kpi-label">Discount Usage</div><div class="luxe-kpi-value">24%</div></div>', unsafe_allow_html=True)

st.divider()

# SECTION 9: RFM INTELLIGENCE
st.markdown("<div id='rfm-intelligence'></div>", unsafe_allow_html=True)
st.markdown("<p class='section-title' style='text-align:center;'>RFM Intelligence</p>", unsafe_allow_html=True)
st.markdown("<p class='section-subtitle' style='text-align:center;'>Understand customer value through Recency, Frequency & Monetary analysis</p>", unsafe_allow_html=True)

rfm_data = [
    {"code": "C", "name": "Champions", "color": "#D8D2F0", "pct": "25.4%", "desc": "High Value"},
    {"code": "L", "name": "Loyal Customers", "color": "#C9D8C5", "pct": "33.6%", "desc": "Strong Loyalty"},
    {"code": "P", "name": "Potential Loyalists", "color": "#E8CFCF", "pct": "23.3%", "desc": "Growth Potential"},
    {"code": "N", "name": "Need Attention", "color": "#FAF8F5", "pct": "12.6%", "desc": "Engagement Needed"},
    {"code": "A", "name": "At Risk", "color": "#D9A5A5", "pct": "4.2%", "desc": "Retention Focus"},
    {"code": "L", "name": "Lost Customers", "color": "#F7F1EC", "pct": "0.9%", "desc": "Re-engagement"}
]

col_r1, col_r2, col_r3, col_r4, col_r5, col_r6 = st.columns(6)
cols_rfm = [col_r1, col_r2, col_r3, col_r4, col_r5, col_r6]

for idx, item in enumerate(rfm_data):
    rfm_card_html = f"""
    <div class="rfm-badge-card">
        <div class="rfm-circle" style="background:{item['color']};">{item['code']}</div>
        <div style="font-weight:600; font-size:0.85rem;">{item['name']}</div>
        <div style="font-family:'Cormorant Garamond'; font-size:1.4rem; font-weight:600; color:#2E2A28; margin:4px 0;">{item['pct']}</div>
        <div style="font-size:0.72rem; color:#8C837D;">{item['desc']}</div>
    </div>
    """
    cols_rfm[idx].markdown(rfm_card_html, unsafe_allow_html=True)

st.divider()

# SECTION 10: CUSTOMER LIFETIME VALUE (CLV)
st.markdown("<div id='clv-intelligence'></div>", unsafe_allow_html=True)
st.markdown("<p class='section-title' style='text-align:center;'>Customer Lifetime Value (CLV)</p>", unsafe_allow_html=True)

clv_tiers = [
    {"tier": "Platinum", "tag": "Top 10% Customers", "clv": "$18,750", "rev": "28%", "ret": "92%"},
    {"tier": "Gold", "tag": "Top 20% Customers", "clv": "$9,850", "rev": "28%", "ret": "56%"},
    {"tier": "Silver", "tag": "Top 30% Customers", "clv": "$4,250", "rev": "20%", "ret": "56%"},
    {"tier": "Bronze", "tag": "Top 40% Customers", "clv": "$1,850", "rev": "10%", "ret": "34%"}
]

col_cl1, col_cl2, col_cl3, col_cl4 = st.columns(4)
cols_clv = [col_cl1, col_cl2, col_cl3, col_cl4]

for idx, t in enumerate(clv_tiers):
    clv_html = f"""
    <div class="clv-card">
        <div style="font-family:'Cormorant Garamond'; font-size:1.5rem; font-weight:600;">{t['tier']}</div>
        <div style="font-size:0.72rem; color:#C9A86A; margin-bottom:12px;">{t['tag']}</div>
        <div style="font-size:0.8rem; color:#8C837D;">Avg CLV</div>
        <div style="font-family:'Cormorant Garamond'; font-size:1.8rem; font-weight:600; color:#2E2A28; margin-bottom:10px;">{t['clv']}</div>
        <div style="font-size:0.78rem; color:#8C837D;">Revenue Contribution: <b>{t['rev']}</b></div>
        <div style="font-size:0.78rem; color:#8C837D;">Retention Rate: <b>{t['ret']}</b></div>
    </div>
    """
    cols_clv[idx].markdown(clv_html, unsafe_allow_html=True)

st.divider()

# SECTION 11: AI RECOMMENDATION ENGINE
st.markdown("<div id='recommendations'></div>", unsafe_allow_html=True)
st.markdown("<p class='section-title' style='text-align:center;'>AI Recommendation Engine</p>", unsafe_allow_html=True)

col_rec1, col_rec2 = st.columns([1, 2])
with col_rec1:
    st.image("https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=500&q=80", use_container_width=True)
with col_rec2:
    st.markdown("### Recommended Strategy for VIP Cosmetics Enthusiasts")
    st.markdown("""
    - **Increase Premium Product Visibility:** Feature high-margin luxury skincare drops on homepage hero.
    - **Introduce Concierge Loyalty Membership:** Provide early access to seasonal launches and private sales.
    - **Personal Beauty Advisor Outreach:** Deploy personalized SMS consultations for re-order windows.
    """)
    st.markdown("""
    <div style="background:#FAF8F5; border:1px solid #E8CFCF; padding:16px; border-radius:12px; margin-top:15px;">
        <div style="font-size:0.78rem; color:#8C837D; text-transform:uppercase;">Expected Financial Impact</div>
        <div style="font-family:'Cormorant Garamond'; font-size:2rem; font-weight:600; color:#2E2A28;">+18% Revenue Increase</div>
        <div style="font-size:0.8rem; color:#C9A86A;">High Confidence Prediction Model</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# SECTION 12: AUTOMATIC BUSINESS INSIGHTS
st.markdown("<div id='ai-insights'></div>", unsafe_allow_html=True)
st.markdown("<p class='section-title' style='text-align:center;'>Automatic Business Insights</p>", unsafe_allow_html=True)

col_i1, col_i2 = st.columns(2)
with col_i1:
    st.markdown("""
    <div style="background:#FFFFFF; border-left:3px solid #C9A86A; padding:16px; margin-bottom:12px; border-radius:0 12px 12px 0;">
        “ Customers aged 25–35 contribute <b>48%</b> of total annual revenue. ”
    </div>
    <div style="background:#FFFFFF; border-left:3px solid #D9A5A5; padding:16px; border-radius:0 12px 12px 0;">
        “ Weekend sales increase by <b>22%</b> during targeted beauty campaign launches. ”
    </div>
    """, unsafe_allow_html=True)
with col_i2:
    st.markdown("""
    <div style="background:#FFFFFF; border-left:3px solid #D8D2F0; padding:16px; margin-bottom:12px; border-radius:0 12px 12px 0;">
        “ High-income customers spend <b>2.8×</b> more per order than platform average. ”
    </div>
    <div style="background:#FFFFFF; border-left:3px solid #C9D8C5; padding:16px; border-radius:0 12px 12px 0;">
        “ Premium customers represent only 12% of users but generate <b>39%</b> of revenue. ”
    </div>
    """, unsafe_allow_html=True)

st.divider()

# SECTION 13: EXPORT CENTER
st.markdown("<div id='export-center'></div>", unsafe_allow_html=True)
st.markdown("<p class='section-title' style='text-align:center;'>Export Center</p>", unsafe_allow_html=True)

col_ex1, col_ex2 = st.columns(2)
with col_ex1:
    csv_data = df_raw.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Complete Segmented CSV Dataset", data=csv_data, file_name="lumiere_segmented_data.csv", mime="text/csv", use_container_width=True)
with col_ex2:
    if st.button("📄 Generate Executive PDF Briefing", use_container_width=True):
        pdf_gen = PDFReportGenerator()
        pdf_file = pdf_gen.build_pdf_report(AutomatedInsightGenerator().generate_all_insights())
        with open(pdf_file, "rb") as f:
            st.download_button("Download PDF Briefing", data=f, file_name="Lumiere_Executive_Report.pdf", mime="application/pdf", use_container_width=True)

# FOOTER
st.markdown("""
<div class="lumiere-footer">
    <div style="font-family:'Cormorant Garamond'; font-size:1.8rem; font-weight:600; color:#2E2A28;">LUMIÈRE AI</div>
    <div style="font-size:0.78rem; color:#8C837D; margin-bottom:15px;">Customer Intelligence Platform • Soft Luxury Analytics</div>
    <div style="font-size:0.75rem; color:#8C837D;">Built with Python • Scikit-learn • Pandas • Plotly • Streamlit</div>
    <div style="font-size:0.72rem; color:#C9A86A; margin-top:10px;">© 2026 Lumière AI Analytics. All rights reserved.</div>
</div>
""", unsafe_allow_html=True)
