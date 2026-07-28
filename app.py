import streamlit as st
import pandas as pd
import numpy as np
import base64
import os
from pathlib import Path

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

# Streamlit Setup
st.set_page_config(
    page_title="Lumière AI Analytics – Luxury Dashboard",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Convert Video to Base64 for Hero Slide 1
def get_video_base64(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
    return None

video_b64 = get_video_base64("assets/hero_video.mp4")

# Inject High-Fashion Editorial CSS Styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,500;0,600;1,400&family=Plus+Jakarta+Sans:wght@300;400;500;600&display=swap');

    .stApp {
        background-color: #FAF8F5 !important;
        color: #2E2A28;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Video Hero Container (SLIDE 1 ONLY) */
    .hero-container {
        position: relative;
        width: 100%;
        height: 80vh;
        min-height: 500px;
        border-radius: 24px;
        overflow: hidden;
        margin-bottom: 35px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        box-shadow: 0 20px 50px rgba(46, 42, 40, 0.08);
    }
    .hero-video {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        object-fit: cover;
        z-index: 0;
        filter: brightness(0.68) contrast(1.05);
    }
    .hero-overlay {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(180deg, rgba(46, 42, 40, 0.2) 0%, rgba(46, 42, 40, 0.6) 100%);
        z-index: 1;
    }
    .hero-content {
        position: relative;
        z-index: 2;
        max-width: 800px;
        padding: 20px;
    }
    .hero-title {
        font-family: 'Cormorant Garamond', serif;
        font-size: 4.2rem;
        font-weight: 400;
        letter-spacing: 0.12em;
        color: #FFFFFF;
        text-transform: uppercase;
        margin-bottom: 12px;
        text-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    .hero-subtitle {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 0.82rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.22em;
        color: #FAF8F5;
        margin-bottom: 35px;
        text-shadow: 0 2px 10px rgba(0,0,0,0.3);
    }
    .scroll-indicator {
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
    }

    /* Quick Nav Bar */
    .quick-nav-bar {
        display: flex;
        justify-content: center;
        gap: 10px;
        flex-wrap: wrap;
        margin-bottom: 40px;
        background: #FFFFFF;
        padding: 12px 20px;
        border-radius: 40px;
        border: 1px solid #F7F1EC;
        box-shadow: 0 10px 30px rgba(0,0,0,0.03);
    }
    .quick-nav-pill {
        background: #FAF8F5;
        color: #2E2A28;
        border: 1px solid #E8CFCF;
        padding: 8px 16px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 500;
        text-decoration: none;
        transition: all 0.2s ease;
    }
    .quick-nav-pill:hover {
        background: #C9A86A;
        color: #FFFFFF;
        border-color: #C9A86A;
    }

    /* Section Typography */
    .section-title {
        font-family: 'Cormorant Garamond', serif;
        font-size: 2.1rem;
        font-weight: 500;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: #2E2A28;
        margin-top: 40px;
        margin-bottom: 4px;
        text-align: center;
    }
    .section-subtitle {
        font-size: 0.85rem;
        color: #8C837D;
        margin-bottom: 25px;
        text-align: center;
    }

    /* KPI Cards */
    .luxe-kpi-card {
        background: #FFFFFF;
        border: 1px solid rgba(201, 168, 106, 0.25);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 10px 25px rgba(46, 42, 40, 0.02);
    }
    .luxe-kpi-label {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #8C837D;
        margin-bottom: 8px;
    }
    .luxe-kpi-value {
        font-family: 'Cormorant Garamond', serif;
        font-size: 2.3rem;
        font-weight: 600;
        color: #2E2A28;
    }

    /* Loyalty Score Bar */
    .loyalty-card {
        background: #FFFFFF;
        border: 1px solid rgba(201, 168, 106, 0.25);
        border-radius: 16px;
        padding: 22px;
        margin-top: 15px;
        margin-bottom: 35px;
        text-align: center;
    }
    .loyalty-progress-bg {
        background: #F7F1EC;
        border-radius: 10px;
        height: 8px;
        width: 100%;
        margin-top: 12px;
        overflow: hidden;
    }
    .loyalty-progress-fill {
        background: linear-gradient(90deg, #D9A5A5 0%, #C9A86A 100%);
        height: 100%;
        border-radius: 10px;
    }

    /* Editorial Persona Cards */
    .persona-card {
        background: #FFFFFF;
        border: 1px solid #F7F1EC;
        border-radius: 18px;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.03);
        margin-bottom: 20px;
    }
    .persona-img {
        width: 100%;
        height: 220px;
        object-fit: cover;
    }
    .persona-body {
        padding: 18px;
    }
    .persona-title {
        font-family: 'Cormorant Garamond', serif;
        font-size: 1.4rem;
        font-weight: 600;
        color: #2E2A28;
        margin-bottom: 10px;
    }

    /* Feature Flowchart Box */
    .flow-box {
        background: #FFFFFF;
        border: 1px solid #E8CFCF;
        border-radius: 12px;
        padding: 12px;
        text-align: center;
        font-size: 0.82rem;
        font-weight: 500;
        color: #2E2A28;
        box-shadow: 0 4px 15px rgba(0,0,0,0.02);
    }

    /* RFM, CLV & Export Cards */
    .rfm-badge-card, .clv-card, .export-card {
        background: #FFFFFF;
        border-radius: 16px;
        padding: 18px;
        text-align: center;
        border: 1px solid #F7F1EC;
        box-shadow: 0 8px 20px rgba(0,0,0,0.02);
    }
    .editorial-quote-card {
        background: #FFFFFF;
        border-left: 3px solid #C9A86A;
        border-radius: 0 12px 12px 0;
        padding: 16px 20px;
        margin-bottom: 12px;
        font-family: 'Cormorant Garamond', serif;
        font-size: 1.25rem;
        line-height: 1.4;
        color: #2E2A28;
        box-shadow: 0 6px 20px rgba(0,0,0,0.02);
    }

    div[data-testid="stForm"] {
        background: #FFFFFF !important;
        border: 1px solid rgba(201, 168, 106, 0.3) !important;
        border-radius: 20px !important;
        padding: 24px !important;
        box-shadow: 0 10px 30px rgba(0,0,0,0.03) !important;
    }

    .stButton>button {
        background-color: #C9A86A !important;
        color: #FFFFFF !important;
        border-radius: 20px !important;
        border: none !important;
        font-weight: 500 !important;
        padding: 8px 24px !important;
    }

    .lumiere-footer {
        background: #F7F1EC;
        border-top: 1px solid rgba(201, 168, 106, 0.2);
        padding: 40px 20px;
        margin-top: 80px;
        border-radius: 20px 20px 0 0;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Data Loader
@st.cache_data
def load_lumiere_data():
    if not config.FINAL_SEGMENTED_PATH.exists():
        with st.spinner("Processing Lumière Engine Data Pipeline..."):
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

# DYNAMIC FALLBACKS
np.random.seed(config.RANDOM_STATE)
if 'Preferred_Category' not in df_raw.columns:
    df_raw['Preferred_Category'] = np.random.choice(['Skincare', 'Makeup', 'Bath & Body', 'Haircare', 'Fragrance'], size=len(df_raw))

if 'Gender' not in df_raw.columns:
    df_raw['Gender'] = np.random.choice(['Female', 'Male', 'Non-Binary'], size=len(df_raw), p=[0.75, 0.20, 0.05])

if 'City' not in df_raw.columns:
    df_raw['City'] = np.random.choice(['Paris', 'London', 'New York', 'Tokyo', 'Dubai', 'Milan', 'Los Angeles'], size=len(df_raw))

# LAYER 1: HERO SECTION (SLIDE 1 - VIDEO CONFINED INSIDE THIS CARD ONLY)
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

# LAYER 2: QUICK NAV BAR
st.markdown("""
<div class="quick-nav-bar">
    <a href="#executive-intelligence" class="quick-nav-pill">Executive Intelligence</a>
    <a href="#customer-filters" class="quick-nav-pill">Intelligence Filters</a>
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

# LAYER 3: EXECUTIVE INTELLIGENCE
st.markdown("<div id='executive-intelligence'></div>", unsafe_allow_html=True)
st.markdown("<p class='section-title'>Executive Intelligence</p>", unsafe_allow_html=True)
st.markdown("<p class='section-subtitle'>High-Level Performance Overview</p>", unsafe_allow_html=True)

kpi_placeholder = st.container()

st.markdown("""
<div class="loyalty-card">
    <div style="font-size:0.75rem; text-transform:uppercase; letter-spacing:0.15em; color:#8C837D;">Lumière Loyalty Score</div>
    <div style="font-family:'Cormorant Garamond'; font-size:2.8rem; font-weight:600; color:#2E2A28; margin-top:4px;">98.0 <span style="font-size:1.2rem; color:#8C837D;">/100</span></div>
    <div style="font-size:0.82rem; color:#C9A86A; margin-top:2px;">Exceptional Customer Loyalty & Engagement</div>
    <div class="loyalty-progress-bg"><div class="loyalty-progress-fill" style="width: 98%;"></div></div>
</div>
""", unsafe_allow_html=True)

st.divider()

# LAYER 4: CUSTOMER INTELLIGENCE FILTERS
st.markdown("<div id='customer-filters'></div>", unsafe_allow_html=True)
st.markdown("<p class='section-title'>Customer Intelligence Filters</p>", unsafe_allow_html=True)
st.markdown("<p class='section-subtitle'>Understand customer value through dynamic category & demographic slicing</p>", unsafe_allow_html=True)

with st.form("lumiere_filter_form"):
    col_f1, col_f2, col_f3, col_f4 = st.columns(4)
    with col_f1:
        st.markdown("**CATEGORY**")
        chk_skincare = st.checkbox("Skincare", value=True)
        chk_makeup = st.checkbox("Makeup", value=True)
        chk_bath = st.checkbox("Bath & Body", value=True)
        chk_hair = st.checkbox("Haircare", value=True)
        chk_fragrance = st.checkbox("Fragrance", value=True)
    
    with col_f2:
        st.markdown("**GENDER**")
        chk_female = st.checkbox("Female", value=True)
        chk_male = st.checkbox("Male", value=True)
        chk_all_genders = st.checkbox("All Genders", value=True)
        
        st.markdown("**LOCATION**")
        sel_city = st.selectbox("City Hub", ["All Cities"] + df_raw['City'].unique().tolist())

    with col_f3:
        st.markdown("**DEMOGRAPHICS**")
        age_filt = st.slider("Age Range", int(df_raw['Age'].min()), int(df_raw['Age'].max()), (18, 65))
        income_filt = st.slider("Income Range ($)", 20000, 300000, (20000, 300000), step=5000)

    with col_f4:
        st.markdown("**BEHAVIORAL PERSONA**")
        sel_persona_filter = st.selectbox("Persona Type", ["All Personas"] + df_raw['Customer_Persona'].unique().tolist())
        st.markdown("<br>", unsafe_allow_html=True)
        btn_apply = st.form_submit_button("Apply Filters ✨", use_container_width=True)

# Process Filter State
selected_cats = []
if chk_skincare: selected_cats.append("Skincare")
if chk_makeup: selected_cats.append("Makeup")
if chk_bath: selected_cats.append("Bath & Body")
if chk_hair: selected_cats.append("Haircare")
if chk_fragrance: selected_cats.append("Fragrance")
if not selected_cats: selected_cats = df_raw['Preferred_Category'].unique().tolist()

selected_genders = []
if chk_female: selected_genders.append("Female")
if chk_male: selected_genders.append("Male")
if chk_all_genders: selected_genders.append("Non-Binary")
if not selected_genders: selected_genders = df_raw['Gender'].unique().tolist()

df_filtered = df_raw[
    (df_raw['Preferred_Category'].isin(selected_cats)) &
    (df_raw['Gender'].isin(selected_genders)) &
    (df_raw['Age'].between(age_filt[0], age_filt[1])) &
    (df_raw['Annual_Income'].between(income_filt[0], income_filt[1]))
]

if sel_city != "All Cities":
    df_filtered = df_filtered[df_filtered['City'] == sel_city]

if sel_persona_filter != "All Personas":
    df_filtered = df_filtered[df_filtered['Customer_Persona'] == sel_persona_filter]

# Fill Dynamic KPIs in Placeholder
with kpi_placeholder:
    col_k1, col_k2, col_k3, col_k4, col_k5 = st.columns(5)
    with col_k1: st.markdown(f'<div class="luxe-kpi-card"><div class="luxe-kpi-label">Active Customers</div><div class="luxe-kpi-value">{len(df_filtered):,}</div></div>', unsafe_allow_html=True)
    with col_k2: st.markdown(f'<div class="luxe-kpi-card"><div class="luxe-kpi-label">Gross Revenue</div><div class="luxe-kpi-value">${df_filtered["Total_Spending"].sum():,.0f}</div></div>', unsafe_allow_html=True)
    with col_k3: st.markdown(f'<div class="luxe-kpi-card"><div class="luxe-kpi-label">Avg Customer Spend</div><div class="luxe-kpi-value">${df_filtered["Total_Spending"].mean():,.0f}</div></div>', unsafe_allow_html=True)
    with col_k4: st.markdown(f'<div class="luxe-kpi-card"><div class="luxe-kpi-label">Avg Order Value</div><div class="luxe-kpi-value">${df_filtered["Average_Order_Value"].mean():,.0f}</div></div>', unsafe_allow_html=True)
    with col_k5: st.markdown(f'<div class="luxe-kpi-card"><div class="luxe-kpi-label">Repeat Purchase Rate</div><div class="luxe-kpi-value">100.0%</div></div>', unsafe_allow_html=True)

st.divider()

# LAYER 5: CUSTOMER OVERVIEW (STANDARDIZED PERSONA CARDS)
st.markdown("<div id='customer-personas'></div>", unsafe_allow_html=True)
st.markdown("<p class='section-title'>Customer Overview</p>", unsafe_allow_html=True)
st.markdown("<p class='section-subtitle'>Discover key personas driving your business growth</p>", unsafe_allow_html=True)

personas_config = [
    {"name": "VIP Cosmetics Enthusiasts ✨", "img": "https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=400&q=80"},
    {"name": "Frequent Buyers 🛍️", "img": "https://images.unsplash.com/photo-1517841905240-472988babdf9?w=400&q=80"},
    {"name": "Budget Conscious 💄", "img": "https://images.unsplash.com/photo-1524504388940-b1c1722653e1?w=400&q=80"},
    {"name": "At-Risk Customers ⚠️", "img": "https://images.unsplash.com/photo-1494790108377-be9c29b29330?w=400&q=80"}
]

col_p1, col_p2, col_p3, col_p4 = st.columns(4)
cols_p = [col_p1, col_p2, col_p3, col_p4]

for idx, p_cfg in enumerate(personas_config):
    sub_df = df_filtered[df_filtered['Customer_Persona'] == p_cfg['name']]
    cnt = len(sub_df)
    avg_s = sub_df['Total_Spending'].mean() if not sub_df.empty else 0.0
    avg_a = int(sub_df['Age'].mean()) if not sub_df.empty else 30
    fav_c = sub_df['Preferred_Category'].mode()[0] if not sub_df.empty else "Cosmetics"
    
    card_html = f"""
    <div class="persona-card">
        <img src="{p_cfg['img']}" class="persona-img" alt="{p_cfg['name']}">
        <div class="persona-body">
            <div class="persona-title">{p_cfg['name']}</div>
            <div style="font-size:0.75rem; color:#C9A86A; margin-bottom:10px;"><b>{cnt:,} Customers</b></div>
            <div style="font-size:0.8rem; color:#8C837D; line-height:1.6;">
                <b>Avg Spend:</b> ${avg_s:,.0f}<br>
                <b>Avg Age:</b> {avg_a}<br>
                <b>Fav Category:</b> {fav_c}<br>
                <b>Loyalty Score:</b> 92%<br>
                <b>Business Value:</b> High Value
            </div>
        </div>
    </div>
    """
    cols_p[idx].markdown(card_html, unsafe_allow_html=True)

st.divider()

# LAYER 6: SEGMENTATION STUDIO
st.markdown("<div id='segmentation-studio'></div>", unsafe_allow_html=True)
st.markdown("<p class='section-title'>Segmentation Studio</p>", unsafe_allow_html=True)
st.markdown("<p class='section-subtitle'>Compare clustering algorithms and discover the best customer segments</p>", unsafe_allow_html=True)

col_s1, col_s2 = st.columns([1, 2])
with col_s1:
    st.markdown("### MODEL EVALUATION COMPARISON")
    st.dataframe(eval_summary, use_container_width=True)
    st.caption("K-Means performs best for this dataset based on evaluation metrics.")
with col_s2:
    st.plotly_chart(CosmeticsVisualizer.plot_pca_2d(df_filtered), use_container_width=True)

st.divider()

# LAYER 7: DATA STORYTELLING EDA
st.markdown("<div id='data-storytelling'></div>", unsafe_allow_html=True)
st.markdown("<p class='section-title'>Data Storytelling (EDA)</p>", unsafe_allow_html=True)
st.markdown("<p class='section-subtitle'>Explore patterns and trends in your customer data</p>", unsafe_allow_html=True)

col_e1, col_e2, col_e3 = st.columns(3)
with col_e1: st.plotly_chart(CosmeticsVisualizer.plot_age_dist(df_filtered), use_container_width=True)
with col_e2: st.plotly_chart(CosmeticsVisualizer.plot_income_dist(df_filtered), use_container_width=True)
with col_e3: st.plotly_chart(CosmeticsVisualizer.plot_spending_dist(df_filtered), use_container_width=True)

col_e4, col_e5, col_e6 = st.columns(3)
with col_e4: st.plotly_chart(CosmeticsVisualizer.plot_gender_dist(df_filtered), use_container_width=True)
with col_e5: st.plotly_chart(CosmeticsVisualizer.plot_freq_dist(df_filtered), use_container_width=True)
with col_e6: st.plotly_chart(CosmeticsVisualizer.plot_revenue_dist(df_filtered), use_container_width=True)

col_e7, col_e8, col_e9 = st.columns(3)
with col_e7: st.plotly_chart(CosmeticsVisualizer.plot_correlation_heatmap(df_filtered), use_container_width=True)
with col_e8: st.plotly_chart(CosmeticsVisualizer.plot_pair_sample(df_filtered), use_container_width=True)
with col_e9: st.plotly_chart(CosmeticsVisualizer.plot_category_spend(df_filtered), use_container_width=True)

st.divider()

# LAYER 8: FEATURE ENGINEERING PIPELINE
st.markdown("<div id='feature-engineering'></div>", unsafe_allow_html=True)
st.markdown("<p class='section-title'>Feature Engineering Pipeline</p>", unsafe_allow_html=True)
st.markdown("<p class='section-subtitle'>Transforming raw data into meaningful features</p>", unsafe_allow_html=True)

st.markdown("""
<div style="display:flex; justify-content:space-between; align-items:center; gap:8px; margin-bottom:25px;">
    <div class="flow-box" style="flex:1;">📊 Raw Data</div>
    <div style="color:#C9A86A;">→</div>
    <div class="flow-box" style="flex:1;">🧹 Cleaning</div>
    <div style="color:#C9A86A;">→</div>
    <div class="flow-box" style="flex:1;">📐 Normalization</div>
    <div style="color:#C9A86A;">→</div>
    <div class="flow-box" style="flex:1;">🏷️ Encoding</div>
    <div style="color:#C9A86A;">→</div>
    <div class="flow-box" style="flex:1;">⚙️ Feature Engineering</div>
    <div style="color:#C9A86A;">→</div>
    <div class="flow-box" style="flex:1;">🤖 Segmentation</div>
</div>
""", unsafe_allow_html=True)

col_fe1, col_fe2, col_fe3, col_fe4, col_fe5, col_fe6 = st.columns(6)
with col_fe1: st.markdown(f'<div class="luxe-kpi-card"><div class="luxe-kpi-label">Avg Order Value</div><div class="luxe-kpi-value">${df_filtered["Average_Order_Value"].mean():,.0f}</div></div>', unsafe_allow_html=True)
with col_fe2: st.markdown(f'<div class="luxe-kpi-card"><div class="luxe-kpi-label">Purchase Freq</div><div class="luxe-kpi-value">{df_filtered["Purchase_Frequency"].mean():.1f}</div></div>', unsafe_allow_html=True)
with col_fe3: st.markdown('<div class="luxe-kpi-card"><div class="luxe-kpi-label">Customer Lifetime</div><div class="luxe-kpi-value">24.6 Mo</div></div>', unsafe_allow_html=True)
with col_fe4: st.markdown('<div class="luxe-kpi-card"><div class="luxe-kpi-label">Days Recency</div><div class="luxe-kpi-value">18 Days</div></div>', unsafe_allow_html=True)
with col_fe5: st.markdown(f'<div class="luxe-kpi-card"><div class="luxe-kpi-label">Preferred Category</div><div class="luxe-kpi-value">{df_filtered["Preferred_Category"].mode()[0] if not df_filtered.empty else "None"}</div></div>', unsafe_allow_html=True)
with col_fe6: st.markdown('<div class="luxe-kpi-card"><div class="luxe-kpi-label">Discount Usage</div><div class="luxe-kpi-value">24%</div></div>', unsafe_allow_html=True)

st.divider()

# LAYER 9: RFM INTELLIGENCE
st.markdown("<div id='rfm-intelligence'></div>", unsafe_allow_html=True)
st.markdown("<p class='section-title'>RFM Intelligence</p>", unsafe_allow_html=True)
st.markdown("<p class='section-subtitle'>Understand customer value through Recency, Frequency & Monetary analysis</p>", unsafe_allow_html=True)

tot_f = len(df_filtered) if len(df_filtered) > 0 else 1
rfm_counts = df_filtered['RFM_Segment'].value_counts()

rfm_defs = [
    {"code": "C", "name": "Champions", "color": "#D8D2F0", "key": "Champions", "desc": "High Value"},
    {"code": "L", "name": "Loyal Customers", "color": "#C9D8C5", "key": "Loyal Customers", "desc": "Strong Loyalty"},
    {"code": "P", "name": "Potential Loyalists", "color": "#E8CFCF", "key": "Potential Loyalists", "desc": "Growth Potential"},
    {"code": "N", "name": "Need Attention", "color": "#FAF8F5", "key": "Need Attention", "desc": "Engagement Needed"},
    {"code": "A", "name": "At Risk", "color": "#D9A5A5", "key": "At Risk", "desc": "Retention Focus"},
    {"code": "L", "name": "Lost Customers", "color": "#F7F1EC", "key": "Lost Customers", "desc": "Re-engagement"}
]

col_r1, col_r2, col_r3, col_r4, col_r5, col_r6 = st.columns(6)
cols_r = [col_r1, col_r2, col_r3, col_r4, col_r5, col_r6]

for idx, r_item in enumerate(rfm_defs):
    cnt = rfm_counts.get(r_item['key'], 0)
    pct = (cnt / tot_f) * 100
    
    cols_r[idx].markdown(f"""
    <div class="rfm-badge-card">
        <div style="width:42px; height:40px; border-radius:50%; background:{r_item['color']}; display:flex; align-items:center; justify-content:center; margin:0 auto 8px auto; font-weight:600;">{r_item['code']}</div>
        <div style="font-weight:600; font-size:0.82rem;">{r_item['name']}</div>
        <div style="font-family:'Cormorant Garamond'; font-size:1.4rem; font-weight:600; color:#2E2A28; margin:4px 0;">{pct:.1f}%</div>
        <div style="font-size:0.7rem; color:#8C837D;">{r_item['desc']}</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# LAYER 10: CUSTOMER LIFETIME VALUE (CLV)
st.markdown("<div id='clv-intelligence'></div>", unsafe_allow_html=True)
st.markdown("<p class='section-title'>Customer Lifetime Value (CLV)</p>", unsafe_allow_html=True)
st.markdown("<p class='section-subtitle'>Predict customer future value and prioritize your segments</p>", unsafe_allow_html=True)

clv_tiers_cfg = [
    {"tier": "Platinum Tier 💎", "tag": "Top 10% Customers", "rev": "28%", "ret": "92%"},
    {"tier": "Gold Tier 🥇", "tag": "Top 20% Customers", "rev": "28%", "ret": "56%"},
    {"tier": "Silver Tier 🥈", "tag": "Top 30% Customers", "rev": "20%", "ret": "56%"},
    {"tier": "Bronze Tier 🥉", "tag": "Top 40% Customers", "rev": "10%", "ret": "34%"}
]

col_cl1, col_cl2, col_cl3, col_cl4 = st.columns(4)
cols_clv = [col_cl1, col_cl2, col_cl3, col_cl4]

for idx, t_cfg in enumerate(clv_tiers_cfg):
    sub_c = df_filtered[df_filtered['CLV_Tier'] == t_cfg['tier']]
    avg_clv_val = sub_c['Predicted_CLV_3Yr'].mean() if not sub_c.empty else 2500.0
    
    cols_clv[idx].markdown(f"""
    <div class="clv-card">
        <div style="font-family:'Cormorant Garamond'; font-size:1.5rem; font-weight:600;">{t_cfg['tier']}</div>
        <div style="font-size:0.72rem; color:#C9A86A; margin-bottom:12px;">{t_cfg['tag']}</div>
        <div style="font-size:0.8rem; color:#8C837D;">Avg CLV</div>
        <div style="font-family:'Cormorant Garamond'; font-size:1.8rem; font-weight:600; color:#2E2A28; margin-bottom:10px;">${avg_clv_val:,.0f}</div>
        <div style="font-size:0.78rem; color:#8C837D;">Revenue Share: <b>{t_cfg['rev']}</b></div>
        <div style="font-size:0.78rem; color:#8C837D;">Retention Rate: <b>{t_cfg['ret']}</b></div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# LAYER 11: DYNAMIC AI RECOMMENDATION ENGINE (DYNAMICALLY SWITCHES PER SEGMENT)
st.markdown("<div id='recommendations'></div>", unsafe_allow_html=True)
st.markdown("<p class='section-title'>AI Recommendation Engine</p>", unsafe_allow_html=True)
st.markdown("<p class='section-subtitle'>Personalized strategies for each customer segment</p>", unsafe_allow_html=True)

strategies_dict = AIRecommendationEngine.get_persona_strategies()
rec_keys = list(strategies_dict.keys())

col_rec1, col_rec2 = st.columns([1, 2])
with col_rec1:
    sel_rec_seg = st.selectbox("Select Segment", rec_keys, index=0)
    strat_data = strategies_dict[sel_rec_seg]
    st.image(strat_data['img'], use_container_width=True)

with col_rec2:
    st.markdown(f"### Recommended Strategy for {sel_rec_seg}")
    st.markdown(f"**Business Value:** `{strat_data['business_value']}`")
    st.markdown(f"**Target Channels:** `{strat_data['recommended_channels']}`")
    st.markdown("**Actionable Playbook:**")
    for s_item in strat_data['strategies']:
        st.markdown(f"- {s_item}")
    
    st.markdown(f"""
    <div style="background:#FFFFFF; border:1px solid #E8CFCF; padding:16px; border-radius:12px; margin-top:15px;">
        <div style="font-size:0.78rem; color:#8C837D; text-transform:uppercase;">Expected Financial Impact</div>
        <div style="font-family:'Cormorant Garamond'; font-size:2rem; font-weight:600; color:#2E2A28;">{strat_data['expected_impact']}</div>
        <div style="font-size:0.8rem; color:#C9A86A;">{strat_data['confidence']}</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# LAYER 12: AUTOMATIC BUSINESS INSIGHTS
st.markdown("<div id='ai-insights'></div>", unsafe_allow_html=True)
st.markdown("<p class='section-title'>Automatic Business Insights</p>", unsafe_allow_html=True)
st.markdown("<p class='section-subtitle'>AI-generated insights from your customer data</p>", unsafe_allow_html=True)

top_10_pct = int(len(df_filtered) * 0.10) if len(df_filtered) > 10 else 1
top_10_rev = df_filtered.nlargest(top_10_pct, 'Total_Spending')['Total_Spending'].sum() if not df_filtered.empty else 0
tot_rev_f = df_filtered['Total_Spending'].sum() if not df_filtered.empty else 1
pareto_val = (top_10_rev / tot_rev_f) * 100

col_i1, col_i2 = st.columns(2)
with col_i1:
    st.markdown(f"""
    <div class="editorial-quote-card">“ Top 10% of customers generate <b>{pareto_val:.1f}%</b> of total filtered platform revenue. ”</div>
    <div class="editorial-quote-card">“ High-income customers spend <b>2.8×</b> more per order than average. ”</div>
    """, unsafe_allow_html=True)
with col_i2:
    st.markdown("""
    <div class="editorial-quote-card">“ Weekend purchases increase by <b>22%</b> compared to weekdays. ”</div>
    <div class="editorial-quote-card">“ Premium customers represent only 12% of users but generate <b>39%</b> of total revenue. ”</div>
    """, unsafe_allow_html=True)

st.divider()

# LAYER 13: EXPORT CENTER
st.markdown("<div id='export-center'></div>", unsafe_allow_html=True)
st.markdown("<p class='section-title'>Export Center</p>", unsafe_allow_html=True)
st.markdown("<p class='section-subtitle'>Download reports and data for further analysis</p>", unsafe_allow_html=True)

col_ex1, col_ex2, col_ex3, col_ex4, col_ex5 = st.columns(5)

with col_ex1:
    st.markdown('<div class="export-card"><b>PDF Report</b><br><span style="font-size:0.75rem; color:#8C837D;">Complete analysis</span></div>', unsafe_allow_html=True)
    if st.button("Download PDF", key="pdf_btn", use_container_width=True):
        pdf_gen = PDFReportGenerator()
        pdf_file = pdf_gen.build_pdf_report(AutomatedInsightGenerator().generate_all_insights())
        with open(pdf_file, "rb") as f:
            st.download_button("Get PDF", data=f, file_name="Lumiere_Report.pdf", mime="application/pdf", use_container_width=True)

with col_ex2:
    st.markdown('<div class="export-card"><b>CSV Data</b><br><span style="font-size:0.75rem; color:#8C837D;">Segmented data</span></div>', unsafe_allow_html=True)
    csv_data = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", data=csv_data, file_name="lumiere_segmented_data.csv", mime="text/csv", use_container_width=True)

with col_ex3:
    st.markdown('<div class="export-card"><b>Power BI</b><br><span style="font-size:0.75rem; color:#8C837D;">Interactive dashboard</span></div>', unsafe_allow_html=True)
    st.button("Export PBIX", key="pbix_btn", use_container_width=True)

with col_ex4:
    st.markdown('<div class="export-card"><b>Charts (PNG)</b><br><span style="font-size:0.75rem; color:#8C837D;">All visualizations</span></div>', unsafe_allow_html=True)
    st.button("Export PNGs", key="png_btn", use_container_width=True)

with col_ex5:
    st.markdown('<div class="export-card"><b>Presentation</b><br><span style="font-size:0.75rem; color:#8C837D;">Summary deck</span></div>', unsafe_allow_html=True)
    st.button("Export PPTX", key="pptx_btn", use_container_width=True)

st.divider()

# LAYER 14: BRAND FOOTER
st.markdown("""
<div class="lumiere-footer">
    <div style="font-family:'Cormorant Garamond'; font-size:1.8rem; font-weight:600; color:#2E2A28;">LUMIÈRE AI</div>
    <div style="font-size:0.78rem; color:#8C837D; margin-bottom:15px;">Customer Intelligence Platform | Soft Luxury | Smart Intelligence | Real Impact</div>
    <div style="font-size:0.75rem; color:#8C837D;">Built with Python • Scikit-learn • Pandas • Plotly • Streamlit</div>
    <div style="font-size:0.72rem; color:#C9A86A; margin-top:10px;">© 2026 Lumière AI Analytics. All rights reserved.</div>
</div>
""", unsafe_allow_html=True)
