# Lumière AI – Soft Luxury Customer Intelligence & Analytics Platform 

[![Live App](https://img.shields.io/badge/LIVE_DEMO-Lumière_AI-C9A86A?style=flat-square&logo=streamlit&logoColor=2E2A28&labelColor=FAF8F5)](https://lumiere-ai-analytics.streamlit.app)
[![Python](https://img.shields.io/badge/PYTHON-3.10+-2E2A28?style=flat-square&logo=python&logoColor=C9A86A&labelColor=FAF8F5)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/SCIKIT_LEARN-ML_ENGINE-2E2A28?style=flat-square&logo=scikit-learn&logoColor=C9A86A&labelColor=FAF8F5)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/PLOTLY-LUXURY_VIZ-2E2A28?style=flat-square&logo=plotly&logoColor=C9A86A&labelColor=FAF8F5)](https://plotly.com/)
[![Kaggle](https://img.shields.io/badge/DATASET-KAGGLE-2E2A28?style=flat-square&logo=kaggle&logoColor=C9A86A&labelColor=FAF8F5)](https://www.kaggle.com/datasets/parvezkhan90/cosmetic-products-sales)

![Lumière AI Banner](assets/banner.png)

An enterprise-grade Customer Intelligence and Customer Segmentation web application built for luxury cosmetics and e-commerce brands. Features unsupervised machine learning (**K-Means**, **Hierarchical**, **DBSCAN**), 2D PCA cluster mapping, dynamic RFM behavioral scoring, 3-Year Customer Lifetime Value (CLV) predictive modeling, automated natural-language insight generation, and executive multi-format exports.

**Live Interactive App:** [https://lumiere-ai-analytics.streamlit.app](https://lumiere-ai-analytics.streamlit.app)

---

##  Dataset Source

This platform is trained and analyzed using the **[Cosmetic Products Sales Dataset](https://www.kaggle.com/datasets/parvezkhan90/cosmetic-products-sales)** from Kaggle, featuring customer purchasing habits, product category preferences, transaction frequency, and revenue metrics across luxury cosmetics and beauty categories.

---

##  Key Features

1. **Luxury Editorial UI/UX**: Designed as a single-page long-scrolling platform with a Base64-encoded hero header, Cormorant Garamond typography, and soft luxury pastel styling.
2. **Data Storytelling & EDA**: 9 interactive Plotly charts analyzing age, annual income, spending velocity, purchase frequency, revenue flow, and feature correlations.
3. **Unsupervised Machine Learning Engine**:
   - Compares **K-Means**, **Agglomerative (Hierarchical)**, and **DBSCAN** algorithms side-by-side.
   - 2D PCA (Principal Component Analysis) dimensional reduction plots using Plotly.
   - Model benchmark evaluation using **Silhouette Score**, **Davies–Bouldin Index**, and **Calinski–Harabasz Index**.
4. **RFM Behavioral Matrix**: Quantile-based Recency, Frequency, and Monetary scoring categorizing users into 6 core segments (*Champions*, *Loyal*, *Potential Loyalists*, *Need Attention*, *At Risk*, *Lost*).
5. **Predictive 3-Year CLV Analytics**: Mathematical lifetime value calculation tiered into Platinum 💎, Gold 🥇, Silver 🥈, and Bronze 🥉 valuation tiers.
6. **Automated AI Recommendation Engine**: Persona-level marketing playbooks providing activation channels, action tactics, and expected financial impact.
7. **Real-Time Dynamic Filter Engine**: Zero metric hallucination — every KPI, distribution chart, persona metric, and insight recalculates live on filter state changes.
8. **Executive Export Center**: Instant multi-format exports:
   -  **PDF Executive Briefing** (Built dynamically via ReportLab)
   -  **Filtered CSV Data**
   -  **Power BI Integration Package** (Structured JSON schema)
   -  **Visual Assets ZIP** (Interactive Plotly HTML charts)
   -  **Executive Presentation Deck** (Standalone HTML slide deck)
  
   - ## Executive Intelligence (KPIs & Loyalty Score)
   -  How it works under the hood:
KPIs recalculate live whenever filters change.
Loyalty Score Formula:
Loyalty Score
=
(
0.40
×
Repeat Purchase Density
)
+
(
0.30
×
Active Recency Rate
)
+
(
0.30
×
Margin Rate
)
Loyalty Score=(0.40×Repeat Purchase Density)+(0.30×Active Recency Rate)+(0.30×Margin Rate)

## Customer Lifetime Value
How it works under the hood: Calculated in src/clv.py using historical monthly spending velocity, a 65% gross profit margin factor, and an 82% annual retention rate:
Predicted 3-Yr CLV
=
Monthly Spend Velocity
×
36
×
0.65
×
0.82
Predicted 3-Yr CLV=Monthly Spend Velocity×36×0.65×0.82

---

##  Tech Stack

- **Core Analytics & Data Engineering**: Python, Pandas, NumPy
- **Machine Learning & Modeling**: Scikit-Learn (K-Means, AgglomerativeClustering, DBSCAN, PCA, StandardScaler)
- **Data Visualization**: Plotly Express, Plotly Graph Objects
- **Web App & UI Framework**: Streamlit, Custom HTML5/CSS3 Styling
- **Document & Asset Export**: ReportLab, Zipfile, JSON

---

##  Machine Learning Benchmark Results

| Algorithm | Silhouette Score (↑) | Davies–Bouldin Index (↓) | Calinski–Harabasz Index (↑) |
| :--- | :---: | :---: | :---: |
| **K-Means (Selected)** | **High Separation** | **Optimal Compactness** | **Strong Density** |
| Agglomerative | Comparable | Moderate | Moderate |
| DBSCAN | Noise Filtered | N/A | N/A |

---

##  Project Structure

```text
customer-segmentation-analytics/
├── assets/                  # Hero banner & background assets
│   ├── img.png              # Executive README Banner
│   └── hero_video.mp4       # Base64 background video
├── data/                    # Raw, cleaned, and segmented CSV datasets
│   ├── cosmetics_customers.csv
│   ├── cleaned_cosmetics_customers.csv
│   └── segmented_cosmetics_customers.csv
├── src/                     # Core analytical & machine learning modules
│   ├── preprocessing.py     # Data cleaning, median imputation & IQR outlier handling
│   ├── feature_engineering.py# Calculated metric pipelines & velocity features
│   ├── rfm.py               # Quantile-based RFM scoring matrix
│   ├── clv.py               # Predictive 3-year CLV calculation engine
│   ├── clustering.py        # K-Means, Agglomerative, DBSCAN & PCA engines
│   ├── insights.py          # Dynamic natural-language insight generator
│   ├── recommendations.py   # AI persona marketing playbooks
│   ├── visualization.py     # Soft luxury Plotly chart generator
│   └── pdf_generator.py     # ReportLab PDF compilation engine
├── .streamlit/              # Streamlit custom theme configuration
│   └── config.toml
├── app.py                   # Main Streamlit web application
├── config.py                # Global directory paths & hyperparameter configs
├── requirements.txt         # Python project dependencies
└── README.md                # Project documentation
