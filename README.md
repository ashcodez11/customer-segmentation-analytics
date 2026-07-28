# Lumière AI – Soft Luxury Customer Intelligence & Analytics Platform 💄✨

An end-to-end Customer Segmentation and Customer Intelligence web application built for luxury cosmetics and fashion e-commerce. Features unsupervised machine learning (K-Means, Hierarchical, DBSCAN), RFM behavioral scoring, 3-Year Customer Lifetime Value (CLV) predictive modeling, automated natural-language insight generation, and executive PDF briefing exports.

---

## 🌟 Key Features

1. **Luxury Editorial UI/UX**: Designed as a single-page long-scrolling platform with an embedded video hero section, Cormorant Garamond typography, and soft luxury pastel styling.
2. **Data Storytelling & EDA**: Visual distributions for age, annual income, spending scores, purchase frequency, revenue flow, and correlation heatmaps.
3. **Unsupervised ML Clustering**:
   - **K-Means**, **Agglomerative (Hierarchical)**, and **DBSCAN** algorithms.
   - 2D & 3D PCA dimensional reduction plots using Plotly.
   - Evaluated via **Silhouette Score**, **Davies-Bouldin Index**, and **Calinski-Harabasz Index**.
4. **RFM Behavioral Matrix**: Quantile-based Recency, Frequency, and Monetary scoring assigning customers to 8 behavioral segments.
5. **3-Year CLV Predictive Analytics**: Historical spend and predictive 3-year valuation tiered into Platinum 💎, Gold 🥇, Silver 🥈, and Bronze 🥉 categories.
6. **Automated AI Marketing Recommendation Engine**: Persona-level marketing playbooks with activation directives and expected revenue impacts.
7. **Executive PDF Export**: Automated PDF briefing generation using FPDF2.

---

## 🛠️ Tech Stack

- **Language**: Python 3.10+
- **Data Engineering**: Pandas, NumPy
- **Machine Learning**: Scikit-Learn, SciPy
- **Data Visualization**: Plotly Express, Plotly Graph Objects
- **Web Framework**: Streamlit
- **PDF Generation**: FPDF2

---

## 🚀 How to Run Locally

1. **Clone Repository**:
   git clone https://github.com/your-username/customer-segmentation.git
   cd customer-segmentation

2. **Set Up Environment**:
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt

3. **Launch Platform**:
   streamlit run app.py
