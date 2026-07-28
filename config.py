import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
EXPORTS_DIR = BASE_DIR / "exports"
REPORTS_DIR = BASE_DIR / "reports"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
EXPORTS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

RAW_DATA_PATH = DATA_DIR / "raw_customer_data.csv"
CLEANED_DATA_PATH = DATA_DIR / "cleaned_customer_data.csv"
FEATURE_DATA_PATH = DATA_DIR / "featured_customer_data.csv"
FINAL_SEGMENTED_PATH = DATA_DIR / "segmented_customer_data.csv"

# Color Palette for Dashboards and Charts
COLOR_PALETTE = {
    'primary': '#1E88E5',
    'secondary': '#FFC107',
    'accent': '#D81B60',
    'background': '#F4F6F9',
    'text': '#212121',
    'clusters': ['#2A9D8F', '#E76F51', '#F4A261', '#E9C46A', '#457B9D', '#1D3557']
}

# General Settings
RANDOM_STATE = 42
DEFAULT_N_CLUSTERS = 4
