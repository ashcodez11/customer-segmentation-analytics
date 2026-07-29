from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ASSETS_DIR = BASE_DIR / "assets"

RAW_DATA_PATH = DATA_DIR / "cosmetics_customers.csv"
CLEAN_DATA_PATH = DATA_DIR / "cleaned_cosmetics_customers.csv"
FINAL_SEGMENTED_PATH = DATA_DIR / "segmented_cosmetics_customers.csv"

RANDOM_STATE = 42
N_CLUSTERS = 4
