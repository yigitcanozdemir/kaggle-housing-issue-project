# ======================================================
# 1. Import library
# ======================================================

import kaggle
import os
from dotenv import find_dotenv, load_dotenv

# ======================================================
# 2. Find .env file with dotenv and import credentials
# ======================================================

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
os.environ["KAGGLE_USERNAME"] = os.getenv("KAGGLE_USERNAME")
os.environ["KAGGLE_KEY"] = os.getenv("KAGGLE_KEY")

# ======================================================
# 3. Download dataset from Kaggle
# ======================================================

kaggle.api.dataset_download_files(
    "yasserh/housing-prices-dataset", path="../../data/raw/", unzip=True
)
