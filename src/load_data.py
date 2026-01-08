"""Functions for loading data."""
import pandas as pd
from pathlib import Path
import os

from .config import PENGUINS_URL, RAW_DATA_FILE, PROCESSED_DATA_FILE


def download_penguins_data(url: str = PENGUINS_URL, 
                           output_path: str = RAW_DATA_FILE) -> pd.DataFrame:
    print(f"Downloading data from {url}")
    df = pd.read_csv(url)
    
    os.makedirs(Path(output_path).parent, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")
    
    return df


def load_raw_data(path: str = RAW_DATA_FILE) -> pd.DataFrame:
    return pd.read_csv(path)


def load_processed_data(path: str = PROCESSED_DATA_FILE) -> pd.DataFrame:
    return pd.read_csv(path)

