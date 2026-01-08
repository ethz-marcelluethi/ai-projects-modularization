"""
Simple unit tests demonstrating testing principles for data preprocessing.

Educational example showing:
- Function behavior testing (checking transformations work correctly)
- Data integrity testing (ensuring no data loss during cleaning)
"""

import pandas as pd

from src import clean_data, encode_features, load_raw_data


def test_clean_data_removes_missing_values():
    raw_data = load_raw_data()    
    cleaned = clean_data(raw_data)
        
    assert not cleaned.isnull().any().any(), "Cleaned data should not contain NaN values"


def test_encode_features_creates_species_mapping():
    raw_data = load_raw_data()
    cleaned_data = clean_data(raw_data)
    
    encoded = encode_features(cleaned_data)
            
    unique_values = encoded['species'].unique()
    for val in unique_values:
        assert val in [0, 1, 2], f"Species should be encoded as 0, 1, or 2, got {val}"
