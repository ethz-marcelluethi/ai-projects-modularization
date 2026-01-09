"""
Simple unit tests demonstrating testing principles for data preprocessing.

Educational example showing:
- Function behavior testing (checking transformations work correctly)
- Data integrity testing (ensuring no data loss during cleaning)
- Uses synthetic data for testing to avoid dependency on raw data files
"""

import pandas as pd
import numpy as np
import pytest

from src import clean_data, encode_features


@pytest.fixture
def penguin_data():
    """Create synthetic penguin data with missing values for testing."""
    data = {
        'species': ['Adelie', 'Gentoo', 'Chinstrap', 'Adelie', 'Gentoo', 'Chinstrap'],
        'island': ['Torgersen', 'Biscoe', 'Dream', 'Torgersen', 'Biscoe', 'Dream'],
        'bill_length_mm': [39.1, np.nan, 48.7, 38.9, 47.3, 49.2],
        'bill_depth_mm': [18.7, 14.5, 15.3, 17.8, 14.2, 15.8],
        'flipper_length_mm': [181.0, 217.0, 193.0, 185.0, 220.0, 195.0],
        'body_mass_g': [3750.0, 5500.0, 3800.0, np.nan, 5400.0, 3900.0],
        'sex': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female']
    }
    
    return pd.DataFrame(data)


@pytest.mark.unit
def test_clean_data_removes_missing_values(penguin_data):
    
    # Verify synthetic data has missing values
    assert penguin_data.isnull().any().any(), "Synthetic data should contain NaN values for testing"
    
    cleaned = clean_data(penguin_data)
        
    assert not cleaned.isnull().any().any(), "Cleaned data should not contain NaN values"


@pytest.mark.unit
def test_encode_features_creates_species_mapping(penguin_data):
    raw_data = penguin_data
    cleaned_data = clean_data(raw_data)
    
    encoded = encode_features(cleaned_data)
            
    unique_values = encoded['species'].unique()
    for val in unique_values:
        assert val in [0, 1, 2], f"Species should be encoded as 0, 1, or 2, got {val}"
