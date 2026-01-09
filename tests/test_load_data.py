"""
Simple unit tests demonstrating testing principles for data loading.

Educational example showing:
- Data validation (checking expected columns)
- Type checking (ensuring correct data types)
"""

import pandas as pd
import pytest

from src import load_raw_data


@pytest.mark.unit
def test_data_has_required_columns():
    """
    DATA VALIDATION: Verify loaded data has all required columns.
    
    This demonstrates a basic data validation test - ensuring the 
    dataset structure matches our expectations.
    """
    df = load_raw_data()
    
    required_columns = ['species', 'island', 'bill_length_mm', 
                      'bill_depth_mm', 'flipper_length_mm', 
                      'body_mass_g', 'sex']
    
    for col in required_columns:
        assert col in df.columns, f"Missing required column: {col}"


@pytest.mark.unit
def test_body_mass_in_valid_range():
    """
    DATA VALIDATION: Verify body mass values are positive and realistic.
    
    This demonstrates range validation - ensuring measurements fall 
    within biologically plausible ranges for penguins.
    """
    df = load_raw_data()
    
    body_mass = df['body_mass_g'].dropna()
    
    assert (body_mass >= 2000).all(), "Body mass should be at least 2000g"
    assert (body_mass <= 6500).all(), "Body mass should not exceed 6500g"
