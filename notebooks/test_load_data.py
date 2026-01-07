"""
Simple unit tests demonstrating testing principles for data loading.

Educational example showing:
- Data validation (checking expected columns)
- Type checking (ensuring correct data types)
"""

import unittest
import pandas as pd

from load_data import load_raw_data


class TestDataLoading(unittest.TestCase):
    """Simple test cases for data loading."""
    
    def test_data_has_required_columns(self):
        """
        DATA VALIDATION: Verify loaded data has all required columns.
        
        This demonstrates a basic data validation test - ensuring the 
        dataset structure matches our expectations.
        """
        # Load the actual raw data file
        df = load_raw_data()
        
        required_columns = ['species', 'island', 'bill_length_mm', 
                          'bill_depth_mm', 'flipper_length_mm', 
                          'body_mass_g', 'sex']
        
        for col in required_columns:
            self.assertIn(col, df.columns, 
                        f"Missing required column: {col}")

    def test_body_mass_in_valid_range(self):
        """
        DATA VALIDATION: Verify body mass values are positive and realistic.
        
        This demonstrates range validation - ensuring measurements fall 
        within biologically plausible ranges for penguins.
        """
        # Load the actual raw data file
        df = load_raw_data()
        
        # Remove NaN values for this check
        body_mass = df['body_mass_g'].dropna()
        
        # All values should be positive
        self.assertTrue((body_mass > 0).all(),
                       "All body mass values should be positive")
        
        # Penguins typically weigh between 2000g and 6500g
        # This checks for data entry errors or outliers
        self.assertTrue((body_mass >= 2000).all(),
                       "Body mass should be at least 2000g")
        self.assertTrue((body_mass <= 6500).all(),
                       "Body mass should not exceed 6500g")


if __name__ == '__main__':
    unittest.main()
