"""
Simple unit tests demonstrating testing principles for data preprocessing.

Educational example showing:
- Function behavior testing (checking transformations work correctly)
- Data integrity testing (ensuring no data loss during cleaning)
"""

import unittest
import pandas as pd

from preprocess import clean_data, encode_features
from load_data import load_raw_data


class TestPreprocessing(unittest.TestCase):

    
    def test_clean_data_removes_missing_values(self):


        raw_data = load_raw_data()    
        cleaned = clean_data(raw_data)
            
        self.assertFalse(cleaned.isnull().any().any(), 
                        "Cleaned data should not contain NaN values")
        
    
    def test_encode_features_creates_species_mapping(self):

        raw_data = load_raw_data()
        cleaned_data = clean_data(raw_data)
        
        encoded = encode_features(cleaned_data)
                
        unique_values = encoded['species'].unique()
        for val in unique_values:
            self.assertIn(val, [0, 1, 2],
                        f"Species should be encoded as 0, 1, or 2, got {val}")


if __name__ == '__main__':
    unittest.main()
