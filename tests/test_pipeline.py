import numpy as np
import pytest

from src import (
    load_raw_data,
    clean_data, encode_features, scale_features
)


@pytest.mark.integration
def test_preprocessing_pipeline_integration():
    """
    INTEGRATION TEST: Verify data preprocessing steps work together.
    """ 
    
    raw_data = load_raw_data()    
    assert len(raw_data) > 0, "Raw data should not be empty"
    
    cleaned_data = clean_data(raw_data)
    assert not cleaned_data.isnull().any().any(), "Cleaned data should have no missing values"
    assert len(cleaned_data) <= len(raw_data), "Cleaning should not add rows"
    
    encoded_data = encode_features(cleaned_data)
    assert 'species' in encoded_data.columns, "Encoded data should have species column"
    assert encoded_data['species'].dtype in [np.int32, np.int64], "Species should be numeric after encoding"
    
    # Stage 4: Scale features (fit new scaler to test integration)
    scaled_data, scaler = scale_features(encoded_data, scaler=None, fit=True)
    feature_cols = [col for col in scaled_data.columns if col.isnumeric() and col != 'species']
    for col in feature_cols:
        assert scaled_data[col].mean() != encoded_data[col].mean(), f"{col} should be transformed by scaling"
    
    X = scaled_data.drop('species', axis=1).values.astype(np.float32)
    y = scaled_data['species'].values.astype(np.int32)
    
    assert X.shape[0] == y.shape[0], "Features and labels should have same number of samples"
    assert X.shape[1] > 0, "Should have at least one feature"
    assert len(np.unique(y)) > 1, "Should have multiple species classes"