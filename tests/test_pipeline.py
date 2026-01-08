import numpy as np

from src import (
    load_raw_data, load_processed_data,
    clean_data, encode_features, scale_features, load_scaler,
    load_model, evaluate_model
)


def test_complete_pipeline_from_raw_to_predictions():
    """
    INTEGRATION TEST: Verify the complete pipeline works end-to-end.
    
    This demonstrates integration testing - processing raw data through
    all stages (clean, encode, scale) and evaluating with the trained model.
    """
    raw_data = load_raw_data()
    assert raw_data is not None, "Raw data should load successfully"
    
    cleaned_data = clean_data(raw_data)
    assert not cleaned_data.isnull().any().any(), "Cleaned data should have no missing values"
    
    encoded_data = encode_features(cleaned_data)
    assert 'species' in encoded_data.columns, "Encoded data should have species column"
    
    scaler = load_scaler()
    scaled_data, _ = scale_features(encoded_data, scaler=scaler, fit=False)
    
    X = scaled_data.drop('species', axis=1).values.astype(np.float32)
    y = scaled_data['species'].values.astype(np.int32)
    
    model = load_model()
    assert model is not None, "Model should load successfully"

    loss, accuracy = evaluate_model(model, X, y)
    
    assert accuracy > 0.85, "Pipeline should achieve >85% accuracy on processed data"
    
    assert loss < 1.0, "Loss should be reasonably low"