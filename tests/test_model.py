import numpy as np
import pytest
from tensorflow import keras

from src import (
    load_model,
    load_train_test_split,
    evaluate_model,
    load_raw_data,
    preprocess_pipeline,
    create_train_test_split,
    create_mlp_model,
    compile_model,
    train_model,
    save_model
)


@pytest.fixture(scope="session")
def setup_test_model():
    """
    Train and save a minimal model before running model evaluation tests.    
    """
    # Load and preprocess data
    raw_data = load_raw_data()
    processed_data, scaler = preprocess_pipeline(raw_data, save_output=True)
    
    # Create train/test split
    splits = create_train_test_split(processed_data, save_output=True)
    
    # Prepare training data
    X_train = splits['X_train'].values.astype(np.float32)
    y_train = splits['y_train'].values.astype(np.int32)
    
    # Create and train model with minimal epochs
    model = create_mlp_model(input_shape=X_train.shape[1])
    model = compile_model(model)
    model, _ = train_model(X_train, y_train, model=model, epochs=50, verbose=0)
    
    # Save model for tests
    save_model(model)
    yield


@pytest.mark.slow
def test_model_evaluation_meets_performance_threshold(setup_test_model):
    """
    MODEL EVALUATION: Verify trained model meets minimum performance standards.
        """
    # Load the saved model and test data
    model = load_model()
    splits = load_train_test_split()
    
    X_test = splits['X_test'].values.astype(np.float32)
    y_test = splits['y_test'].values.astype(np.int32)
    
    loss, accuracy = evaluate_model(model, X_test, y_test)
    
    assert accuracy > 0.85, f"Model accuracy {accuracy:.4f} below threshold of 0.8"
    assert loss < 1.0, f"Model loss {loss:.4f} exceeds threshold of 1.0"


@pytest.mark.slow
def test_body_mass_increase_affects_species_prediction(setup_test_model):
    """
    DIRECTIONAL TEST: Verify heavier penguins predict differently than lighter ones
    """
    model = load_model()
    
    light_penguin = np.array([[
        -0.9,
        0.8,
        -1.0,
        -1.5,
        0,
        1,
        1
    ]])
    
    heavy_penguin = np.array([[
        0.5,
        -1.0,
        1.5,
        2.0,
        0,
        0,
        1
    ]])
    
    light_pred = model.predict(light_penguin, verbose=0)
    heavy_pred = model.predict(heavy_penguin, verbose=0)
    
    light_species = np.argmax(light_pred)
    heavy_species = np.argmax(heavy_pred)
    
    assert light_species != 1, "Light penguin should not be predicted as Chinstrap"
    assert heavy_species != 0, "Heavy penguin should not be predicted as Adelie"


