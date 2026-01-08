import pytest
import numpy as np
from src import (
    download_penguins_data,
    load_raw_data,
    preprocess_pipeline,
    create_train_test_split,
    create_mlp_model,
    compile_model,
    train_model,
    save_model
)


@pytest.fixture(scope="session", autouse=True)
def setup_test_data():
    """
        Download penguins data before running tests.
        This 'fixture' is run automatically before the test session starts
    """
    download_penguins_data()
    yield


@pytest.fixture(scope="session", autouse=True)
def setup_test_model():
    """
    Train and save a minimal model before running tests.
    Uses only 10 epochs to keep CI fast while providing a valid model.
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
    model, _ = train_model(X_train, y_train, model=model, epochs=10, verbose=0)
    
    # Save model for tests
    save_model(model)
    
    yield
