import os
from pathlib import Path

# Base directory is workspace root (where src/ is located)
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
EXTERNAL_DATA_DIR = BASE_DIR / "data" / "external"

# Model directory
MODELS_DIR = BASE_DIR / "models"

PENGUINS_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"
RAW_DATA_FILE = str(RAW_DATA_DIR / "penguins.csv")
PROCESSED_DATA_FILE = str(PROCESSED_DATA_DIR / "penguins_cleaned.csv")
TRAIN_TEST_FILE = str(PROCESSED_DATA_DIR / "penguins_train_test.pkl")

NUMERICAL_FEATURES = [
    'bill_length_mm',
    'bill_depth_mm',
    'flipper_length_mm',
    'body_mass_g'
]

NUM_SPECIES = 3

SCALER_FILE = str(MODELS_DIR / "scaler.pkl")

# Model configuration
MODEL_CONFIG = {
    'hidden_units': 32,
    'activation': 'relu',
    'output_activation': 'softmax',
    'optimizer': 'adam',
    'loss': 'sparse_categorical_crossentropy',
    'metrics': ['accuracy']
}

# Training configuration
TRAIN_CONFIG = {
    'test_size': 0.2,
    'validation_split': 0.2,
    'epochs': 10,
    'batch_size': 32,
    'random_state': 42,
    'verbose': 0
}

# Model artifact
MODEL_FILE = str(MODELS_DIR / "penguin_classifier.keras")