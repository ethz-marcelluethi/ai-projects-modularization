"""Data preprocessing functions."""
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
from pathlib import Path

from config import (
    NUMERICAL_FEATURES, PROCESSED_DATA_FILE, TRAIN_TEST_FILE, TRAIN_CONFIG, SCALER_FILE
)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy()    
    df_clean.dropna(inplace=True)
    return df_clean


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    df_encoded = df.copy()
    
    df_encoded = pd.get_dummies(df_encoded, columns=['island', 'sex'], drop_first=True)

    species_map = {'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2}
    df_encoded['species'] = df_encoded['species'].map(species_map)
    return df_encoded



def scale_features(df: pd.DataFrame, scaler: StandardScaler = None, fit: bool = True) -> tuple[pd.DataFrame, StandardScaler]:

    df_scaled = df.copy()
    
    if fit:
        # Fit a new scaler
        scaler = StandardScaler()
        scaler.fit(df_scaled[NUMERICAL_FEATURES])
        
        # Save the scaler when fitting
        os.makedirs(Path(SCALER_FILE).parent, exist_ok=True)
        with open(SCALER_FILE, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"Scaler saved to {SCALER_FILE}")
    else:
        # Use provided scaler
        if scaler is None:
            raise ValueError("Must provide scaler when fit=False")
    
    df_scaled[NUMERICAL_FEATURES] = scaler.transform(df_scaled[NUMERICAL_FEATURES])
    return df_scaled, scaler


def preprocess_pipeline(df: pd.DataFrame, save_output: bool = True) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Run the full preprocessing pipeline
    """
    print("Starting preprocessing pipeline...")

    df_clean = clean_data(df)
    df_encoded = encode_features(df_clean)
    df_processed, scaler = scale_features(df_encoded)
    if save_output:
        os.makedirs(Path(PROCESSED_DATA_FILE).parent, exist_ok=True)
        df_processed.to_csv(PROCESSED_DATA_FILE, index=False)
        
        # we also need to save the scaler
        os.makedirs(Path(SCALER_FILE).parent, exist_ok=True)        
        with open(SCALER_FILE, 'wb') as f:
            pickle.dump(scaler, f)
    return df_processed, scaler 

def load_scaler(path: str = SCALER_FILE) -> StandardScaler:
 
    with open(path, 'rb') as f:
        scaler = pickle.load(f)
    
    print(f"Loaded scaler from {path}")
    return scaler


def create_train_test_split(df: pd.DataFrame,
                            test_size: float | None = None,
                            random_state: int | None = None,
                            save_output: bool = True) -> dict:
    if test_size is None:
        test_size = TRAIN_CONFIG['test_size']
    if random_state is None:
        random_state = TRAIN_CONFIG['random_state']
    
    # Separate features and target
    y = df['species']
    X = df.drop('species', axis=1)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state
    )
        
    splits = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }
    
    os.makedirs(Path(TRAIN_TEST_FILE).parent, exist_ok=True)
    with open(TRAIN_TEST_FILE, 'wb') as f:
        pickle.dump(splits, f)
    print(f"\nSaved train/test splits to {TRAIN_TEST_FILE}")
    
    return splits


def load_train_test_split(path: str = TRAIN_TEST_FILE) -> dict:
    with open(path, 'rb') as f:
        splits = pickle.load(f)
    
    print(f"Loaded train/test splits from {path}")
    
    return splits


