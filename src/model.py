import tensorflow as tf
import os
from pathlib import Path

from .config import MODEL_CONFIG, MODEL_FILE, NUM_SPECIES

def create_mlp_model(input_shape: int) -> tf.keras.Model:


    hidden_units = MODEL_CONFIG['hidden_units']
    activation = MODEL_CONFIG['activation']
    output_activation = MODEL_CONFIG['output_activation']
    num_classes = NUM_SPECIES
    
    # Define model architecture
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_shape,)),
        tf.keras.layers.Dense(hidden_units, activation=activation),
        tf.keras.layers.Dense(num_classes, activation=output_activation)
    ])
    
    print(f"Created MLP model:")
    print(f"  Input shape: {input_shape}")
    print(f"  Hidden layer: {hidden_units} units, {activation} activation")
    print(f"  Output layer: {num_classes} units, {output_activation} activation")
    
    return model


def compile_model(model: tf.keras.Model) -> tf.keras.Model:
    
    optimizer = MODEL_CONFIG['optimizer']    
    loss = MODEL_CONFIG['loss']    
    metrics = MODEL_CONFIG['metrics']
    
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    
    print(f"\nCompiled model:")
    print(f"  Optimizer: {optimizer}")
    print(f"  Loss: {loss}")
    print(f"  Metrics: {metrics}")
    
    return model


def get_model_summary(model: tf.keras.Model) -> None:
    model.summary()


def train_model(X_train, y_train, model: tf.keras.Model = None, epochs: int = 10, 
                validation_split: float = 0.2, verbose: int = 1) -> tuple[tf.keras.Model, tf.keras.callbacks.History]:
    if model is None:
        model = create_mlp_model(input_shape=X_train.shape[1])
        model = compile_model(model)
    
    history = model.fit(X_train, y_train, epochs=epochs, 
                       validation_split=validation_split, verbose=verbose)
    return model, history


def evaluate_model(model: tf.keras.Model, X_test, y_test) -> tuple[float, float]:
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")
    return loss, accuracy



def save_model(model: tf.keras.Model, filepath: str = MODEL_FILE) -> None:
    os.makedirs(Path(filepath).parent, exist_ok=True)
    model.save(filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath: str = MODEL_FILE) -> tf.keras.Model:
    return tf.keras.models.load_model(filepath)

