import numpy as np
from tensorflow import keras

from notebooks import load_model


def test_body_mass_increase_affects_species_prediction():
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


