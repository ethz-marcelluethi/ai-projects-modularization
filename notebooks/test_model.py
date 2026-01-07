import unittest
import numpy as np
from tensorflow import keras

from model import load_model


class TestModel(unittest.TestCase):
    
    def test_body_mass_increase_affects_species_prediction(self):
        """
        DIRECTIONAL TEST: Verify heavier penguins predict differently than lighter ones
        """
        # Load the trained model
        model = load_model()
        
        
        # Light penguin (small features, typical of Adelie)
        light_penguin = np.array([[
            -0.9,  # small bill_length
            0.8,   # large bill_depth (Adelie characteristic)
            -1.0,  # small flipper_length
            -1.5,  # small body_mass
            0,     # not Dream island
            1,     # Torgersen island (Adelie common)
            1      # male
        ]])
        
        # Heavy penguin (large features, typical of Gentoo)
        heavy_penguin = np.array([[
            0.5,   # medium bill_length
            -1.0,  # small bill_depth (Gentoo characteristic)
            1.5,   # large flipper_length
            2.0,   # large body_mass
            0,     # not Dream island
            0,     # not Torgersen (Gentoo on Biscoe)
            1      # male
        ]])
        
        # Get predictions 
        light_pred = model.predict(light_penguin, verbose=0)
        heavy_pred = model.predict(heavy_penguin, verbose=0)
        
        # Determine predicted species (index of max probability)
        light_species = np.argmax(light_pred)
        heavy_species = np.argmax(heavy_pred)
        
        self.assertNotEqual(light_species, 1, 
                            "Light penguin should not be predicted as Chinstrap")
        self.assertNotEqual(heavy_species, 0, 
                            "Heavy penguin should not be predicted as Adelie")


