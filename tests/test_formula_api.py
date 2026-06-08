import tam as ta

def test_complex_formula_parsing():
    """Ensures the factory builds the correct PyTorch classes from the string."""
    formula = "load ~ s(temperature, k=10) + f(tod, m=4) + te(s(lat), s(lon)) + n(img)"
    model = ta.StaticTAM(formula=formula)
    
    # Check that the target was isolated
    assert model.target_col_ == "load"
    
    # Extract the string names or classes of the built effects
    effect_types = [type(effect).__name__ for effect in model.effects_list_]
    
    # Assert the spectrum was correctly translated
    assert "SplineEffect" in effect_types
    assert "FourierEffect" in effect_types
    assert "TensorProductEffect" in effect_types
    assert "NeuralEffect" in effect_types # Or NEPT / whatever the class is named