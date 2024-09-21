"""
Tests for the constants module.
"""

import fast_particle_collision_time.constants as constants

def test_electron_charge():
    """Test the value of the electron charge constant."""
    expected_value = 1.60217662e-19
    assert constants.ELECTRON_CHARGE == expected_value, "Electron charge value is incorrect"
