"""
This module contains functions for calculating the slowing down and deflection times
of a test particle interacting with a Maxwellian background species.

Please note that we follow the equations 2.14.1 and 2.14.2 in the book:
Wesson, J. Tokamaks, Vol. 149. Oxford University Press, 2011. 
"""

import numpy as np
import scipy.special
from fast_particle_collision_time import constants

def _a_d(background_density, test_charge_state, background_charge_state, test_mass):
    """
    Calculates the A_D function (see page 63 of Wesson, 2011).

    Parameters
    ----------
    background_density : float
        The background density of the background species
        in m^(-3)
    test_charge_state : float
        The charge state of the test particle (dimensionless)
    background_charge_state : float
        The charge state of the background species (dimensionless)
    test_mass : float
        The mass of the test particle in kg

    Returns
    -------
    float
        The value of A_D in m^3 s^(-4)
    """
    return background_density * constants.ELECTRON_CHARGE**4 * test_charge_state**2 \
           * background_charge_state**2 * constants.COULOMB_LOG / \
           (2 * np.pi * constants.PERMITTIVITY_OF_FREE_SPACE**2 * test_mass**2)

def _phi(x):
    """
    Calculates the Phi function on page 62 of Wesson, 2011. Note that it is
    equivalent to the error function.
    """
    return scipy.special.erf(x)

def _phi_prime(x):
    """
    Calculates the derivative of the Phi function on page 62 of Wesson, 2011. Note that it is
    equivalent to the derivative of the error function.
    """
    return 2 * np.exp(-x**2) / np.sqrt(np.pi)

def _psi(x):
    """
    Calculates the Psi function on page 63 of Wesson, 2011.
    """
    return (_phi(x) - x * _phi_prime(x)) / (2 * x**2)

def slowing_down_time(test_speed, background_thermal_speed, test_mass, background_mass,
                      background_density, test_charge_state, background_charge_state):
    """
    Calculates the slowing down time (see page Eq. 2.14.1 of Wesson, 2011).

    Parameters
    ----------
    test_speed : float
        The speed of the test particle in m s^(-1)
    background_thermal_speed : float
        The thermal speed of the background species in m s^(-1)
    test_mass : float
        The mass of the test particle in kg
    background_mass : float
        The mass of the background species in kg
    background_density : float
        The background density of the background species
        in m^(-3)
    test_charge_state : float
        The charge state of the test particle (dimensionless)
    background_charge_state : float
        The charge state of the background species (dimensionless)

    Returns
    -------
    float
        The value of the slowing down time in s
    """
    return 2 * background_thermal_speed**2 * test_speed / \
           ((1 + test_mass / background_mass) * \
            _a_d(background_density, test_charge_state, background_charge_state, test_mass) \
            * _psi(test_speed / (np.sqrt(2) * background_thermal_speed)))

def deflection_time(test_speed, background_thermal_speed, test_mass,
                    background_density, test_charge_state, background_charge_state):
    """
    Calculates the deflection time (see page Eq. 2.14.2 of Wesson, 2011).

    Parameters
    ----------
    test_speed : float
        The speed of the test particle in m s^(-1)
    background_thermal_speed : float
        The thermal speed of the background species in m s^(-1)
    test_mass : float
        The mass of the test particle in kg
    background_mass : float
        The mass of the background species in kg
    background_density : float
        The background density of the background species
        in m^(-3)
    test_charge_state : float
        The charge state of the test particle (dimensionless)
    background_charge_state : float
        The charge state of the background species (dimensionless)

    Returns
    -------
    float
        The value of the deflection time in s
    """
    return test_speed**3 / \
           (_a_d(background_density, test_charge_state, background_charge_state, test_mass) * \
            (_phi(test_speed / (np.sqrt(2) * background_thermal_speed)) - \
             _psi(test_speed / (np.sqrt(2) * background_thermal_speed))))
