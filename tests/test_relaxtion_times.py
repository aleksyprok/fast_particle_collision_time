"""
Tests for the relaxation_times module.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pytest
from fast_particle_collision_time import constants, relaxation_times

def test_slowing_down_time_alpha_vs_electrons():
    """
    Test slowing down time for an alpha particle in an electron background.
    The electrons are assumed to have a temperature of 10 keV and density of 10^20 m^-3.
    We compare the result of the relaxation_times.slowing_down_time to the approximation
    given in Eq. 5.4.3 for speeds up to 3.5 MeV.
    """
    # Parameters
    electron_temperature_ev = 10e3  # 10 keV electron temperature
    electron_temperature_j = electron_temperature_ev * constants.ELECTRON_CHARGE # Joules
    electron_density = 1e20  # m^-3
    alpha_mass = constants.ALPHA_ION_MASS  # kg
    electron_mass = constants.ELECTRON_MASS  # kg
    alpha_charge_state = 2  # charge state of alpha
    electron_charge_state = 1  # charge state of electron

    # Thermal speed of background electrons
    electron_thermal_speed = np.sqrt(electron_temperature_j / electron_mass)

    # Speeds of the alpha particle from 0 to 3.5 MeV
    alpha_energies_mev = np.linspace(0.01, 3.5, 100)  # MeV
    alpha_energies_joules = alpha_energies_mev * 1e6 * constants.ELECTRON_CHARGE
    alpha_speeds = np.sqrt(alpha_energies_joules / alpha_mass)  # m/s

    def slowing_down_time_approx(t_e, n_e, m_e, m_alpha, z_e, z_alpha):
        """
        Analytical approximation for the slowing down time given in Eq. 5.4.3.
        in Wesson (2011).
        """
        def a_d(n_e, z_e, z_alpha, m_alpha):
            return n_e * constants.ELECTRON_CHARGE**4 * constants.COULOMB_LOG \
                   * z_e**2 * z_alpha**2 / \
                   (2 * np.pi * constants.PERMITTIVITY_OF_FREE_SPACE**2 * m_alpha**2)

        return 3 * np.sqrt(2 * np.pi * t_e**3) / \
            (np.sqrt(m_e) * m_alpha * a_d(n_e, z_e, z_alpha, m_alpha))

    # Values for the slowing down time
    tau_slowing_down_times = []
    tau_slowing_down_approx = []

    # Calculate the slowing down time for each alpha speed
    for v_alpha in alpha_speeds:
        tau_s = relaxation_times.slowing_down_time(v_alpha, electron_thermal_speed, alpha_mass,
                                                   electron_mass, electron_density,
                                                   alpha_charge_state, electron_charge_state)
        tau_approx = slowing_down_time_approx(electron_temperature_j, electron_density,
                                              electron_mass, alpha_mass,
                                              electron_charge_state, alpha_charge_state)

        tau_slowing_down_times.append(tau_s)
        tau_slowing_down_approx.append(tau_approx)

    # Check the specific case where energy is 3.5 MeV
    tau_s_3_5_mev = tau_slowing_down_times[-1]
    tau_approx_3_5_mev = tau_slowing_down_approx[-1]

    # Ensure the values are approximately equal within a tolerance
    assert tau_s_3_5_mev == pytest.approx(tau_approx_3_5_mev, rel=0.1), \
        f"Slowing down time at 3.5 MeV differs: tau_s = {tau_s_3_5_mev}," \
        f" tau_approx = {tau_approx_3_5_mev}"

    # Plot the results for visual inspection
    fig, ax = plt.subplots()
    ax.plot(alpha_energies_mev, tau_slowing_down_times,
            label="Numerical slowing down time (Eq. 2.14.1)")
    ax.plot(alpha_energies_mev, tau_slowing_down_approx,
            label="Analytical approximation (Eq. 5.4.3)", linestyle="--")
    ax.set_xlabel("Alpha Energy (MeV)")
    ax.set_ylabel("Slowing Down Time (s)")
    ax.set_title("Slowing Down Time of Alpha Particle in Electron Background (10 keV, 10^20 m^-3)")
    ax.legend()
    ax.grid(True)

    # Save the plot for inspection
    test_dir = os.path.dirname(os.path.abspath(__file__))
    test_plots_dir = os.path.join(test_dir, "test_plots")
    os.makedirs(test_plots_dir, exist_ok=True)
    fig.savefig(os.path.join(test_plots_dir, "slowing_down_time_alpha_vs_electrons.png"),
                dpi=300, bbox_inches="tight")

if __name__ == "__main__":
    test_slowing_down_time_alpha_vs_electrons()
