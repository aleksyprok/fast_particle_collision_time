"""
This module contains functions to compute the DT reactivity and
reaction rate.
"""
import numpy as np

def reactivity(temp_ions):
    """
    This function computes the DT reactivity. Using this paper:
    H.-S. Bosch and G.M. Hale 1992 Nucl. Fusion 32 611

    Input:
      - temp_ions: temperature of the ions in keV

    Output:
      - reactivity: DT reactivity in cm^3/s
    """

    gamov_constant = 34.382  # in sqrt(kev)
    mrc2 = 1124656  # in keV
    const_1 = 1.17302e-9
    const_2 = 1.51361e-2
    const_3 = 7.51886e-2
    const_4 = 4.60643e-3
    const_5 = 1.35000e-2
    const_6 = -1.06750e-4
    const_7 = 1.36600e-5

    # Calculate theta
    denominator = -temp_ions * (const_2 +
                                temp_ions * (const_4 +
                                             temp_ions * const_6))
    denominator /= 1 + temp_ions * (const_3 +
                                    temp_ions * (const_5 +
                                                 temp_ions * const_7))
    denominator += 1
    theta = temp_ions / denominator
    xi_greek = (0.25 * gamov_constant**2 / theta)**(1/3)
    sigma_v = const_1 * theta * np.sqrt(xi_greek / (mrc2 * temp_ions**3))
    sigma_v *= np.exp(-3 * xi_greek)
    return sigma_v

