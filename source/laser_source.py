"""Generate Source Parameters"""

import cmath
import pandas as pd
import numpy as np


def generate_source(
    fwhm_bw: float, central_wavelength: float, number_of_points: int, power: float
) -> pd.DataFrame:
    """Generates a gaussian beam based on input parameters

    Parameters
    ----------
    fwhm_bw : float
        Full width half max bandwidth
    central_wavelength : float
        Central wavelength of the fit
    number_of_points : int
        Number of points to have in fit
    power : float
        Total power of the laser

    Returns
    -------
    pd.DataFrame
        Dataframe containing power per wavelength of the laser
    """

    # Get Bounds of wavelengths
    low_wavelength = central_wavelength - fwhm_bw * 2
    high_wavelength = central_wavelength + fwhm_bw * 2
    wavelengths = np.linspace(low_wavelength, high_wavelength, number_of_points)
    wavelengths_factor = (high_wavelength - low_wavelength) / number_of_points

    # Perform fitting
    c_var = fwhm_bw / (2.35482)
    a_var = power * wavelengths_factor / (np.sqrt(2) * c_var * np.sqrt(cmath.pi))
    data = a_var * np.exp(
        (-4 * ((0.6931 * (wavelengths - central_wavelength) ** 2) / (fwhm_bw**2)))
    )

    return pd.DataFrame({"Wavelength_nm": wavelengths, "Power_W": data})
