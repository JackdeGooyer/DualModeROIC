# Related to all code for the linear section

# DC shot noise - dark current based
# Thernal noise - cap dependent (RC)

import math
import pandas as pd
import scipy.constants as spc


def amp_to_frequency(
    current_df: pd.DataFrame,
    capacitor: float,
    threshold_voltage: float,
    reset_time: float,
    reset_jitter: float = None,
) -> pd.DataFrame:
    """Turns current to frequency

    Parameters
    ----------
    current_df : pd.DataFrame
        Dataframe containing currents to convert
    capacitor : float
        Size of Capacitor
    threshold_voltage : float
        Capacitor reset threshold
    reset_time : float
        Time to reset capacitor
    reset_jitter : float, optional
        Jitter on reset, by default None
        If this is set, then it will be added, ussualy needed for noise

    Returns
    -------
    pd.DataFrame
        Dataframe containing output frequencies
    """

    df = current_df.copy(deep=True)

    for column in df:
        if column == "Wavelength_nm" or column == "Detector":
            pass

        # Perform Conversion
        time_to_charge = df[column] / (capacitor * threshold_voltage)
        df[column] = 1 / (time_to_charge + reset_time)

        # Add reset jitter # TODO Check this
        if reset_jitter is not None:
            df[column] = df[column] + 1 / reset_jitter

    return df


def linear_noise(
    dc_current_df: pd.DataFrame,
    capacitance: pd.DataFrame,
    detector_resistance: float = None,
    temperature: float = 300,
) -> pd.DataFrame:
    """Compute the noise of the linear system

    Parameters
    ----------
    dc_current_df : pd.DataFrame
        Noise currents to have noises added to
    capacitance : pd.DataFrame
        Capacitance of frequency converter cap
    detector_resistance : float
        rd of detector, if measured, by default None
    temperature : float, optional
        Temperature of detector, by default 300

    Returns
    -------
    pd.DataFrame
        Dataframe with noise currents
    """

    df = dc_current_df.copy(deep=True)

    # Thermal Shot Noise = kT/C
    thermal_current_noise = spc.Boltzmann * temperature / capacitance

    # DC Shot Noise  -- get bandwidth (RC?) - is it by the diode resistance + cap?
    for column in df.columns:
        if column == "Wavelength_nm" or column == "Detector":
            pass
        if detector_resistance is None:
            detector_resistance = (
                spc.Boltzmann
                * temperature
                / (dc_current_df[column] * spc.elementary_charge)
            )

        # Add Thermal Noise
        df[column] = thermal_current_noise

        # Add Shot Noise
        bandwidth = 1 / (capacitance * detector_resistance)  # TODO fix this
        diode_shot_noise = math.sqrt(
            spc.elementary_charge * 2 * dc_current_df["column"]
        ) * (bandwidth)

        # Save Result
        df[column] = df[column] + diode_shot_noise

    return df
