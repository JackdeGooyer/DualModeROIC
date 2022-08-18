"""Related to all code for linear electronic operations"""

# DC shot noise - dark current based
# Thernal noise - cap dependent (RC)

import math
import pandas as pd
import numpy as np
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

    out_df = current_df.copy(deep=True)

    for column in out_df:
        if column == "Wavelength_nm" or column == "Detector":
            pass

        # Perform Conversion
        time_to_charge = out_df[column] / (capacitor * threshold_voltage)
        out_df[column] = 1 / (time_to_charge + reset_time)

        # Add reset jitter # TODO Change this
        if reset_jitter is not None:
            out_df[column] = out_df[column] + (1 / reset_jitter)

    return out_df


def add_thermal_noise(
    current_df: pd.DataFrame,
    resistance: float,
    capacitance: float,
    temperature: float = 300,
) -> pd.DataFrame:
    """

    Parameters
    ----------
    current_df : pd.DataFrame
        Template dataframe
    resistance : float
        Resistance of detector
    capacitance : float
        Measuring capacitor
    temperature : float, optional
        Operating temperature, by default 300

    Returns
    -------
    pd.DataFrame
        _description_
    """
    out_df = current_df.copy(deep=True)
    thermal_voltage = np.sqrt(spc.Boltzmann * temperature / capacitance)
    thermal_current = thermal_voltage / resistance
    out_df["Thermal_Noise_A_rms"] = thermal_current
    out_df = out_df.loc[:, ["Wavelength_nm", "Detector", "Thermal_Noise_A_rms"]]
    return out_df


def add_shot_noise(
    current_df: pd.DataFrame,
    resistance: float,
    capacitance: float,
) -> pd.DataFrame:
    """Adds the shot noise to the detector

    Parameters
    ----------
    current_df : pd.DataFrame
        Current to get DC shot noise from
    resistance : float
        Detector Resistance
    capacitance : float
        Capacitance on detector

    Returns
    -------
    pd.DataFrame
        Dataframe containing shot noise per detector
    """

    total_current_per_detector = current_df.loc[
        :, ~current_df.columns.str.contains("Wavelength_nm|Detector")
    ].sum(axis=1)
    out_df = current_df.copy(deep=True)
    bandwidth = 1 / (capacitance * resistance)
    out_df["Shot_Noise_A_rms"] = np.sqrt(
        spc.elementary_charge * 2 * total_current_per_detector * bandwidth
    )
    out_df = out_df.loc[:, ["Wavelength_nm", "Detector", "Shot_Noise_A_rms"]].copy(
        deep=True
    )
    return out_df


# def add_reset_noise(
#     current_df: pd.DataFrame,
#     capacitance: float,
#     reset_voltage: float,
#     reset_time: float,
# ) -> pd.DataFrame:
#     total_current_per_detector = current_df.loc[
#         :, ~current_df.columns.str.contains("Wavelength_nm|Detector")
#     ].sum(axis=1)
#     # SNR is 1 when the reset time is equal to the smallest bit
#     # Ask peter how to do
#     return
