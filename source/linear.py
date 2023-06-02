"""Related to all code for linear electronic operations"""

# DC shot noise - dark current based
# Thernal noise - cap dependent (RC)

import copy
from typing import Dict, Union
import pandas as pd
import numpy as np
import scipy.constants as spc
import constants as c


def add_thermal_noise(
    current_df: pd.DataFrame,
    capacitance: float,
    temperature: float = 300,
) -> pd.DataFrame:
    """

    Parameters
    ----------
    current_df : pd.DataFrame
        Template dataframe
    capacitance : float
        Measuring capacitor
    temperature : float, optional
        Operating temperature, by default 300

    Returns
    -------
    pd.DataFrame
        Dataframe contianing thermal voltage noise
    """
    out_df = current_df.copy(deep=True)
    out_df["Thermal_Noise_V_rms"] = np.sqrt(spc.Boltzmann * temperature / capacitance)
    out_df = out_df.loc[:, ["Wavelength_nm", "Detector", "Thermal_Noise_V_rms"]]
    return out_df


def add_amplifier_noise(
    current_df: pd.DataFrame,
    bandwidth: float,
    default_noise: float,
) -> pd.DataFrame:
    """

    Parameters
    ----------
    current_df : pd.DataFrame
        Template dataframe
    bandwidth : float
        Bandwidth of device
    default_noise: float
        Voltage noise on interface (V/sqrt(Hz))

    Returns
    -------
    pd.DataFrame
        Dataframe contianing amplifier thermal voltage noise
    """
    out_df = current_df.copy(deep=True)
    out_df["Amplifier_Noise_V_rms"] = np.sqrt(bandwidth) * default_noise
    out_df = out_df.loc[:, ["Wavelength_nm", "Detector", "Amplifier_Noise_V_rms"]]
    return out_df


def add_shot_noise(
    current_df: pd.DataFrame,
    resistance: float,
    capacitance: float,
    override_bandwidth: float = None,
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
    override_bandwidth : float, by default None
        Banndwidth of system input

    Returns
    -------
    pd.DataFrame
        Dataframe containing shot noise per detector
    """

    # Determine Bandwidth
    if override_bandwidth is None:
        bandwidth = 1 / (2 * np.pi * capacitance * resistance)
    else:
        bandwidth = override_bandwidth

    total_current_per_detector = get_total_current(current_df)
    out_df = current_df.copy(deep=True)
    out_df["Shot_Noise_A_rms"] = np.sqrt(
        spc.elementary_charge * 2 * total_current_per_detector * bandwidth
    )
    out_df = out_df.loc[:, ["Wavelength_nm", "Detector", "Shot_Noise_A_rms"]].copy(
        deep=True
    )
    return out_df


def amp_to_period(
    current_df: pd.DataFrame,
    capacitor: float,
    threshold_voltage: float,
) -> pd.Series:
    """Turns current to period

    Parameters
    ----------
    current_df : pd.DataFrame
        Dataframe containing currents to convert
    capacitor : float
        Size of Capacitor
    threshold_voltage : float
        Capacitor reset threshold

    Returns
    -------
    pd.Series
        Dataframe containing output period
    """
    return copy.deepcopy(
        (capacitor * threshold_voltage) / get_total_current(current_df)
    )


def voltage_noise_to_jitter(
    noise: Union[pd.DataFrame, c.RecursiveDict, None],
    current_df: pd.DataFrame,
    capacitor: float,
) -> Union[pd.DataFrame, c.RecursiveDict, None]:
    """Converts voltage noise on capacitor to pulse jitter

    Parameters
    ----------
    noise_df : Union[pd.DataFrame, c.RecursiveDict, None]
        Voltage noise to convert
    current_df : pd.DataFrame
        Dataframe containing DC signal current
    capacitor : float
        Integrating capacitor

    Returns
    -------
    Union[pd.DataFrame, c.RecursiveDict, None]
        Structure containing jitter
    """
    out_noise = copy.deepcopy(noise)
    out_noise = _voltage_noise_to_jitter(out_noise, current_df, capacitor)
    return out_noise


def _voltage_noise_to_jitter(
    noise: Union[pd.DataFrame, c.RecursiveDict, None],
    current_df: pd.DataFrame,
    capacitor: float,
) -> pd.DataFrame:
    """Recursive function to get jitter from noise

    Parameters
    ----------
    noise_df : Union[pd.DataFrame, c.RecursiveDict, None]
        Voltage noise to convert
    current_df : pd.DataFrame
        Dataframe containing DC signal current
    capacitor : float
        Integrating capacitor

    Returns
    -------
    Union[pd.DataFrame, c.RecursiveDict, None]
        Structure containing jitter
    """

    # Check for type
    if isinstance(noise, Dict):
        for signal in noise:
            noise[signal] = _voltage_noise_to_jitter(
                noise[signal], current_df, capacitor
            )
        return noise
    if noise is None:
        return None

    out_df = noise[["Detector", "Wavelength_nm"]].copy(deep=True)
    total_current = get_total_current(current_df)
    voltage_rate = total_current / capacitor
    for column in noise[noise.columns.difference(["Detector", "Wavelength_nm"])]:
        new_column = column.replace("_V_rms", "_s")
        out_df[new_column] = np.sqrt((noise[column] ** 2) * (1 / (voltage_rate**2)))
    return out_df


def de_voltage_noise_to_jitter(
    noise: Union[pd.DataFrame, c.RecursiveDict, None],
    current_df: pd.DataFrame,
    capacitor: float,
) -> Union[pd.DataFrame, c.RecursiveDict, None]:
    """Converts jitter on comparator to voltage

    Parameters
    ----------
    noise_df : Union[pd.DataFrame, c.RecursiveDict, None]
        jitter noise to convert
    jitter_df : pd.DataFrame
        Dataframe containing DC signal current
    capacitor : float
        Integrating capacitor

    Returns
    -------
    Union[pd.DataFrame, c.RecursiveDict, None]
        Structure containing voltage noise
    """
    noise = copy.deepcopy(noise)
    out_noise = _de_voltage_noise_to_jitter(noise, current_df, capacitor)
    return out_noise


def _de_voltage_noise_to_jitter(
    noise: Union[pd.DataFrame, c.RecursiveDict, None],
    current_df: pd.DataFrame,
    capacitor: float,
) -> pd.DataFrame:
    """Recursive function to get voltage noise from jitter

    Parameters
    ----------
    noise_df : Union[pd.DataFrame, c.RecursiveDict, None]
        Voltage noise to convert
    current_df : pd.DataFrame
        Dataframe containing DC signal current
    capacitor : float
        Integrating capacitor

    Returns
    -------
    Union[pd.DataFrame, c.RecursiveDict, None]
        Structure containing jitter
    """

    # Check for type
    if isinstance(noise, Dict):
        for signal in noise:
            noise[signal] = _de_voltage_noise_to_jitter(
                noise[signal], current_df, capacitor
            )
        return noise
    if noise is None:
        return None

    out_df = noise[["Detector", "Wavelength_nm"]].copy(deep=True)
    total_current = get_total_current(current_df)
    voltage_rate = total_current / capacitor
    for column in noise[noise.columns.difference(["Detector", "Wavelength_nm"])]:
        new_column = column.replace("_s", "_V_rms")
        out_df[new_column] = np.sqrt(noise[column] ** 2 * voltage_rate**2)
    return out_df


def integrate_current_noise_on_capacitor(
    current_df_dict: Union[c.RecursiveDict, pd.DataFrame, None],
    cap: float,
    reset_frequency: float,
) -> Union[c.RecursiveDict, Dict]:
    """Integrates current on capacitor

    Parameters
    ----------
    current_df_dict : Union[c.RecursiveDict, pd.DataFrame, None]
        Current to integrate into the capacitor
    cap : float
        Capacitor size
    reset_frequency : float
        Frequency of reset (sample rate)

    Returns
    -------
    Union[c.RecursiveDict, Dict]
        Voltages coresponding to currents
    """
    current_df_dict = copy.deepcopy(current_df_dict)
    voltage_df_dict = _integrate_current_noise_on_capacitor(
        current_df_dict, cap, reset_frequency
    )
    return voltage_df_dict


def _integrate_current_noise_on_capacitor(
    current_dict: Union[c.RecursiveDict, pd.DataFrame, None],
    cap: float,
    reset_frequency: float,
) -> Union[pd.DataFrame, Dict]:
    # Check for type
    if isinstance(current_dict, Dict):
        for signal in current_dict:
            current_dict[signal] = _integrate_current_noise_on_capacitor(
                current_dict[signal], cap, reset_frequency
            )
        return current_dict
    if current_dict is None:
        return None

    # Setup DataFrames
    current_df = current_dict.copy(deep=True)
    voltage_df = current_df[["Detector", "Wavelength_nm"]].copy(deep=True)
    for column in current_df[
        current_df.columns.difference(["Detector", "Wavelength_nm"])
    ]:
        new_column = column.replace("_A", "_V")
        voltage_df[new_column] = np.sqrt(
            ((1 / (cap * reset_frequency)) ** 2) * current_df[column] ** 2
        )
    return voltage_df


def de_integrate_current_on_capacitor(
    voltage_df_dict: Union[c.RecursiveDict, pd.DataFrame, None],
    cap: float,
    reset_frequency: float,
) -> Union[c.RecursiveDict, Dict]:
    """De Integrates current on capacitor, Voltage to Current

    Parameters
    ----------
    voltage_df_dict : Union[c.RecursiveDict, pd.DataFrame, None]
        Voltage to de_integrate into the capacitor
    cap : float
        Capacitor size
    reset_frequency : float
        Frequency of reset (sample rate)

    Returns
    -------
    Union[c.RecursiveDict, Dict]
        Currents corresponding to voltages
    """
    voltage_df_dict = copy.deepcopy(voltage_df_dict)
    current_df_dict = _de_integrate_current_on_capacitor(
        voltage_df_dict, cap, reset_frequency
    )
    return current_df_dict


def _de_integrate_current_on_capacitor(
    voltage_dict: Union[c.RecursiveDict, pd.DataFrame, None],
    cap: float,
    reset_frequency: float,
) -> Union[pd.DataFrame, Dict]:
    # Check for type
    if isinstance(voltage_dict, Dict):
        for signal in voltage_dict:
            voltage_dict[signal] = _de_integrate_current_on_capacitor(
                voltage_dict[signal], cap, reset_frequency
            )
        return voltage_dict
    if voltage_dict is None:
        return None

    # Setup DataFrames
    voltage_df = voltage_dict.copy(deep=True)
    current_df = voltage_df[["Detector", "Wavelength_nm"]].copy(deep=True)
    for column in voltage_df[
        voltage_df.columns.difference(["Detector", "Wavelength_nm"])
    ]:
        new_column = column.replace("_V", "_A")
        current_df[new_column] = np.sqrt(
            ((cap * reset_frequency) ** 2) * voltage_df[column] ** 2
        )
    return current_df


def get_total_current(df: pd.DataFrame) -> pd.Series:
    """Gets all current in a dataframe

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to get current from

    Returns
    -------
    pd.DataFrame
        Total current in detector channels
    """
    return df.loc[:, ~df.columns.str.contains("Wavelength_nm|Detector|_[^A]$")].sum(
        axis=1
    )


def get_total_noise(df: pd.DataFrame) -> pd.Series:
    """Gets all current in a dataframe given independent noises

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to get noise from

    Returns
    -------
    pd.Series
        Total noise current in detector channels
    """
    return np.sqrt(
        df.loc[:, ~df.columns.str.contains("Wavelength_nm|Detector")].pow(2).sum(axis=1)
    )


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
