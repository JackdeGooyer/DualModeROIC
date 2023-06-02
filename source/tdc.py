# Script for all digital encoding

import copy
import pandas as pd
import numpy as np
import constants as c
from typing import Union, Dict


def add_tdc_jitter(template: pd.DataFrame, jitter: float) -> pd.DataFrame:
    """Add the jitter from the TDC

    Parameters
    ----------
    template : pd.DataFrame
        Template for detector number and wavelengths
    jitter : float
        Total jitter

    Returns
    -------
    pd.DataFrame
        Dataframe containing formatted jitter
    """
    out_df = template[["Detector", "Wavelength_nm"]].copy(deep=True)
    out_df["Quantization_Noise_s"] = jitter
    return out_df


def period_signal_to_bits(
    freq_df: pd.DataFrame, min_sample_freq: float, lsb: float, bits: int
) -> pd.DataFrame:
    """Convert Frequency Signal to Bits

    Parameters
    ----------
    freq_df : pd.DataFrame
        Input frequency data
    min_sample_freq : float
        Min sample frequency/ Max allowed time period
    lsb : float
        Size of least significant bit
    bits : int
        Total number of bits

    Returns
    -------
    pd.DataFrame
        Dataframe containing signal in bits
    """
    # Get unrounded bits

    # # Round bits
    # max_reading = (1 / min_sample_freq) / lsb
    # out_df["rounded_bits"] = np.round(out_df["un_rounded_bits"])
    # out_df[out_df["rounded_bits"] > max_reading] = max_reading
    # out_df[out_df["rounded_bits"] > np.power(bits, 2)] = np.power(bits, 2)\

    return freq_df["Signal_s"] / lsb


def period_noise_to_bits(
    freq_df_dict: Union[c.RecursiveDict, pd.DataFrame, None], lsb: float
) -> Union[c.RecursiveDict, Dict]:
    """Frequency to Bits

    Parameters
    ----------
    freq_df_dict : Union[c.RecursiveDict, pd.DataFrame, None]
        Frequencies to quantize
    lsb : float
        Size of LSB in seconds

    Returns
    -------
    Union[c.RecursiveDict, Dict]
        Quantizations coresponding to frequency
    """
    freq_df_dict = copy.deepcopy(freq_df_dict)
    bit_df_dict = _period_noise_to_bits(freq_df_dict, lsb)
    return bit_df_dict


def _period_noise_to_bits(
    freq_dict: Union[c.RecursiveDict, pd.DataFrame, None], lsb: float
) -> Union[pd.DataFrame, Dict]:
    # Check for type
    if isinstance(freq_dict, Dict):
        for signal in freq_dict:
            freq_dict[signal] = _period_noise_to_bits(freq_dict[signal], lsb)
        return freq_dict
    if freq_dict is None:
        return None

    # Setup DataFrames
    freq_df = freq_dict.copy(deep=True)
    bit_df = freq_df[["Detector", "Wavelength_nm"]].copy(deep=True)
    for column in freq_df[freq_df.columns.difference(["Detector", "Wavelength_nm"])]:
        new_column = column.replace("_s", "_B")
        bit_df[new_column] = freq_df[column] / lsb
    return bit_df


def de_period_noise_to_bits(
    bit_df_dict: Union[c.RecursiveDict, pd.DataFrame, None], lsb: float
) -> Union[c.RecursiveDict, Dict]:
    """Frequency to Bits

    Parameters
    ----------
    bit_df_dict : Union[c.RecursiveDict, pd.DataFrame, None]
        Bits to become frequencies again
    lsb : float
        Size of LSB in seconds

    Returns
    -------
    Union[c.RecursiveDict, Dict]
        Quantizations coresponding to frequency
    """
    bit_df_dict = copy.deepcopy(bit_df_dict)
    freq_df_dict = _de_period_noise_to_bits(bit_df_dict, lsb)
    return freq_df_dict


def _de_period_noise_to_bits(
    bit_dict: Union[c.RecursiveDict, pd.DataFrame, None], lsb: float
) -> Union[pd.DataFrame, Dict]:
    # Check for type
    if isinstance(bit_dict, Dict):
        for signal in bit_dict:
            bit_dict[signal] = _de_period_noise_to_bits(bit_dict[signal], lsb)
        return bit_dict
    if bit_dict is None:
        return None

    # Setup DataFrames
    bit_df = bit_dict.copy(deep=True)
    freq_df = bit_df[["Detector", "Wavelength_nm"]].copy(deep=True)
    for column in bit_df[bit_df.columns.difference(["Detector", "Wavelength_nm"])]:
        new_column = column.replace("_B", "_s")
        freq_df[new_column] = bit_df[column] * lsb
    return freq_df
