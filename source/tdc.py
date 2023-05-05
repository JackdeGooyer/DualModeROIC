# Script for all digital encoding

import pandas as pd
import numpy as np
import constants as c


def frequency_signal_to_bits(
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
        _description_
    """
    # Get unrounded bits
    out_df = freq_df[["Detector", "Wavelength_nm"]].copy(deep=True)
    out_df["un_rounded_bits"] = freq_df["Period_s"] / lsb

    # Round bits
    max_reading = (1 / min_sample_freq) / lsb
    out_df["rounded_bits"] = np.round(out_df["un_rounded_bits"])
    out_df[out_df["rounded_bits"] > max_reading] = max_reading
    out_df[out_df["rounded_bits"] > np.power(bits, 2)] = np.power(bits, 2)
    return out_df


def frequency_noise_to_bits():
    # Use LSB to get noise in bits

    pass


def de_frequency_to_bits():
    # Use unrounded bits to convert back
    pass
