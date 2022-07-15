""" File for handling beam parameters

    Format of Dataframe - Can be for signal or noise

     Wavelength_nm  Depth_0  Depth_1  Depth_2 ... Depth_n
     [m]         [w]      [w]      [w]         [w]
"""

from typing import Dict
from scipy.constants import constants as spc
import numpy as np
import pandas as pd


def beam_split(
    input_df: pd.DataFrame = None, split_pct_ref: float = 50
) -> Dict[str, pd.DataFrame]:
    """Splits Dataframe into reference and source data

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to be Split
    split_pct_ref : float, optional
        Amount of signal going to the ref

    Returns
    -------
    Dict[pd.DataFrame]
        Reference DataFrame, Source DataFrame
    """

    ref_df = input_df.copy(deep=True).loc[:, input_df.columns != "Wavelength_nm"] * (
        split_pct_ref / 100
    )
    sample_df = input_df.copy(deep=True).loc[:, input_df.columns != "Wavelength_nm"] * (
        (1 - (split_pct_ref / 100))
    )
    out_dict = {"ref": ref_df, "sample": sample_df}

    return out_dict


def light_loss(input_df: pd.DataFrame = None, loss_pct: float = 10) -> pd.DataFrame:
    """Simulates loss of light in some medium (no phase change)

    Parameters
    ----------
    input_df : pd.DataFrame, optional
        Dataframe to have loss, by default None
    loss_pct : float, optional
        Percentage lost, by default 10

    Returns
    -------
    pd.DataFrame
        Dataframe with loss
    """
    return input_df.copy(deep=True).loc[:, input_df.columns != "Wavelength_nm"] * (
        loss_pct / 100
    )


def reflection_loss(
    input_df: pd.DataFrame = None, reflection_loss_pct: float = 10
) -> pd.DataFrame:
    """Simulates refection and loss of light in some medium (no phase change)

    Parameters
    ----------
    input_df : pd.DataFrame, optional
        Dataframe to be reflected, by default None
    loss_pct : float, optional
        Percentage lost, by default 10

    Returns
    -------
    pd.DataFrame
        Dataframe with reflection loss
    """
    return light_loss(input_df, reflection_loss_pct)


def get_time_delay(
    input_df: pd.DataFrame = None,
    depth_points: float = 100,
    max_depth: float = 10e-03,
    ref_index: float = 1.5,
    preserve_name: bool = False,
) -> pd.DataFrame:
    """Appends a series of time delayed signals

    Parameters
    ----------
    input_df : pd.DataFrame, optional
        Dataframe to have depths added, by default None
    depth_points : float, optional
        Amount of points in, by default 100
    max_depth : float, optional
        Maximum depth of sample, by default 10e-03
    ref_index : float, optional
        Index of refraction of material, by default 1.5
    preserve_name : bool, optional
        Used when doing recursion, appends original column name to depth, by default False

    Returns
    -------
    pd.DataFrame
        Datarame with depth info added
    """

    df = input_df.copy(deep=True)

    # Iterate for each depth
    for column in df.columns:

        # Get the frequency of at the wavelength
        if column == "Wavelength_nm":
            freq = spc.speed_of_light / (df["Wavelength_nm"].to_numpy() * 1e-09)
            continue

        # Iterate for each depth for delay
        for depth in np.arange(0, max_depth, max_depth / depth_points):

            # Determine proper column name
            col_name = (
                (column + "Depth_" + str(1000 * depth) + "mm")
                if preserve_name
                else ("Depth_" + str(1000 * depth) + "mm")
            )

            # Calculate frequency shift and append new column TODO Performance
            seen_depth = depth * ref_index
            df[col_name] = df[column].to_numpy() * np.exp(
                -1j * 2 * spc.pi * freq * seen_depth * spc.speed_of_light
            )

    return df
