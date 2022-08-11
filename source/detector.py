"""Functions to convert from photons to electrons"""

import constants as c
import copy
import math
import pandas as pd
import numpy as np
from scipy import signal
from typing import Union, Dict


def convert_em_to_channels(
    em_dict: c.RecursiveDict,
    responsivity_df: pd.DataFrame,
    bias: float,
    no_detectors: int,
    detector_area: float,
    add_dc_dark_current: bool = False,
) -> c.RecursiveDict:
    """Converts em data to currents from responsivity graph

    Parameters
    ----------
    em_dict : c.RecursiveDict
        Dictionary of data to convert
    responsivity_df : pd.DataFrame
        Dataframe containing responsivity of detector
    bias : float
        Bias voltage of detector
    no_detectors : int
        Total number of detectors (resolution)
    detector_area : float
        Area of photodetector
    add_dc_dark_current : bool, optional
        Add dark current to current (for DC current only), by default False

    Returns
    -------
    c.RecursiveDict
        Recursive dicitonary containing currents for each signal type and detector
    """
    detector_current = copy.deepcopy(em_dict)
    responsivity = get_responsivity(responsivity_df, bias, add_dc_dark_current)
    _convert_em_to_channels(
        detector_current,
        responsivity,
        bias,
        no_detectors,
        detector_area,
        add_dc_dark_current,
    )
    return detector_current


def _convert_em_to_channels(
    em_dict: Union[c.RecursiveDict, pd.DataFrame, None],
    responsivity: pd.DataFrame,
    bias: float,
    no_detectors: int,
    detector_area: float,
    add_dc_dark_current: bool = False,
) -> c.RecursiveDict:

    # Check for type
    if isinstance(em_dict, Dict):
        for signal in em_dict:
            _convert_em_to_channels(
                em_dict[signal],
                responsivity,
                bias,
                no_detectors,
                detector_area,
                add_dc_dark_current,
            )
        return
    if em_dict is None:
        return

    # Setup DataFrames
    detector_df = pd.DataFrame({"Detector": pd.Series(range(0, no_detectors))})
    em_df = em_dict.copy(deep=True)

    # Compute current
    for column in em_df:
        if column == "Wavelength_nm":
            upscale_factor = math.ceil(len(em_df.index) / no_detectors)
            max_w = em_df["Wavelength_nm"].max()
            min_w = em_df["Wavelength_nm"].min()
            wavelength = np.arange(min_w, max_w, (max_w - min_w) / no_detectors)
            em_df["Wavelength_nm"] = wavelength
            continue
        if column == "Power_W":
            continue

        # Take real part of signals
        em_data = np.real(em_df[column])

        # Resample to multiple of detecotrs
        # TODO - Test
        em_data = signal.resample(em_data, upscale_factor * no_detectors)

        # Sum energy and scale
        em_data = np.array([i.sum() for i in em_data])
        em_data = em_data * ((len(em_df.index)) / (no_detectors * upscale_factor))

        # Integrate over detectors
        em_df[column] = em_data * responsivity

        # TODO - There are some cross correlations I am missing

        # TODO - Remove aditonal dark currents

    return em_df


def get_responsivity(
    responsivity_df: pd.DataFrame,
    bias: float = 3.0,
    detector_area: float = 200e-06,
) -> pd.DataFrame:
    """Get the responsivity dataframe for a given bias with or without dark current

    Parameters
    ----------
    responsivity_df : pd.DataFrame
        Dataframe containing responsivity data
    bias : float, optional
        Bias voltage of detectror, by default 3.0
    detector_area : float, optional
        Area of detector, by default 200e-06e

    Returns
    -------
    pd.DataFrame
        DataFrame containing corrected and interpolated responsivity for bias
    """
    responsivity = interpolate_responsivity(responsivity_df, bias)
    responsivity["Dark_Current_A"] = responsivity["Dark_Current_A_m^2"] * detector_area
    responsivity["Responsivity_A_W"] = (
        responsivity["Responsivity_A_W_m^2"] * detector_area
    )
    return responsivity


def interpolate_responsivity(
    responsivity_df: pd.DataFrame, bias: float = 3.0
) -> pd.DataFrame:
    """Interpolates the responsivity to a given bias

    Parameters
    ----------
    responsivity_df : pd.DataFrame
        Responsivities to interpolate
    bias : float, optional
        Bias Voltage to Interpolate to, by default 3.0

    Returns
    -------
    pd.DataFrame
        Dataframe containing interpolated data

    Raises
    ------
    RuntimeError
        Rasied if bias is out of range of sampled data
    """

    # Check to see if it is a known bias
    if bias in responsivity_df["Bias_V"]:
        return responsivity_df.loc[responsivity_df["Bias_V"] == bias].reset_index(
            drop=True
        )

    # Get series bounds
    responsivity_higher_df = responsivity_df[responsivity_df["Bias_V"] >= bias]
    responsivity_higher_df = responsivity_higher_df[
        responsivity_higher_df["Bias_V"] == responsivity_higher_df["Bias_V"].min()
    ]
    responsivity_lower_df = responsivity_df[responsivity_df["Bias_V"] <= bias]
    responsivity_lower_df = responsivity_lower_df[
        responsivity_lower_df["Bias_V"] == responsivity_lower_df["Bias_V"].max()
    ]
    if responsivity_lower_df.empty or responsivity_higher_df.empty:
        raise RuntimeError(f"Out of Bounds Bias Voltage for Responsivity {bias}")

    # Generate NaN dataframe in centre
    responsivity_interp_df = responsivity_higher_df.copy()
    responsivity_interp_df["Bias_V"] = bias
    responsivity_interp_df["Dark_Current_A_m^2"] = np.NaN
    responsivity_interp_df["Responsivity_A_W_m^2"] = np.NaN

    # Combine and Interpolate
    pd_combined = pd.concat(
        [responsivity_lower_df, responsivity_interp_df, responsivity_higher_df]
    )
    dark_current = (
        pd_combined.pivot(
            index="Bias_V", columns=["Wavelength_nm"], values="Dark_Current_A_m^2"
        )
        .interpolate(method="values")
        .transpose()[bias]
    )
    responsivity = (
        pd_combined.pivot(
            index="Bias_V", columns=["Wavelength_nm"], values="Responsivity_A_W_m^2"
        )
        .interpolate(method="values")
        .transpose()[bias]
    )
    return pd.DataFrame(
        {
            "Bias_V": bias,
            "Wavelength_nm": dark_current.index,
            "Dark_Current_A_m^2": dark_current.values,
            "Responsivity_A_W_m^2": responsivity.values,
        }
    )
