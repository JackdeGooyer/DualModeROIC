"""Functions to convert from photons to electrons"""

import copy
import re
import constants as c
import pandas as pd
import numpy as np
from scipy import interpolate
from typing import Union, Dict


def convert_em_to_channels(
    em_dict: c.RecursiveDict,
    responsivity_df: pd.DataFrame,
    bias: float,
    no_detectors: int,
    detector_area: float,
    include_dc_dark_current: bool = False,
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
    include_dc_dark_current : bool, optional
        Add dark current to current (for DC current only), by default False

    Returns
    -------
    c.RecursiveDict
        Recursive dicitonary containing currents for each signal type and detector
    """
    detector_current = copy.deepcopy(em_dict)
    responsivity = get_responsivity(responsivity_df, bias, detector_area)
    detector_current = _convert_em_to_channels(
        detector_current,
        responsivity,
        bias,
        no_detectors,
        detector_area,
        include_dc_dark_current,
    )
    return detector_current


def _convert_em_to_channels(
    em_dict: Union[c.RecursiveDict, pd.DataFrame, None],
    responsivity: pd.DataFrame,
    bias: float,
    no_detectors: int,
    detector_area: float,
    include_dc_dark_current: bool = False,
) -> Union[c.RecursiveDict, pd.DataFrame, None]:

    # Check for type
    if isinstance(em_dict, Dict):
        for signal in em_dict:
            em_dict[signal] = _convert_em_to_channels(
                em_dict[signal],
                responsivity,
                bias,
                no_detectors,
                detector_area,
                include_dc_dark_current,
            )
        return em_dict
    if em_dict is None:
        return None

    # Setup DataFrames
    detector_df = pd.DataFrame({"Detector": pd.Series(range(0, no_detectors))})
    em_df = em_dict.copy(deep=True)

    # Compute Wavelengths and Upscale
    upscale_factor = len(em_df.index) / no_detectors
    max_w = em_df["Wavelength_nm"].max()
    min_w = em_df["Wavelength_nm"].min()
    wavelength = np.arange(min_w, max_w, (max_w - min_w) / no_detectors)
    detector_df["Wavelength_nm"] = wavelength

    # Interpolate Responsivity (again...)
    interpt_responsivity = interpolate.interp1d(
        responsivity["Wavelength_nm"],
        responsivity["Responsivity_A_W"],
        bounds_error=False,
        fill_value="extrapolate",
    )
    interpt_responsivity = interpt_responsivity(detector_df["Wavelength_nm"])

    # Compute Current
    for column in em_df:
        if column == "Wavelength_nm":
            continue
        if column == "Power_W":
            continue
        # Change Column name to _W to _A
        new_col_name = re.sub("_W$", "_A", column)

        # Resample input current to detectors
        interpolate_data = interpolate.interp1d(
            em_df["Wavelength_nm"], em_df[column], kind="linear"
        )
        detector_df[new_col_name] = interpolate_data(detector_df["Wavelength_nm"])

        # Integrate over detectors
        detector_df[new_col_name] = (
            detector_df[new_col_name] * interpt_responsivity * upscale_factor
        )

    # Add dark current (optional)
    if include_dc_dark_current:
        interpt_dark_current = interpolate.interp1d(
            responsivity["Wavelength_nm"],
            responsivity["Dark_Current_A"],
            bounds_error=False,
            fill_value="extrapolate",
        )
        detector_df["Dark_Current_A"] = interpt_dark_current(
            detector_df["Wavelength_nm"]
        )

    return detector_df


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
    responsivity = interpolate_bias_responsivity(responsivity_df, bias)
    responsivity["Dark_Current_A"] = responsivity["Dark_Current_A_m^2"] * detector_area
    responsivity["Responsivity_A_W"] = (
        responsivity["Responsivity_A_W_m^2"] * detector_area
    )
    return responsivity


def interpolate_bias_responsivity(
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
