"""Functions to convert from photons to electrons"""

import math
import pandas as pd
import numpy as np
from scipy import signal


def convert_em_to_channels(
    em_df: pd.DataFrame,
    responsivity_df: pd.DataFrame,
    bias: float,
    no_detectors: int,
    dark_current: bool = False,
) -> pd.DataFrame:
    """Converts the EM Wattage to Amps

    Parameters
    ----------
    em_df : pd.DataFrame
        Dataframe containing df to convert
    responsivity_df : pd.DataFrame
        Dataframe containing detector responsivity
    bias : float
        Detector Bias
    no_detectors : int
        Bumber of Detectors
    dark_current : bool, optional
        Should this include the dark current, by default False

    Returns
    -------
    pd.DataFrame
        Dataframe with depths and amps
    """

    df = pd.DataFrame({"Detector": pd.Series(range(0, no_detectors))})

    # Get the responsivity for given bias
    responsivity_series = responsivity_df.loc[responsivity_df["Bias_V"] == bias]
    responsivity = responsivity_series["1W_Responsivity_A"].iloc[0]
    if not dark_current:
        responsivity = responsivity - responsivity_series["Dark_Current"].iloc[0]

    # Compute current
    for column in em_df:
        if column == "Wavelength_nm":
            upscale_factor = math.ceil(len(em_df.index) / no_detectors)
            max_w = em_df["Wavelength_nm"].max()
            min_w = em_df["Wavelength_nm"].min()
            wavelength = np.arange(min_w, max_w, (max_w - min_w) / no_detectors)
            df["Wavelength_nm"] = wavelength
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
        df[column] = em_data * responsivity

        # TODO - There are some cross correlations I am missing
        pass

        # TODO - Remove aditonal dark currents
        pass

    return df
