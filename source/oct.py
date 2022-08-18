""" File for handling beam parameters

    Format of Dataframe - Can be for signal or noise

     Wavelength_nm  Depth_0  Depth_1  Depth_2 ... Depth_n
     [m]         [w]      [w]      [w]         [w]
"""
import copy
from typing import Dict, Tuple, Union
import constants as c
from scipy.constants import constants as spc
import numpy as np
import pandas as pd


def beam_split(
    input_dict: c.RecursiveDict = None, split_pct_ref: float = 50
) -> Tuple[c.RecursiveDict]:
    """Splits the beam into component parts

    Parameters
    ----------
    input_dict : c.RecursiveDict, optional
        beam to split, by default None
    split_pct_ref : float, optional
        ratio to split by, by default 50

    Returns
    -------
    Tuple[c.RecursiveDict]
        Tupple containing split beams
    """

    ref_dict = copy.deepcopy(input_dict)
    sample_dict = copy.deepcopy(input_dict)

    _reduce_signal(ref_dict, split_pct_ref)
    _reduce_signal(sample_dict, 100 - split_pct_ref)

    return ref_dict, sample_dict


def light_loss(input_dict: c.RecursiveDict, loss_pct: float = 10) -> pd.DataFrame:
    """Simulates loss of light in some medium (no phase change)

    Parameters
    ----------
    input_dict : c.RecursiveDict
        Signal to have loss, by default None
    loss_pct : float, optional
        Percentage lost, by default 10

    Returns
    -------
    pd.DataFrame
        Dataframe with loss
    """
    update_dict = copy.deepcopy(input_dict)
    _reduce_signal(update_dict, loss_pct)
    return update_dict


def reflection_loss(
    input_dict: c.RecursiveDict, reflection_pct: float = 10
) -> pd.DataFrame:
    """Simulates refection and loss of light in some medium (no phase change)

    Parameters
    ----------
    input_dict : c.RecursiveDict
        Signal to be reflected, by default None
    reflection_pct : float, optional
        Percent to reflect, by default 10

    Returns
    -------
    pd.DataFrame
        Dataframe with reflection loss
    """
    update_dict = copy.deepcopy(input_dict)
    _reduce_signal(update_dict, (100 - reflection_pct))
    return update_dict


def get_time_delay(
    input_df_or_dict: c.RecursiveDict = None,
    depth_points: float = 100,
    max_depth: float = 10e-03,
    ref_index: float = 1.5,
    preserve_name: bool = False,
) -> c.RecursiveDict:
    """Appends a series of time delayed signals for each depth
       Adds energy to the system

    Parameters
    ----------
    input_df_or_dict : c.RecursiveDict, optional
        Data to be time delayed, by default None
    depth_points : float, optional
        How many points to delay by, by default 100
    max_depth : float, optional
        Maximal depth to attain, by default 10e-03
    ref_index : float, optional
        Refractive index of material, by default 1.5
    preserve_name : bool, optional
        Used when doing recursion, appends origianl column name to depth by default False

    Returns
    -------
    c.RecursiveDict
        _description_
    """

    dict_copy = copy.deepcopy(input_df_or_dict)
    _get_time_delay(dict_copy, depth_points, max_depth, ref_index, preserve_name)
    return dict_copy


def add_em_noise(
    input_dict: pd.DataFrame, rin_noise_pct: float = 100, shot_noise: bool = True
) -> pd.DataFrame:
    """Adds all noise to em signal data

    Parameters
    ----------
    input_dict : pd.DataFrame
        Dicitonary to add noise to
    rin_noise_pct : float, optional
        Percentage of signal to add as RIN, by default 100
    shot_noise : bool, optional
        Percentage of signal to add as photon shot noise, by default True

    Returns
    -------
    pd.DataFrame
        _description_
    """
    dict_copy = copy.deepcopy(input_dict)
    _add_em_noise(dict_copy, rin_noise_pct, shot_noise)
    return dict_copy


def generate_rin_noise(signal: pd.DataFrame, rin_noise_pct: float) -> pd.DataFrame:
    """Creates a dataframe with a rin_noise added

    Parameters
    ----------
    signal : pd.DataFrame
        signal to have rin noise added
    rin_noise_pct : float
        Rin noise in percentage

    Returns
    -------
    pd.DataFrame
        Dataframe containing noise
    """

    rin_df = signal.copy(deep=True)
    for column in signal.columns:

        # Get photon energy
        if column == "Wavelength_nm":
            continue

        # Get RiN Noise TODO check formula
        rin_df[column] = rin_df[column] * rin_noise_pct / 100
    return rin_df


def generate_shot_noise(signal: pd.DataFrame, shot_noise: bool) -> pd.DataFrame:
    """Generates a dataframe with shot noise

    Parameters
    ----------
    signal : pd.DataFrame
        Signal to generate shot noise from
    shot_noise : bool
        Boolean to determine to add shot noise or not

    Returns
    -------
    pd.DataFrame
        Dataframe containing shot noise
    """

    shot_df = signal.copy(deep=True)
    for column in signal.columns:

        # Get photon energy
        if column == "Wavelength_nm":
            energy = (
                spc.Planck
                * spc.speed_of_light
                / (shot_df["Wavelength_nm"].to_numpy() * 1e-09)
            )
            continue

        # Get Photon Shot Noise (sqrt(Photon Number)) TODO Check formula
        if shot_noise:
            photon_shot_noise = energy * np.sqrt(shot_df[column] / energy)
        else:
            photon_shot_noise = 0
        shot_df[column] = photon_shot_noise
    return shot_df


def select_depths(em_data: c.RecursiveDict, depths: list) -> c.RecursiveDict:
    """Get subset of depths from array

    Parameters
    ----------
    em_data : c.RecursiveDict
        Dictionary to get depths from
    depths : list
        Depths to choose from em_data array

    Returns
    -------
    c.RecursiveDict
        em_data with selected depths remaining
    """
    em_data_copy = copy.deepcopy(em_data)
    if depths is None:
        return em_data_copy
    _select_depths(em_data_copy, depths)
    return em_data_copy


def combine_em_data(
    source_power: pd.DataFrame,
    ref_reflectivity: pd.DataFrame,
    sample_reflectivity: pd.DataFrame,
    small_signal: bool = False,
) -> pd.DataFrame:
    """Combines em_data and reflectivity to get power per detector

    Parameters
    ----------
    source_power : pd.DataFrame
        Power of the source
    ref_reflectivity : pd.DataFrame
        Power reflectivity of the reference
    sample_reflectivity : pd.DataFrame
        Power reflectivity of the sample
    small_signal : bool, optional
        Parameter to adjust for small signal (multply by 2), by default False

    Returns
    -------
    pd.DataFrame
        Output dataframe containing DC, Cross, and Autocorrelation Terms
    """

    output = source_power.loc[:, ["Wavelength_nm"]]

    ref_depth = _get_depth(ref_reflectivity).sum(axis=1)
    ref_exp = ref_depth / ref_depth.abs()
    ref_mag = ref_depth.abs()
    sample_depths = _get_depth(sample_reflectivity)

    # Get DC Terms
    output["DC_Terms_W"] = source_power["Power_W"] * (
        ref_mag + sample_depths.abs().sum(axis=1)
    )

    # Get Cross Terms
    output["Cross_Terms_W"] = 0
    for column in sample_depths.columns:
        sample_mag = sample_depths[column].abs()
        sample_exp = sample_depths[column] / sample_mag
        output["Cross_Terms_W"] = output["Cross_Terms_W"] + np.sqrt(
            sample_mag * ref_mag
        ) * (((ref_exp / sample_exp) + (sample_exp / ref_exp)).abs())
    output["Cross_Terms_W"] = output["Cross_Terms_W"] * source_power["Power_W"]

    # Get Auto Terms (TODO: divide by 2 ?)
    output["Auto_Terms_W"] = 0
    for n_column in sample_depths.columns:
        n_mag = sample_depths[n_column].abs()
        n_exp = sample_depths[n_column] / n_mag
        for m_column in sample_depths.columns:
            if n_column == m_column:
                pass
            m_mag = sample_depths[m_column].abs()
            m_exp = sample_depths[m_column] / m_mag
            output["Auto_Terms_W"] = output["Auto_Terms_W"] + np.sqrt(n_mag * m_mag) * (
                ((n_exp / m_exp) + (m_exp / n_exp)).abs()
            )
    output["Auto_Terms_W"] = output["Auto_Terms_W"] * source_power["Power_W"]

    # IF this is the small noise, use aproximation
    if small_signal:
        output["DC_Terms_W"] = output["DC_Terms_W"] * 2
        output["Cross_Terms_W"] = output["Cross_Terms_W"] * 2
        output["Auto_Terms_W"] = output["Auto_Terms_W"] * 2
    return output


def _add_em_noise(input_dict: pd.DataFrame, rin_noise_pct, shot_noise) -> pd.DataFrame:
    """Adds power for poission noise, rin noise, and other source noises

    Parameters
    ----------
    input_df : pd.DataFrame
        Dataframe of current laser powers at various depths
    rin_noise_pct : float, optional
        Relative intensity noise, by percent, by default 100
    shot_noise : bool, optional
        Flag to determine weather to calculate shot noise, by default True

    Returns
    -------
    pd.DataFrame
        Dataframe showing power of noise per depth
    """

    try:
        signal = input_dict["signal"]
        input_dict["noise"]["rin"] = generate_rin_noise(signal, rin_noise_pct)
        input_dict["noise"]["photon_shot"] = generate_shot_noise(signal, shot_noise)
        return
    except (KeyError, AttributeError):
        pass

    # Check for type
    if isinstance(input_dict, Dict):
        for signal in input_dict.keys():
            _add_em_noise(input_dict[signal], rin_noise_pct, shot_noise)
        return
    if input_dict is None:
        return
    return


def _get_time_delay(
    input_df_or_dict: Union[c.RecursiveDict, Dict, None],
    depth_points: float,
    max_depth: float,
    ref_index: float,
    preserve_name: bool,
):
    """Appends a series of time delayed signals of equal power

    Parameters
    ----------
    input_df : pd.DataFrame
        Dataframe to have depths added
    depth_points : float
        Amount of points in
    max_depth : float
        Maximum depth of sample
    ref_index : float
        Index of refraction of material
    preserve_name : bool
        Used when doing recursion, appends original column name to depth
    """

    # Check for type
    if isinstance(input_df_or_dict, Dict):
        for signal in input_df_or_dict:
            _get_time_delay(
                input_df_or_dict[signal],
                depth_points,
                max_depth,
                ref_index,
                preserve_name,
            )
        return
    if input_df_or_dict is None:
        return

    # Iterate for each depth
    for column in input_df_or_dict.columns:

        # Get the frequency of at the wavelength
        if column == "Wavelength_nm":
            freq = spc.speed_of_light / (
                input_df_or_dict["Wavelength_nm"].to_numpy() * 1e-09
            )
            continue

        # Iterate for each depth for delay
        for depth in np.arange(0, max_depth, max_depth / depth_points):

            # Determine proper column name
            col_name = (
                (column + "Depth_" + str(1000 * round(depth, 16)) + "mm")
                if preserve_name
                else ("Depth_" + str(1000 * round(depth, 16)) + "mm")
            )

            # Calculate frequency shift and append new column
            seen_depth = depth * ref_index * 2  # For both in and out of sample
            input_df_or_dict[col_name] = input_df_or_dict[column].to_numpy() * np.exp(
                -1j * 2 * spc.pi * freq * seen_depth / spc.speed_of_light
            )
    return


def _reduce_signal(
    input_dict_or_df: Union[c.RecursiveDict, Dict, None],
    reduce_pct: float,
):
    """Recursive function to reduce all dataframes by some pct
    Parameters
    ----------
    input_dict_or_df : Union[c.RecursiveDict, Dict, None]
        input dataframe or dictionary to be reduced
    reduce_pct : float
        what percent to reduce by
    """

    # Perform type checking
    if isinstance(input_dict_or_df, Dict):
        for key in input_dict_or_df:
            _reduce_signal(input_dict_or_df[key], reduce_pct)
        return
    if input_dict_or_df is None:
        return

    # Perfrom operation
    input_dict_or_df.loc[
        :, input_dict_or_df.columns != "Wavelength_nm"
    ] = input_dict_or_df.loc[:, input_dict_or_df.columns != "Wavelength_nm"] * (
        (100 - reduce_pct) / 100
    )
    return


def _select_depths(
    em_data: Union[c.RecursiveDict, pd.DataFrame, None], depth_points: list
):
    """Recursive function to select depths in em_data

    Parameters
    ----------
    em_data : Union[c.RecursiveDict, pd.DataFrame, None]
        Recursively searched em_data object
    depth_points : list
        Which points will be kept

    Raises
    ------
    RuntimeError
        Error is raised when not all depths can be grabbed
    """
    if isinstance(em_data, Dict):
        for key in em_data:
            _select_depths(em_data[key], depth_points)
        return
    if em_data is None:
        return
    only_depths = em_data.loc[:, em_data.columns.str.contains("Depth")]
    non_depth = ~em_data.columns.str.contains("Depth")
    drop_depths = only_depths.iloc[:, depth_points]
    rows_to_remove = ~np.bitwise_or(
        em_data.columns.str.contains("|".join(list(drop_depths.columns))), non_depth
    )
    em_data.drop(list(em_data.iloc[:, rows_to_remove].columns), axis=1, inplace=True)

    # drop_depths = only_depths.iloc[:, depth_points]
    # em_data = em_data.loc[
    #     :,
    #     np.bitwise_or(
    #         em_data.columns.str.contains("|".join(list(only_depths.columns))), non_depth
    #     ),
    # ]
    return


def _get_depth(depth_df: pd.DataFrame) -> pd.DataFrame:
    return depth_df.loc[:, depth_df.columns.str.contains("Depth")]
