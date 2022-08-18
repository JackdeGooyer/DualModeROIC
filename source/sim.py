"""Simulate Design of Device -- Done in Fourier Domain
"""

from pathlib import Path
from oct import (
    beam_split,
    reflection_loss,
    get_time_delay,
    add_em_noise,
    select_depths,
    combine_em_data,
)
from detector import convert_em_to_channels
from linear import add_shot_noise, add_thermal_noise
import constants as c
import pandas as pd
import numpy as np
import plotly.express as px
from laser_source import generate_source


class OCTSim:
    """Class for Simulation Data Handling"""

    # Data dict -- Keys: Physical Location (Sample, Ref, etc), Subtype (Signal, Noise)
    em_data: c.RecursiveDict = None
    em_data_template: c.RecursiveDict = {"input": {"signal": True, "noise": {}}}
    detector_data: c.RecursiveDict = None
    frequency_data: c.RecursiveDict = None

    # Files
    _responsivity_path: Path = c.DEFAULT_RESPONSIVITY_FILEPATH
    _source_path: Path = c.DEFAULT_SOURCE_FILEPATH

    reflectivity: pd.DataFrame = None
    responsivity: pd.DataFrame = None
    source_power_df: pd.DataFrame = None

    # Source Characteristics
    fwhm_bw: float = c.FULL_WIDTH_HALF_MAX_BW
    central_wl: float = c.CENTRAL_WAVELENGTH
    source_point: int = c.SOUCE_POINTS
    total_source_power: float = c.SOURCE_POWER

    # Sim Depth
    sample_depths: int = c.SAMPLE_DEPTHS
    sample_depth: float = c.DEFAULT_SAMPLE_DEPTH

    # Engineering Parameter
    sample_rate: float = c.SAMPLE_RATE
    number_of_detectors: int = c.DETECTOR_NUMBER
    detector_area: int = c.DETECTOR_AREA
    detector_bias_voltage: float = c.DETECTOR_VOLTAGE
    detector_resistance: float = c.DETECTOR_RESISTANCE
    temperature: float = c.TEMPERATURE

    # OCT Beam Params
    ref_reflect_pct: float = c.REFERENCE_REFLECTIVITY_PCT
    bs_ref_pct: float = c.BEAM_SPLIT_REF_PCT
    beam_losses_pct: float = c.BEAM_LOSSES_PCT
    rin_noise_pct: float = c.RIN_NOISE_PCT

    # Linear Values
    linear_cap: float = c.CAPACITOR
    reset_time: float = c.RESET_TIME
    reset_jitter: float = c.RESET_JITTER
    buffer_gain: float = c.BUFFER_GAIN
    threshold_voltage: float = c.VOLTAGE_THRESHOLD

    # Sample Specs
    sample_reflect_pct: float = c.SAMPLE_REFLECTIVITY_PCT
    sample_ref_index: float = c.SAMPLE_REFRACTIVE_INDEX
    tissue_depth: float = c.DEFAULT_SAMPLE_DEPTH

    def __init__(self):
        """Reads all data files into Detector Class"""
        self.responsivity = pd.read_csv(self._responsivity_path, index_col=False)
        self.source_power_df = pd.read_csv(self._source_path, index_col=False)
        self.em_data = dict()
        self.reflectivity = dict()
        self.detector_data = dict()
        self.frequency_data = dict()
        return

    def perform_oct_sim(self):
        """Perform all required simulation for the OCT portion of the device"""

        # Generate Data
        self.em_data = self._generate_em_data(self.em_data_template)
        self.reflectivity = self._generate_relectivity(self.em_data)

        # Split the beam
        self.reflectivity["ref"], self.reflectivity["sample"] = beam_split(
            self.reflectivity["input"], split_pct_ref=self.bs_ref_pct
        )

        # Losses from reflections
        self.reflectivity["ref"] = reflection_loss(
            self.reflectivity["ref"], self.ref_reflect_pct
        )
        self.reflectivity["sample"] = reflection_loss(
            self.reflectivity["sample"], self.sample_reflect_pct
        )

        # Add time delay - No complex numbers returned
        self.reflectivity["sample"] = get_time_delay(
            self.reflectivity["sample"],
            depth_points=self.sample_depths,
            max_depth=self.sample_depth,
            ref_index=self.sample_ref_index,
        )
        self.reflectivity["ref"] = get_time_delay(
            self.reflectivity["ref"],
            depth_points=self.sample_depths,
            max_depth=self.sample_depth,
            ref_index=self.sample_ref_index,
        )

        # Losses from returning through beamsplitter
        # TODO: Double check this is still right!
        self.reflectivity["ref"] = reflection_loss(
            self.reflectivity["ref"], self.bs_ref_pct
        )
        self.reflectivity["sample"] = reflection_loss(
            self.reflectivity["sample"], 100 - self.bs_ref_pct
        )

        # Add the noise TODO: This will likely be by varying the em_data
        # TODO: Check these equations as they are likely NEP
        self.em_data = add_em_noise(
            self.em_data, rin_noise_pct=self.rin_noise_pct, shot_noise=True
        )
        return

    def interfere_waves(self):
        """Interferes the waves to create the signal seen on the detectors"""

        # Select Depths TODO: Make depths variate
        self.reflectivity["sample"] = select_depths(self.reflectivity["sample"], [0, 1])
        self.reflectivity["ref"] = select_depths(self.reflectivity["ref"], [0])

        # Combine the EM Data
        self.em_data["output"] = dict()
        self.em_data["output"]["signal"] = combine_em_data(
            self.em_data["input"]["signal"],
            self.reflectivity["ref"]["signal"],
            self.reflectivity["sample"]["signal"],
            small_signal=False,
        )

        # Get Noise converted using small signal aprox
        self.em_data["output"]["noise"] = dict()
        for noise_type in self.em_data["input"]["noise"]:
            self.em_data["output"]["noise"][noise_type] = combine_em_data(
                self.em_data["input"]["noise"][noise_type],
                self.reflectivity["ref"]["signal"],
                self.reflectivity["sample"]["signal"],
                small_signal=True,
            )
        return

    def perform_detector_sampling(self):
        """Perform all required simulation for Spectrometor Beam to Detector Channels"""
        # Convert Each Individually
        self.detector_data["signal"] = convert_em_to_channels(
            self.em_data["output"]["signal"],
            self.responsivity,
            self.detector_bias_voltage,
            self.number_of_detectors,
            self.detector_area,
            include_dc_dark_current=True,
        )
        self.detector_data["noise"] = convert_em_to_channels(
            self.em_data["output"]["noise"],
            self.responsivity,
            self.detector_bias_voltage,
            self.number_of_detectors,
            self.detector_area,
            include_dc_dark_current=False,
        )
        return

    def add_linear_noise(self):
        """Add noises from various features"""
        self.detector_data["noise"]["shot_noise"] = add_shot_noise(
            self.detector_data["signal"],
            resistance=self.detector_resistance,
            capacitance=self.linear_cap,
        )
        self.detector_data["noise"]["thermal_noise"] = add_thermal_noise(
            self.detector_data["signal"],
            temperature=self.temperature,
            resistance=self.detector_resistance,
            capacitance=self.linear_cap,
        )
        return

    def _generate_em_data(
        self, dictionary: c.RecursiveDict, input_df: pd.DataFrame = None
    ) -> c.RecursiveDict:
        """Generates a Dataframe for EM data based on an input dictionary

        Parameters
        ----------
        dictionary : RecursiveDict
            Dictionary to mimic, if an entry is "True", it will fill this with
            the input_df
        input_df : pd.DataFrame, by defaylt None
            For use with external data set of input data

        Returns
        -------
        RecursiveDict
            Filled dictionary
        """
        em_data = dict()
        for physical_loc in dictionary.keys():
            em_data[physical_loc] = dict()
            for sig_type in dictionary[physical_loc].keys():
                if dictionary[physical_loc][sig_type] is True:
                    if input_df is None:
                        em_data[physical_loc][sig_type] = generate_source(
                            self.fwhm_bw,
                            self.central_wl,
                            self.source_point,
                            self.total_source_power,
                        )
                    else:
                        em_data[physical_loc][sig_type] = input_df.copy(deep=True)
                else:
                    em_data[physical_loc][sig_type] = dict()
        return em_data

    @staticmethod
    def _generate_relectivity(dictionary: c.RecursiveDict = None) -> c.RecursiveDict:
        """Generates a dataframe to contain reflectivity data

        Parameters
        ----------
        dictionary : c.RecursiveDict, optional
            Data to model relefctivity from, by default None

        Returns
        -------
        c.RecursiveDict
            Dictionary containing reflectivity data set to 1 for all entries
        """
        relfectivity = dict()
        for physical_loc in dictionary.keys():
            relfectivity[physical_loc] = dict()
            for sig_type in dictionary[physical_loc].keys():
                if sig_type == "noise":
                    continue
                if isinstance(dictionary[physical_loc][sig_type], dict):
                    relfectivity[physical_loc][sig_type] = dict()
                    continue
                relfectivity[physical_loc][sig_type] = dictionary[physical_loc][
                    sig_type
                ].copy(deep=True)
                for column in relfectivity[physical_loc][sig_type].columns:
                    if column == "Wavelength_nm":
                        continue
                    relfectivity[physical_loc][sig_type][column] = 1
        return relfectivity

    def run(self):
        """Performs simulation of device"""

        # Interpolate TODO

        # Perform simulations of source to bs
        self.perform_oct_sim()

        # Interfere waves
        self.interfere_waves()

        # Convert to Linear Detections
        self.perform_detector_sampling()

        # Peform Linear Measurements
        self.add_linear_noise()

        # Convert to APD Detectins

        # Perform APD Measurements

        # Determine timing jitter to noise constant

        # Get noise at each output

        return


if __name__ == "__main__":

    for input_power in np.arange(0, 5.2e-03, 0.1e-03):
        DetectorSim = OCTSim()
        DetectorSim.total_source_power = input_power
        DetectorSim.run()
