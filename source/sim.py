"""Simulate Design of Device -- Done in Fourier Domain
"""

from pathlib import Path
import copy
from oct import (
    beam_split,
    reflection_loss,
    get_time_delay,
    add_em_noise,
    select_depths,
    combine_em_data,
)
from detector import convert_em_to_channels
import constants as c
import pandas as pd
import numpy as np
import plotly.express as px
from laser_source import generate_source
from linear import (
    get_total_current,
    get_total_noise,
    integrate_current_on_capacitor,
    amp_to_frequency,
    voltage_noise_to_jitter,
    add_shot_noise,
    add_thermal_noise,
    add_tdc_jitter,
    de_integrate_current_on_capacitor,
    de_voltage_noise_to_jitter,
)


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
    TDC_jitter: float = c.TDC_JITTER
    buffer_gain: float = c.BUFFER_GAIN
    threshold_voltage: float = c.VOLTAGE_THRESHOLD

    # Sample Specs
    sample_reflect_pct: float = c.SAMPLE_REFLECTIVITY_PCT
    sample_ref_index: float = c.SAMPLE_REFRACTIVE_INDEX
    tissue_depth: float = c.DEFAULT_SAMPLE_DEPTH

    def __init__(self, responsivity: pd.DataFrame = None):
        """Reads all data files into Detector Class"""
        if responsivity is None:
            self.responsivity = pd.read_csv(self._responsivity_path, index_col=False)
        else:
            self.responsivity = copy.deepcopy(responsivity)
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
        self.detector_data["current_noise"] = convert_em_to_channels(
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
        self.detector_data["current_noise"]["shot_noise"] = add_shot_noise(
            self.detector_data["signal"],
            resistance=self.detector_resistance,
            capacitance=self.linear_cap,
        )
        self.detector_data["current_noise"]["thermal_noise"] = add_thermal_noise(
            self.detector_data["signal"],
            temperature=self.temperature,
            resistance=self.detector_resistance,
            capacitance=self.linear_cap,
        )
        return

    def integrate_current_noise(self):
        """Turn current noise into voltage noise"""
        self.detector_data["voltage_noise"] = dict()
        total_current = get_total_current(self.detector_data["signal"])
        self.detector_data["voltage_noise"] = integrate_current_on_capacitor(
            self.detector_data["current_noise"],
            total_current,
            self.linear_cap,
            self.threshold_voltage,
        )
        return

    def perform_frequency_conversion(self):
        """Turn current and voltage noise into freq and jitter"""
        self.frequency_data["signal"] = amp_to_frequency(
            self.detector_data["signal"], self.linear_cap, self.threshold_voltage
        )
        self.frequency_data["jitter"] = voltage_noise_to_jitter(
            self.detector_data["voltage_noise"],
            self.detector_data["signal"],
            self.linear_cap,
        )
        self.frequency_data["jitter"]["TDC"] = add_tdc_jitter(
            self.frequency_data["jitter"][list(self.frequency_data["jitter"])[0]],
            self.TDC_jitter,
        )
        return

    def back_convert_noises(self):
        """Back propogates noise to input"""
        self.detector_data["voltage_noise"] = de_voltage_noise_to_jitter(
            self.frequency_data["jitter"], self.detector_data["signal"], self.linear_cap
        )
        total_current = get_total_current(self.detector_data["signal"])
        self.detector_data["current_noise"] = de_integrate_current_on_capacitor(
            self.detector_data["voltage_noise"],
            total_current,
            self.linear_cap,
            self.threshold_voltage,
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

        # Perform Integration on Capacitor
        self.integrate_current_noise()

        # Convert to Frequencies and Jitter
        self.perform_frequency_conversion()

        # Back convert all types
        self.back_convert_noises()

        # Convert to APD Detectins

        # Perform APD Measurements

        # Determine timing jitter to noise constant

        # Get noise at each output

        return


if __name__ == "__main__":

    detector_data = pd.DataFrame(
        columns=[
            "Magnitude [A]",
            "Input Power [W]",
            "Signal Type",
            "Capacitance [pF]",
        ]
    )
    detector_snr_data = pd.DataFrame(
        columns=[
            "SNR",
            "Input Power [W]",
            "Signal Type",
            "Capacitance [pF]",
        ]
    )

    selected_detector = 512
    for input_power in np.logspace(-15, 1, 100, base=10):
        for capacitance in np.arange(0.1e-12, 1e-12, 0.1e-12):

            capacitance = 1e-9

            # Run Sim
            DetectorSim = OCTSim()
            DetectorSim.total_source_power = input_power
            DetectorSim.linear_cap = capacitance
            DetectorSim.run()

            #  Get Signal
            cap_data = []
            snr_data = []
            signal = get_total_current(
                DetectorSim.detector_data["signal"][
                    ["DC_Terms_A", "Cross_Terms_A", "Auto_Terms_A"]
                ]
            )[selected_detector]
            cap_data.append(
                {
                    "Signal Magnitude [A]": np.log(signal),
                    "Input Power [W]": np.log(input_power),
                    "Signal Type": "Signal",
                    "Capacitance [pF]": capacitance,
                }
            )

            # Get Noises
            total_noise = 0
            for column in ["photon_shot", "shot_noise", "thermal_noise", "TDC", "rin"]:
                noise = get_total_noise(
                    DetectorSim.detector_data["current_noise"][column]
                )[selected_detector]
                total_noise = np.sqrt(np.power(total_noise, 2) + np.power(noise, 2))
                cap_data.append(
                    {
                        "Signal Magnitude [A]": np.log(noise),
                        "Input Power [W]": np.log(input_power),
                        "Signal Type": column,
                        "Capacitance [pF]": capacitance,
                    }
                )
                snr_data.append(
                    {
                        "SNR": 20 * np.log(signal / noise),
                        "Input Power [W]": np.log(input_power),
                        "Signal Type": column,
                        "Capacitance [pF]": capacitance,
                    }
                )
            cap_data.append(
                {
                    "Signal Magnitude [A]": np.log(total_noise),
                    "Input Power [W]": np.log(input_power),
                    "Signal Type": "Total Noise",
                    "Capacitance [pF]": capacitance,
                }
            )
            snr_data.append(
                {
                    "SNR": 20 * np.log(signal / total_noise),
                    "Input Power [W]": np.log(input_power),
                    "Signal Type": "Total Noise",
                    "Capacitance [pF]": capacitance,
                }
            )
            detector_data = pd.concat([detector_data, pd.DataFrame(cap_data)])
            detector_snr_data = pd.concat([detector_snr_data, pd.DataFrame(snr_data)])

            break  # Remove for multi cap

    detector_data["Capacitance [pF]"] = detector_data["Capacitance [pF]"] * 1e12
    fig = px.line(
        detector_data,
        x="Input Power [W]",
        y="Signal Magnitude [A]",
        color="Signal Type",
    )
    fig.show()
    fig = px.line(
        detector_snr_data,
        x="Input Power [W]",
        y="SNR",
        color="Signal Type",
    )
    fig.show()
    # fig = px.scatter_3d(
    #     detector_data,
    #     x="Input Power [W]",
    #     z="Signal Magnitude [A]",
    #     y="Capacitance [pF]",
    #     color="Signal Type",
    # )
    # fig.show()

    # fig = px.scatter_3d(
    #     detector_snr_data,
    #     x="Input Power [W]",
    #     z="SNR",
    #     y="Capacitance [pF]",
    #     color="Signal Type",
    # )
    # fig.show()

    # Plot SNR versus signal 2d, SNR, input power, for a particular capacitance
    # Start of input

    # Number of bits in the counter -- add this
    # N = 10
    # time LSB = 1/framerate / 2^(bits in counter)
    # seconds^2 quantization = time LSB^2 / 12
