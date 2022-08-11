"""Simulate Design of Device -- Done in Fourier Domain
"""

from pathlib import Path
from oct import beam_split, reflection_loss, get_time_delay, add_em_noise
from detector import convert_em_to_channels
from linear import amp_to_frequency, linear_noise
import constants as c
import pandas as pd
from source import generate_source


"""TODO:
        1. Add independent noise channels - no more sums (dict?)
        2. Create numerical generator for source
        3. Create numerical generator for detector (?)
        4. Perform full interpolations for bias levels, etc
        5. Fix signal equation
        6. Add testing
        7. Remove depths -- useless dimension :)
"""


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

    responsivity: pd.DataFrame = None
    source_power_df: pd.DataFrame = None

    # Source Characteristics
    fwhm_bw: float = c.FULL_WIDTH_HALF_MAX_BW
    central_wl: float = c.CENTRAL_WAVELENGTH
    source_point: int = c.SOUCE_POINTS
    total_source_power: float = c.SOURCE_POWER

    # Sim Depth
    sample_points: int = c.SAMPLE_POINTS
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
        self.detector_data = dict()
        self.frequency_data = dict()
        return

    def perform_oct_sim(self):
        """Perform all required simulation for the OCT portion of the device"""

        self.generate_em_data(self.em_data_template)
        self.em_data["ref"], self.em_data["sample"] = beam_split(
            self.em_data["input"], split_pct_ref=self.bs_ref_pct
        )

        # Losses from reflections
        self.em_data["ref"] = reflection_loss(self.em_data["ref"], self.ref_reflect_pct)
        self.em_data["sample"] = reflection_loss(
            self.em_data["sample"], self.sample_reflect_pct
        )

        # Add time delay
        self.em_data["sample"] = get_time_delay(
            self.em_data["sample"],
            depth_points=self.sample_depths,
            max_depth=self.sample_depth,
            ref_index=self.sample_ref_index,
        )
        self.em_data["ref"] = get_time_delay(
            self.em_data["ref"],
            depth_points=self.sample_depths,
            max_depth=self.sample_depth,
            ref_index=0,
        )

        # Losses from returning through beamsplitter  -- TODO: Add to after recombined data!!
        # self.em_data["ref"] = reflection_loss(self.em_data["ref"], self.bs_ref_pct)
        # self.em_data["sample"] = reflection_loss(
        #     self.em_data["sample"], 100 - self.bs_ref_pct
        # )

        # TODO: Convert to field amplitudes (add before or after noise)
        # self.combined_em_data = combine_em_data(self.em_data)

        # Add the noise
        # TODO: Check these equations as they are likely NEP
        self.em_data["sample"] = add_em_noise(
            self.em_data["sample"], rin_noise_pct=self.rin_noise_pct, shot_noise=True
        )
        self.em_data["ref"] = add_em_noise(
            self.em_data["ref"], rin_noise_pct=self.rin_noise_pct, shot_noise=True
        )
        return

    def perform_detector_sampling(self):
        """Perform all required simulation for Spectrometor Beam to Detector Channels"""

        # Combine all together

        # Convert Each Individually
        self.detector_data["Signal"] = convert_em_to_channels(
            self.em_data["sample"],
            self.responsivity,
            self.detector_bias_voltage,
            self.number_of_detectors,
            self.detector_area,
        )
        self.detector_data["DC"] = convert_em_to_channels(
            self.em_data["ref"],
            self.responsivity,
            self.detector_bias_voltage,
            self.number_of_detectors,
            self.detector_area,
        )
        self.detector_data["Signal_Noise"] = convert_em_to_channels(
            self.em_data["sample_noise"],
            self.responsivity,
            self.detector_bias_voltage,
            self.number_of_detectors,
            self.detector_area,
        )
        self.detector_data["DC_Noise"] = convert_em_to _channels(
            self.em_data["ref_noise"],
            self.responsivity,
            self.detector_bias_voltage,
            self.number_of_detectors,
            self.detector_area,
        )
        return

    def perform_linear_measurements(self):
        """Perform Amp to Frequency Conversion"""
        self.frequency_data["Signal"] = amp_to_frequency(
            self.detector_data["Signal"] + self.detector_data["DC"],
            self.linear_cap,
            self.threshold_voltage,
            self.reset_time,
        )  # Includes DC, which is bias.
        self.detector_data["DC_Noise"] = self.detector_data["DC_Noise"] + linear_noise(
            self.detector_data["DC"] + self.detector_data["Signal"],
            self.linear_cap,
            self.detector_resistance,
            self.temperature,
        )
        self.frequency_data["Noise"] = amp_to_frequency(
            self.detector_data["DC_Noise"]
            + self.detector_data["sample_noise"]
            + self.detector_data["ref_noise"],
            self.linear_cap,
            self.threshold_voltage,
            self.reset_time,
            self.reset_jitter,
        )
        return

    def generate_em_data(
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
        for physical_loc in dictionary.keys():
            self.em_data[physical_loc] = dict()
            for sig_type in dictionary[physical_loc].keys():
                if dictionary[physical_loc][sig_type] is True:
                    if input_df is None:
                        self.em_data[physical_loc][sig_type] = generate_source(
                            self.fwhm_bw,
                            self.central_wl,
                            self.source_point,
                            self.total_source_power,
                        )
                    else:
                        self.em_data[physical_loc][sig_type] = input_df.copy(deep=True)
                else:
                    self.em_data[physical_loc][sig_type] = dict()
        return

    def run(self):
        """Performs simulation of device"""

        # Interpolate TODO

        # Perform simulations of source to bs
        self.perform_oct_sim()

        # Convert to Linear Detections
        self.perform_detector_sampling()

        # Peform Linear Measurements
        self.perform_linear_measurements()

        # Convert to APD Detectins

        # Perform APD Measurements

        # Determine timing jitter to noise constant

        # Get noise at each output

        return


if __name__ == "__main__":
    DetectorSim = OCTSim()
    DetectorSim.run()
