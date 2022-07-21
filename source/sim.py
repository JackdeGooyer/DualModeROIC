"""Simulate Design of Device -- Done in Fourier Domain
"""

from pathlib import Path
from typing import Dict
from oct import beam_split, reflection_loss, get_time_delay, get_em_noise
from detector import convert_em_to_channels
from linear import amp_to_frequency, linear_noise
import pandas as pd

# Points in transform
SAMPLE_POINTS = 1000
SAMPLE_DEPTHS = 760  # For 1.3mm sample, this is

# Temperature
TEMPERATURE = 300.0  # K

# Engineering Parameter
SAMPLE_RATE = 400000.0  # Hz
DETECTOR_HEIGHT = 10e-06  # m
DETECTOR_WIDTH = 200e-06  # m
DETECTOR_NUMBER = 1024  # pixels
DETECTOR_VOLTAGE = 3.3  # V
DETECTOR_RESISTANCE = 0.2  # Ohms TODO: is this calculated?
DETECTOR_AREA = DETECTOR_HEIGHT * DETECTOR_WIDTH  # m^2

# OCT Beam Params
REFERENCE_REFLECTIVITY_PCT = 100.0
BEAM_SPLIT_REF_PCT = 50.0  # Sample/Reference
BEAM_LOSSES_PCT = 10.0
RIN_NOISE_PCT = 1.0

# Sample Specs
SAMPLE_REFLECTIVITY_PCT = 1.0
SAMPLE_REFRACTIVE_INDEX = 1.5  # n
DEFAULT_SAMPLE_DEPTH = 1.3e-03  # m

# Linear Specs
CAPACITOR = 1e-12  # F
RESET_TIME = 10e-09  # s
RESET_JITTER = 1e-09  # s
BUFFER_GAIN = 1.0  # V/V
VOLTAGE_THRESHOLD = 3.3  # V

# File Paths
BASEDIR = Path(__file__).resolve().parent.parent
DEFAULT_RESPONSIVITY_FILEPATH = BASEDIR.joinpath(r"Data\DetectorResponsivity.csv")
DEFAULT_SOURCE_FILEPATH = BASEDIR.joinpath(r"Data\SourcePower.csv")


class Detector:
    """Class for Detector Data Handling"""

    # Data
    em_data: Dict[str, pd.DataFrame] = None
    detector_data: Dict[str, pd.DataFrame] = None
    frequency_data: Dict[str, pd.DataFrame] = None

    # Files
    _responsivity_path: Path = DEFAULT_RESPONSIVITY_FILEPATH
    _source_path: Path = DEFAULT_SOURCE_FILEPATH

    responsivity: pd.DataFrame = None
    source_power: pd.DataFrame = None

    # Sim Depth
    sample_points: int = SAMPLE_POINTS
    sample_depths: int = SAMPLE_DEPTHS

    # Engineering Parameter
    sample_rate: float = SAMPLE_RATE
    number_of_detectors: int = DETECTOR_NUMBER
    detector_area: int = DETECTOR_AREA
    detector_bias_voltage: float = DETECTOR_VOLTAGE
    detector_resistance: float = DETECTOR_RESISTANCE
    temperature: float = TEMPERATURE

    # OCT Beam Params
    ref_reflect_pct: float = REFERENCE_REFLECTIVITY_PCT
    bs_ref_pct: float = BEAM_SPLIT_REF_PCT
    beam_losses_pct: float = BEAM_LOSSES_PCT
    rin_noise_pct: float = RIN_NOISE_PCT

    # Linear Values
    linear_cap: float = CAPACITOR
    reset_time: float = RESET_TIME
    reset_jitter: float = RESET_JITTER
    buffer_gain: float = BUFFER_GAIN
    threshold_voltage: float = VOLTAGE_THRESHOLD

    # Sample Specs
    sample_reflect_pct: float = SAMPLE_REFLECTIVITY_PCT
    sample_ref_index: float = SAMPLE_REFRACTIVE_INDEX
    tissue_depth: float = DEFAULT_SAMPLE_DEPTH

    def __init__(self, sample_depth: float = DEFAULT_SAMPLE_DEPTH):
        """Reads all data files into Detector Class

        Parameters
        ----------
        sample_depth : float, optional
            Tissue Depth of Sample, by default DEFAULT_SAMPLE_DEPTH
        """
        self.responsivity = pd.read_csv(self._responsivity_path, index_col=False)
        self.source_power = pd.read_csv(self._source_path, index_col=False)
        # TODO Add resample of source power for more points (interpolate)
        self.sample_depth = sample_depth
        self.em_data = dict()
        self.detector_data = dict()
        self.frequency_data = dict()
        return

    def perform_oct_sim(self):
        """Perform all required simulation for the OCT portion of the device"""
        self.em_data = beam_split(self.source_power, split_pct_ref=self.bs_ref_pct)
        self.em_data["ref"] = reflection_loss(self.em_data["ref"], self.ref_reflect_pct)
        self.em_data["sample"] = reflection_loss(
            self.em_data["sample"], self.sample_reflect_pct
        )
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
        self.em_data["ref_noise"] = get_em_noise(
            self.em_data["ref"], rin_noise_pct=self.rin_noise_pct, shot_noise=True
        )
        self.em_data["sample_noise"] = get_em_noise(
            self.em_data["sample"], rin_noise_pct=self.rin_noise_pct, shot_noise=True
        )
        return

    def perform_detector_sampling(self):
        """Perform all required simulation for Spectrometor Beam to Detector Channels"""
        self.detector_data["Signal"] = convert_em_to_channels(
            self.em_data["sample"],
            self.responsivity,
            self.detector_bias_voltage,
            self.number_of_detectors,
        )
        self.detector_data["DC"] = convert_em_to_channels(
            self.em_data["ref"],
            self.responsivity,
            self.detector_bias_voltage,
            self.number_of_detectors,
            dark_current=True,
        )
        self.detector_data["Signal_Noise"] = convert_em_to_channels(
            self.em_data["sample_noise"],
            self.responsivity,
            self.detector_bias_voltage,
            self.number_of_detectors,
        )
        self.detector_data["DC_Noise"] = convert_em_to_channels(
            self.em_data["ref_noise"],
            self.responsivity,
            self.detector_bias_voltage,
            self.number_of_detectors,
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
    DetectorSim = Detector()
    DetectorSim.run()
