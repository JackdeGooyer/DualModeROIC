"""Contains all configuration and globals"""

from pathlib import Path
from typing import Dict, Union
import pandas as pd

RecursiveDict = Dict[str, Union["RecursiveDict", pd.DataFrame]]

# Souce Parameters
FULL_WIDTH_HALF_MAX_BW = 129  # nm
CENTRAL_WAVELENGTH = 785  # nm
SOUCE_POINTS = 1000
SOURCE_POWER = 5.2e-03  # W

# Points in transform
SAMPLE_DEPTHS = 10  # For 1.3mm sample, set to 760

# Temperature
TEMPERATURE = 300.0  # K

# Engineering Parameter
SAMPLE_RATE = 400000.0  # Hz
DETECTOR_HEIGHT = 10e-06  # m
DETECTOR_WIDTH = 200e-06  # m
DETECTOR_NUMBER = 1024  # pixels
DETECTOR_VOLTAGE = 2  # V
DETECTOR_RESISTANCE = 10e07  # Ohms TODO: is this calculated?
DETECTOR_AREA = DETECTOR_HEIGHT * DETECTOR_WIDTH  # m^2

# OCT Beam Params
REFERENCE_REFLECTIVITY_PCT = 100.0
BEAM_SPLIT_REF_PCT = 50.0  # Sample/Reference
BEAM_LOSSES_PCT = 10.0
RIN_NOISE_PCT = 0.05  # Changed from 1

# Sample Specs
SAMPLE_REFLECTIVITY_PCT = 1.0
SAMPLE_REFRACTIVE_INDEX = 1.5  # n
DEFAULT_SAMPLE_DEPTH = 1.3e-05  # m

# Linear Specs
CAPACITOR = 1e-12  # F
RESET_TIME = 10e-09  # s
TDC_JITTER = 1e-09  # s
BUFFER_GAIN = 1.0  # V/V
VOLTAGE_THRESHOLD = 3.3  # V

# ADC Specs
MIN_SAMPLE_RATE = 400000.0  # Hz
ADC_BITS = 15  # Bits (Ceil(Log2((1/(MIN_SAMPLE_RATE)/TDC_RESOLUTION))))
TDC_RESOLUTION = 100 * 10 ^ -12  # S

# File Paths
BASEDIR = Path(__file__).resolve().parent.parent
DEFAULT_RESPONSIVITY_FILEPATH = BASEDIR.joinpath(r"Data\DetectorResponsivity.csv")
DEFAULT_SOURCE_FILEPATH = BASEDIR.joinpath(r"Data\SourcePower.csv")
