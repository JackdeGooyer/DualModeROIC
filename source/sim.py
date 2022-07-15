"""Simulate Design of Device -- Done in Fourier Domain
"""

from pathlib import Path
from typing import Dict
from oct import beam_split, reflection_loss, get_time_delay
import pandas as pd

# Points in transform
SAMPLE_POINTS = 100000
SAMPLE_DEPTHS = 100

# Engineering Parameter
SAMPLE_RATE = 400000  # Hz
DETECTOR_HEIGHT = 20e-06  # m
DETECTOR_WIDTH = 100e-06  # m
DETECTOR_NUMBER = 60  # pixels

DETECTOR_AREA = DETECTOR_HEIGHT * DETECTOR_WIDTH

# OCT Beam Params
REFERENCE_REFLECTIVITY_PCT = 100
BEAM_SPLIT_REF_PCT = 50  # Sample/Reference
BEAM_LOSSES_PCT = 10

# Sample Specs
SAMPLE_REFLECTIVITY_PCT = 1
SAMPLE_REFRACTIVE_INDEX = 1.5  # n
DEFAULT_SAMPLE_DEPTH = 10e-03  # m

# File Paths
BASEDIR = Path(__file__).resolve().parent.parent
DEFAULT_RESPONSIVITY_FILEPATH = BASEDIR.joinpath(r"Data\DetectorResponsivity.csv")
DEFAULT_SOURCE_FILEPATH = BASEDIR.joinpath(r"Data\SourcePower.csv")


class Detector:
    """Class for Detector Data Handling"""

    # Data
    em_data: Dict[str, pd.DataFrame] = None
    detectors_data: Dict[str, pd.DataFrame] = None

    # Files
    _responsivity_path: Path = DEFAULT_RESPONSIVITY_FILEPATH
    _source_path: Path = DEFAULT_SOURCE_FILEPATH

    responsivity: pd.DataFrame = None
    source_power: pd.DataFrame = None

    # Sim Depth
    sample_points: int = SAMPLE_POINTS
    sample_depths: int = SAMPLE_DEPTHS

    # Engineering Parameter
    sample_rate: int = SAMPLE_RATE
    number_of_detectors: int = DETECTOR_NUMBER
    detector_area: int = DETECTOR_AREA

    # OCT Beam Params
    ref_reflect_pct: float = REFERENCE_REFLECTIVITY_PCT
    bs_ref_pct: float = BEAM_SPLIT_REF_PCT
    beam_losses_pct: float = BEAM_LOSSES_PCT

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
        self.responsivity = pd.read_csv(self._responsivity_path)
        self.source_power = pd.read_csv(self._source_path)
        self.sample_depth = sample_depth
        return

    def run(self):
        """Performs simulation of device"""

        # Interpolate TODO

        # Perform simulations of source to bs
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

        # Convert to Linear Detections (add the noise too)

        # Peform Linear Measurements

        # Convert to APD Detectins

        # Perform APD Measurements

        # Determine timing jitter to noise constant

        # Get noise at each output

        return


if __name__ == "__main__":
    DetectorSim = Detector()
    DetectorSim.run()
