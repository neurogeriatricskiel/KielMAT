from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

VALID_CHANNEL_TYPES = {"ACCEL", "ANGACCEL", "GYRO", "JNTANG", "LATENCY", "MAGN", "MISC", "ORNT", "POS", "VEL"}
VALID_COMPONENT_TYPES = {"x", "y", "z", "quat_x", "quat_y", "quat_z", "quat_w", "n/a"}

@dataclass
class FileInfo:
    TaskName: str
    SamplingFrequency: float
    TaskDescription: Optional[str] = None
    Instructions: Optional[str] = None
    Manufacturer: Optional[str] = None
    ManufacturersModelName: Optional[str] = None
    MissingValues: Optional[str] = None
    RotationOrder: Optional[str] = None
    RotationRule: Optional[str] = None
    SamplingFrequencyEffective: Optional[float] = None
    SpatialAxes: Optional[str] = None
    TrackedPointsCount: Optional[int] = None
    TrackingSystemName: Optional[str] = None
    Channels: List[ChannelMetaData] = field(default_factory=list)


@dataclass
class ChannelMetaData:
    name: str
    component: str
    ch_type: str
    tracked_point: str
    units: str
    placement: Optional[str] = None
    description: Optional[str] = None
    sampling_frequency: Optional[float] = None
    status: Optional[str] = None
    status_description: Optional[str] = None
    additional_columns: Optional[dict] = field(default_factory=dict)

    def __post_init__(self):
    if self.ch_type not in VALID_CHANNEL_TYPES:
        raise ValueError(f"Invalid channel type {self.type}. Must be one of {VALID_CHANNEL_TYPES}")
    if self.component not in VALID_COMPONENT_TYPES:
        raise ValueError(f"Invalid component type {self.component}. Must be one of {VALID_COMPONENT_TYPES}")


@dataclass
class MotionData:
    info: FileInfo
    times: np.ndarray  # Can be a 1D array representing timestamps
    channel_names: List[str]  # Can be a list of channel names
    time_series: np.ndarray  # Can be a 2D array where each row is a channel

    def __post_init__(self):
        if len(self.times) != self.time_series.shape[1]:
            raise ValueError("The length of `times` should match the number of columns in `time_series`")
        
        if len(self.channel_names) != self.time_series.shape[0]:
            raise ValueError("The number of `channel_names` should match the number of rows in `time_series`")
        
    @classmethod
    def import_time_series(self, file: str):
        # Placeholder function to be implemented
        pass
