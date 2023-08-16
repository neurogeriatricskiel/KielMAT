import numpy as np
from typing import Optional, List
from dataclasses import dataclass, field

@dataclass(kw_only=True)
class IMURecording:
    type: str
    units: str
    fs: float
    data: np.array

@dataclass(kw_only=True)
class IMUDevice:
    tracked_point: str
    device_manufacturer: Optional[str] = ""
    device_model: Optional[str] = ""
    recordings: List[IMURecording] = field(default_factory=list)

@dataclass(kw_only=True)
class IMUDataset:
    subject_id: str  # TODO: replace with relevant subject info
    devices: List[IMUDevice] = field(default_factory=list)