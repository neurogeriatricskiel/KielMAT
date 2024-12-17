from bids_validator import BIDSValidator
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, List, Optional, Union, Sequence

REQUIRED_COLUMNS = [
    "name",
    "component",
    "type",
    "tracked_point",
    "units",
    "sampling_frequency",
]

# See: https://bids-specification.readthedocs.io/en/stable/modality-specific-files/motion.html#restricted-keyword-list-for-channel-type
VALID_CHANNEL_TYPES = {
    "ACCEL",
    "ANGACCEL",
    "BARO",
    "GYRO",
    "JNTANG",
    "LATENCY",
    "MAGN",
    "MISC",
    "ORNT",
    "POS",
    "VEL",
}

# See: https://bids-specification.readthedocs.io/en/stable/modality-specific-files/motion.html#restricted-keyword-list-for-channel-component
VALID_COMPONENT_TYPES = {
    "x",
    "y",
    "z",
    "quat_x",
    "quat_y",
    "quat_z",
    "quat_w",
    "n/a",
    "NaN",
    "nan",
}

# See https://bids-specification.readthedocs.io/en/stable/modality-agnostic-files.html#participants-file
VALID_INFO_KEYS = {
    "Subject",
    "Session",
    "Task",
}


VALID_CHANNEL_STATUS_VALUES = ["good", "bad", "n/a", "NaN", "nan"]


@dataclass(kw_only=True)
class KielMATRecording:
    """Dataclass to hold any data and associated infos for a KielMAT recording.

    Attributes:
        data (dict): The data is stored as a pandas DataFrame for each unique tracking system.
        channels (dict): The channels descriptions are stored as a pandas DataFrame for each unique tracking system.
        info (dict): The infos on the subject, task, and more, are stored as a nested dictionary.
        events (dict): The events are stored as a pandas DataFrame for each unique tracking system.
        events_info (dict): The event infos are stored as a nested dictionary.
    """

    data: dict[str, pd.DataFrame]
    channels: dict[str, pd.DataFrame]
    info: None | dict[str, Any] = None
    events: None | dict[str, pd.DataFrame] = None
    events_info: None | dict[str, Any] = None

    def __post_init__(self):
        # Validate channels when an instance is created
        self.validate_channels()

    def validate_channels(self) -> str:
        """
        Validates the channel dataframes for each system.

        This function checks if the channel dataframes have the required columns in the correct order,
        and if the data types of the columns are valid. It also performs additional value checks for
        optional columns.

        Raises:
            ValueError: If the channel dataframe does not have the required columns in the correct order,
                or if the 'component' column contains invalid values, or if the 'type' column is not
                uppercase strings, or if the 'status' column contains invalid values.
            TypeError: If the 'name' column is not of type string.

        Returns:
            Confirmation message indicating that all channel dataframes are valid.
        """
        for system_name, df in self.channels.items():
            # Check required columns and their order
            if not df.columns.tolist()[:6] == REQUIRED_COLUMNS:
                raise ValueError(
                    f"Channels dataframe for '{system_name}' does not have the required columns in correct order. The correct order is: {REQUIRED_COLUMNS}."
                )

            # Check data types
            if not all(isinstance(name, str) for name in df["name"]):
                raise TypeError(
                    f"Column 'name' in '{system_name}' must be of type string."
                )
            invalid_components = set(
                [
                    item
                    for item in df["component"]
                    if item not in VALID_COMPONENT_TYPES and not pd.isna(item)
                ]
            )
            if invalid_components:
                raise ValueError(
                    f"Column 'component' in '{system_name}' contains invalid values: {invalid_components}."
                )
            if not all(isinstance(typ, str) and typ.isupper() for typ in df["type"]):
                raise ValueError(
                    f"Column 'type' in '{system_name}' must be uppercase strings."
                )

            # Additional value checks for optional columns
            if "status" in df.columns and not all(
                s in VALID_CHANNEL_STATUS_VALUES for s in df["status"] if s != "n/a"
            ):
                raise ValueError(
                    f"Column 'status' in '{system_name}' contains invalid values."
                )

        return "All channel dataframes are valid."

    def add_events(self, tracking_system: str, new_events: pd.DataFrame) -> None:
        """Add events to the recording for a specific tracking system.

        Args:
            tracking_system (str): Tracking system for which events are to be added.
            new_events (pd.DataFrame): Events to be added in BIDS format.
        """
        if self.events is None:
            self.events = {}

        if tracking_system not in self.events:
            self.events[tracking_system] = new_events
        else:
            existing_events = self.events[tracking_system]
            self.events[tracking_system] = pd.concat(
                [existing_events, new_events], ignore_index=True
            )

    def add_info(self, key: str, value: Any) -> None:
        """Add information to the info dictionary. Valid keys are : 'Subject', 'Session', 'Task'.

        Args:
            key (str): The key for the information.
            value (Any): The value of the information.

        Raises:
            ValueError: If the provided 'key' is not one of the valid info keys.

        Examples:
            >>> recording.add_info("Subject", "01")
        """
        if self.info is None:
            self.info = {}

        # Check if the key belongs to a list of keywords
        if key not in VALID_INFO_KEYS:
            print(
                f"Warning: Invalid info key '{key}'. Valid info keys are: {VALID_INFO_KEYS}"
            )

        # add the key-value pair to the info dictionary
        self.info[key] = value

        # Check if the value are lower case, if not, convert to lower case and give warning
        if isinstance(value, str):
            self.info[key] = value.lower()
            print(
                f"Warning: The value of the key '{key}' should be lower case. Converted to lower case."
            )

        # check if value contains underscore or space, if yes, remove and give warning
        if "_" in value or " " in value:
            self.info[key] = value.replace("_", "").replace(" ", "")
            print(
                f"Warning: The value of the key '{key}' should not contain underscore or space. Removed underscore and space."
            )

    def export_events(
        self,
        file_path: str,
        tracking_system: Optional[str] = None,
        file_name: Optional[str] = None,
        bids_compatible_fname: Optional[bool] = False,
    ) -> None:
        """Export events for a specific tracking system to a file.

        Args:
            tracking_system (Optional[str]): Tracking system for which events are to be exported.
                If None, events from all tracking systems will be exported (default is None).
            file_path (str): Path to the directory where the file should be saved.
            file_name (Optional[str]): Name of the file to be exported. If None, a default name will be used.
            bids_compatible_fname (bool): Flag indicating whether the exported filename should be BIDS compatible (default is False).
        """
        if self.events is not None:
            if tracking_system is None:
                all_events = pd.concat(
                    self.events.values(),
                    keys=self.events.keys(),
                    names=["tracking_system"],
                )
                if file_name is None:
                    file_name = "all_events.csv"
                if bids_compatible_fname:
                    # Construct the filename using subject ID and task name
                    subject_id = self.info.get("Subject", "")
                    task_name = self.info.get("Task", "")
                    # check if subject_id and task_name are present in the info dictionary
                    if subject_id == None or task_name == None:
                        raise ValueError(
                            "Subject ID and Task Name should be specified in the info dictionary."
                        )
                    file_name = f"sub-{subject_id}_task-{task_name}_events.csv"
                    # check if session is present in the info dictionary
                    session = self.info.get("Session")
                    if session != None:
                        file_name = f"sub-{subject_id}_ses-{session}_task-{task_name}_events.csv"
                    file_path = Path(file_path).joinpath(file_name)
                    all_events.to_csv(file_path, sep="\t", index=False)
                else:
                    file_path = Path(file_path).joinpath(file_name)
                    all_events.to_csv(file_path, index=False)
            elif tracking_system in self.events:
                if file_name is None:
                    file_name = f"{tracking_system}_events.csv"
                if bids_compatible_fname:
                    file_name = file_name.replace(".csv", "_events.tsv")
                    file_path = Path(file_path).joinpath(file_name)
                    self.events[tracking_system].to_csv(
                        file_path, sep="\t", index=False
                    )
                else:
                    file_path = Path(file_path).joinpath(file_name)
                    self.events[tracking_system].to_csv(file_path, index=False)

            # check if file_path is BIDS compatible
            if bids_compatible_fname:
                # validate the file_path
                validator = BIDSValidator()
                errors = validator.is_bids(file_path)
                if errors:
                    raise ValueError(f"File path '{file_path}' is not BIDS compatible.")
