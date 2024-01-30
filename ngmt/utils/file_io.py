from pathlib import Path
import json
from json_repair import repair_json


def get_unit_from_type(channel_types):
    """This function returns the unit of most common channel types automatically.
    For example ACCEL is always m/s^2, GYRO is always deg/s, MAG is always T, etc.


    Args:
        channel_types (list): List of channel types

    Returns:
        list: List of units
    """

    # create list of units as long as types
    units = [None] * len(channel_types)

    # automatically assign units if common channel types are found
    for i, ch_type in enumerate(channel_types):
        if ch_type == "ACCEL":
            units[i] = "m/s^2"
        elif ch_type == "GYRO":
            units[i] = "deg/s"
        elif ch_type == "MAGN":
            units[i] = "T"
        else:
            units[i] = None

    return units


def fix_json_file(file_path: str) -> None:
    """
    Fix a JSON file and save the fixed version.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        None
    """
    file_path = Path(file_path)
    json_string = file_path.read_text()

    try:
        decoded_object = repair_json(json_string)
        # save the decoded object as a JSON file with the original name
        with open(file_path.name, "w") as write_file:
            json_object = json.loads(decoded_object)
            json.dump(json_object, write_file, indent=4)
    except Exception:
        # Not even this library could fix this JSON
        print("Could not fix JSON file")
