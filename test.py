import numpy as np
import matplotlib.pyplot as plt
import os
from ngmt.datasets import (
    # keepcontrol,
    mobilised,
)

# from ngmt.modules.gsd import GSDA


def main():
    base_path = "/mnt/neurogeriatrics_data/Mobilise-D/rawdata"
    sub_ids = [
        folder_name
        for folder_name in os.listdir(base_path)
        if folder_name.startswith("sub-")
        and os.path.isdir(os.path.join(base_path, folder_name))
    ]

    for idx_sub, sub_id in enumerate(sub_ids):
        session_name = "Free-living"
        file_path = os.path.join(base_path, sub_id, session_name, "data.mat")

        motion_data = mobilised._load_file(file_path=file_path)
        break

    return


if __name__ == "__main__":
    main()
