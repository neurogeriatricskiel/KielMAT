import os
from ngmt.datasets._keepcontrol import load_recording


def main():
    path_name = "Z:\\Keep Control\\Data\\lab dataset\\rawdata\\sub-pp035\\motion"
    file_name = "sub-pp035_task-homePart1_run-off_tracksys-imu_motion.tsv"

    recording = load_recording(
        os.path.join(path_name, file_name),
        tracking_systems=["imu", "omc"],
        tracked_points={
            "imu": ["pelvis", "sternum"],
            "omc": [f"{lr}_{ap}sis" for ap in ["a", "p"] for lr in ["l", "r"]]
            + [f"m_ster{i}" for i in range(1, 4)],
        },
    )
    return


if __name__ == "__main__":
    main()
