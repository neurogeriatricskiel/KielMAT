from ngmt.datasets import mobilised


def main() -> None:
    # Load the dataset
    # mobilised.load_dataset(progressbar=True)
    mobilised.load_file(
        tracking_systems=[
            # "SU",
            "PressureInsoles_raw",
        ],
        tracked_points={
            # "SU": ["LowerBack"],
            "PressureInsoles_raw": ["LeftFoot", "RightFoot"],
        },
    )
    return


if __name__ == "__main__":
    main()
