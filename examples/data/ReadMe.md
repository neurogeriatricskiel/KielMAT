# Sleep Analysis Data Files

This repository contains two Parquet files that will be used for testing and providing examples for a **sleep analysis algorithm**. These files include acceleration data and metadata recorded from a wearable device positioned at the lower back.

---

## Files

### 1. `acceleration_data_sleep_analysis.parquet`
This file contains the raw acceleration data collected during the study.

- **Purpose**: 
  - To test and validate the sleep analysis algorithm.
  - To provide an example dataset for demonstrating how the algorithm works.

- **Structure**:
  - **`timestamp`**: Timestamps indicating the time of each recorded data point.
  - **`LowerBack_ACCEL_x`**: Acceleration values along the x-axis.
  - **`LowerBack_ACCEL_y`**: Acceleration values along the y-axis.
  - **`LowerBack_ACCEL_z`**: Acceleration values along the z-axis.

- **Details**:
  - Unit of acceleration: `g`
  - Sampling frequency: `128 Hz`

---

### 2. `channels_info_sleep_analysis.parquet`
This file contains metadata about the acceleration data, describing the recorded channels and their properties.

- **Purpose**:
  - To provide context for the acceleration data.
  - To define units, sampling frequency, and other metadata used in the analysis.

- **Structure**:
  - **`name`**: Names of the acceleration channels (`LowerBack_ACCEL_x`, `LowerBack_ACCEL_y`, `LowerBack_ACCEL_z`).
  - **`component`**: The axis associated with each channel (`x`, `y`, `z`).
  - **`type`**: The type of sensor data (`ACCEL` for acceleration).
  - **`tracked_point`**: The body location where the sensor was attached (`LowerBack`).
  - **`units`**: The unit of measurement (`g`).
  - **`sampling_frequency`**: The sampling frequency of the data (`128 Hz`).

---

## Usage

### Sleep Analysis Algorithm Testing
- These files serve as input data for testing the **sleep analysis algorithm**.

### Example File
- The data files are also used in example scripts to demonstrate how to:
  - Load the data.
  - Process acceleration signals.
  - Extract features for sleep analysis.

---

## How to Use

1. **Load the Data**:
   Use Python to load the Parquet files with the following code:
   ```python
   import pandas as pd
   
   # Load acceleration data
   accel_df = pd.read_parquet("acceleration_data_sleep_analysis.parquet")

   # Load metadata
   channels_df = pd.read_parquet("channels_info_sleep_analysis.parquet")

   # Extract timestamps
   dt_data = pd.to_datetime(accel_df["timestamp"])
   accel_df = accel_df.drop(columns=["timestamp"])
