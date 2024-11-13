# Overview

This part of the project documentation focuses on the available **utilities** that assist in various preprocessing, importing, and data handling tasks within the KielMAT toolbox. The following utilities provide functionality to import data from various formats, perform preprocessing, and estimate orientations for movement analysis.


### [Data Preprocessing](preprocessing.md)

The **Data Preprocessing** provide a set of utilities designed to prepare motion data for analysis. These functions are essential for cleaning, filtering, and transforming raw sensor data into a suitable format for subsequent analysis.


### [Data Import](importers.md)

The **Data Import** within KielMAT are designed to handle data from various sources and formats. This makes it easy to load and integrate different datasets into the toolbox. Below are examples of the import functions available for different sensor data sources:

- **Axivity Data Import**  
  The `import_axivity` function imports Axivity data from specified files. It reads Axivity data files, performs lowpass filtering, and calibrates for gravity, ensuring that the data is ready for analysis. The function outputs the data along with the associated channel information in a format compatible with the KielMAT `KielMATRecording` class. This function allows easy integration of Axivity data, including accelerometer information for a specific tracked point.

- **APDM Mobility Lab Data Import**  
  The `import_mobilityLab` function imports data from the APDM Mobility Lab system, which is commonly used for gait analysis. It reads data from sensor files (in HDF5 format) and extracts accelerometer, gyroscope, and magnetometer readings for specific tracked points such as lumbar. The function outputs the data along with channel information, which is formatted according to the required specifications, including sensor name, component type, units, and sampling frequency. It handles multiple tracked points and can process data from various sensor types in one operation.

These import functions facilitate the integration of complex sensor data into KielMAT for analysis and ensure that data from different systems can be processed consistently and efficiently.

### [Orientation estimation](orientation_estimation.md)

The **Orientation estimation** utilities help estimate the orientation of the tracked points in space, crucial for movement analysis involving angular data. These functions apply algorithms that use accelerometer and gyroscope data to estimate the orientation of the sensors, providing insights into body posture and movement. The utility ensures that orientation data is available for further analysis, including gait and posture assessment.
