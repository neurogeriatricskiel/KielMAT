# Overview

This section contains a series of examples designed to demonstrate key features and functionalities within the KielMAT toolbox. These examples will guide users through common workflows and illustrate how to apply different modules to motion analysis data.

## Construct Your DataClass

The `DataClass` is the central data structure in KielMAT, which stores and organizes motion data along with associated events. The following examples guide you through the process of loading and structuring your data.

### [Example 1: Load data into KielMAT](basic_01_load_Data_into_KielMAT.md)
In this example, you will learn how to load motion data into `KielMAT`'s `DataClass`. This step is essential for processing any motion capture data, whether it's from IMUs, C3D files, or other data formats.

### [Example 2: Load datasets](basic_02_load_dataset.md)
This example demonstrates how to load datasets into `KielMAT`. You will learn how to import data from different sources, handle multiple datasets, and integrate them into a unified structure for analysis.

### [Example 3: Events in DataClass](basic_03_events.md)
The `DataClass` not only stores motion data but also provides functionality to mark and organize events (such as gait sequences, initial contacts, or other notable movement occurrences). This example shows how to tag specific events within the `DataClass`, allowing you to analyze them in the context of the motion data.

## Run Modules

KielMAT includes several pre-built modules to analyze motion data for different tasks. The following examples demonstrate how to apply each module to extract meaningful information from your data.

### [Example 4: Gait Sequence Detection](modules_04_gsd.md)
This example introduces the [Gait Sequence Detection](https://neurogeriatricskiel.github.io/KielMAT/modules/gsd/) module. This module identifies gait sequences using 3D accelerometer data from a lower back sensor.

### [Example 5: Initial Contact Detection](modules_05_icd.md)
This example introduces the [Initial Contact Detection](https://neurogeriatricskiel.github.io/KielMAT/modules/icd/) module. It identifies and characterizes initial contacts within each detected gait sequence using the gait sequence detection module.

### [Example 6: Physical Activity Monitoring](modules_06_pam.md)
This example introduces the [Physical Activity Monitoring](https://neurogeriatricskiel.github.io/KielMAT/modules/pam/) module. The example shows how the module is implemented on sample 3D acceleration data from an IMU sensor to monitor physical activity levels.

### [Example 7: Postural Transition Detection](modules_07_ptd.md)
This example introduces the [Postural Transition Detection](https://neurogeriatricskiel.github.io/KielMAT/modules/ptd/) module. It demonstrates how the module is implemented on sample 3D acceleration and 3D angular velocity data from a lower back IMU sensor to detect postural transitions, such as sit-to-stand or stand-to-sit.

### [Example 8: Turn Detection](modules_08_td.md)
This example introduces the [Turn Detection](https://neurogeriatricskiel.github.io/KielMAT/modules/td/) module. It demonstrates how the module is implemented on sample 3D acceleration and 3D angular velocity data from a lower back IMU sensor to detect turns.
