# Overview

This section of the project documentation presents the available **modules** within the KielMAT toolbox. These modules are developed to analyze motion data and assist in movement analysis-related activities. Each module is briefly introduced below, with links to more comprehensive guides.


## [Gait Sequence Detection (Paraschiv-Ionescu)](gsd.md)

The **Gait Sequence Detection** module is based on the [Paraschiv-Ionescu](https://ieeexplore.ieee.org/document/9176281) algorithm, which aims to identify gait sequences from motion data. Specifically, it uses 3D accelerometer data from lower back IMU sensors to detect these gait sequences.


## [Initial Contact Detection (Paraschiv-Ionescu)](icd.md)

The **Initial Contact Detection** module is also based on the [Paraschiv-Ionescu](https://ieeexplore.ieee.org/document/9176281) algorithm and is designed to identify the initial contact in each gait sequence. The outputs of this module are essential for accurately measuring temporal gait parameters, such as stride time and gait symmetry.

## [Physical Activity Monitoring](pam.md)

The **Physical Activity Monitoring** module tracks and analyzes physical activity levels using 3D acceleration data from IMU sensors. It provides outputs such as activity intensity and duration, enabling a comprehensive assessment of an individual's movement behavior throughout the day.

## [Postural Transition Detection (Pham)](ptd.md)

The **Postural Transition Detection** module is based on the work of [Pham](https://www.frontiersin.org/journals/neurology/articles/10.3389/fneur.2018.00652/full) and aims to identify and analyze postural transitions, such as sit-to-stand or stand-to-sit movements. The module detects these transitions using 3D acceleration and 3D angular velocity data from lower back IMU sensors. It also calculates key spatial-temporal parameters, such as the angle of postural transition and maximum flexion velocity.

## [Turn Detection (Pham)](td.md)

The **Turn Detection** module is based on the work of [Pham](https://pubmed.ncbi.nlm.nih.gov/28443059/) and aims to identify and characterize body turns using 3D acceleration and angular velocity data from lower back IMU sensors. The module also calculates key spatial-temporal parameters, such as the angle of turn and peak angular velocity.