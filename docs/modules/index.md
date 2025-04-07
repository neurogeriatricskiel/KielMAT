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


## [Initial Contact Classification](ic_rlc.md)

KielMAT provides two alternative methods for classifying the laterality of initial contacts (ICs)—that is, determining whether each foot-ground contact corresponds to the left or right foot. Both methods rely on gyroscope data recorded from a lower-back IMU sensor and require pre-detected IC timestamps (e.g., from Paraschiv-Ionescu or another algorithm).

- The **McCamley method** is a rule-based approach that uses the sign of the filtered angular velocity signal to determine foot laterality. Users can choose between vertical, anterior-posterior, or combined axes. This method is based on the work of [McCamley et al. (2012)](https://doi.org/10.1016/j.gaitpost.2012.02.019) and provides a lightweight, interpretable classification mechanism.

- The **Ullrich method** is a machine learning-based approach that uses a model trained on six features extracted from vertical and anterior-posterior gyroscope signals (original values, first derivatives, and second derivatives). The method supports multiple classifiers (e.g., random forest, SVM, k-NN) and is based on the work of [Ullrich et al. (2021)](https://ieeexplore.ieee.org/document/9630653).


## [Gait Spatio-temporal Parameters](gait_param.md)
The Gait Spatio-temporal Parameters module provides a comprehensive calculation of clinically relevant gait metrics based on pre-detected gait events. These include initial contacts (IC), final contacts (FC), and gait sequences derived from wearable IMU data.