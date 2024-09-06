---
title: "KielMAT: Kiel Motion Analysis Toolbox - An Open-Source Python Toolbox for Analyzing Neurological Motion Data from Various Recording Modalities"
tags:
  - Python
  - Motion capture
  - Neurology
  - Accelerometer
  - Gyroscope
authors:
  - name: Masoud Abedinifar^[co-first author]
    orcid: 0000-0002-4050-9835
    affiliation: 1
  - name: Julius Welzel^[co-first author]
    orcid: 0000-0001-8958-0934
    affiliation: 1
  - name: Clint Hansen
    orcid: 0000-0003-4813-3868
    affiliation: 1
  - name: Walter Maetzler
    orcid: 0000-0002-5945-4694
    affiliation: 1
  - name: Robbin Romijnders^[corresponding author]
    orcid: 0000-0002-2507-0924
    affiliation: 1
affiliations:
  - name: Neurogeriatrics, Department of Neurology, University Hospital Schleswig-Holstein (USKH), Kiel Germany
    index: 1
date: 08 Feb 2024
bibliography: references.bib
output:
  quarto::pdf_document:
    latex_engine: xelatex
    keep_tex: true
---



<div style="text-align: justify;">

# Summary
The Kiel Motion Analysis Toolbox (KielMAT) is an open-source Python-based toolbox designed for processing human motion data, following open-science practices. KielMAT offers a range of algorithms for the processing of motion data in neuroscience and biomechanics and currently includes implementations for gait sequence detection, initial contact detection, physical activity monitoring, sit to stand and stand to sit detection algorithms. These algorithms aid in identifying patterns in human motion data on different time scales. The KielMAT is versatile in accepting motion data from various recording modalities, including IMUs that provide acceleration data from specific body locations such as the pelvis or wrist. This flexibility allows researchers to analyze data captured using different hardware setups, ensuring broad applicability across studies. Some of the toolbox algorithms have been developed and validated in clinical cohorts, allowing extracted patters to be used in a clinical context. The modular design of KielMAT allows the toolbox to be easily extended to incorporate relevant algorithms which will be developed in the research community. The toolbox is designed to be user-friendly and is accompanied by a comprehensive documentation and practical examples, while the underlying data structures build on the Motion BIDS specification [@jeung:2024]. The KielMAT toolbox is intended to be used by researchers and clinicians to analyze human motion data from various recording modalities and to promote the utilization of open-source software in the field of human motion analysis.

# Statement of need
Physical mobility is an essential aspect of health, as impairment in mobility is associated with reduced quality of life, falls, hospitalization, mortality, and other adverse events in many chronic conditions. Traditional mobility measures include patient-reported outcomes, objective clinical assessments, and subjective clinical assessments. These measures are linked to the perception and capacity aspects of health, which often fail to show relevant effects on daily function at an individual level [@maetzler:2021]. Perception involves surveys and patient-reported outcomes that capture how individuals feel about their own functional abilities, while capacity refers to clinical assessments of an individual's ability to perform various tasks. To complement both patient-reported (perception) and clinical (capacity) assessment approaches, digital health technology (DHT) introduces a new paradigm for assessing daily function. By using wearable devices, DHT provides objective insights into an individual's functional performance, directly linking it to the International Classification of Functioning, Disability and Health (ICF) framework [@ICF:2001; @ustun:2003] for assessing how people perform in everyday life activities. [@warmerdam:2020; @fasano:2020; @maetzler:2021; @hansen:2018; @buckley:2019; @celik:2021]. DHT allows an objective impression of how patients function in everyday life and their ability to routinely perform everyday activities [@hansen:2018; @buckley:2019; @celik:2021]. Nonetheless, due to several persisting challenges in this field, current tools and techniques are still in their infancy [@micoamigo:2023]. Many studies often used proprietary software to clinically relevant features of mobility. The development of easy-to-use and open-source software is imperative for transparent features extraction in research and clinical settings. KielMAT addresses this gap by providing software for human mobility analysis, to be used by motion researchers and clinicians, while promoting open-source practices. The conceptual framework builds on Findable, Accessible, Interoperable and Reusable (FAIR) data principles to encourage the use of open source software as well as facilitate data sharing and reproducibility in the field of human motion analysis [@wilkinson:2016]. The KielMAT comprises several modules which are implemented and validated with different dataset and each serving distinct purposes in human motion analysis:

1. Gait Sequence Detection (GSD): Identifies walking bouts to analyze gait patterns and abnormalities, crucial for neurological and biomechanical assessments.

2. Initial Contact Detection (ICD): Pinpoints the moment of initial foot contact during walking, aiding in understanding gait dynamics and stability.

3. Physical Activity Monitoring (PAM): Determines the intensity level of physical activities based on accelerometer signals.

These modules are pivotal because they enable researchers and clinicians to extract meaningful insights from motion data captured in various environments and conditions. These modules are designed to process data from wearable devices, which offer distinct advantages over vision-based approaches. wearable devices such as IMUs provide continuous monitoring capabilities, enabling users to wear them throughout the day in diverse settings without logistical constraints posed by camera-based systems.

# State of the field
With the growing availability of digital health data, open-source implementations of relevant algorithms are increasingly becoming available. From the Mobilise-D consortium, the recommended algorithms for assessing real-world gait were released, but these algorithms were developed in MATLAB, that is not free to use [@mobilised:2019; @mobilised:2023]. Likewise, an algorithm for the estimation of gait quality was released, but it is also only available in MATLAB [@gaitqualitycomposite:2016; @MATLAB:2022]. Alternatively, open-source, Python packages are available, for example to detect gait and extract gait features from a low back-worn inertial measurement unit (IMU) [@czech:2019], or from two feet-worn IMUs [@kuederle:2024]. These advancements facilitate broader accessibility and usability across research and clinical applications. Additionally, innovative approaches like Mobile GaitLab focus on video input for predicting key gait parameters such as walking speed, cadence, knee flexion angle at maximum extension, and the Gait Deviation Index, leveraging open-source principles and designed to be accessible to non-computer science specialists [@kidzinski:2020; @mobile-gaitlab:2020]. Moreover, tools such as Sit2Stand and Sports2D contribute to this landscape by offering user-friendly platforms for assessing physical function through automated analysis of movements like sit-to-stand transitions and joint angles from smartphone videos (Sports2D) [@Boswell:2023; @Pagnon:2023]. KielMAT builds forth on these toolboxes by providing a module software package that goes beyond the analysis of merely gait, and extends these analyses by additionally allowing for the physical activity monitoring [@van:2013] and other daily life-relevant movements, such as sit-to-stand and stand-to-sit transitions [@pham:2017] as well as turns [@pham:2018].

# Provided Functionality
KielMAT offers a comprehensive suite of algorithms for motion data processing in neuroscience and biomechanics. Currently, the toolbox includes implementations for gait sequence detection (GSD) and initial contact detection (ICD), whereas algorithms for postural transition analysis [@pham:2017] and turns [@pham:2018] are under current development. KielMAT is built on principles from the Brain Imaging Data Structure (BIDS) [@gorgolewski:2016] and for the motion analysis data are organized similar to the Motion-BIDS specifications [@jeung:2024].

## Dataclass
Supporting the data curation as specified in BIDS, data are organized in recordings, where recordings can be simultaneously collected with different tracking systems (e.g., an camera-based optical motion capture system and a set of IMUs). A tracking system is defined as a group of motion channels that share hardware properties (the recording device) and software properties (the recording duration and number of samples). Loading of a recording returns a `KielMATRecording` object, that holds both `data` and `channels`. Here, `data` are the actual time series data, where `channels` provide information (meta-data) on the time series type, component, the sampling frequency, and the units in which the time series (channel) are recorded.

## Modules
The data can be passed to algorithms that are organized in different modules, such as GSD and ICD. For example, the accelerometer data from a lower back-worn IMU can be passed to the gait sequence detection algorithm [@paraschiv:2019;@paraschiv:2020]. Next, the data can be passed to the initial contact detection algorithm [@paraschiv:2019] to returns the timings of initial contacts within each gait sequence (Figure [1](example_data.png)).

![](example_data.png)
<div style="text-align:center;">
<b>Figure 1:</b> A representative snippet of acceleration data from a low back-worn with the detected gait sequences (pink-shaded background) and the detected initial contacts (red triangles).
</div>

# Installation and usage
The KielMAT package is implemented in Python (>=3.10) and is freely available under a Non-Profit Open Software License version 3.0. The stable version of the package can be installed from PyPI.org using `pip install kielmat`. Users and developers can also install the toolbox from source from GitHub. The documentation of the toolbox provides detailed instructions on [installation](https://neurogeriatricskiel.github.io/KielMAT/#installation), [conceptual framework](https://neurogeriatricskiel.github.io/KielMAT/#data-classes-conceptual-framework) and [tutorial notebooks](https://neurogeriatricskiel.github.io/KielMAT/examples/) for basic usage and specific algorithms. Data used in the examples have been collected in accordance with the Declaration of Helsinki.

# How to contribute
KielMAT is a community effort, and any contribution is welcomed. The project is hosted on [https://github.com/neurogeriatricskiel/KielMAT](https://github.com/neurogeriatricskiel/KielMAT). In case you want to add new algorithms, it is suggested to fork the project and, after finalizing the changes, to [create a pull request from a fork](https://docs.github.com/de/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork).

# Acknowledgements
The authors would like to thank every person who provided data which has been used in the development and validation of the algorithms in the KielMAT toolbox.
The authors declare no competing interests.

# References
