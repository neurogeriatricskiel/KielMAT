---
title: "NGMT: NeuroGeriatric Motion Toolbox - An Open-Source Python Toolbox for Analyzing Neurological Motion Data from Various Recording Modalities"
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
The NeuroGeriatric Motion Toolbox (NGMT) is an open-source Python-based toolbox designed for processing human motion data, following open-science practices. NGMT offers a wide range of algorithms for the processing of motion data in neuroscience and biomechanics and currently includes implementations for gait sequence detection, initial contact detection, physical activity monitoring, sit to stand and stand to sit detection algorithms. These algorithms aid in identifying patterns in human motion data on different time scales. Some of the toolbox algorithms have been developed and validated in clinical cohorts, allowing extracted patters to be used in a clinical context. The modular design of NGMT allows the toolbox to be easily extended to incorporate relevant algorithms which will be developed in the research community. The toolbox is designed to be user-friendly and is accompanied by a comprehensive documentation and practical examples, while the underlying data structures build on the Motion BIDS specification [@jeung:2023]. The NGMT toolbox is intended to be used by researchers and clinicians to analyze human motion data from various recording modalities and to promote the utilization of open-source software in the field of human motion analysis.

# Statement of need
Physical mobility is an essential aspect of health, since impairment of mobility is associated with reduced quality of life, falls, hospitalization, mortality, and other adverse events in many chronic conditions. Traditional mobility measures include patient-reported outcomes, objective clinical assessments, and subjective clinical assessments. These measures are associated with the perception and capacity aspects of health that frequently fail to show any relevant effect on daily function at an individual level [@maetzler:2021]. To complement both patient-reported (perception) and clinical (capacity) assessment approaches, digital health technology (DHT), including body-worn or wearable devices, offers a new dimension of measuring daily function, that is, performance [@warmerdam:2020; @fasano:2020; @maetzler:2021]. DHT allows an objective impression of how patients function in everyday life and their ability to routinely perform everyday activities [@hansen:2018; @buckley:2019; @celik:2021]. Nonetheless, due to several persisting challenges in this field, current tools and techniques are still in their infancy [@micoamigo:2023]. Many studies often used proprietary software to clinically relevant features of mobility. The development of easy-to-use and open-source software is imperative for transparent features extraction in research and clinical settings. NeuroGeriatric Motion Toolbox (NGMT) addresses this gap by providing diverse for human mobility analysis, catering to motion researchers and clinicians and promoting the utilization of open-source software build on FAIR data principles. The toolbox provides algorithms for gait sequence detection, initial contact detection, and physical activity monitoring. The conceptual framework builds on FAIR data principles to encourage the use of open source software as well as facilitate data sharing and reproducibility in the field of human motion analysis.

# State of the field
With the growing availability of digital health data, open-source implementations of relevant algorithms are increasingly becoming available. From the Mobilise-D consortium, the recommended algorithms for assessing real-world gait were released, but these algorithms were developed in MATLAB, that is not free to use [@mobilised:2023]. Likewise, an algorithm for the estimation of gait quality was released, but it is also only available in MATLAB [@gaitqualitycomposite:2016].  Alternatively, open-source, Python packages are available, for example to detect gait and extract gait features from a low back-worn inertial measurement unit (IMU) [@czech:2019], or from two feet-worn IMUs [@kuederle:2024]. NGMT builds forth on these toolboxes by providing a module software package that goes beyond the analysis of merely gait, and extends these analyses by additionally allowing for the analysis of general physical activity and other daily life-relevant movements, such as sit-to-stand and stand-to-sit transitions [@pham:2017] as well as turns [@pham:2018].

# Provided Functionality
NGMT offers a comprehensive suite of algorithms for motion data processing in neuroscience and biomechanics. Currently, the toolbox includes implementations for gait sequence detection (GSD) and initial contact detection (ICD), whereas algorithms for postural transition analysis [@pham:2017] and turns [@pham:2018] are under current development. NGMT is built on principles from the Brain Imaging Data Structure (BIDS) [gorgolewski:2016] and for the motion analysis data are organized similar to the Motion-BIDS specifications [@jeung:2023]. 

## Dataclass
Practically, this means that data are organized in recordings, where recordings can be simultaneously collected with different tracking systems (e.g., an camera-based optical motion capture system and a set of IMUs). A tracking system is defined as a group of motion channels that share hardware properties (the recording device) and software properties (the recording duration and number of samples). Loading of a recording returns a `NGMTRecording` object, that holds both `data` and `channels`. Here, `data` are the actual time series data, where `channels` provide information on the time series type, component, the sampling frequency, and the units in which the time series are recorded.

## Modules
The data can be passed to algorithms that are organized in different modules, such as GSD and ICD. For example, the accelerometer data from a low back-worn IMU can be passed to the gait sequence detection algorithm [@paraschiv:2019;@paraschiv:2020]. Next, the data can be passed to the initial contact detection algorithm [@paraschiv:2019] to returns the timings of initial contacts within each gait sequence (Figure [1](my_figure.png)).

![](my_figure.png)
<div style="text-align:center;">
<b>Figure 1:</b> A representative snipped of acceleration data from a low back-worn with the detected gait sequences (pink-shaded background) and the detected initial contacts (red triangles).
</div>

# Installation and usage
The NGMT package is implemented in Python (>=3.10) and is freely available under a Non-Profit Open Software License version 3.0. The stable version of the package can be installed from PyPI.org using `pip install ngmt`. Users and developers can also install the oolbxo from source from GitHub. The documentation of the toolbox provides detailed instructions on [installation](https://neurogeriatricskiel.github.io/NGMT/#installation), [conceptual framework](https://neurogeriatricskiel.github.io/NGMT/#data-classes-conceptual-framework) and [tutorial notebooks](https://neurogeriatricskiel.github.io/NGMT/examples/) for basic usage and specific algorithms.

# How to contribute
NGMT is a community effort, and any contribution is welcomed. The project is hosted on [https://github.com/neurogeriatricskiel/NGMT](https://github.com/neurogeriatricskiel/NGMT). In case you want to add new algorithms, it is suggested to fork the project and, after finalizing the changes, to [create a pull request from a fork](https://docs.github.com/de/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork).

# Acknowledgements

# References