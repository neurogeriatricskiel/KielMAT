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
The NeuroGeriatric Motion Toolbox (NGMT) is an open-source Python-based toolbox designed for processing human motion data, following open-science practices. NGMT offers a wide range of algorithms for the processing of motion data in neuroscience and biomechanics and currently includes implementations for gait sequence detection, initial contact detection, and physical activity monitoring. These algorithms aid in identifying patterns in human motion data on different time scales. Some of the toolbox algorithms have been developed and validated in clinical cohorts, allowing extracted patters to be used in a clinical context. However, the modular design of NGMT allows the toolbox to be easily extended to incorporate relevant algorithms which will be developed in the research community. The toolbox is designed to be user-friendly and is accompanied by a comprehensive documentation and practical examples, while the underlying data structures build on the Motion BIDS specification [@jeung2023motion]. The NGMT toolbox is intended to be used by researchers and clinicians to analyze human motion data from various recording modalities and to promote the utilization of open-source software in the field of human motion analysis.

# Statement of need
Human motion characterization is crucial for overall well-being, encompassing our physical, mental, and social dimensions. The increasing prevalence of mobility-limiting diseases, such as Parkinson's disease (PD), poses a serious burden on healthcare systems [@mahlknecht2013prevalence]. While motion data becomes more avaliable and the analysis of human movement is critical for neurological assessment [@micoamigo_2023], a lack of validated algorithms can be observed in the literature. Many studies often used not freely available software to extract data to extract features of motion, (such as step time) for research puposes. The development of easy-to-use and open-source code implementations is imperative for transparent data extraction in research and clinical settings. NGMT addresses this gap by providing diverse algorithms for human motion data analysis, catering to motion researchers and clinicians and promoting the utilization of open-source software build on FAIR data principles. The toolbox provides algorithms for gait sequence detection, initial contact detection, and physical activity monitoring, which are essential for the analysis of human motion data. The toolbox conceptual framework builds on FAIR data principles to encourage the use of open source software as well as facilitate data sharing and reproducibility in the field of human motion analysis.

# Provided Functionality
NGMT provides a comprehensive suite of algorithms for motion data processing in neuroscience and biomechanics, currently including implementations for gait sequence detection, initial contact detection, and physical activity monitoring. The toolbox offers practical examples demonstrating the application of currently implemented algorithms, focusing on gait sequence detection and initial contact detection. The toolbox utilizes IMU sensor data from clinical cohorts, such as those with congestive heart failure (CHF) [@micoamigo_2023]. Participants undergo real-world assessments, engaging in daily activities and specific tasks, including outdoor walking, navigating slopes and stairs, and moving between rooms.

The data were processed using our gait sequence detection module, based on the Paraschiv-Ionescu algorithm[@paraschiv2019locomotion,@paraschiv2020real], to identify gait sequences within the time series. Subsequently, our initial contact detection module , also based on the Paraschiv-Ionescu algorithm [@paraschiv2019locomotion,@paraschiv2020real], was applied to identify initial contacts within the detected gait sequences. Utilizing lower back acceleration data from the Mobilise-D dataset, we demonstrate the accurate identification of gait events such as gait onset, gait duration, and initial contacts. NGMT offers practical examples demonstrating the application of currently implemented algorithms, focusing on gait sequence detection and initial contact detection. This example illustrates the practical application of our toolbox in analyzing human gait patterns which is shown in figure 1.

![](figure_1.png){#fig:figure1}
<div style="text-align:center;">
<b>Figure 1:</b> Acceleration Data and Detected Gait Events using NGMT Modules
</div>

In figure 1, we present a detailed analysis of acceleration data from the lower back, highlighting the key gait events detected by our modules. The green vertical line indicates the onset of a gait sequence, while the shaded gray region represents the duration of the gait sequence. Additionally, blue dashed lines denote the detected initial contacts within the gait sequence.

# State of the field
With the growing avaliability of motion data data, there are implementations of open source algorithms avaliable for the processing of motion data. In Python there is the [GaitMap](https://gaitmap.readthedocs.io/en/latest/index.html) package which has a special focus on the processing of feet worn IMUs. The GaitLink package includes algorithms for ....
For video based processing of motion data, there are a number of computer vision packages avaliable such as OpenCV, OpenPose, DeepLabCut, and others. However, the NGMT package is unique in that it is designed to be a comprehensive toolbox for the processing of motion data from a wide range of recording modalities, currently focusing on data from inertial measurement units and optical motion capture.

# Example Use Case

# Installation and usage
The NGMT package is implemented in Python (>=3.10) and is freely available under a Non-Profit Open Software License version 3.0. The stable version of the package can be installed from PyPI.org using `pip install ngmt`. For developmental puposes, the toolbox can also be installed via GitHub. The documentation of the toolbox provides detailed instructions on [installation](https://neurogeriatricskiel.github.io/NGMT/#installation), [conceptual framework](https://neurogeriatricskiel.github.io/NGMT/#data-classes-conceptual-framework) and [tutorial notebooks](https://neurogeriatricskiel.github.io/NGMT/examples/) for basic usage and specific algorithms.

# Acknowledgements

# References