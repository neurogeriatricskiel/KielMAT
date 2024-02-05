---
title: "NGMT: NeuroGeriatric Motion Toolbox - An Open-Source Python Toolbox to Analyze Neurological Motion Data from Various Recording Modalities"  
tags:
  - Python
  - Motion capture
  - Neurology
  - Accelerometer
  - Gyroscope
authors:
  - name: Masoud Abedinifar^[corresponding author]   
    orcid: 0000-0002-4050-9835  
    affiliation: 1  
  - name: Julius Welzel  
    orcid: 0000-0001-8958-0934  
    affiliation: 1  
  - name: Clint Hansen  
    orcid: 0000-0003-4813-3868  
    affiliation: 1
  - name: Walter Maetzler  
    orcid: 0000-0002-5945-4694  
    affiliation: 1
  - name: Robbin Romijnders  
    orcid: 0000-0002-2507-0924  
    affiliation: 1
affiliations:
  - name: Neurogeriatrics, Department of Neurology, University Hospital Schleswig-Holstein (USKH), Kiel Germany  
    index: 1
date: 31 January 2024  
bibliography: references.bib
---

# Summary
The NeuroGeriatric Motion Toolbox (NGMT) is a Python-based, open-source toolbox designed for processing human motion data. Specifically targeting motion researchers and clinicians, NGMT aims to provide a comprehensive suite of algorithms for motion data processing in neuroscience and biomechanics. This includes algorithms like gait sequence detection, initial contact detection, and more.

# Statement of need
The analysis of human movement is a critical part of the neurological assessment [@micoamigo_2023]. Human motion characterization is crucial for our overall well-being, encompassing our physical, mental, and social dimensions. The increasing prevalence of mobility-limiting diseases, such as Parkinson's disease (PD), poses a serious burden on healthcare systems [mahlknecht2013prevalence]. Wearable devices such as inertial measurement units (IMUs) allow for long-term monitoring of disease progression and could, therefore, be used to track changes in gait [mazza2021technical]. 

While many studies have focused on a single IMU worn on the lower back, they often used not freely available software to extract data [mico2023assessing]. However, the development of easy-to-use and open-source code implementations is imperative for large-scale data extraction in both research and clinical settings. Addressing this gap, NGMT is introduced, providing diverse algorithms for human motion data analysis. It incorporates various algorithms for the analysis of human motion data.This toolbox caters to motion researchers and clinicians, promoting the utilization of open-source software. NGMT encompasses a broad range of validated algorithms such as gait sequence detection, initial contact detection, and physical activity monitoring. While not exhaustive, ongoing efforts aim to include additional validated algorithms, such as sit-to-stand, stand-to-sit, etc.

# Provided Functionality
NGMT offers practical examples demonstrating the application of currently implemented algorithms, focusing on gait sequence detection and initial contact detection. The toolbox utilizes IMU sensor data from clinical cohorts, such as those with congestive heart failure (CHF) [@mico2023assessing]. Participants undergo real-world assessments, engaging in daily activities and specific tasks, including outdoor walking, navigating slopes and stairs, and moving between rooms.

# Installation and usage
The NGMT package is implemented in Python and is freely available under a Non-Profit Open Software License version 3.0. The stable version of the package can be installed from PyPI.org using `pip install ngmt`. Our documentation provides detailed instructions on installation and some tutorial notebooks.

# Acknowledgements

# References