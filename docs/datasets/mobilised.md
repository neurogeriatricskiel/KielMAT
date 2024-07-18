## Mobilise-D

The Mobilise-D dataset is derived from the [Mobilise-D](https://mobilise-d.eu/) consortium, which was a European project aimed at developing digital mobility outcomes to monitor the daily life gait of people with various mobility problems. For monitoring daily life gait a low back-worn inertial measurement unit (IMU) was used. 

Example data were made publicly available as Micó-Amigo *et al*., Zenodo, 2023, Assessing real-world gait with digital technology? Validation, insights and recommendations from the Mobilise-D consortium [[Data set]](https://zenodo.org/records/7547125), doi: [10.5281/zenodo.7547125](https://doi.org/10.5281/zenodo.7547125) and results for the entire dataset were published as Micó-Amigo *et al*., Journal of NeuroEngineering and Rehabilitation, 2023, Assessing real-world gait with digital technology? Validation, insights and recommendations from the Mobilise-D consortium, doi: [10.1186/s12984-023-01198-5](https://doi.org/10.1186/s12984-023-01198-5).

The example data can be fetched from the Zenodo repository, and data are saved in the following structure:
```
── ngmt/
│   ├── datasets/
│   │   ├── _mobilised/
│   │   │   ├── CHF/
│   │   │   │   ├── data.mat
│   │   │   │   └── infoForAlgo.mat
│   │   │   ├── ...
│   │   │   ├── PFF/
│   │   │   │   ├── data.mat
│   │   │   │   └── infoForAlgo.mat
│   ├── modules/
│   ├── ...
│   ├── __init__.py
│   └── config.py
│   ...
```

With for each of the cohorts (congestive heart failure (CHF), chronic obstructive pulmonary disease (COPD), healthy adult (HA), multiple sclerosis (MS), Parkinson’s disease (PD) and proximal femoral fracture (PFF)) an example data file (`data.mat`) and a file with information of the participant (`infoForAlgo.mat`). 

The recording can be loaded into a `NGMTRecording` with the corresponding function:


::: datasets.mobilised
