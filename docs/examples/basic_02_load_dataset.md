# Tutorial: load datasets

**Author:** Robbin Romijnders  
**Last update:** Tue 16 Jan 2024

## Learning objectives
By the end of this tutorial:

- you can load data from a recording that belongs to one of the available datasets,
- you know which attributes are available for an instance of the `KielMATRecording`
- you can do some basic selecting and slicing of data

## Imports

We start by importing some Python libraries. You should be familiar with most of them, and we will not discuss them here.


```python
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from kielmat.datasets import mobilised
```

## Import data

Let us consider a single recording, namely of the randomly selected subject `sub-3011` from the `Mobilise-D` dataset, and load the data. For that we use the `load_recording()` function that is available in the `kielmat.datasets.mobilised` module.


```python
# load the data
recording = mobilised.load_recording()
```

We have loaded the data for two tracking systems, `SU` and `SU_INDIP`, and we have specified three tracked points. The data is assigned to the variable `recording`, so let us take a look at what we have got.


```python
recording.__dict__
```




    {'data': {'SU':         LowerBack_ACCEL_x  LowerBack_ACCEL_y  LowerBack_ACCEL_z  \
      0                0.933334           0.084820          -0.302665   
      1                0.932675           0.084844          -0.300591   
      2                0.932350           0.082886          -0.310576   
      3                0.929716           0.081786          -0.303551   
      4                0.932825           0.077879          -0.308859   
      ...                   ...                ...                ...   
      693471          -0.192553          -0.016052          -0.984290   
      693472          -0.189575          -0.016449          -0.988130   
      693473          -0.191176          -0.017954          -0.983820   
      693474          -0.189691          -0.014539          -0.986376   
      693475          -0.192993          -0.015306          -0.989452   
      
              LowerBack_GYRO_x  LowerBack_GYRO_y  LowerBack_GYRO_z  \
      0               5.600066          1.120697          0.489152   
      1               5.440734          1.401663          0.279477   
      2               5.196312          1.168802          0.435765   
      3               5.553083          1.116346          0.383447   
      4               5.437505          0.892803         -0.150115   
      ...                  ...               ...               ...   
      693471         -0.225874          0.832856          0.704711   
      693472         -0.393438          0.598116          0.522755   
      693473         -0.430749          0.417541          0.282336   
      693474         -0.279277          0.559122          0.418693   
      693475         -0.563741          0.478618          0.411295   
      
              LowerBack_MAGN_x  LowerBack_MAGN_y  LowerBack_MAGN_z  \
      0             -93.972011        -25.023998         44.675028   
      1             -93.958012        -25.016007         44.610055   
      2             -93.946010        -25.000014         44.520078   
      3             -93.938007        -24.980018         44.411097   
      4             -93.935003        -24.957021         44.287113   
      ...                  ...               ...               ...   
      693471        -50.718928        -36.997006         34.111960   
      693472        -50.649929        -37.003005         34.072972   
      693473        -50.579936        -37.008003         34.044986   
      693474        -50.515946        -37.011000         34.031004   
      693475        -50.460961        -37.010996         34.035025   
      
              LowerBack_BARO_n/a  
      0               990.394600  
      1               990.395100  
      2               990.395600  
      3               990.396199  
      4               990.396700  
      ...                    ...  
      693471          990.204600  
      693472          990.204900  
      693473          990.205200  
      693474          990.205500  
      693475          990.205800  
      
      [693476 rows x 10 columns]},
     'channels': {'SU':                  name component   type tracked_point  units  \
      0   LowerBack_ACCEL_x         x  ACCEL     LowerBack      g   
      1   LowerBack_ACCEL_y         y  ACCEL     LowerBack      g   
      2   LowerBack_ACCEL_z         z  ACCEL     LowerBack      g   
      3    LowerBack_GYRO_x         x   GYRO     LowerBack  deg/s   
      4    LowerBack_GYRO_y         y   GYRO     LowerBack  deg/s   
      5    LowerBack_GYRO_z         z   GYRO     LowerBack  deg/s   
      6    LowerBack_MAGN_x         x   MAGN     LowerBack     µT   
      7    LowerBack_MAGN_y         y   MAGN     LowerBack     µT   
      8    LowerBack_MAGN_z         z   MAGN     LowerBack     µT   
      9  LowerBack_BARO_n/a       n/a   BARO     LowerBack    hPa   
      
         sampling_frequency  
      0               100.0  
      1               100.0  
      2               100.0  
      3               100.0  
      4               100.0  
      5               100.0  
      6               100.0  
      7               100.0  
      8               100.0  
      9               100.0  },
     'info': None,
     'events': None,
     'events_info': None}



That is a whole lot of output, so let us take a look at the attributes of instance one by one. First, print a list of all available attributes.


```python
print(recording.__dict__.keys())
```

    dict_keys(['data', 'channels', 'info', 'events', 'events_info'])
    

The contents of any individual attribute can be accessed in two ways, namely via the `__dict__` or with `dot` indexing.


```python
print(recording.data)  # print(recording.__dict__["data"])
```

    {'SU':         LowerBack_ACCEL_x  LowerBack_ACCEL_y  LowerBack_ACCEL_z  \
    0                0.933334           0.084820          -0.302665   
    1                0.932675           0.084844          -0.300591   
    2                0.932350           0.082886          -0.310576   
    3                0.929716           0.081786          -0.303551   
    4                0.932825           0.077879          -0.308859   
    ...                   ...                ...                ...   
    693471          -0.192553          -0.016052          -0.984290   
    693472          -0.189575          -0.016449          -0.988130   
    693473          -0.191176          -0.017954          -0.983820   
    693474          -0.189691          -0.014539          -0.986376   
    693475          -0.192993          -0.015306          -0.989452   
    
            LowerBack_GYRO_x  LowerBack_GYRO_y  LowerBack_GYRO_z  \
    0               5.600066          1.120697          0.489152   
    1               5.440734          1.401663          0.279477   
    2               5.196312          1.168802          0.435765   
    3               5.553083          1.116346          0.383447   
    4               5.437505          0.892803         -0.150115   
    ...                  ...               ...               ...   
    693471         -0.225874          0.832856          0.704711   
    693472         -0.393438          0.598116          0.522755   
    693473         -0.430749          0.417541          0.282336   
    693474         -0.279277          0.559122          0.418693   
    693475         -0.563741          0.478618          0.411295   
    
            LowerBack_MAGN_x  LowerBack_MAGN_y  LowerBack_MAGN_z  \
    0             -93.972011        -25.023998         44.675028   
    1             -93.958012        -25.016007         44.610055   
    2             -93.946010        -25.000014         44.520078   
    3             -93.938007        -24.980018         44.411097   
    4             -93.935003        -24.957021         44.287113   
    ...                  ...               ...               ...   
    693471        -50.718928        -36.997006         34.111960   
    693472        -50.649929        -37.003005         34.072972   
    693473        -50.579936        -37.008003         34.044986   
    693474        -50.515946        -37.011000         34.031004   
    693475        -50.460961        -37.010996         34.035025   
    
            LowerBack_BARO_n/a  
    0               990.394600  
    1               990.395100  
    2               990.395600  
    3               990.396199  
    4               990.396700  
    ...                    ...  
    693471          990.204600  
    693472          990.204900  
    693473          990.205200  
    693474          990.205500  
    693475          990.205800  
    
    [693476 rows x 10 columns]}
    

We see that that `data` attribute is in the form of a Python `dict`, where the keys correspond to the tracking systems that we have requested when calling the `load_recording()` function. KielMAT is setup so that the keys of the `channels` attribute match with these keys, so that the channel descriptions are availbale per tracking system.


```python
print(f"We have the following keys in recording.data: {recording.data.keys()}")

print(f"We have the same keys in recordings.channels: {recording.channels.keys()}")
```

    We have the following keys in recording.data: dict_keys(['SU'])
    We have the same keys in recordings.channels: dict_keys(['SU'])
    
