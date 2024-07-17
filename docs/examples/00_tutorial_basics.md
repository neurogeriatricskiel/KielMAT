# Tutorial: the basics of KMAT

**Author:** Robbin Romijnders  
**Last update:** Tue 16 Jan 2024

## Learning objectives
By the end of this tutorial:
- you can load data from a recording that belongs to one of the available datasets,
- you know which attributes are available for an instance of the `KMATRecording`
- you can do some basic selecting and slicing of data

## Imports

We start by importing some Python libraries. You should be familiar with most of them, and we will not discuss them here.


```python
import numpy as np
import matplotlib.pyplot as plt
import os
from kmat.datasets import mobilised
```

## Import data

Let us consider a single recording, namely of the randomly selected subject `sub-3011` from the `Mobilise-D` dataset, and load the data. For that we use the `load_recording()` function that is available in the `kmat.datasets.mobilised` module.


```python
# Set the filepath
file_path = "Z:\\Mobilise-D\\rawdata\\sub-3011\\Free-living\\data.mat"

# Load the recording
recording = mobilised.load_recording(
    file_name=file_path, tracking_systems=["SU", "SU_INDIP"], 
    tracked_points=["LowerBack", "LeftFoot", "RightFoot"]
)
```

We have loaded the data for two tracking systems, `SU` and `SU_INDIP`, and we have specified three tracked points. The data is assigned to the variable `recording`, so let us take a look at what we have got.


```python
recording.__dict__
```




    {'data': {'SU':         LowerBack_ACCEL_x  LowerBack_ACCEL_y  LowerBack_ACCEL_z  \
      0                0.967784           0.045886          -0.260760   
      1                0.969667           0.044949          -0.256890   
      2                0.968848           0.045823          -0.259559   
      3                0.968601           0.045996          -0.260772   
      4                0.970748           0.044825          -0.259930   
      ...                   ...                ...                ...   
      993019           0.966575          -0.062226           0.273807   
      993020           0.968330          -0.065081           0.268255   
      993021           0.971785          -0.068572           0.269635   
      993022           0.970579          -0.069420           0.269644   
      993023           0.960542          -0.069657           0.270160   
      
              LowerBack_ANGVEL_x  LowerBack_ANGVEL_y  LowerBack_ANGVEL_z  \
      0                -2.566851           -2.177240            0.297938   
      1                -2.429344           -2.188699            0.034383   
      2                -2.521010           -2.051195           -0.320841   
      3                -2.532473           -2.028272           -0.366690   
      4                -2.085604           -2.108478           -0.000031   
      ...                    ...                 ...                 ...   
      993019            0.011351           -1.021059           -0.011785   
      993020            0.000108           -1.142992           -0.044537   
      993021            0.010811           -0.836410            0.089080   
      993022           -0.058050           -0.824628           -0.183023   
      993023           -0.136219           -0.771909           -0.147571   
      
              LowerBack_MAGN_x  LowerBack_MAGN_y  LowerBack_MAGN_z  \
      0               5.491000         11.878000          4.327000   
      1               5.490000         11.881000          4.331000   
      2               5.490000         11.881000          4.334000   
      3               5.491000         11.879000          4.337000   
      4               5.493000         11.875000          4.339000   
      ...                  ...               ...               ...   
      993019          4.118991          7.638019          1.819981   
      993020          4.117991          7.640009          1.817943   
      993021          4.116991          7.641000          1.811906   
      993022          4.115981          7.641000          1.801868   
      993023          4.075389          7.569287          1.771219   
      
              LowerBack_BARO_n/a  
      0              1011.628100  
      1              1011.628400  
      2              1011.628800  
      3              1011.629200  
      4              1011.629600  
      ...                    ...  
      993019         1012.077903  
      993020         1012.078202  
      993021         1012.078403  
      993022         1012.078703  
      993023         1002.580321  
      
      [993024 rows x 10 columns],
      'SU_INDIP':         LowerBack_ACCEL_x  LowerBack_ACCEL_y  LowerBack_ACCEL_z  \
      0                0.967986           0.074256          -0.227576   
      1                0.963671           0.072786          -0.222888   
      2                0.967964           0.076218          -0.229266   
      3                0.964173           0.080834          -0.225058   
      4                0.968537           0.073739          -0.224440   
      ...                   ...                ...                ...   
      993019           0.952616          -0.036956           0.276052   
      993020           0.950713          -0.039025           0.274731   
      993021           0.951084          -0.041569           0.273339   
      993022           0.951656          -0.043793           0.273537   
      993023           0.955107          -0.043583           0.274676   
      
              LowerBack_ANGVEL_x  LowerBack_ANGVEL_y  LowerBack_ANGVEL_z  \
      0                 2.112411            0.790947            0.541712   
      1              -132.991561          -72.074004          104.116796   
      2                40.206686           20.650084          -16.748853   
      3                37.499673           18.407235          -17.670796   
      4                 6.486804            2.543410           -3.971685   
      ...                    ...                 ...                 ...   
      993019            0.874026           -0.039631           -0.363585   
      993020            0.736097           -0.079306           -0.373310   
      993021            0.592863           -0.078105           -0.326287   
      993022            0.483162            0.062394           -0.319518   
      993023            0.449540            0.202463           -0.350353   
      
              LowerBack_MAGN_x  LowerBack_MAGN_y  LowerBack_MAGN_z  \
      0              -9.285130         75.022617         -5.902833   
      1             -11.966959         18.729473          9.501037   
      2             -11.456134         18.993760          9.626272   
      3             -11.839253         19.390191          9.751507   
      4             -11.456134         19.522334          9.375802   
      ...                  ...               ...               ...   
      993019         -0.345702          0.335094        -18.075647   
      993020         -0.396785          0.361523        -17.449474   
      993021         -0.757200          0.875414        -17.521831   
      993022         -1.111939          0.757953        -17.987983   
      993023         -1.048086          0.361523        -18.050600   
      
              LeftFoot_ACCEL_x  ...  LeftFoot_MAGN_z  RightFoot_ACCEL_x  \
      0               0.934771  ...        -3.994010           0.910403   
      1               0.933119  ...        -3.994010           0.905757   
      2               0.933472  ...        -4.007887           0.908215   
      3               0.930711  ...        -4.132780           0.907955   
      4               0.929001  ...        -4.105026           0.907808   
      ...                  ...  ...              ...                ...   
      993019          0.849485  ...       -20.058072           0.894927   
      993020          0.850188  ...       -20.252351           0.895039   
      993021          0.850738  ...       -20.230148           0.895129   
      993022          0.850454  ...       -20.417488           0.895036   
      993023          0.848690  ...       -20.264840           0.893635   
      
              RightFoot_ACCEL_y  RightFoot_ACCEL_z  RightFoot_ANGVEL_x  \
      0               -0.213594          -0.360157            0.199539   
      1               -0.213015          -0.358808            0.160327   
      2               -0.214753          -0.358071            0.164578   
      3               -0.216700          -0.357703            0.127464   
      4               -0.217405          -0.354327            0.053191   
      ...                   ...                ...                 ...   
      993019          -0.232745          -0.385142           -0.273263   
      993020          -0.230868          -0.385270           -0.244655   
      993021          -0.230846          -0.384043           -0.215061   
      993022          -0.232099          -0.384561           -0.236093   
      993023          -0.231213          -0.385298           -0.240081   
      
              RightFoot_ANGVEL_y  RightFoot_ANGVEL_z  RightFoot_MAGN_x  \
      0                 0.415360            1.012113        -14.106638   
      1                 0.559909            1.031959        -13.982228   
      2                 0.862008            0.954256        -14.202222   
      3                 0.794218            0.781762        -14.270497   
      4                 0.623394            0.505714        -14.407045   
      ...                    ...                 ...               ...   
      993019            0.091395           -0.207776        -23.842547   
      993020            0.139509           -0.200464        -23.938131   
      993021            0.168476           -0.226669        -23.866822   
      993022            0.147898           -0.208713        -23.828892   
      993023            0.144888           -0.270695        -23.828892   
      
              RightFoot_MAGN_y  RightFoot_MAGN_z  
      0              23.532360         10.766563  
      1              23.576285         10.871915  
      2              23.278221         10.724733  
      3              23.278221         10.585297  
      4              23.490003         10.445862  
      ...                  ...               ...  
      993019         23.334696         -2.884172  
      993020         23.292340         -2.912059  
      993021         23.532360         -3.246704  
      993022         23.726886         -3.519378  
      993023         23.670411         -3.296281  
      
      [993024 rows x 27 columns]},
     'channels': {'SU':                  name type component tracked_point  units  sampling_frequency
      0   LowerBack_ACCEL_x  Acc         x     LowerBack      g               100.0
      1   LowerBack_ACCEL_y  Acc         y     LowerBack      g               100.0
      2   LowerBack_ACCEL_z  Acc         z     LowerBack      g               100.0
      3  LowerBack_ANGVEL_x  Gyr         x     LowerBack  deg/s               100.0
      4  LowerBack_ANGVEL_y  Gyr         y     LowerBack  deg/s               100.0
      5  LowerBack_ANGVEL_z  Gyr         z     LowerBack  deg/s               100.0
      6    LowerBack_MAGN_x  Mag         x     LowerBack     µT               100.0
      7    LowerBack_MAGN_y  Mag         y     LowerBack     µT               100.0
      8    LowerBack_MAGN_z  Mag         z     LowerBack     µT               100.0
      9  LowerBack_BARO_n/a  Bar       n/a     LowerBack    hPa               100.0,
      'SU_INDIP':                   name type component tracked_point  units  sampling_frequency
      0    LowerBack_ACCEL_x  Acc         x     LowerBack      g               100.0
      1    LowerBack_ACCEL_y  Acc         y     LowerBack      g               100.0
      2    LowerBack_ACCEL_z  Acc         z     LowerBack      g               100.0
      3   LowerBack_ANGVEL_x  Gyr         x     LowerBack  deg/s               100.0
      4   LowerBack_ANGVEL_y  Gyr         y     LowerBack  deg/s               100.0
      5   LowerBack_ANGVEL_z  Gyr         z     LowerBack  deg/s               100.0
      6     LowerBack_MAGN_x  Mag         x     LowerBack     µT               100.0
      7     LowerBack_MAGN_y  Mag         y     LowerBack     µT               100.0
      8     LowerBack_MAGN_z  Mag         z     LowerBack     µT               100.0
      9     LeftFoot_ACCEL_x  Acc         x      LeftFoot      g               100.0
      10    LeftFoot_ACCEL_y  Acc         y      LeftFoot      g               100.0
      11    LeftFoot_ACCEL_z  Acc         z      LeftFoot      g               100.0
      12   LeftFoot_ANGVEL_x  Gyr         x      LeftFoot  deg/s               100.0
      13   LeftFoot_ANGVEL_y  Gyr         y      LeftFoot  deg/s               100.0
      14   LeftFoot_ANGVEL_z  Gyr         z      LeftFoot  deg/s               100.0
      15     LeftFoot_MAGN_x  Mag         x      LeftFoot     µT               100.0
      16     LeftFoot_MAGN_y  Mag         y      LeftFoot     µT               100.0
      17     LeftFoot_MAGN_z  Mag         z      LeftFoot     µT               100.0
      18   RightFoot_ACCEL_x  Acc         x     RightFoot      g               100.0
      19   RightFoot_ACCEL_y  Acc         y     RightFoot      g               100.0
      20   RightFoot_ACCEL_z  Acc         z     RightFoot      g               100.0
      21  RightFoot_ANGVEL_x  Gyr         x     RightFoot  deg/s               100.0
      22  RightFoot_ANGVEL_y  Gyr         y     RightFoot  deg/s               100.0
      23  RightFoot_ANGVEL_z  Gyr         z     RightFoot  deg/s               100.0
      24    RightFoot_MAGN_x  Mag         x     RightFoot     µT               100.0
      25    RightFoot_MAGN_y  Mag         y     RightFoot     µT               100.0
      26    RightFoot_MAGN_z  Mag         z     RightFoot     µT               100.0},
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
    0                0.967784           0.045886          -0.260760   
    1                0.969667           0.044949          -0.256890   
    2                0.968848           0.045823          -0.259559   
    3                0.968601           0.045996          -0.260772   
    4                0.970748           0.044825          -0.259930   
    ...                   ...                ...                ...   
    993019           0.966575          -0.062226           0.273807   
    993020           0.968330          -0.065081           0.268255   
    993021           0.971785          -0.068572           0.269635   
    993022           0.970579          -0.069420           0.269644   
    993023           0.960542          -0.069657           0.270160   
    
            LowerBack_ANGVEL_x  LowerBack_ANGVEL_y  LowerBack_ANGVEL_z  \
    0                -2.566851           -2.177240            0.297938   
    1                -2.429344           -2.188699            0.034383   
    2                -2.521010           -2.051195           -0.320841   
    3                -2.532473           -2.028272           -0.366690   
    4                -2.085604           -2.108478           -0.000031   
    ...                    ...                 ...                 ...   
    993019            0.011351           -1.021059           -0.011785   
    993020            0.000108           -1.142992           -0.044537   
    993021            0.010811           -0.836410            0.089080   
    993022           -0.058050           -0.824628           -0.183023   
    993023           -0.136219           -0.771909           -0.147571   
    
            LowerBack_MAGN_x  LowerBack_MAGN_y  LowerBack_MAGN_z  \
    0               5.491000         11.878000          4.327000   
    1               5.490000         11.881000          4.331000   
    2               5.490000         11.881000          4.334000   
    3               5.491000         11.879000          4.337000   
    4               5.493000         11.875000          4.339000   
    ...                  ...               ...               ...   
    993019          4.118991          7.638019          1.819981   
    993020          4.117991          7.640009          1.817943   
    993021          4.116991          7.641000          1.811906   
    993022          4.115981          7.641000          1.801868   
    993023          4.075389          7.569287          1.771219   
    
            LowerBack_BARO_n/a  
    0              1011.628100  
    1              1011.628400  
    2              1011.628800  
    3              1011.629200  
    4              1011.629600  
    ...                    ...  
    993019         1012.077903  
    993020         1012.078202  
    993021         1012.078403  
    993022         1012.078703  
    993023         1002.580321  
    
    [993024 rows x 10 columns], 'SU_INDIP':         LowerBack_ACCEL_x  LowerBack_ACCEL_y  LowerBack_ACCEL_z  \
    0                0.967986           0.074256          -0.227576   
    1                0.963671           0.072786          -0.222888   
    2                0.967964           0.076218          -0.229266   
    3                0.964173           0.080834          -0.225058   
    4                0.968537           0.073739          -0.224440   
    ...                   ...                ...                ...   
    993019           0.952616          -0.036956           0.276052   
    993020           0.950713          -0.039025           0.274731   
    993021           0.951084          -0.041569           0.273339   
    993022           0.951656          -0.043793           0.273537   
    993023           0.955107          -0.043583           0.274676   
    
            LowerBack_ANGVEL_x  LowerBack_ANGVEL_y  LowerBack_ANGVEL_z  \
    0                 2.112411            0.790947            0.541712   
    1              -132.991561          -72.074004          104.116796   
    2                40.206686           20.650084          -16.748853   
    3                37.499673           18.407235          -17.670796   
    4                 6.486804            2.543410           -3.971685   
    ...                    ...                 ...                 ...   
    993019            0.874026           -0.039631           -0.363585   
    993020            0.736097           -0.079306           -0.373310   
    993021            0.592863           -0.078105           -0.326287   
    993022            0.483162            0.062394           -0.319518   
    993023            0.449540            0.202463           -0.350353   
    
            LowerBack_MAGN_x  LowerBack_MAGN_y  LowerBack_MAGN_z  \
    0              -9.285130         75.022617         -5.902833   
    1             -11.966959         18.729473          9.501037   
    2             -11.456134         18.993760          9.626272   
    3             -11.839253         19.390191          9.751507   
    4             -11.456134         19.522334          9.375802   
    ...                  ...               ...               ...   
    993019         -0.345702          0.335094        -18.075647   
    993020         -0.396785          0.361523        -17.449474   
    993021         -0.757200          0.875414        -17.521831   
    993022         -1.111939          0.757953        -17.987983   
    993023         -1.048086          0.361523        -18.050600   
    
            LeftFoot_ACCEL_x  ...  LeftFoot_MAGN_z  RightFoot_ACCEL_x  \
    0               0.934771  ...        -3.994010           0.910403   
    1               0.933119  ...        -3.994010           0.905757   
    2               0.933472  ...        -4.007887           0.908215   
    3               0.930711  ...        -4.132780           0.907955   
    4               0.929001  ...        -4.105026           0.907808   
    ...                  ...  ...              ...                ...   
    993019          0.849485  ...       -20.058072           0.894927   
    993020          0.850188  ...       -20.252351           0.895039   
    993021          0.850738  ...       -20.230148           0.895129   
    993022          0.850454  ...       -20.417488           0.895036   
    993023          0.848690  ...       -20.264840           0.893635   
    
            RightFoot_ACCEL_y  RightFoot_ACCEL_z  RightFoot_ANGVEL_x  \
    0               -0.213594          -0.360157            0.199539   
    1               -0.213015          -0.358808            0.160327   
    2               -0.214753          -0.358071            0.164578   
    3               -0.216700          -0.357703            0.127464   
    4               -0.217405          -0.354327            0.053191   
    ...                   ...                ...                 ...   
    993019          -0.232745          -0.385142           -0.273263   
    993020          -0.230868          -0.385270           -0.244655   
    993021          -0.230846          -0.384043           -0.215061   
    993022          -0.232099          -0.384561           -0.236093   
    993023          -0.231213          -0.385298           -0.240081   
    
            RightFoot_ANGVEL_y  RightFoot_ANGVEL_z  RightFoot_MAGN_x  \
    0                 0.415360            1.012113        -14.106638   
    1                 0.559909            1.031959        -13.982228   
    2                 0.862008            0.954256        -14.202222   
    3                 0.794218            0.781762        -14.270497   
    4                 0.623394            0.505714        -14.407045   
    ...                    ...                 ...               ...   
    993019            0.091395           -0.207776        -23.842547   
    993020            0.139509           -0.200464        -23.938131   
    993021            0.168476           -0.226669        -23.866822   
    993022            0.147898           -0.208713        -23.828892   
    993023            0.144888           -0.270695        -23.828892   
    
            RightFoot_MAGN_y  RightFoot_MAGN_z  
    0              23.532360         10.766563  
    1              23.576285         10.871915  
    2              23.278221         10.724733  
    3              23.278221         10.585297  
    4              23.490003         10.445862  
    ...                  ...               ...  
    993019         23.334696         -2.884172  
    993020         23.292340         -2.912059  
    993021         23.532360         -3.246704  
    993022         23.726886         -3.519378  
    993023         23.670411         -3.296281  
    
    [993024 rows x 27 columns]}
    

We see that that `data` attribute is in the form of a Python `dict`, where the keys correspond to the tracking systems that we have requested when calling the `load_recording()` function. KMAT is setup so that the keys of the `channels` attribute match with these keys, so that the channel descriptions are availbale per tracking system.


```python
print(recording.data.keys(), recording.channels.keys())
```

    dict_keys(['SU', 'SU_INDIP']) dict_keys(['SU', 'SU_INDIP'])
    


```python

```
