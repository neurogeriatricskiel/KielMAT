
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
# Set the filepath
file_path = Path(os.getcwd()).parent.joinpath("examples","data","exMobiliseFreeLiving.mat")

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




    {'data': {'SU':          LowerBack_ACCEL_x  LowerBack_ACCEL_y  LowerBack_ACCEL_z  \
      0                 0.942178           0.007510          -0.288112   
      1                 0.942810           0.010240          -0.289560   
      2                 0.943393           0.006346          -0.295584   
      3                 0.940556           0.002275          -0.291497   
      4                 0.941915           0.005847          -0.292111   
      ...                    ...                ...                ...   
      1074280           0.971606           0.054683          -0.130928   
      1074281           0.970852           0.056564          -0.135593   
      1074282           0.969576           0.056643          -0.132909   
      1074283           0.973411           0.056775          -0.136337   
      1074284           0.972761           0.054689          -0.136837   
      
               LowerBack_GYRO_x  LowerBack_GYRO_y  LowerBack_GYRO_z  \
      0                0.137440          0.916976          0.332333   
      1                0.137510          0.676277          0.561320   
      2                0.252010          0.595940          0.389748   
      3               -0.091394          0.550077          0.469760   
      4               -0.183270          0.630187          0.435476   
      ...                   ...               ...               ...   
      1074280          0.607320          0.401071          0.217741   
      1074281          0.389616          0.412534          0.458358   
      1074282          0.469829          0.492739          0.297937   
      1074283          0.561494          0.355235          0.275020   
      1074284          0.275020          0.423989          0.286479   
      
               LowerBack_MAGN_x  LowerBack_MAGN_y  LowerBack_MAGN_z  \
      0               -3.220996         -5.058998         -7.994002   
      1               -3.224997         -5.059000         -7.991002   
      2               -3.229996         -5.059000         -7.986004   
      3               -3.234996         -5.056002         -7.980005   
      4               -3.238997         -5.052003         -7.973006   
      ...                   ...               ...               ...   
      1074280         -2.690000         -3.682000         -7.408000   
      1074281         -2.690000         -3.683000         -7.406000   
      1074282         -2.689000         -3.685000         -7.405000   
      1074283         -2.688000         -3.686000         -7.404000   
      1074284         -2.688000         -3.688000         -7.403000   
      
               LowerBack_BARO_n/a  
      0                 1001.0884  
      1                 1001.0881  
      2                 1001.0877  
      3                 1001.0874  
      4                 1001.0872  
      ...                     ...  
      1074280           1000.1650  
      1074281           1000.1656  
      1074282           1000.1662  
      1074283           1000.1668  
      1074284           1000.1674  
      
      [1074285 rows x 10 columns],
      'SU_INDIP':          LowerBack_ACCEL_x  LowerBack_ACCEL_y  LowerBack_ACCEL_z  \
      0                 0.952360           0.025137          -0.303656   
      1                 0.954792           0.025641          -0.305136   
      2                 0.956515           0.027100          -0.303936   
      3                 0.958129           0.022116          -0.299751   
      4                 0.953879           0.024071          -0.303579   
      ...                    ...                ...                ...   
      1074280           0.982439           0.083063          -0.144056   
      1074281           0.985635           0.084257          -0.142461   
      1074282           0.984442           0.084510          -0.141719   
      1074283           0.983147           0.084963          -0.144032   
      1074284           0.983330           0.082541          -0.145596   
      
               LowerBack_GYRO_x  LowerBack_GYRO_y  LowerBack_GYRO_z  \
      0               -0.502367          1.005961          0.315477   
      1               -0.505664          0.872392          0.107265   
      2               -0.502459          0.867674          0.316193   
      3               -0.593967          0.737457          0.241306   
      4               -0.796316          0.793064          0.173697   
      ...                   ...               ...               ...   
      1074280         -0.019817          0.252715          0.125780   
      1074281         -0.117261          0.349371          0.105506   
      1074282         -0.158850          0.444370          0.105493   
      1074283         -0.269020          0.513086          0.162121   
      1074284         -0.351246          0.524813          0.232652   
      
               LowerBack_MAGN_x  LowerBack_MAGN_y  LowerBack_MAGN_z  \
      0              -40.721165         -9.478952          4.755029   
      1              -41.104489         -8.947342          4.881202   
      2              -41.104489         -8.947342          5.385897   
      3              -40.836163         -9.518823          5.045228   
      4              -40.759498         -9.811208          5.423749   
      ...                   ...               ...               ...   
      1074280        -34.179100          7.306624         12.968930   
      1074281        -34.242988          7.080690         12.905844   
      1074282        -34.005895          7.311054         13.026410   
      1074283        -34.051326          7.080690         13.183426   
      1074284        -34.179100          6.788305         13.107721   
      
               LeftFoot_ACCEL_x  ...  LeftFoot_MAGN_z  RightFoot_ACCEL_x  \
      0                0.846680  ...        14.824899           0.804422   
      1                0.845092  ...        14.995658           0.801165   
      2                0.847735  ...        15.023101           0.799507   
      3                0.848488  ...        15.323606           0.799081   
      4                0.847645  ...        15.412797           0.799564   
      ...                   ...  ...              ...                ...   
      1074280         -0.243655  ...        15.185932          -0.262768   
      1074281         -0.251149  ...        15.127386          -0.263699   
      1074282         -0.262728  ...        14.885885          -0.265521   
      1074283         -0.267446  ...        14.765134          -0.263540   
      1074284         -0.271330  ...        14.748668          -0.261535   
      
               RightFoot_ACCEL_y  RightFoot_ACCEL_z  RightFoot_GYRO_x  \
      0                -0.181028          -0.574083          0.072533   
      1                -0.181296          -0.573567          0.044996   
      2                -0.181399          -0.570716          0.040708   
      3                -0.179668          -0.571936          0.077413   
      4                -0.181624          -0.574851          0.075845   
      ...                    ...                ...               ...   
      1074280           0.208634          -0.938649          0.329448   
      1074281           0.209702          -0.938480          0.166966   
      1074282           0.211253          -0.938072          0.019466   
      1074283           0.210406          -0.937846         -0.036720   
      1074284           0.210334          -0.937432         -0.077651   
      
               RightFoot_GYRO_y  RightFoot_GYRO_z  RightFoot_MAGN_x  \
      0               -0.264021         -0.700053        -18.117186   
      1               -0.222030         -0.686232        -18.092139   
      2               -0.237381         -0.576345        -18.142233   
      3               -0.265936         -0.566602        -17.894268   
      4               -0.264927         -0.638134        -17.518564   
      ...                   ...               ...               ...   
      1074280         -0.089383         -0.656585          9.537144   
      1074281         -0.092967         -0.404968          9.634827   
      1074282         -0.156837         -0.264162          9.394562   
      1074283         -0.217445         -0.180748          9.372669   
      1074284         -0.172167         -0.120255          9.564696   
      
               RightFoot_MAGN_y  RightFoot_MAGN_z  
      0               -3.691539         -3.127904  
      1               -3.638682         -3.383316  
      2               -3.057251         -3.178986  
      3               -2.835249         -3.012969  
      4               -3.041393         -3.367992  
      ...                   ...               ...  
      1074280          7.807590          3.694157  
      1074281          7.469303          3.402987  
      1074282          7.594497          3.625385  
      1074283          7.796138          4.164967  
      1074284          7.831376          4.156453  
      
      [1074285 rows x 27 columns]},
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
      9               100.0  ,
      'SU_INDIP':                  name component   type tracked_point  units  \
      0   LowerBack_ACCEL_x         x  ACCEL     LowerBack      g   
      1   LowerBack_ACCEL_y         y  ACCEL     LowerBack      g   
      2   LowerBack_ACCEL_z         z  ACCEL     LowerBack      g   
      3    LowerBack_GYRO_x         x   GYRO     LowerBack  deg/s   
      4    LowerBack_GYRO_y         y   GYRO     LowerBack  deg/s   
      5    LowerBack_GYRO_z         z   GYRO     LowerBack  deg/s   
      6    LowerBack_MAGN_x         x   MAGN     LowerBack     µT   
      7    LowerBack_MAGN_y         y   MAGN     LowerBack     µT   
      8    LowerBack_MAGN_z         z   MAGN     LowerBack     µT   
      9    LeftFoot_ACCEL_x         x  ACCEL      LeftFoot      g   
      10   LeftFoot_ACCEL_y         y  ACCEL      LeftFoot      g   
      11   LeftFoot_ACCEL_z         z  ACCEL      LeftFoot      g   
      12    LeftFoot_GYRO_x         x   GYRO      LeftFoot  deg/s   
      13    LeftFoot_GYRO_y         y   GYRO      LeftFoot  deg/s   
      14    LeftFoot_GYRO_z         z   GYRO      LeftFoot  deg/s   
      15    LeftFoot_MAGN_x         x   MAGN      LeftFoot     µT   
      16    LeftFoot_MAGN_y         y   MAGN      LeftFoot     µT   
      17    LeftFoot_MAGN_z         z   MAGN      LeftFoot     µT   
      18  RightFoot_ACCEL_x         x  ACCEL     RightFoot      g   
      19  RightFoot_ACCEL_y         y  ACCEL     RightFoot      g   
      20  RightFoot_ACCEL_z         z  ACCEL     RightFoot      g   
      21   RightFoot_GYRO_x         x   GYRO     RightFoot  deg/s   
      22   RightFoot_GYRO_y         y   GYRO     RightFoot  deg/s   
      23   RightFoot_GYRO_z         z   GYRO     RightFoot  deg/s   
      24   RightFoot_MAGN_x         x   MAGN     RightFoot     µT   
      25   RightFoot_MAGN_y         y   MAGN     RightFoot     µT   
      26   RightFoot_MAGN_z         z   MAGN     RightFoot     µT   
      
          sampling_frequency  
      0                100.0  
      1                100.0  
      2                100.0  
      3                100.0  
      4                100.0  
      5                100.0  
      6                100.0  
      7                100.0  
      8                100.0  
      9                100.0  
      10               100.0  
      11               100.0  
      12               100.0  
      13               100.0  
      14               100.0  
      15               100.0  
      16               100.0  
      17               100.0  
      18               100.0  
      19               100.0  
      20               100.0  
      21               100.0  
      22               100.0  
      23               100.0  
      24               100.0  
      25               100.0  
      26               100.0  },
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

    {'SU':          LowerBack_ACCEL_x  LowerBack_ACCEL_y  LowerBack_ACCEL_z  \
    0                 0.942178           0.007510          -0.288112   
    1                 0.942810           0.010240          -0.289560   
    2                 0.943393           0.006346          -0.295584   
    3                 0.940556           0.002275          -0.291497   
    4                 0.941915           0.005847          -0.292111   
    ...                    ...                ...                ...   
    1074280           0.971606           0.054683          -0.130928   
    1074281           0.970852           0.056564          -0.135593   
    1074282           0.969576           0.056643          -0.132909   
    1074283           0.973411           0.056775          -0.136337   
    1074284           0.972761           0.054689          -0.136837   
    
             LowerBack_GYRO_x  LowerBack_GYRO_y  LowerBack_GYRO_z  \
    0                0.137440          0.916976          0.332333   
    1                0.137510          0.676277          0.561320   
    2                0.252010          0.595940          0.389748   
    3               -0.091394          0.550077          0.469760   
    4               -0.183270          0.630187          0.435476   
    ...                   ...               ...               ...   
    1074280          0.607320          0.401071          0.217741   
    1074281          0.389616          0.412534          0.458358   
    1074282          0.469829          0.492739          0.297937   
    1074283          0.561494          0.355235          0.275020   
    1074284          0.275020          0.423989          0.286479   
    
             LowerBack_MAGN_x  LowerBack_MAGN_y  LowerBack_MAGN_z  \
    0               -3.220996         -5.058998         -7.994002   
    1               -3.224997         -5.059000         -7.991002   
    2               -3.229996         -5.059000         -7.986004   
    3               -3.234996         -5.056002         -7.980005   
    4               -3.238997         -5.052003         -7.973006   
    ...                   ...               ...               ...   
    1074280         -2.690000         -3.682000         -7.408000   
    1074281         -2.690000         -3.683000         -7.406000   
    1074282         -2.689000         -3.685000         -7.405000   
    1074283         -2.688000         -3.686000         -7.404000   
    1074284         -2.688000         -3.688000         -7.403000   
    
             LowerBack_BARO_n/a  
    0                 1001.0884  
    1                 1001.0881  
    2                 1001.0877  
    3                 1001.0874  
    4                 1001.0872  
    ...                     ...  
    1074280           1000.1650  
    1074281           1000.1656  
    1074282           1000.1662  
    1074283           1000.1668  
    1074284           1000.1674  
    
    [1074285 rows x 10 columns], 'SU_INDIP':          LowerBack_ACCEL_x  LowerBack_ACCEL_y  LowerBack_ACCEL_z  \
    0                 0.952360           0.025137          -0.303656   
    1                 0.954792           0.025641          -0.305136   
    2                 0.956515           0.027100          -0.303936   
    3                 0.958129           0.022116          -0.299751   
    4                 0.953879           0.024071          -0.303579   
    ...                    ...                ...                ...   
    1074280           0.982439           0.083063          -0.144056   
    1074281           0.985635           0.084257          -0.142461   
    1074282           0.984442           0.084510          -0.141719   
    1074283           0.983147           0.084963          -0.144032   
    1074284           0.983330           0.082541          -0.145596   
    
             LowerBack_GYRO_x  LowerBack_GYRO_y  LowerBack_GYRO_z  \
    0               -0.502367          1.005961          0.315477   
    1               -0.505664          0.872392          0.107265   
    2               -0.502459          0.867674          0.316193   
    3               -0.593967          0.737457          0.241306   
    4               -0.796316          0.793064          0.173697   
    ...                   ...               ...               ...   
    1074280         -0.019817          0.252715          0.125780   
    1074281         -0.117261          0.349371          0.105506   
    1074282         -0.158850          0.444370          0.105493   
    1074283         -0.269020          0.513086          0.162121   
    1074284         -0.351246          0.524813          0.232652   
    
             LowerBack_MAGN_x  LowerBack_MAGN_y  LowerBack_MAGN_z  \
    0              -40.721165         -9.478952          4.755029   
    1              -41.104489         -8.947342          4.881202   
    2              -41.104489         -8.947342          5.385897   
    3              -40.836163         -9.518823          5.045228   
    4              -40.759498         -9.811208          5.423749   
    ...                   ...               ...               ...   
    1074280        -34.179100          7.306624         12.968930   
    1074281        -34.242988          7.080690         12.905844   
    1074282        -34.005895          7.311054         13.026410   
    1074283        -34.051326          7.080690         13.183426   
    1074284        -34.179100          6.788305         13.107721   
    
             LeftFoot_ACCEL_x  ...  LeftFoot_MAGN_z  RightFoot_ACCEL_x  \
    0                0.846680  ...        14.824899           0.804422   
    1                0.845092  ...        14.995658           0.801165   
    2                0.847735  ...        15.023101           0.799507   
    3                0.848488  ...        15.323606           0.799081   
    4                0.847645  ...        15.412797           0.799564   
    ...                   ...  ...              ...                ...   
    1074280         -0.243655  ...        15.185932          -0.262768   
    1074281         -0.251149  ...        15.127386          -0.263699   
    1074282         -0.262728  ...        14.885885          -0.265521   
    1074283         -0.267446  ...        14.765134          -0.263540   
    1074284         -0.271330  ...        14.748668          -0.261535   
    
             RightFoot_ACCEL_y  RightFoot_ACCEL_z  RightFoot_GYRO_x  \
    0                -0.181028          -0.574083          0.072533   
    1                -0.181296          -0.573567          0.044996   
    2                -0.181399          -0.570716          0.040708   
    3                -0.179668          -0.571936          0.077413   
    4                -0.181624          -0.574851          0.075845   
    ...                    ...                ...               ...   
    1074280           0.208634          -0.938649          0.329448   
    1074281           0.209702          -0.938480          0.166966   
    1074282           0.211253          -0.938072          0.019466   
    1074283           0.210406          -0.937846         -0.036720   
    1074284           0.210334          -0.937432         -0.077651   
    
             RightFoot_GYRO_y  RightFoot_GYRO_z  RightFoot_MAGN_x  \
    0               -0.264021         -0.700053        -18.117186   
    1               -0.222030         -0.686232        -18.092139   
    2               -0.237381         -0.576345        -18.142233   
    3               -0.265936         -0.566602        -17.894268   
    4               -0.264927         -0.638134        -17.518564   
    ...                   ...               ...               ...   
    1074280         -0.089383         -0.656585          9.537144   
    1074281         -0.092967         -0.404968          9.634827   
    1074282         -0.156837         -0.264162          9.394562   
    1074283         -0.217445         -0.180748          9.372669   
    1074284         -0.172167         -0.120255          9.564696   
    
             RightFoot_MAGN_y  RightFoot_MAGN_z  
    0               -3.691539         -3.127904  
    1               -3.638682         -3.383316  
    2               -3.057251         -3.178986  
    3               -2.835249         -3.012969  
    4               -3.041393         -3.367992  
    ...                   ...               ...  
    1074280          7.807590          3.694157  
    1074281          7.469303          3.402987  
    1074282          7.594497          3.625385  
    1074283          7.796138          4.164967  
    1074284          7.831376          4.156453  
    
    [1074285 rows x 27 columns]}
    

We see that that `data` attribute is in the form of a Python `dict`, where the keys correspond to the tracking systems that we have requested when calling the `load_recording()` function. KielMAT is setup so that the keys of the `channels` attribute match with these keys, so that the channel descriptions are availbale per tracking system.


```python
print(f"We have the following keys in recording.data: {recording.data.keys()}")

print(f"We have the same keys in recordings.channels: {recording.channels.keys()}")
```

    We have the following keys in recording.data: dict_keys(['SU', 'SU_INDIP'])
    We have the same keys in recordings.channels: dict_keys(['SU', 'SU_INDIP'])
    
