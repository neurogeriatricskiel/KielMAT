## The output of KielMAT does not make sense

If the output you are getting from KielMAT does not make sense, please check the following:

- Is the sampling frequency correct?
- Are the units correct?
- Did you accidentally mix up the accelerometer and gyroscope data?

## I am not sure what units my data is in

If you are unsure about the units of your data, you can do a quick and dirty check by looking at the data. For example, if you are looking at accelerometer data, you can check if the values are in the range of -1 to 1. If they are, the data is likely in g. If the values are in the range of -9.81 to 9.81, the data is likely in m/s^2. Similarly, for gyroscope data, if the values are in the range of -180 to 180, the data is likely in degrees per second.
