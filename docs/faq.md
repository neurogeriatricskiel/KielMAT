## Why does the output of KielMAT not make sense?

If the output you are getting from KielMAT does not make sense, please check the following:

- Is the sampling frequency correct?
- Are the units correct?
- Did you accidentally mix up the accelerometer and gyroscope data?

## What units are my data in?

If you are unsure about the units of your data, you can do a quick and dirty check by looking at the data. For example, if you are looking at accelerometer data, you can check if the values are in the range of -1 to 1. If they are, the data is likely in g. If the values are in the range of -9.81 to 9.81, the data is likely in m/s^2. Similarly, for gyroscope data, if the values are in the range of -180 to 180, the data is likely in degrees per second.

## Why am I encountering a `SSLCertVerificationError`?

The `SSLCertVerificationError` occurs when Python’s SSL module encounters an issue verifying the SSL certificate of a remote server. This can happen in secure environments like institutional networks where network security policies might be more restrictive. To address this error, you may want to try the following:

- **Update the certificate store**: make sure that the certificate store used by Python is up-to-date. This can be done by updating the `certifi` package, which Python's `requests` library (commonly used for HTTP requests) relies on:  
  
  ```bash
  pip install --upgrade certifi
  ```

- **Download the datasets manually**: you can also manually download the datasets for the respective websites, and see if that works. For that purpose, navigate in your webbrowser to [Zenodo for the Mobilise-D dataset](https://zenodo.org/records/7547125) or [OpenNeuro for the Keep Control dataset](https://openneuro.org/datasets/ds005258/versions/1.0.5), and  then download the datasets to a local folder. If you are not allowed to download the datasets from the respective websites, then it may help to contact your local IT department.
