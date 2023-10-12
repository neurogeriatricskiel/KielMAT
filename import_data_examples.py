import matplotlib.pyplot as plt
from ngmt.motiondata import MotionData
from ngmt.utils.importers import import_polar_watch
from pathlib import Path

# set Path and import motion data form HasoMed IMU sensor
fpath = Path(r"C:\Users\User\Desktop\kiel\NGMT\examples\data\exDataHasomed2.csv")
hm_test = MotionData.import_hasomed_imu(fpath)

# set Path and import training data from POLAR-M200 Smartwatch
fpath = Path(r"C:\Users\juliu\Desktop\kiel\NGMT\examples\data\exDataPolarm200.csv")
polar_test = import_polar_watch(fpath)
