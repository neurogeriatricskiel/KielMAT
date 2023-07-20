import matplotlib.pyplot as plt
from ngmt.motiondata import MotionData
from pathlib import Path

fpath = Path(r'C:\Users\User\Desktop\kiel\NGMT\examples\data\exDataHasomed2.csv')
hm_test = MotionData.import_hasomed_imu(fpath)

# plot a channel
plt.plot(hm_test.time_series[0:3,:].T)