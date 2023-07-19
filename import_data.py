from ngmt.motiondata import MotionData
from pathlib import Path

fpath = Path(r'C:\Users\User\Desktop\kiel\NGMT\examples\data\exDataHasomed.csv')
hm_test = MotionData.import_hasomed_imu(fpath)

hm_test.info