import matplotlib.pyplot as plt
import numpy as np

# get 3 colors from viridis
raw_xyz_color = plt.cm.viridis(np.linspace(0, 1, 3))
prep_xyz_color = plt.cm.magma(np.linspace(0, 1, 3))

cfg_colors = {"raw": raw_xyz_color, "prep": prep_xyz_color}
