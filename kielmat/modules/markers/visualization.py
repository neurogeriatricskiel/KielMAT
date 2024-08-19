import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def show_marker_gaps(marker_data: pd.DataFrame, skip_markers: None | str | list[str] = None) -> None:
    # Parse skip_markers
    skip_markers = skip_markers or []
    skip_markers = [skip_markers] if isinstance(skip_markers, str) else skip_markers
    
    # Get the marker position values, and duplicate the values to similate additional color channels
    X = marker_data.to_numpy()  # recommended
    X = X[:, :, np.newaxis]
    X = np.repeat(X, 3, axis=-1)

    Y = np.where(np.isnan(X), 1, X)
    return