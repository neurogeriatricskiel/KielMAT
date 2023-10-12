def get_unit_from_type(channel_types):
    """This function returns the unit of most common channel types automatically.
    For example ACCEL is always m/s^2, GYRO is always deg/s, MAG is always T, etc.


    Args:
        channel_types (list): List of channel types

    Returns:
        list: List of units
    """

    # create list of units as long as types
    units = [None] * len(channel_types)

    # automatically assign units if common channel types are found
    for i, ch_type in enumerate(channel_types):
        if ch_type == "ACCEL":
            units[i] = "m/s^2"
        elif ch_type == "GYRO":
            units[i] = "deg/s"
        elif ch_type == "MAGN":
            units[i] = "T"
        else:
            units[i] = None

    return units
