import numpy as np
import pandas as pd


def calculate_blink_rate(blinks, missing_values, sampling_freq):
    """
    Calculate the average blink rate.

    Parameters:
    blinks (dict): Dictionary containing blink onset and offset timestamps.
    missing_values (np.array): Boolean array indicating missing values (True if missing, False otherwise).
    sampling_freq (float): The sampling frequency (samples per second).

    Returns:
    float: Average blink rate (blinks per minute).
    """
    # Sets total_blinks equal to the number of blink onsets
    total_blinks = len(blinks["blink_onset"])

    # Calculate the number of non-missing values
    num_of_non_missing_values = np.sum(~missing_values)

    # Calculate the duration in seconds
    duration_in_seconds = num_of_non_missing_values / sampling_freq

    # Convert the adjusted duration to minutes
    duration_in_minutes = duration_in_seconds / 60

    # Calculate the average blink rate
    average_blink_rate = total_blinks / duration_in_minutes

    return average_blink_rate


def calculate_average_blink_duration(blinks):
    """
    Calculate the average blink duration from blink onset and offset times.

    Parameters:
    blinks (dict): dict containing 'blink_onset' and 'blink_offset' keys with the timestamps as the values.

    Returns:
    float: Average blink duration in seconds.
    """
    blink_durations = np.array(blinks['blink_offset']) - np.array(blinks['blink_onset'])
    average_blink_duration = np.mean(blink_durations)
    return average_blink_duration


def calculate_blink_duration_variability(blinks):
    """
    Calculate the blink duration variability (standard deviation of blink durations).

    Parameters:
    blinks (dict): dict containing 'blink_onset' and 'blink_offset' keys with the timestamps as the values.

    Returns:
    float: Standard deviation of blink durations in seconds.
    """
    blink_durations = np.array(blinks['blink_offset']) - np.array(blinks['blink_onset'])
    blink_duration_variability = np.std(blink_durations)
    return blink_duration_variability


def calculate_inter_blink_interval(blinks):
    """
    Calculate the inter-blink interval (IBI) from blink onset times.

    Parameters:
    blinks (dict): dict containing 'blink_onset' and 'blink_offset' keys with the timestamps as the values.

    Returns:
    ibi (list): List of inter-blink intervals in seconds.
    """
    # Ensure blink onsets are sorted
    blink_onsets = sorted(blinks["blink_onset"])

    # Calculate the intervals between consecutive blinks
    ibi = np.diff(blink_onsets)

    return ibi


def mean_inter_blink_interval(ibi):
    """
    Calculate the mean inter-blink interval (IBI) excluding the blink periods.

    Parameters:
    ibi (list): List of inter-blink intervals in seconds.
    (This is generated using the calculate_inter_blink_interval function.)

    Returns:
    mean_ibi (float): The mean inter-blink interval.
    """
    mean_ibi = np.mean(ibi)
    return mean_ibi


def average_pupil_size_without_blinks(pupil_size, timestamps, blinks):
    """
    Calculate the average pupil size excluding the blink periods.

    Parameters:
    pupil_size (list): List containing a timeseries of the pupil sizes of one eye.
    timestamps (list): List of timestamps corresponding to the list of pupil sizes.
    blinks (dict): dict containing 'blink_onset' and 'blink_offset' keys with the timestamps as the values.

    Returns:
    float: The average pupil size excluding the blink periods.
    """
    # Remove NaN values from pupil_size and the corresponding timestamps
    # We must remove the nan values because adding an nan value to a numerical value results in nan.
    # Also, nan means the pupil size was not recorded, meaning no valuable pupil size data is lost in
    # removing nan periods
    valid_mask = ~pupil_size.isna()
    pupil_size = pupil_size[valid_mask]
    timestamps = timestamps[valid_mask]

    total_size = 0.0
    valid_count = 0

    for size, timestamp in zip(pupil_size, timestamps):
        in_blink_period = False
        for onset, offset in zip(blinks["blink_onset"], blinks["blink_offset"]):
            if onset <= timestamp <= offset:
                in_blink_period = True
                break
        if not in_blink_period:
            total_size += size
            valid_count += 1

    average_size = total_size / valid_count if valid_count > 0 else float('nan')
    return average_size


def pupil_size_variability(pupil_size, timestamps, blinks):
    """
    Calculate the pupil size variability excluding the blink periods.

    Parameters:
    pupil_size (pd.Series): Series containing a timeseries of the pupil sizes of one eye.
    timestamps (pd.Series): Series of timestamps corresponding to the list of pupil sizes.
    blinks (dict): dict containing 'blink_onset' and 'blink_offset' keys with the timestamps as the values.

    Returns:
    float: The standard deviation of the pupil size excluding the blink periods.
    """
    blink_onset = blinks['blink_onset']
    blink_offset = blinks['blink_offset']

    if blink_onset.empty or blink_offset.empty:
        print("Blink onset or offset series is empty.")
        return float('nan')

    # Remove NaN values from pupil_size and the corresponding timestamps
    valid_mask = ~pupil_size.isna()
    pupil_size = pupil_size[valid_mask]
    timestamps = timestamps[valid_mask]

    valid_pupil_sizes = []

    for size, timestamp in zip(pupil_size, timestamps):
        in_blink_period = False
        for onset, offset in zip(blink_onset, blink_offset):
            if onset <= timestamp <= offset:
                in_blink_period = True
                break
        if not in_blink_period:
            valid_pupil_sizes.append(size)

    if len(valid_pupil_sizes) == 0:
        return float('nan')

    variability = np.std(valid_pupil_sizes)
    return variability


def missing_data_excluding_blinks_both_pupils(pupil_size_left, pupil_size_right, blinks, timestamps):
    """
    Identify samples with missing data, excluding periods marked as blinks. Useful when you have two
    pupil sizes which you want to take into account

    Parameters:
    pupil_size_left (np.ndarray): Array of pupil sizes for the left eye, with NaN indicating missing values.
    pupil_size_right (np.ndarray): Array of pupil sizes for the right eye, with NaN indicating missing values.
    blinks (dict): dict containing 'blink_onset' and 'blink_offset' keys with the timestamps as the values.
    timestamps (np.ndarray): Array of timestamps for the samples.

    Returns:
    np.ndarray: Boolean array where True indicates missing data not within blink periods.
    """
    onsets = blinks["blink_onset"]
    offsets = blinks["blink_offset"]

    # Initialize an array to hold missing data flags
    missing_pupil_size = np.isnan(pupil_size_left) & np.isnan(pupil_size_right)

    # Initialize array and iterate through missing data points
    missing_data = np.zeros_like(missing_pupil_size, dtype=bool)

    # Iterate over each sample to check if it's missing and not within a blink period
    for i, is_nan in enumerate(missing_pupil_size):
        if is_nan:
            timestamp = timestamps[i]
            # Check if the timestamp falls within any blink period
            in_blink = any(blink_onset <= timestamp <= blink_offset for blink_onset, blink_offset in
                           zip(onsets, offsets))
            if not in_blink:
                missing_data[i] = True

    return missing_data


def missing_data_excluding_blinks_single_pupil(pupil_size, blinks, timestamps):
    """
    Identify indices of missing data that are not caused by blinks.

    This function considers a sample missing if the pupil size data
    is NaN (Not a Number). It then uses blink detection results
    to exclude missing data points occurring during blinks.

    Parameters:
    pupil_size (np.array): Array containing pupil size data.
    blinks (dict): dict containing 'blink_onset' and 'blink_offset' keys with the timestamps as the values.
    timestamps (np.array): Array containing timestamps for each data point.

    Returns:
    np.array: Boolean array indicating missing data indices (True)
              excluding those caused by blinks (False).
    """
    onsets = blinks["blink_onset"]
    offsets = blinks["blink_offset"]

    # Detect missing data
    missing_pupil_size = np.isnan(pupil_size)

    # Initialize array and iterate through missing data points
    missing_data = np.zeros_like(missing_pupil_size, dtype=bool)
    for i, is_nan in enumerate(missing_pupil_size):
        if is_nan:
            timestamp = timestamps[i]
            in_blink = any(blink_onset <= timestamp <= blink_offset for blink_onset, blink_offset in
                           zip(onsets, offsets))
            if not in_blink:
                missing_data[i] = True

    return missing_data


def missing_data_excluding_time_range(bool_array_of_missing_values, sampling_freq):
    """
    This function iterates through a boolean array representing missing data
    and identifies consecutive groups of True values. It then filters out groups
    shorter than a specified duration (considered irrelevant gaps because a blink could
    not occur during the gap).

    Parameters:
    bool_array_of_missing_values (np.array): Boolean array where True indicates missing data.
    (Generated from missing_data_excluding_blinks_single_pupil or missing_data_excluding_blinks_both_pupils)
    sampling_freq (int): Sampling frequency of the data in Hz.

    Returns:
    np.array: Boolean array with the same shape as the input,
              where True indicates filtered missing data intervals
              (excluding short gaps).
    """

    # Minimum number of samples representing 100ms (considered noise)
    min_samples_for_100ms = int(sampling_freq * 0.1)  # 100ms = 0.1 seconds

    # Create a copy of the input array for modification
    filtered_missing_values = np.copy(bool_array_of_missing_values)

    # Iterate through the boolean array to find and filter short missing data groups
    i = 0
    while i < len(bool_array_of_missing_values):
        if bool_array_of_missing_values[i]:
            # Start of a missing data group
            start = i
            while i < len(bool_array_of_missing_values) and bool_array_of_missing_values[i]:
                i += 1
            end = i  # end is exclusive

            # Check the duration of the missing data group (in number of samples)
            num_samples = end - start
            if num_samples < min_samples_for_100ms:
                # Set these values to False in the output array (considered noise)
                filtered_missing_values[start:end] = False
        else:
            i += 1

    return filtered_missing_values
