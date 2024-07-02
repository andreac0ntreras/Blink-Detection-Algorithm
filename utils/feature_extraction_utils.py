import numpy as np


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


def calculate_concat_blink_rate(concat_onsets, missing_values, sampling_freq):
    """
    Calculate the average blink rate. Useful for when you only have a list of onsets/offsets

    Parameters:
    concat_onsets (list): List containing concatenated blink onsets (or offsets).
    missing_values (np.array): Boolean array indicating missing values (True if missing, False otherwise).
    sampling_freq (float): The sampling frequency (samples per second).

    Returns:
    float: Average blink rate (blinks per minute).
    """
    # Sets total_blinks equal to the number of blink onsets
    total_blinks = len(concat_onsets)

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
    blinks (DataFrame): DataFrame containing 'blink_onset' and 'blink_offset' columns.

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
    blinks (DataFrame): DataFrame containing 'blink_onset' and 'blink_offset' columns.

    Returns:
    float: Standard deviation of blink durations in seconds.
    """
    blink_durations = np.array(blinks['blink_offset']) - np.array(blinks['blink_onset'])
    blink_duration_variability = np.std(blink_durations)
    return blink_duration_variability


def calculate_inter_blink_interval(blink_onsets):
    """
    Calculate the inter-blink interval (IBI) from blink onset times.

    Parameters:
    blink_onsets (list): List or array of blink onset times.

    Returns:
    list: List of inter-blink intervals in seconds.
    """
    # Ensure blink onsets are sorted
    blink_onsets = sorted(blink_onsets)

    # Calculate the intervals between consecutive blinks
    ibi = np.diff(blink_onsets)

    return ibi


def missing_data_excluding_blinks_both_pupils(pupil_size_left, pupil_size_right, onsets, offsets, timestamps):
    """
    Identify samples with missing data, excluding periods marked as blinks. Useful when you have two
    pupil sizes which you want to take into account

    Parameters:
    pupil_size_left (np.ndarray): Array of pupil sizes for the left eye, with NaN indicating missing values.
    pupil_size_right (np.ndarray): Array of pupil sizes for the right eye, with NaN indicating missing values.
    onsets (np.ndarray): Array of blink onset timestamps.
    offsets (np.ndarray): Array of blink offset timestamps.
    timestamps (np.ndarray): Array of timestamps for the samples.

    Returns:
    np.ndarray: Boolean array where True indicates missing data not within blink periods.
    """
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


def missing_data_excluding_time_range(bool_array_of_missing_values, sampling_freq):
    """
    This function iterates through a boolean array representing missing data
    and identifies consecutive groups of True values. It then filters out groups
    shorter than a specified duration (considered irrelevant gaps because a blink could
    not occur during the gap).

    Parameters:
    bool_array_of_missing_values (np.array): Boolean array where True indicates missing data.
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


def missing_data_excluding_blinks_single_pupil(pupil_size, onsets, offsets, timestamps):
    """
    Identify indices of missing data that are not caused by blinks.

    This function considers a sample missing if the pupil size data
    is NaN (Not a Number). It then uses blink detection results
    to exclude missing data points occurring during blinks.

    Parameters:
    pupil_size (np.array): Array containing pupil size data.
    sampling_freq (int): Sampling frequency of the data in Hz.
    timestamps (np.array): Array containing timestamps for each data point.

    Returns:
    np.array: Boolean array indicating missing data indices (True)
              excluding those caused by blinks (False).
    """

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
