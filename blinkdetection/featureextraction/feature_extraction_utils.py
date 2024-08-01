import numpy as np
import blinkdetection as bd


def calculate_blink_rate(blinks, missing_values, sampling_freq):
    """
    Calculate the average blink rate.

    Parameters:
        blinks (dict): Dictionary containing blink onset and offset timestamps.
        missing_values (np.array): Boolean array indicating missing values (True if missing, False otherwise).
        sampling_freq (float): The sampling frequency (samples per second).

    Returns:
        average_blink_rate (float): Average blink rate (blinks per minute).
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
        average_blink_duration (float): Average blink duration in seconds.
    """
    # Calculate the durations of each blink by subtracting onset times from offset times
    blink_durations = np.array(blinks['blink_offset']) - np.array(blinks['blink_onset'])
    # Calculate the average of these durations
    average_blink_duration = np.mean(blink_durations)
    return average_blink_duration


def calculate_blink_duration_variability(blinks):
    """
    Calculate the blink duration variability (standard deviation of blink durations).

    Parameters:
        blinks (dict): dict containing 'blink_onset' and 'blink_offset' keys with the timestamps as the values.

    Returns:
        blink_duration_variability (float): Standard deviation of blink durations in seconds.
    """
    # Calculate the durations of each blink by subtracting onset times from offset times
    blink_durations = np.array(blinks['blink_offset']) - np.array(blinks['blink_onset'])
    # Calculate the standard deviation of these durations
    blink_duration_variability = np.std(blink_durations)
    return blink_duration_variability


def onset_difference(left_blinks, right_blinks, tolerance=.15):
    """
    Calculates the difference in seconds between the same blinks characterized in the left and right eyes.

    Parameters:
        left_blinks (dict): dict containing 'blink_onset' and 'blink_offset' keys for the left eye with the timestamps
        as the values.
        right_blinks (dict): dict containing 'blink_onset' and 'blink_offset' keys for the right eye with the timestamps
        as the values.
        tolerance (float): The maximum allowed difference (in seconds) between left and right blink onsets
        to consider them as a single blink event. Default is 0.15 seconds.

    Returns:
        difference_of_onsets (float): The mean difference between the time at which the left eye
        register an offset vs the right eye.
    """
    left_onsets = np.array(left_blinks["blink_onset"])
    right_onsets = np.array(right_blinks["blink_onset"])
    summed_differences = 0

    i, j = 0, 0
    counter = 0
    # Iterate through the onsets of both eyes
    while i < len(left_onsets) and j < len(right_onsets):
        # If the onsets are within the tolerance, calculate the difference
        if abs(left_onsets[i] - right_onsets[j]) <= tolerance:
            summed_differences += abs(left_onsets[i] - right_onsets[j])
            i += 1
            j += 1
            counter += 1
        # If the left onset is earlier, move to the next left onset
        elif left_onsets[i] < right_onsets[j]:
            i += 1
        # If the right onset is earlier, move to the next right onset
        else:
            j += 1

    # If no matching blinks are found, return NaN
    if counter == 0:
        return float('nan')

    # Calculate the mean difference of the onsets
    difference_of_onsets = summed_differences / counter
    return difference_of_onsets


def offset_difference(left_blinks, right_blinks, tolerance=.15):
    """
    Calculates the difference in seconds between the same blinks characterized in the left and right eyes.

    Parameters:
        left_blinks (dict): dict containing 'blink_onset' and 'blink_offset' keys for the left eye with the timestamps
        as the values.
        right_blinks (dict): dict containing 'blink_onset' and 'blink_offset' keys for the right eye with the timestamps
        as the values.
        tolerance (float): The maximum allowed difference (in seconds) between left and right blink onsets
        to consider them as a single blink event. Default is 0.15 seconds.

    Returns:
        difference_of_offsets (float): The mean difference between the time at which the left eye
        register an offset vs the right eye.
    """
    left_offsets = np.array(left_blinks["blink_offset"])
    right_offsets = np.array(right_blinks["blink_offset"])
    summed_differences = 0

    i, j = 0, 0
    counter = 0
    # Iterate through the offsets of both eyes
    while i < len(left_offsets) and j < len(right_offsets):
        # If the offsets are within the tolerance, calculate the difference
        if abs(left_offsets[i] - right_offsets[j]) <= tolerance:
            summed_differences += abs(left_offsets[i] - right_offsets[j])
            i += 1
            j += 1
            counter += 1
        # If the left offset is earlier, move to the next left offset
        elif left_offsets[i] < right_offsets[j]:
            i += 1
        # If the right offset is earlier, move to the next right offset
        else:
            j += 1

    # If no matching blinks are found, return NaN


def duration_difference(left_blinks, right_blinks, tolerance=.15):
    """
    Calculates the difference in seconds between the durations of the same blinks characterized in the left and right
    eyes.

    Parameters:
        left_blinks (dict): Dictionary containing 'blink_onset' and 'blink_offset' keys for the left eye with the
        timestamps as the values.
        right_blinks (dict): Dictionary containing 'blink_onset' and 'blink_offset' keys for the right eye with the
        timestamps as the values.
        tolerance (float): The maximum allowed difference (in seconds) between left and right blink onsets to consider
        them as a single blink event. Default is 0.15 seconds.

    Returns:
        duration_diff (float): The mean difference between the durations of the blinks for the left eye vs the
        right eye.
    """
    left_onsets = np.array(left_blinks["blink_onset"])
    left_offsets = np.array(left_blinks["blink_offset"])
    right_onsets = np.array(right_blinks["blink_onset"])
    right_offsets = np.array(right_blinks["blink_offset"])

    # Calculate the durations of blinks for both eyes
    left_durations = left_offsets - left_onsets
    right_durations = right_offsets - right_onsets

    summed_differences = 0
    i, j = 0, 0
    counter = 0
    # Iterate through the onsets of both eyes
    while i < len(left_onsets) and j < len(right_onsets):
        # If the onsets are within the tolerance, calculate the difference in durations
        if abs(left_onsets[i] - right_onsets[j]) <= tolerance:
            summed_differences += abs(left_durations[i] - right_durations[j])
            i += 1
            j += 1
            counter += 1
        # If the left onset is earlier, move to the next left onset
        elif left_onsets[i] < right_onsets[j]:
            i += 1
        # If the right onset is earlier, move to the next right onset
        else:
            j += 1

    # If no matching blinks are found, return NaN
    if counter == 0:
        return float('nan')

    # Calculate the mean difference of the durations
    duration_diff = summed_differences / counter
    return duration_diff


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
    Calculate the mean inter-blink interval (IBI).

    Parameters:
        ibi (list): List of inter-blink intervals in seconds.
        (This is generated using the calculate_inter_blink_interval function.)

    Returns:
        mean_ibi (float): The mean inter-blink interval.
    """
    mean_ibi = np.mean(ibi)
    return mean_ibi


def inter_blink_interval_variability(ibi):
    """
    Calculate the inter-blink interval (IBI) variability.

    Parameters:
        ibi (list): List of inter-blink intervals in seconds.
        (This is generated using the calculate_inter_blink_interval function.)

    Returns:
        ibi_var (float): The variability of the inter-blink interval.
    """
    ibi_var = np.std(ibi)
    return ibi_var


def average_pupil_size_without_blinks(pupil_size, timestamps, blinks):
    """
    Calculate the average pupil size excluding the blink periods.

    Parameters:
        pupil_size (list): List containing a timeseries of the pupil sizes of one eye.
        timestamps (list): List of timestamps corresponding to the list of pupil sizes.
        blinks (dict): dict containing 'blink_onset' and 'blink_offset' keys with the timestamps as the values.

    Returns:
        average_size (float): The average pupil size excluding the blink periods.
    """
    # Remove NaN values from pupil_size and the corresponding timestamps
    # We must remove the NaN values because adding a NaN value to a numerical value results in NaN.
    # Also, NaN means the pupil size was not recorded, meaning no valuable pupil size data is lost in
    # removing NaN periods
    valid_mask = ~pupil_size.isna()
    pupil_size = pupil_size[valid_mask]
    timestamps = timestamps[valid_mask]

    # Initialize total_size and valid_count for calculating the average
    total_size = 0.0
    valid_count = 0

    # Iterate over each pupil size and timestamp
    for size, timestamp in zip(pupil_size, timestamps):
        in_blink_period = False
        # Check if the timestamp falls within any blink period
        for onset, offset in zip(blinks["blink_onset"], blinks["blink_offset"]):
            if onset <= timestamp <= offset:
                in_blink_period = True
                break
        # If the timestamp is not within a blink period, add the size to total_size and increment valid_count
        if not in_blink_period:
            total_size += size
            valid_count += 1

    # Calculate the average size
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
        variability (float): The standard deviation of the pupil size excluding the blink periods.
    """
    blink_onset = blinks['blink_onset']
    blink_offset = blinks['blink_offset']

    # Remove NaN values from pupil_size and the corresponding timestamps
    valid_mask = ~pupil_size.isna()
    pupil_size = pupil_size[valid_mask]
    timestamps = timestamps[valid_mask]

    # Initialize list to store valid pupil sizes
    valid_pupil_sizes = []

    # Iterate over each pupil size and timestamp
    for size, timestamp in zip(pupil_size, timestamps):
        in_blink_period = False
        # Check if the timestamp falls within any blink period
        for onset, offset in zip(blink_onset, blink_offset):
            if onset <= timestamp <= offset:
                in_blink_period = True
                break
        # If the timestamp is not within a blink period, add the size to valid_pupil_sizes
        if not in_blink_period:
            valid_pupil_sizes.append(size)

    # If no valid pupil sizes are found, return NaN
    if len(valid_pupil_sizes) == 0:
        return float('nan')

    # Calculate the standard deviation of the valid pupil sizes
    variability = np.std(valid_pupil_sizes)
    return variability


def missing_data_excluding_blinks_both_pupils(pupil_size_left, pupil_size_right, blinks, timestamps):
    """
    Identify samples with missing data, excluding periods marked as blinks. Useful when you have two
    pupil sizes which you want to take into account.

    Parameters:
        pupil_size_left (np.ndarray): Array of pupil sizes for the left eye, with NaN indicating missing values.
        pupil_size_right (np.ndarray): Array of pupil sizes for the right eye, with NaN indicating missing values.
        blinks (dict): dict containing 'blink_onset' and 'blink_offset' keys with the timestamps as the values.
        timestamps (np.ndarray): Array of timestamps for the samples.

    Returns:
        missing_data (np.ndarray): Boolean array where True indicates missing data not within blink periods.
    """
    onsets = blinks["blink_onset"]
    offsets = blinks["blink_offset"]

    # Initialize an array to hold missing data flags
    missing_pupil_size = np.isnan(pupil_size_left) & np.isnan(pupil_size_right)

    # Initialize array to store missing data flags excluding blinks
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
        missing_data (np.array): Boolean array indicating missing data indices (True)
              excluding those caused by blinks (False).
    """
    onsets = blinks["blink_onset"]
    offsets = blinks["blink_offset"]

    # Detect missing data
    missing_pupil_size = np.isnan(pupil_size)

    # Initialize array to store missing data flags excluding blinks
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


def calculate_missing_data_excluding_blinks(df, sampling_freq, interval_time=2):
    """
    Calculate the percentage of missing data excluding blinks from pupil size data.

    This function processes pupil size data and timestamps to calculate the percentage
    of missing data points that are not caused by blinks within specified intervals.

    Parameters:
        df (pd.DataFrame): DataFrame containing columns 'pupil_size_left', 'pupil_size_right', and 'timestamps'.
        sampling_freq (int): Sampling frequency of the data in Hz.
        interval_time (int, optional): Time interval for processing data in seconds. Defaults to 2 seconds.

    Returns:
        total_missing_data_excluding_blinks (np.ndarray): Boolean array indicating missing values
        (True if missing, False otherwise).
        missing_data_percentage_excluding_blinks (float): The percentage of missing data excluding blinks.
    """

    # Extract pupil sizes and timestamps from the dataframe
    pupil_size_left = df["pupil_size_left"]
    pupil_size_right = df["pupil_size_right"]
    timestamps = df["timestamps"]

    # Calculate the interval size in samples
    interval_size = int(interval_time * sampling_freq)

    # Initialize a list to store missing data flags excluding blinks
    total_missing_data_excluding_blinks = []

    # Iterate through the data in specified intervals
    for start in range(0, len(pupil_size_left), interval_size):
        end = min(start + interval_size, len(pupil_size_left))

        # Extract interval data for both pupils
        interval_left = pupil_size_left[start:end]
        interval_right = pupil_size_right[start:end]
        interval_timestamps = timestamps[start:end]

        # Detect blinks in the interval for both pupils
        blinks_left = bd.single_pupil_blink_detection(interval_left, sampling_freq, interval_timestamps)
        blinks_right = bd.single_pupil_blink_detection(interval_right, sampling_freq, interval_timestamps)

        # Identify missing data excluding blinks for both pupils
        missing_excluding_blinks_left = missing_data_excluding_blinks_single_pupil(interval_left, blinks_left, interval_timestamps)
        missing_excluding_blinks_right = missing_data_excluding_blinks_single_pupil(interval_right, blinks_right, interval_timestamps)

        # Determine which pupil has less missing data and use its result
        if np.sum(np.isnan(interval_left)) <= np.sum(np.isnan(interval_right)):
            total_missing_data_excluding_blinks.extend(missing_excluding_blinks_left)
        else:
            total_missing_data_excluding_blinks.extend(missing_excluding_blinks_right)

    # Calculate the percentage of missing data excluding blinks
    missing_data_percentage_excluding_blinks = np.mean(total_missing_data_excluding_blinks) * 100

    return total_missing_data_excluding_blinks, missing_data_percentage_excluding_blinks
