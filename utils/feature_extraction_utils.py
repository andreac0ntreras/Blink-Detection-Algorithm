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
