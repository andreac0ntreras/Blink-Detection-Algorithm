import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from utils import blink_detection_utils


def process_individual_csv(csv_file, folder):
    """
    Process an individual CSV file to calculate the average blink rate.

    Parameters:
    csv_file (str): Name of the CSV file to process.
    folder (str): Path to the folder containing the CSV file.

    Returns:
    dict: Dictionary containing subject ID, day number, and average blink rate.
    """

    # Extract subject ID and day number from filename components
    subject_id, day_number = csv_file.split('_')[0], csv_file.split('_')[4].split('.')[0]

    # Construct the full file path based on folder and filename
    file_path = os.path.join(folder, csv_file)

    # Read the CSV file using pandas
    dataframe = pd.read_csv(file_path)

    # Extract relevant columns by header name
    timestamps = dataframe['timestamps']
    pupil_size_left = dataframe['pupil_size_left']
    pupil_size_right = dataframe['pupil_size_right']

    # Detect blinks using a separate function (not included here)
    left_blinks = blink_detection_utils.single_pupil_blink_detection(pupil_size_left, 600, timestamps)
    right_blinks = blink_detection_utils.single_pupil_blink_detection(pupil_size_right, 600, timestamps)

    # Identify missing data indices for both pupil size columns
    left_missing_data = missing_data_index(pupil_size_left, 600, timestamps)

    # Filter missing data indices to only include the relevant missing data (data where a blink can occur: over 100ms)
    left_missing_data_correct = missing_data_time_range(left_missing_data, 600)

    # Calculate the average blink rate considering missing data periods
    left_average_blink_rate = calculate_blink_rate(left_blinks, left_missing_data_correct, 600)

    # Identify missing data indices for both pupil size columns
    right_missing_data = missing_data_index(pupil_size_right, 600, timestamps)

    # Filter missing data indices to only include the relevant missing data (data where a blink can occur: over 100ms)
    right_missing_data_correct = missing_data_time_range(right_missing_data, 600)

    # Calculate the average blink rate considering missing data periods
    right_average_blink_rate = calculate_blink_rate(right_blinks, right_missing_data_correct, 600)

    # Calculate average blink duration and blink duration variability
    left_average_blink_duration = calculate_average_blink_duration(left_blinks)
    right_average_blink_duration = calculate_average_blink_duration(right_blinks)
    left_blink_duration_variability = calculate_blink_duration_variability(left_blinks)
    right_blink_duration_variability = calculate_blink_duration_variability(right_blinks)

    # Calculate the concatenated onsets and offsets based on how closely the onsets anf offsets of the left and right
    # pupil match
    concat_onsets, concat_offsets = identify_concat_blinks(left_blinks, right_blinks)

    return {
        'subject': subject_id,
        'day': day_number,
        'left_blink_onsets': left_blinks["blink_onset"].values,
        'left_blink_offsets': left_blinks["blink_offset"].values,
        'right_blink_onsets': right_blinks["blink_onset"].values,
        'right_blink_offsets': right_blinks["blink_offset"].values,
        'concatenated_onsets': concat_onsets,
        'concatenated_offsets': concat_offsets,
        'left_average_blink_rate': left_average_blink_rate,
        'right_average_blink_rate': right_average_blink_rate,
        'left_average_blink_duration': left_average_blink_duration,
        'right_average_blink_duration': right_average_blink_duration,
        'left_blink_duration_variability': left_blink_duration_variability,
        'right_blink_duration_variability': right_blink_duration_variability,
        'left_missing_data_percentage': np.mean(left_missing_data),
        'right_missing_data_percentage': np.mean(right_missing_data)
    }


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


def identify_concat_blinks(left_blinks, right_blinks, tolerance=.1):
    """
        Identify and concatenate blinks from left and right eye blink data.

        This function takes blink onset and offset times for left and right eye blinks,
        compares them within a given tolerance, and identifies concurrent blinks.
        The identified concurrent blinks are then concatenated into single onsets
        and offsets using the maximum of the overlapping values.

        Parameters:
        left_blinks (DataFrame): A DataFrame containing 'blink_onset' and 'blink_offset' columns for left eye blinks.
        right_blinks (DataFrame): A DataFrame containing 'blink_onset' and 'blink_offset' columns for right eye blinks.
        tolerance (float): The maximum allowed difference (in seconds) between left and right blink onsets
                           to consider them as a single blink event. Default is 0.1 seconds.

        Returns:
        concat_onsets (list): A list of concatenated blink onset times.
        concat_offsets (list): A list of concatenated blink offset times.
    """
    left_onsets = np.array(left_blinks["blink_onset"])
    left_offsets = np.array(left_blinks["blink_offset"])
    right_onsets = np.array(right_blinks["blink_onset"])
    right_offsets = np.array(left_blinks["blink_offset"])
    concat_onsets = []
    concat_offsets = []

    i, j = 0, 0
    while i < len(left_onsets) and j < len(right_onsets):
        if abs(left_onsets[i] - right_onsets[j]) <= tolerance:
            concat_onsets.append(np.mean([left_onsets[i], right_onsets[j]]))
            if i < len(left_offsets) and j < len(right_offsets):
                concat_offsets.append(np.mean([left_offsets[i], right_offsets[j]]))
            i += 1
            j += 1
        elif left_onsets[i] < right_onsets[j]:
            i += 1
        else:
            j += 1

    return concat_onsets, concat_offsets


def process_csv_files(folder):
    """
    Process all CSV files in the specified folder and save blink rate results to a CSV file.

    Parameters:
    folder (str): Path to the folder containing CSV files.
    """
    # Initialize list of results
    results = []
    for csv_file in os.listdir(folder):
        if csv_file.endswith('.csv'):
            # Process each individual CSV file
            result = process_individual_csv(csv_file, folder)
            results.append(result)

    # Create a DataFrame from the results
    results_df = pd.DataFrame(results)

    # Save the DataFrame to a CSV file
    results_df.to_csv(os.path.join(folder, 'features/compiled_left_right_blink_rates.csv'), index=False)


def missing_data_index(pupil_size, sampling_freq, timestamps):
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
    print(np.mean(missing_pupil_size))

    # Detect blinks
    blinks = blink_detection_utils.single_pupil_blink_detection(pupil_size, sampling_freq, timestamps)
    blink_onsets = blinks['blink_onset']
    blink_offsets = blinks['blink_offset']

    # Initialize array and iterate through missing data points
    missing_data = np.zeros_like(missing_pupil_size, dtype=bool)
    for i, is_nan in enumerate(missing_pupil_size):
        if is_nan:
            timestamp = timestamps[i]
            in_blink = any(blink_onset <= timestamp <= blink_offset for blink_onset, blink_offset in
                           zip(blink_onsets, blink_offsets))
            if not in_blink:
                missing_data[i] = True

    return missing_data


def missing_data_time_range(bool_array_of_missing_values, sampling_freq):
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


def plot_pupil_size_v_time(csv_file):
    """
    This function reads a CSV file containing pupil size data, detects blinks,
    and plots pupil size (left and right eye) vs time.

    Args:
        csv_file (str): Path to the CSV file containing pupil size data.

    """

    # Read the CSV file using pandas
    dataframe = pd.read_csv(csv_file)

    # Extract columns by header name
    timestamps = dataframe['timestamps']
    pupil_size_left = dataframe['pupil_size_left']
    pupil_size_right = dataframe['pupil_size_right']

    # Detect blinks using a separate function (not included here)
    left_blinks = blink_detection_utils.single_pupil_blink_detection(pupil_size_left, 600, timestamps)
    right_blinks = blink_detection_utils.single_pupil_blink_detection(pupil_size_right, 600, timestamps)

    # Plot pupil size vs time for left and right eye
    plt.plot(timestamps, pupil_size_left, label='Left Size')
    plt.plot(timestamps, pupil_size_right, label='Right Size')

    # Mark blink onsets and offsets with vertical lines
    for blink_onset in left_blinks["blink_onset"]:
        plt.axvline(blink_onset, color='green')
    for blink_offset in left_blinks["blink_offset"]:
        plt.axvline(blink_offset, color='pink')
    for blink_onset in right_blinks["blink_onset"]:
        plt.axvline(blink_onset, color='yellow')
    for blink_offset in right_blinks["blink_offset"]:
        plt.axvline(blink_offset, color='red')

    # Label axes and set title
    plt.xlabel('Time (s)')
    plt.ylabel('Size')
    subject_id = csv_file.split('_')[0].split("/")[-1]
    day_number = csv_file.split('_')[4].split('.')[0]
    plt.title(f'{subject_id}_{day_number} Pupil Size')
    plt.legend()
    plt.show()


output_folder = 'output'

# Create a new CSV file with all participants, days, and blink rates
process_csv_files("output")

plot_pupil_size_v_time("output/sub-025_rseeg_block_1_d03_20230720_eyetracker.csv")
