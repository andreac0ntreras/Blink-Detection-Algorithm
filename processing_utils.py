import feature_extraction_utils
import blink_detection_utils
import pandas as pd
import os
import numpy as np


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
    blinks = blink_detection_utils.based_noise_blinks_detection(pupil_size_left, pupil_size_right, 600, timestamps)

    # Identify missing data indices for both pupil size columns
    missing_data = missing_data_index(pupil_size_left, pupil_size_right, 600, timestamps)

    # Filter missing data indices to only include the relevant missing data (data where a blink can occur: over 100ms)
    missing_data_correct = missing_data_time_range(missing_data, 600)

    # Calculate the average blink rate considering missing data periods
    average_blink_rate = feature_extraction_utils.calculate_blink_rate(blinks, missing_data_correct, 600)

    # Print information about missing data percentage and average blink rate
    print(f"{csv_file} Percentage of Missing Data: {np.mean(missing_data_correct)}")
    print("Average Blink Rate for", csv_file, "(blinks per minute):", average_blink_rate)

    # Return results as a dictionary
    return {'Subject': subject_id, 'Day': day_number, 'Average Blink Rate': average_blink_rate,
            'Percentage Missing Data': np.mean(missing_data_correct)}


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
    results_df.to_csv(os.path.join(folder, 'features/compiled_blink_rates.csv'), index=False)


def missing_data_index(pupil_size_left, pupil_size_right, sampling_freq, timestamps):
    """
    Identify indices of missing data that are not caused by blinks.

    This function combines pupil size data from left and right eye
    and considers a sample missing only if both left and right eye
    data are NaN (Not a Number). It then uses blink detection results
    to exclude missing data points occurring during blinks.

    Parameters:
    pupil_size_left (np.array): Array containing pupil size data for left eye.
    pupil_size_right (np.array): Array containing pupil size data for right eye.
    sampling_freq (int): Sampling frequency of the data in Hz.
    timestamps (np.array): Array containing timestamps for each data point.

    Returns:
    np.array: Boolean array indicating missing data indices (True)
              excluding those caused by blinks (False).
    """

    # Combine pupil size data and detect blinks (functions not included)
    missing_pupil_size = np.isnan(pupil_size_left) & np.isnan(pupil_size_right)
    blinks = blink_detection_utils.based_noise_blinks_detection(pupil_size_left, pupil_size_right, sampling_freq,
                                                                timestamps)
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
