import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


def diff(series):
    """
    Python implementation of matlab's diff function

    Computes the difference between consecutive elements in a series
    """
    return series[1:] - series[:-1]


def smooth(x, window_len):
    """
    Python implementation of matlab's smooth function

    Smoothes the input data using a simple moving average.
    """
    if window_len < 3:
        return x

    # Window length must be odd
    if window_len % 2 == 0:
        window_len += 1

    # Create a window of ones for averaging
    w = np.ones(window_len) / window_len

    # Perform convolution with the window
    y = np.convolve(w, x, mode='same')

    return y


def pupil_separated_blink_detection(pupil_size, sampling_freq, timestamps):
    """
    Function to find blinks and return blink onset and offset indices
    Adapted from: R. Hershman, A. Henik, and N. Cohen, “A novel blink detection method based on pupillometry noise,”
    Behav. Res. Methods, vol. 50, no. 1, pp. 107–114, 2018.

    Input:
        pupil_size: A numpy array or list containing pupil size data for one eye.
        sampling_freq: The sampling frequency of the eye-tracking hardware, given in Hz.
        timestamps: A numpy array or list containing the timestamps corresponding to the pupil size
    Output:
        blinks: [dictionary] {"blink_onset", "blink_offset"}
        containing numpy array/list of blink onset and offset indices
    """

    # sampling_interval represents the interval between samples in milliseconds. 1000/600=1.667ms
    sampling_interval = 1000 / sampling_freq

    # concat_gap_interval set to 50, representing the gap interval to concatenate close blinks or missing data periods.
    # This means that 20*1.667=83.33ms. Time gap between blink offset and new blink onset must be more than 83.33ms
    # to register as different blinks

    # initializes output
    blink_onset = []
    blink_offset = []
    blinks = {"blink_onset": blink_onset, "blink_offset": blink_offset}

    # convert input to numpy arrays
    processed_pupil_size = np.asarray(pupil_size)

    # missing_data is an array where each element is 1 if pupil_size is NaN at that index, and 0 otherwise.
    missing_data = np.array(np.isnan(processed_pupil_size), dtype="float32")

    # difference is the difference between consecutive elements in missing data,
    # highlighting transitions between 0 and 1
    difference = np.diff(missing_data)

    # blink_onset contains indices where difference is 1 (indicating a transition from non-missing to missing data)
    blink_onset = (np.where(difference == 1)[0])

    # blink_offset contains indices where difference is -1 (indicating a transition from missing to non-missing data,
    # plus one index to capture the full blink)
    blink_offset = (np.where(difference == -1)[0] + 1)

    length_blinks = len(blink_offset) + len(blink_onset)

    # Edge Case 1: No blinks
    if length_blinks == 0:
        return blinks

    # Edge Case 2: the data starts with a blink. In this case, blink onset will be defined as the first missing value.
    if (len(blink_onset) < len(blink_offset)) or ((len(blink_onset) == len(blink_offset)) and
                                                  (blink_onset[0] > blink_offset[0])):
        blink_onset = np.hstack((0, blink_onset))

    # Edge Case 3: the data ends with a blink. In this case, blink offset will be defined as the last missing sample
    if len(blink_offset) < len(blink_onset):
        blink_offset = np.hstack((blink_offset, len(processed_pupil_size) - 1))

    # Smoothing the data in order to increase the difference between the measurement noise and the eyelid signal.

    # This sets the smoothing window to 10 milliseconds. This is a 10ms window adapted from Hershman et al.
    ms_4_smoothing = 10

    # samples2smooth calculates the number of samples corresponding to the 10-millisecond window,
    # given the sampling frequency. (6 ms)
    samples2smooth = int(ms_4_smoothing / sampling_interval)

    # smooth_pupil_size applies the smooth function to the pupil size data using the
    # calculated window length (samples2smooth). The result is converted to a numpy array
    smooth_pupil_size = np.array(smooth(processed_pupil_size, samples2smooth), dtype='float32')

    # Compute the difference (smooth_pupil_size_diff) of the smoothed data.
    smooth_pupil_size_diff = np.diff(smooth_pupil_size)

    monotonically_dec = smooth_pupil_size_diff <= 0
    monotonically_inc = smooth_pupil_size_diff >= 0

    # Finding correct blink onsets and offsets using monotonically increasing and decreasing arrays
    for i in range(len(blink_onset)):
        if blink_onset[i] != 0:
            j = blink_onset[i] - 1
            if monotonically_inc[j]:
                blink_onset = np.delete(blink_onset, i)
                blink_offset = np.delete(blink_offset, i)
            while j > 0 and monotonically_dec[j]:
                j -= 1
            blink_onset[i] = j + 1

        # If data ends with blink we do not update it and let ending blink index be the last
        # index of the data
        if blink_offset[i] != len(processed_pupil_size) - 1:
            j = blink_offset[i]
            if monotonically_dec[j]:
                blink_onset = np.delete(blink_onset, i)
                blink_offset = np.delete(blink_offset, i)
            while j < len(monotonically_inc) and monotonically_inc[j]:
                j += 1

            blink_offset[i] = j

    # creating empty array the size of the blink onsets and offsets combined
    c = np.empty((len(blink_onset) + len(blink_offset),), dtype=blink_onset.dtype)

    # places blink_onset values at even indices (0, 2, 4, ...).
    c[0::2] = blink_onset

    # places blink_offset values at odd indices (1, 3, 5, ...).
    c[1::2] = blink_offset
    c = list(c)

    # Loop to remove events with durations not between 0.1s and 0.5s
    i = 0
    while i < len(c) - 1:
        duration = c[i + 1] - c[i]
        if duration < .1*sampling_freq or duration > .5*sampling_freq:
            c.pop(i + 1)
            c.pop(i)
        else:
            i += 2

    temp = np.reshape(c, (-1, 2), order='C')

    blinks["blink_onset"] = timestamps[temp[:, 0]]
    blinks["blink_offset"] = timestamps[temp[:, 1]]

    return blinks


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
    left_blinks = pupil_separated_blink_detection(pupil_size_left, 600, timestamps)
    right_blinks = pupil_separated_blink_detection(pupil_size_right, 600, timestamps)

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

    return {
        'subject': subject_id,
        'day': day_number,
        'left_blink_onsets': left_blinks["blink_onset"].values,
        'left_blink_offsets': left_blinks["blink_offset"].values,
        'right_blink_onsets': right_blinks["blink_onset"].values,
        'right_blink_offsets': right_blinks["blink_offset"].values,
        "left_average_blink_rate": left_average_blink_rate,
        'right_average_blink_rate': right_average_blink_rate,
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
    blinks = pupil_separated_blink_detection(pupil_size, sampling_freq, timestamps)
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
    left_blinks = pupil_separated_blink_detection(pupil_size_left, 600, timestamps)
    right_blinks = pupil_separated_blink_detection(pupil_size_right, 600, timestamps)

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

plot_pupil_size_v_time("output/sub-027_rseeg_block_1_d02_20230824_eyetracker.csv")
