import os
import pyxdf
import numpy as np
import pandas as pd


def load_xdf_file(file_path):
    """
    Load XDF file using pyxdf.

    Parameters:
    file_path (str): Path to the XDF file.

    Returns:
    tuple: Data and header extracted from the XDF file.
    """
    return pyxdf.load_xdf(file_path)


def extract_streams(data):
    """
    Extract eyetracker and event streams from the loaded XDF data.

    Parameters:
    data (list): List containing streams from the XDF file.

    Returns:
    tuple containing the following:
     eyetracker_channel_data: streams of both the left and right eye x-position, y-position, and size.
     eyetracker_timestamps: stream of timestamps that correspond with the eyetracker_channel_data stream.
     event_data: list of event codes: [101, 102, 201, 202, 101, 102]:
        101: beginning of first resting state recording
        102: end of first resting state recording
        201: beginning of rest period
        202: end of rest period
        101: beginning of second resting state recording
        101: end of second resting state recording
     event_timestamps: stream of timestamps that correspond with the eyetracker_channel_data stream.
    """

    eyetracker_channel_data = None
    eyetracker_timestamps = None
    event_data = None
    event_timestamps = None

    # Iterate through each stream in the XDF data
    for stream in data:
        # Check if the stream name indicates eyetracker data
        if stream['info']['name'][0] == 'Tobii':
            eyetracker_channel_data = stream['time_series']
            eyetracker_timestamps = stream['time_stamps']
        # Check if the stream name indicates event data
        elif stream['info']['name'][0] == 'OWDM_Task_Events':
            event_data = stream['time_series']
            event_timestamps = stream['time_stamps']

    return eyetracker_channel_data, eyetracker_timestamps, event_data, event_timestamps


def get_resting_state_timestamps(event_timestamps, rs_recording=1):
    """
    Get start and end timestamps for a resting state.

    Parameters:
    event_timestamps (list): List of event timestamps.
    rs_recording (int 1 or 2): Which resting state recording are you looking to pull from? (default 1)

    Returns:
    tuple: Start and end timestamps for the second resting state.

    Note: Event codes are hard coded because they are the same across participants, however, we are not
    able to create a dictionary out of them, because the first resting state and the second resting state have the
    same event codes, so the second resting state would override the first.
    To combat this, decimals (.1, and .2) are added at the end of the 101 and 102 codes to specify what
    resting state measurement they represent
    """
    # Create a modified list of event data tuples with decimals to differentiate resting states
    event_data_modified = [(101.1,), (102.1,), (201.0,), (202.0,), (101.2,), (102.2,)]

    # Create a dictionary mapping modified event data to their corresponding timestamps
    event_dict = dict(zip(event_data_modified, event_timestamps))

    # Initialize output
    start_time = 0.0
    end_time = 0.0

    # Index to extract the timestamps correlating to the first resting state measurement
    if rs_recording == 1:
        start_time = event_dict[(101.1,)]
        end_time = event_dict[(102.1,)]

    if rs_recording == 2:
        start_time = event_dict[(101.2,)]
        end_time = event_dict[(102.2,)]

    return start_time, end_time


def filter_resting_state_data(channel_data, timestamps, start_time, end_time):
    """
    Filter out data from the second resting state.

    Parameters:
    channel_data (np.array): Array containing channel data.
    timestamps (np.array): Array containing timestamps.
    start_time (float): Start time of the second resting state.
    end_time (float): End time of the second resting state.

    Returns:
    tuple: Filtered pupil sizes and timestamps for the second resting state.
    """
    # Filter timestamps within the resting state window
    resting_timestamps = (timestamps >= start_time) & (timestamps <= end_time)

    # Extract pupil size data for left and right eye based on channel indices (assuming indices 2 and 5)
    pupil_size_left = np.array(channel_data[:, 2], dtype="float32")[resting_timestamps]
    pupil_size_right = np.array(channel_data[:, 5], dtype="float32")[resting_timestamps]

    # Filter timestamps corresponding to the extracted data
    timestamps = timestamps[resting_timestamps]

    return pupil_size_left, pupil_size_right, timestamps


def save_to_csv(timestamps, pupil_size_left, pupil_size_right, output_csv):
    """
    Save the filtered data to a CSV file.

    Parameters:
    timestamps (np.array): Array containing timestamps.
    pupil_size_left (np.array): Array containing left pupil sizes.
    pupil_size_right (np.array): Array containing right pupil sizes.
    output_csv (str): Output CSV file path.
    """
    # Creates dataframe with specified column names
    df = pd.DataFrame({
        'timestamps': timestamps,
        'pupil_size_left': pupil_size_left,
        'pupil_size_right': pupil_size_right
    })

    # Creates a csv file from the dataframe
    df.to_csv(output_csv, index=False)


def process_xdf_files(folder_path, output_folder):
    """
    Process all XDF files in the specified folder and save the filtered data to CSV files.

    Parameters:
    folder_path (str): Path to the folder containing XDF files.
    output_folder (str): Path to the output folder for CSV files.
    """
    os.makedirs(output_folder, exist_ok=True)

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.xdf'):
            file_name_full = os.path.join(folder_path, file_name)

            # Load the XDF file
            data, header = load_xdf_file(file_name_full)

            # Extract the relevant streams
            channel_data, timestamps, event_data, event_timestamps = extract_streams(data)

            # Get resting state start and end timestamps
            start_time, end_time = get_resting_state_timestamps(event_timestamps)

            # Filter the data for the resting state
            pupil_size_left, pupil_size_right, timestamps = filter_resting_state_data(channel_data, timestamps,
                                                                                      start_time, end_time)

            # Define the output CSV file name
            output_csv = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}_eyetracker.csv")

            # Save the filtered data to a CSV file
            save_to_csv(timestamps, pupil_size_left, pupil_size_right, output_csv)

            print(f"Processed file: {file_name} -> {output_csv}")
