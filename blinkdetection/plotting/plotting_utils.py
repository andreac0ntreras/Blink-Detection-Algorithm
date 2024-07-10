import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
from blinkdetection.blinkdetection import blink_detection_utils


def plot_pupil_size_v_time_w_concat_onsets_and_offsets(csv_file, show=False):
    """
    This function reads a CSV file containing pupil size data, detects blinks,
    and plots pupil size (left and right eye) vs time.

    Args:
        csv_file (str): Path to the CSV file containing pupil size data.
        show (bool): indication of whether to display the plot
    """

    # Read the CSV file using pandas
    dataframe = pd.read_csv(csv_file)

    # Extract columns by header name
    timestamps = dataframe['timestamps']
    pupil_size_left = dataframe['pupil_size_left']
    pupil_size_right = dataframe['pupil_size_right']

    # Detect blinks using a separate function (not included here)
    blinks = blink_detection_utils.both_pupils_blink_detection(pupil_size_left, pupil_size_right, 600, timestamps)

    # Plot pupil size vs time for left and right eye
    plt.plot(timestamps, pupil_size_left, label='Left Size')
    plt.plot(timestamps, pupil_size_right, label='Right Size', color='darkorange')

    # Mark blink onsets and offsets with vertical lines
    for blink_onset in blinks["blink_onset"]:
        plt.axvline(blink_onset, color='green')
    for blink_offset in blinks["blink_offset"]:
        plt.axvline(blink_offset, color='pink')

    # Label axes and set title
    plt.xlabel('Time (s)')
    plt.ylabel('Size (AU)')
    subject_id = csv_file.split('_')[0].split("/")[-1]
    day_number = csv_file.split('_')[4].split('.')[0]
    plt.title(f'{subject_id}_{day_number} Pupil Size')
    plt.legend()

    # Create output directory if it doesn't exist
    output_dir = '../../output/plots'
    os.makedirs(output_dir, exist_ok=True)

    # Generate filename and save the plot
    plot_filename = os.path.join(output_dir, f"{os.path.basename(csv_file)}_pupil_size_vs_time.png")
    plt.savefig(plot_filename, dpi=1000)

    if show is True:
        plt.show()

    plt.clf()


def plot_pupil_size_v_time_w_both_eyes_blink_onset_and_offsets(csv_file, show=False):
    """
    This function reads a CSV file containing pupil size data, detects blinks,
    and plots pupil size (left and right eye) vs time.

    Args:
        csv_file (str): Path to the CSV file containing pupil size data.
        show (bool): indication of whether to display the plot
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
    plt.ylabel('Size (AU)')
    subject_id = csv_file.split('_')[0].split("/")[-1]
    day_number = csv_file.split('_')[4].split('.')[0]
    plt.title(f'{subject_id}_{day_number} Pupil Size')
    plt.legend()

    if show is True:
        plt.show()

    # Create output directory if it doesn't exist
    output_dir = '../../output/plots'
    os.makedirs(output_dir, exist_ok=True)

    # Generate filename and save the plot
    plot_filename = os.path.join(output_dir, f"{os.path.basename(csv_file)}_pupil_size_vs_time.png")
    plt.savefig(plot_filename, dpi=1000)
    plt.clf()


def plot_all_time_v_pupil_size_csv_files_in_directory(folder, show=False):
    """
    Plot pupil size data over time with blink annotations for all CSV files in the specified folder.

    Parameters:
    folder (str): Path to the folder containing CSV files.
    show (bool): indication of whether to display the plot.
    """
    # Get a list of all files in the specified folder
    files = os.listdir(folder)

    # Filter the list to include only CSV files
    csv_files = [f for f in files if f.endswith('.csv')]

    # Loop through each CSV file and plot the pupil size data
    for csv_file in csv_files:
        # Construct the full file path
        file_path = os.path.join(folder, csv_file)

        # Plot the pupil size data for the current CSV file
        plot_pupil_size_v_time_w_both_eyes_blink_onset_and_offsets(file_path)

    if show is True:
        plt.show()

    plt.clf()


def plot_feature_over_three_days(compiled_df, show=False):
    """
    Plot blink rate and missing data percentage across three days for each subject.

    Args:
        compiled_df (pd.DataFrame): The return of process_individual_csv, which is a DataFrame containing columns
                                    corresponding to different features.
        show (bool): indication of whether to display the plot

    The function performs the following steps:
        1. Transforms the blink rate data to a wide format where each day is a separate column.
        2. Transforms the missing data percentage to a wide format where each day is a separate column.
        3. Melts the wide-format blink rate data back to a long format suitable for plotting and ANOVA.
        4. Melts the wide-format missing data percentage back to a long format suitable for plotting and ANOVA.
        5. Creates a line plot for blink rate across different days for each subject.
        6. Optionally displays the blink rate plot if `show` is True.
        7. Creates a line plot for missing data percentage across different days for each subject.
        8. Optionally displays the missing data plot if `show` is True.
    """

    # Transform the data to long format if needed
    blink_rate_data_long = compiled_df.pivot(index='subject', columns='day',
                                             values='concat_average_blink_rate').reset_index()
    blink_rate_data_long.columns = ['subject', 'day1', 'day2', 'day3']

    # Transform the data to long format if needed
    missing_data_long = (compiled_df.pivot(index='subject', columns='day',
                                           values='left_missing_data_percentage_excluding_blinks_and_min_time_range').
                         reset_index())
    missing_data_long.columns = ['subject', 'day1', 'day2', 'day3']

    # Melt the dataframe back to long format for ANOVA
    blink_rate_data_melted = pd.melt(blink_rate_data_long, id_vars=['subject'], value_vars=['day1', 'day2', 'day3'],
                                     var_name='day', value_name='concat_average_blink_rate')

    missing_data_melted = pd.melt(missing_data_long, id_vars=['subject'], value_vars=['day1', 'day2', 'day3'],
                                  var_name='day',
                                  value_name='left_missing_data_percentage_excluding_blinks_and_min_time_range')

    # Line plot
    sns.lineplot(x='day', y='concat_average_blink_rate', data=blink_rate_data_melted, hue='subject', marker='o')
    plt.title('Blink Rate Across Different Days for Each Subject')
    plt.xlabel('Day')
    plt.ylabel('Average Blink Rate')

    if show is True:
        plt.show()

    sns.lineplot(x='day', y='left_missing_data_percentage_excluding_blinks_and_min_time_range',
                 data=missing_data_melted,
                 hue='subject', marker='o')
    plt.title('Missing Data Across Different Days for Each Subject')
    plt.xlabel('Day')
    plt.ylabel('Percentage of Missing Data')

    if show is True:
        plt.show()
    plt.clf()