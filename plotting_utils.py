import matplotlib.pyplot as plt
import os
import pandas as pd
import blink_detection_utils


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
    blinks = blink_detection_utils.based_noise_blinks_detection(pupil_size_left, pupil_size_right, 600, timestamps)

    # Create a figure for plotting
    plt.figure()

    # Plot pupil size vs time for left and right eye
    plt.plot(timestamps, pupil_size_left, label='Left Size')
    plt.plot(timestamps, pupil_size_right, label='Right Size')

    # Mark blink onsets and offsets with vertical lines
    for blink_onset in blinks["blink_onset"]:
        plt.axvline(blink_onset, color='green')
    for blink_offset in blinks["blink_offset"]:
        plt.axvline(blink_offset, color='pink')

    # Label axes and set title
    plt.xlabel('Time (s)')
    plt.ylabel('Size')
    subject_id = csv_file.split('_')[0].split("/")[-1]
    day_number = csv_file.split('_')[4].split('.')[0]
    plt.title(f'{subject_id}_{day_number} Pupil Size')
    plt.legend()

    # Create output directory if it doesn't exist
    output_dir = 'output/plots'
    os.makedirs(output_dir, exist_ok=True)

    # Generate filename and save the plot
    plot_filename = os.path.join(output_dir, f"{os.path.basename(csv_file)}_pupil_size_vs_time.png")
    plt.savefig(plot_filename, dpi=1000)


def plot_all_time_v_pupil_size_csv_files_in_directory(folder):
    """
    Plot pupil size data over time with blink annotations for all CSV files in the specified folder.

    Parameters:
    folder (str): Path to the folder containing CSV files.
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
        plot_pupil_size_v_time(file_path)

    plt.show()
