import os
import blinkdetection as bd
import pandas as pd

# Path to the directory containing the CSV files
input_directory = 'output/interim'

# Get a list of all CSV files in the directory
csv_files = [f for f in os.listdir(input_directory) if f.endswith('.csv')]

# Define the sampling frequency
sampling_freq = 600  # Example sampling frequency in Hz

# Loop through each CSV file
for csv_file in sorted(csv_files):
    # Create the full file path
    file_path = os.path.join(input_directory, csv_file)

    # Read the CSV file
    df = pd.read_csv(file_path)

    # Calculate the total number of blinks
    blinks = bd.calculate_total_blinks_and_missing_data(df, sampling_freq, 2)
    length_blinks = len(blinks["blink_onset"])
    print(f"Total number of blinks in {csv_file}: {length_blinks}")

    # Plot the pupil size with blink onsets and offsets
    bd.plot_pupil_size_v_time(file_path, True)
