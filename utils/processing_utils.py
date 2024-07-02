from utils import feature_extraction_utils, blink_detection_utils
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
    blinks = blink_detection_utils.both_pupils_blink_detection(pupil_size_left, pupil_size_right, 600, timestamps)

    # Percentage of total missing data
    left_pupil_missing_data = np.mean(np.isnan(pupil_size_left))
    right_pupil_missing_data = np.mean(np.isnan(pupil_size_right))

    # Identify missing data indices for both pupil size columns
    missing_data = feature_extraction_utils.missing_data_excluding_blinks_both_pupils(pupil_size_left, pupil_size_right,
                                                                                      blinks["blink_onset"],
                                                                                      blinks["blink_offset"],
                                                                                      timestamps)

    # Filter missing data indices to only include the relevant missing data (data where a blink can occur: over 100ms)
    missing_data_correct = feature_extraction_utils.missing_data_excluding_time_range(missing_data, 600)

    # Calculate the average blink rate considering missing data periods
    average_blink_rate = feature_extraction_utils.calculate_blink_rate(blinks, missing_data_correct, 600)

    # Print information about missing data percentage and average blink rate
    print(f"{csv_file} Percentage of Missing Data: {np.mean(missing_data_correct)}")
    print("Average Blink Rate for", csv_file, "(blinks per minute):", average_blink_rate)

    # Return results as a dictionary
    return {'subject': subject_id, 'day': day_number, 'blink_rate_mean': average_blink_rate,
            'percentage_missing_data': np.mean(missing_data_correct),
            'left_pupil_missing_data': left_pupil_missing_data,
            'right_pupil_missing_data': right_pupil_missing_data}


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
