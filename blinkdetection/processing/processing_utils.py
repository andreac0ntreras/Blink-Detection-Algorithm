from blinkdetection.featureextraction import feature_extraction_utils
from blinkdetection.blinkdetection import blink_detection_utils
import pandas as pd
import os
import numpy as np


def process_individual_csv(csv_file, folder):
    """
    Process an individual CSV file to calculate the average blink rate.

    Parameters:
    csv_file (str): Name of the CSV file to process.
    folder (str): Path to the folder containing the CSV file.

    dict: A dictionary containing the following keys:
        - 'subject': Subject ID extracted from the filename.
        - 'day': Day number extracted from the filename.
        - 'left_blink_onsets': Array of blink onset times for the left pupil.
        - 'left_blink_offsets': Array of blink offset times for the left pupil.
        - 'right_blink_onsets': Array of blink onset times for the right pupil.
        - 'right_blink_offsets': Array of blink offset times for the right pupil.
        - 'concatenated_onsets': Array of concatenated blink onset times.
        - 'concatenated_offsets': Array of concatenated blink offset times.
        - 'left_average_blink_rate': Average blink rate for the left pupil.
        - 'right_average_blink_rate': Average blink rate for the right pupil.
        - 'concat_average_blink_rate': Average blink rate considering concatenated blinks.
        - 'left_average_blink_duration': Average blink duration for the left pupil.
        - 'right_average_blink_duration': Average blink duration for the right pupil.
        - 'left_blink_duration_variability': Variability of blink duration for the left pupil.
        - 'right_blink_duration_variability': Variability of blink duration for the right pupil.
        - 'left_average_inter_blink_interval': Average inter-blink interval (IBI) for the left pupil.
        - 'right_average_inter_blink_interval': Average inter-blink interval (IBI) for the right pupil.
        - 'concat_average_inter_blink_interval': Average inter-blink interval (IBI) for concatenated blinks.
        - 'left_average_pupil_size': Average pupil size for the left pupil excluding blink periods.
        - 'right_average_pupil_size': Average pupil size for the right pupil excluding blink periods.
        - 'left_missing_data_percentage': Percentage of missing data for the left pupil.
        - 'right_missing_data_percentage': Percentage of missing data for the right pupil.
        - 'left_missing_data_percentage_excluding_blinks': Percentage of missing data for the left pupil excluding
        blink periods.
        - 'right_missing_data_percentage_excluding_blinks': Percentage of missing data for the right pupil excluding
        blink periods.
        - 'left_missing_data_percentage_excluding_blinks_and_min_time_range': Percentage of missing data for the left
        pupil excluding blink periods and a minimum time range (100ms).
        - 'right_missing_data_percentage_excluding_blinks_and_min_time_range': Percentage of missing data for the right
        pupil excluding blink periods and a minimum time range (100ms).
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

    # Detect left and right blink onsets and offsets using a separate function
    left_blinks = blink_detection_utils.single_pupil_blink_detection(pupil_size_left, 600, timestamps)
    right_blinks = blink_detection_utils.single_pupil_blink_detection(pupil_size_right, 600, timestamps)

    # Calculate the concatenated onsets and offsets based on how closely the onsets and offsets of the left and right
    # pupil match
    blinks = blink_detection_utils.identify_concat_blinks(left_blinks, right_blinks)

    concat_onsets = blinks["blink_onset"]
    concat_offsets = blinks["blink_offset"]

    # Identify missing data indices for both pupil size columns
    left_missing_data_excluding_blinks = feature_extraction_utils.missing_data_excluding_blinks_single_pupil(
        pupil_size_left,
        left_blinks["blink_onset"],
        left_blinks["blink_offset"],
        timestamps)

    # Filter missing data indices to only include the relevant missing data (data where a blink can occur: over 100ms)
    left_missing_data_excluding_blinks_and_time_window = feature_extraction_utils.missing_data_excluding_time_range(
        left_missing_data_excluding_blinks, 600)

    # Calculate the average blink rate considering missing data periods
    left_average_blink_rate = feature_extraction_utils.calculate_blink_rate(
        left_blinks, left_missing_data_excluding_blinks_and_time_window, 600)

    # Identify missing data indices for both pupil size columns
    right_missing_data_excluding_blinks = feature_extraction_utils.missing_data_excluding_blinks_single_pupil(
        pupil_size_right, right_blinks["blink_onset"], right_blinks["blink_offset"], timestamps)

    # Filter missing data indices to only include the relevant missing data (data where a blink can occur: over 100ms)
    right_missing_data_excluding_blinks_and_time_window = feature_extraction_utils.missing_data_excluding_time_range(
        right_missing_data_excluding_blinks, 600)

    # Calculate the average blink rate considering missing data periods
    right_average_blink_rate = feature_extraction_utils.calculate_blink_rate(
        right_blinks, right_missing_data_excluding_blinks_and_time_window, 600)

    concat_missing_data_excluding_blinks = feature_extraction_utils.missing_data_excluding_blinks_both_pupils(
        pupil_size_left, pupil_size_right, concat_onsets, concat_offsets, timestamps)

    average_blink_rate = feature_extraction_utils.calculate_concat_blink_rate(concat_onsets,
                                                                              concat_missing_data_excluding_blinks, 600)

    # Calculate average blink duration and blink duration variability
    left_average_blink_duration = feature_extraction_utils.calculate_average_blink_duration(left_blinks)
    right_average_blink_duration = feature_extraction_utils.calculate_average_blink_duration(right_blinks)
    left_blink_duration_variability = feature_extraction_utils.calculate_blink_duration_variability(left_blinks)
    right_blink_duration_variability = feature_extraction_utils.calculate_blink_duration_variability(right_blinks)

    # Calculate inter-blink intervals
    left_ibi = feature_extraction_utils.calculate_inter_blink_interval(left_blinks["blink_onset"].values)
    right_ibi = feature_extraction_utils.calculate_inter_blink_interval(right_blinks["blink_onset"].values)
    concat_ibi = feature_extraction_utils.calculate_inter_blink_interval(concat_onsets)

    left_avg_ibi = feature_extraction_utils.mean_inter_blink_interval(left_ibi)
    right_avg_ibi = feature_extraction_utils.mean_inter_blink_interval(right_ibi)
    concat_avg_ibi = feature_extraction_utils.mean_inter_blink_interval(concat_ibi)

    # Calculate average pupil size
    left_avg_pupil_size = feature_extraction_utils.average_pupil_size_without_blinks(pupil_size_left,
                                                                                     left_blinks["blink_onset"],
                                                                                     left_blinks["blink_offset"])
    right_avg_pupil_size = feature_extraction_utils.average_pupil_size_without_blinks(pupil_size_right,
                                                                                      right_blinks["blink_onset"],
                                                                                      right_blinks["blink_offset"])

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
        'concat_average_blink_rate': average_blink_rate,
        'left_average_blink_duration': left_average_blink_duration,
        'right_average_blink_duration': right_average_blink_duration,
        'left_blink_duration_variability': left_blink_duration_variability,
        'right_blink_duration_variability': right_blink_duration_variability,
        'left_average_inter_blink_interval': left_avg_ibi,
        'right_average_inter_blink_interval': right_avg_ibi,
        'concat_average_inter_blink_interval': concat_avg_ibi,
        'left_average_pupil_size': left_avg_pupil_size,
        'right_average_pupil_size': right_avg_pupil_size,
        'left_missing_data_percentage': np.mean(np.isnan(pupil_size_left)),
        'right_missing_data_percentage': np.mean(np.isnan(pupil_size_right)),
        'left_missing_data_percentage_excluding_blinks': np.mean(left_missing_data_excluding_blinks),
        'right_missing_data_percentage_excluding_blinks': np.mean(right_missing_data_excluding_blinks),
        'left_missing_data_percentage_excluding_blinks_and_min_time_range': np.mean(
            left_missing_data_excluding_blinks_and_time_window),
        'right_missing_data_percentage_excluding_blinks_and_min_time_range': np.mean(
            right_missing_data_excluding_blinks_and_time_window)
    }


def process_csv_files(folder):
    """
    Process all CSV files in the specified folder and save blink rate and other feature results to a CSV file.

    Parameters:
    folder (str): Path to the folder containing CSV files.

    Returns:
    results_df (pd.DataFrame): Dataframe containing blink related variables for each subject in the folder
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
    results_df.to_csv('output/features/compiled_feature_extraction.csv', index=False)

    return results_df
