from blinkdetection.featureextraction import feature_extraction_utils
from blinkdetection.blinkdetection import blink_detection_utils
import pandas as pd
import os
import numpy as np


def process_individual_feature_extraction_csv(csv_file, folder):
    """
    Process an individual CSV file to calculate various features within the blink and pupil size data.

    Parameters:
    csv_file (str): Name of the CSV file to process.
    folder (str): Path to the folder containing the CSV file.

    Returns:
    dict: A dictionary containing the following keys:
        - 'subject': Subject ID extracted from the filename.
        - 'day': Day number extracted from the filename.
        - 'left_ebr': Average blink rate for the left pupil.
        - 'right_ebr': Average blink rate for the right pupil.
        - 'concat_ebr': Average blink rate considering concatenated blinks.
        - 'left_ibi': Average inter-blink interval (IBI) for the left pupil.
        - 'right_ibi': Average inter-blink interval (IBI) for the right pupil.
        - 'concat_ibi': Average inter-blink interval (IBI) for concatenated blinks.
        - 'left_ibi_var': Variability of inter-blink interval for the left pupil.
        - 'right_ibi_var': Variability of inter-blink interval for the right pupil.
        - 'concat_ibi_var': Variability of inter-blink interval for concatenated blinks.
        - 'onset_diff': Mean difference between the time at which the left eye registered an onset vs the right eye.
        - 'offset_diff': Mean difference between the time at which the left eye registered an offset vs the right eye.
        - 'duration_diff': Mean difference between the durations of the blinks for the left eye vs the right eye.
        - 'left_bd': Average blink duration for the left pupil.
        - 'right_bd': Average blink duration for the right pupil.
        - 'left_bdv': Variability of blink duration for the left pupil.
        - 'right_bdv': Variability of blink duration for the right pupil.
        - 'left_pupil_size': Average pupil size for the left pupil excluding blink periods.
        - 'right_pupil_size': Average pupil size for the right pupil excluding blink periods.
        - 'left_pupil_size_var': Variability of pupil size for the left pupil excluding blink periods.
        - 'right_pupil_size_var': Variability of pupil size for the right pupil excluding blink periods.
        - 'left_missing': Percentage of missing data for the left pupil.
        - 'right_missing': Percentage of missing data for the right pupil.
        - 'left_missing_exb': Percentage of missing data for the left pupil excluding blink periods.
        - 'right_missing_exb': Percentage of missing data for the right pupil excluding blink periods.
        - 'left_missing_exb_ext': Percentage of missing data for the left pupil excluding blink periods and a minimum time range (100ms).
        - 'right_missing_exb_ext': Percentage of missing data for the right pupil excluding blink periods and a minimum time range (100ms).
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

    # Identify missing data indices for both pupil size columns
    left_missing_data_excluding_blinks = feature_extraction_utils.missing_data_excluding_blinks_single_pupil(
        pupil_size_left, left_blinks, timestamps)

    # Filter missing data indices to only include the relevant missing data (data where a blink can occur: over 100ms)
    left_missing_data_excluding_blinks_and_time_window = feature_extraction_utils.missing_data_excluding_time_range(
        left_missing_data_excluding_blinks, 600)

    # Calculate the average blink rate considering missing data periods
    left_average_blink_rate = feature_extraction_utils.calculate_blink_rate(
        left_blinks, left_missing_data_excluding_blinks_and_time_window, 600)

    # Identify missing data indices for both pupil size columns
    right_missing_data_excluding_blinks = feature_extraction_utils.missing_data_excluding_blinks_single_pupil(
        pupil_size_right, right_blinks, timestamps)

    # Filter missing data indices to only include the relevant missing data (data where a blink can occur: over 100ms)
    right_missing_data_excluding_blinks_and_time_window = feature_extraction_utils.missing_data_excluding_time_range(
        right_missing_data_excluding_blinks, 600)

    # Calculate the average blink rate considering missing data periods
    right_average_blink_rate = feature_extraction_utils.calculate_blink_rate(
        right_blinks, right_missing_data_excluding_blinks_and_time_window, 600)

    concat_missing_data_excluding_blinks = feature_extraction_utils.missing_data_excluding_blinks_both_pupils(
        pupil_size_left, pupil_size_right, blinks, timestamps)

    average_blink_rate = feature_extraction_utils.calculate_blink_rate(blinks,
                                                                       concat_missing_data_excluding_blinks, 600)

    # Calculate average blink duration and blink duration variability
    left_average_blink_duration = feature_extraction_utils.calculate_average_blink_duration(left_blinks)
    right_average_blink_duration = feature_extraction_utils.calculate_average_blink_duration(right_blinks)
    left_blink_duration_variability = feature_extraction_utils.calculate_blink_duration_variability(left_blinks)
    right_blink_duration_variability = feature_extraction_utils.calculate_blink_duration_variability(right_blinks)

    # Calculate inter-blink intervals
    left_ibi = feature_extraction_utils.calculate_inter_blink_interval(left_blinks)
    right_ibi = feature_extraction_utils.calculate_inter_blink_interval(right_blinks)
    concat_ibi = feature_extraction_utils.calculate_inter_blink_interval(blinks)

    left_avg_ibi = feature_extraction_utils.mean_inter_blink_interval(left_ibi)
    right_avg_ibi = feature_extraction_utils.mean_inter_blink_interval(right_ibi)
    concat_avg_ibi = feature_extraction_utils.mean_inter_blink_interval(concat_ibi)

    left_ibi_var = feature_extraction_utils.inter_blink_interval_variability(left_ibi)
    right_ibi_var = feature_extraction_utils.inter_blink_interval_variability(right_ibi)
    concat_ibi_var = feature_extraction_utils.inter_blink_interval_variability(concat_ibi)

    # Calculate sync of data
    onset_diff = feature_extraction_utils.onset_difference(left_blinks, right_blinks)
    offset_diff = feature_extraction_utils.offset_difference(left_blinks, right_blinks)
    duration_diff = feature_extraction_utils.duration_difference(left_blinks, right_blinks)

    # Calculate average pupil size
    left_avg_pupil_size = feature_extraction_utils.average_pupil_size_without_blinks(pupil_size_left, timestamps,
                                                                                     left_blinks)
    right_avg_pupil_size = feature_extraction_utils.average_pupil_size_without_blinks(pupil_size_right, timestamps,
                                                                                      right_blinks)

    # Calculate pupil size variability
    left_pupil_size_variability = feature_extraction_utils.pupil_size_variability(pupil_size_left, timestamps,
                                                                                  left_blinks)
    right_pupil_size_variability = feature_extraction_utils.pupil_size_variability(pupil_size_right, timestamps,
                                                                                   right_blinks)

    return {
        'subject': subject_id,
        'day': day_number,
        'left_ebr': left_average_blink_rate,
        'right_ebr': right_average_blink_rate,
        'concat_ebr': average_blink_rate,
        'left_ibi': left_avg_ibi,
        'right_ibi': right_avg_ibi,
        'concat_ibi': concat_avg_ibi,
        'left_ibi_var': left_ibi_var,
        'right_ibi_var': right_ibi_var,
        'concat_ibi_var': concat_ibi_var,
        'onset_diff': onset_diff,
        'offset_diff' : offset_diff,
        'duration_diff': duration_diff,
        'left_bd': left_average_blink_duration,
        'right_bd': right_average_blink_duration,
        'left_bdv': left_blink_duration_variability,
        'right_bdv': right_blink_duration_variability,
        'left_pupil_size': left_avg_pupil_size,
        'right_pupil_size': right_avg_pupil_size,
        'left_pupil_size_var': left_pupil_size_variability,
        'right_pupil_size_var': right_pupil_size_variability,
        'left_missing': np.mean(np.isnan(pupil_size_left)),
        'right_missing': np.mean(np.isnan(pupil_size_right)),
        'left_missing_exb': np.mean(left_missing_data_excluding_blinks),
        'right_missing_exb': np.mean(right_missing_data_excluding_blinks),
        'left_missing_exb_ext': np.mean(
            left_missing_data_excluding_blinks_and_time_window),
        'right_missing_exb_ext': np.mean(
            right_missing_data_excluding_blinks_and_time_window)
    }


def process_individual_blink_csv(csv_file, folder):
    """
    Process an individual CSV file to calculate the blink onsets and offsets for the left and right eye and list the
    ones that are consistent among the two.

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

    concat_onsets = np.array(blinks["blink_onset"])
    concat_offsets = np.array(blinks["blink_offset"])

    return {
        'subject': subject_id,
        'day': day_number,
        'left_blink_onsets': left_blinks["blink_onset"].values,
        'left_blink_offsets': left_blinks["blink_offset"].values,
        'right_blink_onsets': right_blinks["blink_onset"].values,
        'right_blink_offsets': right_blinks["blink_offset"].values,
        'concat_onsets': concat_onsets,
        'concat_offsets': concat_offsets,
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
    results_feature_extraction = []
    results_blinks = []
    for csv_file in sorted(os.listdir(folder)):
        if csv_file.endswith('.csv'):
            # Process each individual CSV file's features
            result = process_individual_feature_extraction_csv(csv_file, folder)
            results_feature_extraction.append(result)

            # Process each individual CSV file's blinks
            result_blink = process_individual_blink_csv(csv_file, folder)
            results_blinks.append(result_blink)

    # Create a DataFrame from the results
    results_df = pd.DataFrame(results_feature_extraction)
    results_blinks_df = pd.DataFrame(results_blinks)

    # Save the DataFrame to a CSV file
    results_df.to_csv('output/features/compiled_feature_extraction.csv', index=False)
    results_blinks_df.to_csv('output/features/compiled_blink_extraction.csv', index=False)

    return results_df, results_blinks_df
