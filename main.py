from utils import processing_utils, plotting_utils, sourcing_utils

# Define folder paths
folder_path = 'data/raw/XDF Files'
output_folder = 'output'

# Process the XDF files
# sourcing_utils.process_xdf_files(folder_path, output_folder)

# # Create a new CSV file with all participants, days, and blink rates
# processing_utils.process_csv_files(output_folder)
#
# # Plot time vs pupil size for all participants and save to output/plots
# plotting_utils.plot_all_time_v_pupil_size_csv_files_in_directory(output_folder)

plotting_utils.plot_pupil_size_v_time_w_concat_onsets_and_offsets("output/sub-025_rseeg_block_1_d01_20230718_eyetracker.csv")
