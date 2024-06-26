import pandas as pd
import numpy as np


def parse_blink_times(blink_times_str):
    # Remove unwanted characters and split by spaces
    blink_times_str = blink_times_str.strip('[]').replace('\n', ' ').replace('  ', ' ')
    # Split the string by spaces and convert to a list of floats
    return np.array([float(x) for x in blink_times_str.split()])


def find_common_blinks(left_onsets, left_offsets, right_onsets, right_offsets, tolerance=100):
    concat_onsets = []
    concat_offsets = []

    i, j = 0, 0
    while i < len(left_onsets) and j < len(right_onsets):
        if abs(left_onsets[i] - right_onsets[j]) <= tolerance:
            concat_onsets.append(max(left_onsets[i], right_onsets[j]))
            if i < len(left_offsets) and j < len(right_offsets):
                concat_offsets.append(max(left_offsets[i], right_offsets[j]))
            i += 1
            j += 1
        elif left_onsets[i] < right_onsets[j]:
            i += 1
        else:
            j += 1

    return concat_onsets, concat_offsets


def process_blinks(csv_file):
    df = pd.read_csv(csv_file)

    concat_onsets = []
    concat_offsets = []

    for index, row in df.iterrows():
        left_onsets = parse_blink_times(row['left_blink_onsets'])
        left_offsets = parse_blink_times(row['left_blink_offsets'])
        right_onsets = parse_blink_times(row['right_blink_onsets'])
        right_offsets = parse_blink_times(row['right_blink_offsets'])

        onsets, offsets = find_common_blinks(left_onsets, left_offsets, right_onsets, right_offsets)
        concat_onsets.append(onsets)
        concat_offsets.append(offsets)

    df['concat_blink_onsets'] = concat_onsets
    df['concat_blink_offsets'] = concat_offsets

    # Save the updated DataFrame back to a CSV file
    df.to_csv(csv_file, index=False)


file = 'output/features/compiled_left_right_blink_rates.csv'
process_blinks(file)
