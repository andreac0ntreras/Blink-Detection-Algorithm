"""
THIS USES CODE FROM RESEACRH PAPER HERSHMAN TO DETECT BLINKS
This adaptation to Python was made with the supervision and encouragement of Upamanyu Ghose
For more information about this adaptation and for more Python solutions, don't hesitate to contact him:
Email: titoghose@gmail.com
Github code repository: github.com/titoghose
"""
import numpy as np


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


def preprocess_nan_periods(pupil_size_left, pupil_size_right, sampling_freq):
    """
    Preprocesses pupil size data by converting isolated non-NaN values to NaNs

    Input:
        pupil_size_left: A numpy array containing pupil size data for the left eye.
        pupil_size_right: A numpy array containing pupil size data for the right eye.
        sampling_freq: The sampling frequency of the eye-tracking hardware, given in Hz.

    Output:
        pupil_size_left_processed: Processed pupil size data for the left eye.
        pupil_size_right_processed: Processed pupil size data for the right eye.
    """
    # Convert input to numpy arrays
    pupil_size_left = np.asarray(pupil_size_left)
    pupil_size_right = np.asarray(pupil_size_right)

    # Copy pupil size data
    pupil_size_left_processed = pupil_size_left.copy()
    pupil_size_right_processed = pupil_size_right.copy()

    # Calculate number of samples corresponding to 50ms see
    samples_to_nan = int(0.035 * sampling_freq)

    # Flag indices where both left and right pupil sizes are NaN
    missing_data = np.isnan(pupil_size_left) & np.isnan(pupil_size_right)

    # Find transitions in missing data
    diff_missing_data = diff(missing_data.astype(int))

    if missing_data[0]:
        diff_missing_data[0] = 1

    # Indices where missing data transitions from 0 to 1 (NaN appears)
    nan_onset = np.array(np.where(diff_missing_data == 1)[0])

    # Indices where missing data transitions from 1 to 0 (NaN ends)
    nan_offset = np.array(np.where(diff_missing_data == -1)[0] + 1)

    # Iterate through NaN periods and convert isolated non-NaN values to NaNs
    for onset1, onset2, offset1, offset2 in zip(nan_onset, nan_onset[1:], nan_offset, nan_offset[1:]):
        if onset2 - offset1 <= 4:
            if (onset2 - onset1 >= samples_to_nan) and (offset2 - offset1 >= samples_to_nan):
                pupil_size_left_processed[onset1:offset2] = np.nan
                pupil_size_right_processed[onset1:offset2] = np.nan

    return pupil_size_left_processed, pupil_size_right_processed


def based_noise_blinks_detection(pupil_size_left, pupil_size_right, sampling_freq, timestamps):
    """
    Function to find blinks and return blink onset and offset indices
    Adapted from: R. Hershman, A. Henik, and N. Cohen, “A novel blink detection method based on pupillometry noise,”
    Behav. Res. Methods, vol. 50, no. 1, pp. 107–114, 2018.

    Input:
        pupil_size_left: A numpy array or list containing pupil size data for the left eye.
        pupil_size_right: A numpy array or list containing pupil size data for the right eye.
        sampling_freq: The sampling frequency of the eye-tracking hardware, given in Hz.
        timestamps: A numpy array or list containing the timestamps corresponding to the pupil size
    Output:
        blinks: [dictionary] {"blink_onset", "blink_offset"}
        containing numpy array/list of blink onset and offset indices
    """

    processed_pupil_size_left, processed_pupil_size_right = preprocess_nan_periods(pupil_size_left, pupil_size_right,
                                                                                   sampling_freq)

    # sampling_interval represents the interval between samples in milliseconds. 1000/600=1.667ms
    sampling_interval = 1000 / sampling_freq

    # concat_gap_interval set to 50, representing the gap interval to concatenate close blinks or missing data periods.
    # This means that 20*1.667=83.33ms. Time gap between blink offset and new blink onset must be more than 83.33ms
    # to register as different blinks
    concat_gap_interval = 0

    # initializes output
    blink_onset = []
    blink_offset = []
    blinks = {"blink_onset": blink_onset, "blink_offset": blink_offset}

    # convert input to numpy arrays
    processed_pupil_size_left = np.asarray(processed_pupil_size_left)
    processed_pupil_size_right = np.asarray(processed_pupil_size_right)

    # missing_data is an array where each element is 1 if both pupil_size_left and pupil_size_right are NaNs
    # at that index, and 0 otherwise.
    missing_data = np.array(np.isnan(processed_pupil_size_right) & np.isnan(processed_pupil_size_left), dtype="float32")

    # difference is the difference between consecutive elements in missing data,
    # highlighting transitions between 0 and 1
    difference = diff(missing_data)

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
    """
        Two possible situations may cause this:
            i.  starts with a blink but does not end with a blink ---> len(blink_onset) < len(blink_offset)
            ii. starts with a blink and ends with a blink---> len(blink_onset) == len(blink_offset) 
            and (blink_onset[0] == blink_offset[0])
    """
    if (len(blink_onset) < len(blink_offset)) or ((len(blink_onset) == len(blink_offset)) and
                                                  (blink_onset[0] > blink_offset[0])):
        blink_onset = np.hstack((0, blink_onset))

    # Edge Case 3: the data ends with a blink. In this case, blink offset will be defined as the last missing sample
    """
        Two possible situations may cause this:
            i.  ends with a blink but does not start with a blink ---> len(blink_offset) < len(blink_onset)
            ii. ends with a blink and starts with a blink---> Already handled "start with blink" in Edge case 2 
            so it reduces to i (previous case)
    """
    if len(blink_offset) < len(blink_onset):
        blink_offset = np.hstack((blink_offset, len(processed_pupil_size_right) - 1))

    # Smoothing the data in order to increase the difference between the measurement noise and the eyelid signal.

    # This sets the smoothing window to 10 milliseconds.
    # Meaning, we smooth the data over 10-millisecond intervals to reduce noise.
    ms_4_smoothing = 20

    # samples2smooth calculates the number of samples corresponding to the 10-millisecond window,
    # given the sampling frequency.
    samples2smooth = ms_4_smoothing / sampling_interval

    samples2smooth = int(samples2smooth)

    # smooth_pupil_size applies the smooth function to the right pupil size data using the
    # calculated window length (samples2smooth). The result is converted to a numpy array
    smooth_right_pupil_size = np.array(smooth(processed_pupil_size_right, samples2smooth), dtype='float32')

    # smooth_pupil_size applies the smooth function to the left pupil size data using the
    # calculated window length (samples2smooth). The result is converted to a numpy array
    smooth_left_pupil_size = np.array(smooth(processed_pupil_size_left, samples2smooth), dtype='float32')

    # Combine smoothed left and right pupil size data
    smooth_pupil_size = (smooth_left_pupil_size + smooth_right_pupil_size) / 2

    # What if blink on one and missing on other????
    # See how timestamps line up

    # Compute the difference (smooth_pupil_size_diff) of the smoothed data.
    smooth_pupil_size_diff = diff(smooth_pupil_size)
    """
    Finding values <=0 and >=0 in order to find monotonically increasing and decreasing sections 
    of smoothened pupil data

            Eg. a =     [2, 1, 2, 8, 7, 6, 5, 4, 4, 0, 0, 0, 0, 0, 3, 3, 3, 8, 9, 10, 2, 3, 10]
                                  ----------------  S           E  =================
            diff(a)=   [-1  1  6 -1 -1 -1 -1  0 -4  0  0  0  0  3  0  0  5  1  1  -8  1  7]

    monotonically_dec = [T  F  F  T  T  T  T  T  T  T  T  T  T  F  T  T  F  F  F   T  F  F]   (T=True, F=False)
    monotonically_dec = [F  T  T  F  F  F  F  T  F  T  T  T  T  T  T  T  T  T  T   F  T  T]

    ---> The monotonically decreasing sequence before the blink is underlined with -- and the monotonically increasing 
    sequence after the blink with ==
    ---> S : denotes the initially detected onset of blink
    ---> E : denotes the initially detected offset of blink

    >> Looking at diff(a), all values in the montonically decreasing sequence should be <= 0 and those included in the 
    monotonically increasing sequence >= 0
    >> Hence, by moving left from the initially detected onset while T(True) values are encountered in monotonically_dec 
    we can update the onset to the start of monotonically_dec seq
    >> By moving right from the initially detected offset while T(True) values are encountered in monotonically_inc we 
    can update the offset to the end of monotonically_inc seq + 1
    """

    # smooth_pupil_size_diff contains the differences between consecutive elements in the smoothed pupil size data.
    # These two lines create boolean arrays indicating where the smoothed data is monotonically decreasing
    # (monotonically_dec) or monotonically increasing (monotonically_inc).
    # This helps identify sections where the pupil size is consistently decreasing or increasing,
    # which is useful for accurately determining blink onset and offset points.

    monotonically_dec = smooth_pupil_size_diff <= 0
    monotonically_inc = smooth_pupil_size_diff >= 0

    # Finding correct blink onsets and offsets using monotonically increasing and decreasing arrays
    for i in range(len(blink_onset)):
        # Edge Case 2: If data starts with blink we do not update it and let starting blink index be 0
        if blink_onset[i] != 0:
            j = blink_onset[i] - 1  # j:36125
            # blink onsets rn are places where the data is there but in the next
            # index it is nan

            # if monotonically_dec at that index is True, that means the diff is negative
            # at that index, meaning the value after that index is less.
            # This assigns j to the index before the first blink "onset", which right now
            # is just the last point of data before a group of nans and sees if everything before that is
            # decreasing and continuously makes the one before it the blink onset until it stops decreasing
            while j > 0 and monotonically_dec[j]:
                j -= 1
            blink_onset[i] = j + 1

        # Edge Case 3: If data ends with blink we do not update it and let ending blink index be the last
        # index of the data
        if blink_offset[i] != len(processed_pupil_size_right) - 1:
            j = blink_offset[i]
            while j < len(monotonically_inc) and monotonically_inc[j]:
                j += 1
            blink_offset[i] = j

    # Removing duplications (in case of consecutive sets): [a, b, b, c] => [a, c] or if inter
    # blink interval is less than concat_gap_interval
    c = np.empty((len(blink_onset) + len(blink_offset),), dtype=blink_onset.dtype)

    # places blink_onset values at even indices (0, 2, 4, ...).
    c[0::2] = blink_onset

    # places blink_offset values at odd indices (1, 3, 5, ...).
    c[1::2] = blink_offset
    c = list(c)

    # Checks if the gap between consecutive offset and onset events (c[i + 1] - c[i]) is less than
    # or equal to concat_gap_interval. If the gap is small enough (indicating a duplicate or near-duplicate blink),
    # the events are removed from the list (c[i:i + 2] = []).
    i = 1
    while i < len(c) - 1:
        if c[i + 1] - c[i] <= concat_gap_interval:
            c[i:i + 2] = []
        else:
            i += 2

    # Minimum amplitude check
    '''
    min_amplitude is set to a value that represents the minimum significant change in pupil size needed to 
    consider an event as a blink. '''
    min_amplitude = 0.2
    i = 0
    while i < len(c) - 1:
        blink_start = c[i]
        blink_end = c[i + 1]
        amplitude_change = np.abs(smooth_pupil_size[blink_end] - smooth_pupil_size[blink_start])
        if amplitude_change < min_amplitude:
            c.pop(i + 1)
            c.pop(i)
        else:
            i += 2

    # Loop to remove events with durations not between 0.1s and 0.5s
    i = 0
    while i < len(c) - 1:
        duration = c[i + 1] - c[i]
        if duration < 60 or duration > 420:
            # if the duration is less than 100ms (60 recordings) or more than 500ms +200ms (300+ 120 for buffer)
            # c[i:i + 2] = [] this just removes the associated blink offset and onset and combines the outer two blinks
            # Remove the items at index i and i+1
            c.pop(i + 1)
            c.pop(i)
        else:
            i += 2

    temp = np.reshape(c, (-1, 2), order='C')

    """
    NOTE:edit the lines below to only temp[:, 0] and temp[:, 1] in case you are interested in the indices of 
    blinks and not realtime values
    """

    blinks["blink_onset"] = timestamps[temp[:, 0]]
    blinks["blink_offset"] = timestamps[temp[:, 1]]

    return blinks
