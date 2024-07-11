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


def both_pupils_blink_detection(pupil_size_left, pupil_size_right, sampling_freq, timestamps):
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
        containing numpy array/list of blink onset and offset timestamps
    """

    # sampling_interval represents the interval between samples in milliseconds. 1000/600=1.667ms
    sampling_interval = 1000 / sampling_freq

    # initializes output
    blink_onset = []
    blink_offset = []
    blinks = {"blink_onset": blink_onset, "blink_offset": blink_offset}

    # convert input to numpy arrays
    processed_pupil_size_left = np.asarray(pupil_size_left)
    processed_pupil_size_right = np.asarray(pupil_size_right)

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
    ms_4_smoothing = 10

    # samples2smooth calculates the number of samples corresponding to the 10-millisecond window,
    # given the sampling frequency.
    samples2smooth = int(ms_4_smoothing / sampling_interval)

    # smooth_pupil_size applies the smooth function to the right pupil size data using the
    # calculated window length (samples2smooth). The result is converted to a numpy array
    smooth_right_pupil_size = np.array(smooth(processed_pupil_size_right, samples2smooth), dtype='float32')

    # smooth_pupil_size applies the smooth function to the left pupil size data using the
    # calculated window length (samples2smooth). The result is converted to a numpy array
    smooth_left_pupil_size = np.array(smooth(processed_pupil_size_left, samples2smooth), dtype='float32')

    # Combine smoothed left and right pupil size data
    """
        This is where the blink detection algorithms are different. Smooth pupil sizes are combined 
        (by taking the average) to eventually see if both the left and right pupil size follow the same 
        trend. (Characteristic downward and upward slope of a blink).
        This is where this algorithm needs help. Is this the correct way to do it?
    """
    smooth_pupil_size = (smooth_left_pupil_size + smooth_right_pupil_size) / 2

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
    i = 0
    while i < len(blink_onset):
        # Edge Case 2: If data starts with blink we do not update it and let starting blink index be 0
        if blink_onset[i] != 0:
            b_o = blink_onset[i] - 1
            # blink onsets rn are places where the data is there but in the next
            # index it is nan

            # if monotonically_dec at that index is True, that means the diff is negative
            # at that index, meaning the value after that index is less.
            # This assigns j to the index before the first blink "onset", which right now
            # is just the last point of data before a group of nans and sees if everything before that is
            # decreasing and continuously makes the one before it the blink onset until it stops decreasing
            while b_o > 0 and monotonically_dec[b_o]:
                b_o -= 1
            blink_onset[i] = b_o + 1

        # Edge Case 3: If data ends with blink we do not update it and let ending blink index be the last
        # index of the data
        if blink_offset[i] != len(processed_pupil_size_right) - 1:
            b_off = blink_offset[i]
            while b_off < len(monotonically_inc) and monotonically_inc[b_off]:
                # Reassigns blink offset as the end of the monotonically_inc sequence
                b_off += 1
            blink_offset[i] = b_off
        # if not (monotonically_inc[b_off] and monotonically_dec[b_o]):
        #     # If blink offset is not a part of a monotonically_inc sequence, delete it from the list
        #     # of blink offsets and corresponding blink onset
        #     blink_onset = np.delete(blink_onset, i)
        #     blink_offset = np.delete(blink_offset, i)
        #     i -= 1
        i += 1

    # Creating empty array the size of the blink onsets and offsets combined
    c = np.empty((len(blink_onset) + len(blink_offset),), dtype=blink_onset.dtype)

    # places blink_onset values at even indices (0, 2, 4, ...).
    c[0::2] = blink_onset

    # places blink_offset values at odd indices (1, 3, 5, ...).
    c[1::2] = blink_offset
    c = list(c)

    # Loop to remove events with durations not between 0.1s and 0.5s
    i = 0
    while i < len(c) - 1:
        duration = c[i + 1] - c[i]
        if duration < .1 * sampling_freq or duration > .5 * sampling_freq:
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


def single_pupil_blink_detection(pupil_size, sampling_freq, timestamps):
    """
    Function to find blinks and return blink onset and offset indices from only one pupil.
    Adapted from: R. Hershman, A. Henik, and N. Cohen, “A novel blink detection method based on pupillometry noise,”
    Behav. Res. Methods, vol. 50, no. 1, pp. 107–114, 2018.

    Input:
        pupil_size: A numpy array or list containing pupil size data for one eye.
        sampling_freq: The sampling frequency of the eye-tracking hardware, given in Hz.
        timestamps: A numpy array or list containing the timestamps corresponding to the pupil size
    Output:
        blinks: [dictionary] {"blink_onset", "blink_offset"}
        containing numpy array/list of blink onset and offset timestamps
    """

    # sampling_interval represents the interval between samples in milliseconds. 1000/600=1.667ms
    sampling_interval = 1000 / sampling_freq

    # initializes output
    blink_onset = []
    blink_offset = []
    blinks = {"blink_onset": blink_onset, "blink_offset": blink_offset}

    # convert input to numpy arrays
    processed_pupil_size = np.asarray(pupil_size)

    # missing_data is an array where each element is 1 if pupil_size is NaN at that index, and 0 otherwise.
    missing_data = np.array(np.isnan(processed_pupil_size), dtype="float32")

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
        blink_offset = np.hstack((blink_offset, len(processed_pupil_size) - 1))

    # Smoothing the data in order to increase the difference between the measurement noise and the eyelid signal.

    # This sets the smoothing window to 10 milliseconds.
    # This is a 10ms window adapted from Hershman et al. (2018)
    ms_4_smoothing = 10

    # samples2smooth calculates the number of samples corresponding to the 10-millisecond window,
    # given the sampling frequency. (10/1.667 = 6 ms)
    samples2smooth = int(ms_4_smoothing / sampling_interval)

    # smooth_pupil_size applies the smooth function to the pupil size data using the
    # calculated window length (samples2smooth). The result is converted to a numpy array
    smooth_pupil_size = np.array(smooth(processed_pupil_size, samples2smooth), dtype='float32')

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

        ---> The monotonically decreasing sequence before the blink is underlined with -- and the 
        monotonically increasing sequence after the blink with ==
        ---> S : denotes the initially detected onset of blink
        ---> E : denotes the initially detected offset of blink

        >> Looking at diff(a), all values in the montonically decreasing sequence should be <= 0 and those 
        included in the monotonically increasing sequence >= 0
        >> Hence, by moving left from the initially detected onset while T(True) values are 
        encountered in monotonically_dec we can update the onset to the start of monotonically_dec seq
        >> By moving right from the initially detected offset while T(True) values are encountered in 
        monotonically_inc we can update the offset to the end of monotonically_inc seq + 1
    """
    monotonically_dec = smooth_pupil_size_diff <= 0
    monotonically_inc = smooth_pupil_size_diff >= 0

    # Currently, blink onsets are indices where the pupil size is present but the next index has missing data

    # Finding correct blink onsets and offsets using monotonically increasing and decreasing arrays
    i = 0
    while i < len(blink_onset):
        # Edge Case 2: If data starts with blink we do not update it and let starting blink index be 0
        if blink_onset[i] != 0:
            b_o = blink_onset[i] - 1
            # blink onsets rn are places where the data is there but in the next
            # index it is nan

            # if monotonically_dec at that index is True, that means the diff is negative
            # at that index, meaning the value after that index is less.
            # This assigns j to the index before the first blink "onset", which right now
            # is just the last point of data before a group of nans and sees if everything before that is
            # decreasing and continuously makes the one before it the blink onset until it stops decreasing
            while b_o > 0 and monotonically_dec[b_o]:
                b_o -= 1
            blink_onset[i] = b_o + 1

        # Edge Case 3: If data ends with blink we do not update it and let ending blink index be the last
        # index of the data
        if blink_offset[i] != len(processed_pupil_size) - 1:
            b_off = blink_offset[i]
            while b_off < len(monotonically_inc) and monotonically_inc[b_off]:
                # Reassigns blink offset as the end of the monotonically_inc sequence
                b_off += 1
            blink_offset[i] = b_off
        # if not (monotonically_inc[b_off] and monotonically_dec[b_o]):
        #     # If blink offset is not a part of a monotonically_inc sequence, delete it from the list
        #     # of blink offsets and corresponding blink onset
        #     blink_onset = np.delete(blink_onset, i)
        #     blink_offset = np.delete(blink_offset, i)
        #     i -= 1
        i += 1

    # Creating empty array the size of the blink onsets and offsets combined
    c = np.empty((len(blink_onset) + len(blink_offset),), dtype=blink_onset.dtype)

    # places blink_onset values at even indices (0, 2, 4, ...).
    c[0::2] = blink_onset

    # places blink_offset values at odd indices (1, 3, 5, ...).
    c[1::2] = blink_offset
    c = list(c)

    # Loop to remove events with durations not between 0.1s and 0.5s
    i = 0
    while i < len(c) - 1:
        duration = c[i + 1] - c[i]
        if duration < .1 * sampling_freq or duration > .5 * sampling_freq:
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


def identify_concat_blinks(left_blinks, right_blinks, tolerance=.15):
    """
        Identify and concatenate blinks from left and right eye blink data.

        This function takes blink onset and offset timestamps for left and right eye blinks
        (generated by single_pupil_blink_detection),
        compares them within a given tolerance, and identifies concurrent blinks.
        The identified concurrent blinks are then concatenated into single onsets
        and offsets using the average of the overlapping values.

        Parameters:
        left_blinks (DataFrame): A DataFrame containing 'blink_onset' and 'blink_offset' columns for left eye blinks.
        right_blinks (DataFrame): A DataFrame containing 'blink_onset' and 'blink_offset' columns for right eye blinks.
        tolerance (float): The maximum allowed difference (in seconds) between left and right blink onsets
                           to consider them as a single blink event. Default is 0.15 seconds.

        Output:
        blinks: [dictionary] {"blink_onset", "blink_offset"}
        containing numpy array/list of concat blink onset and offset timestamps

        Returns:
        concat_onsets (list): A list of concatenated blink onset times.
        concat_offsets (list): A list of concatenated blink offset times.
    """
    left_onsets = np.array(left_blinks["blink_onset"])
    left_offsets = np.array(left_blinks["blink_offset"])
    right_onsets = np.array(right_blinks["blink_onset"])
    right_offsets = np.array(right_blinks["blink_offset"])
    concat_onsets = []
    concat_offsets = []

    # initializes output
    blink_onset = []
    blink_offset = []
    blinks = {"blink_onset": blink_onset, "blink_offset": blink_offset}

    i, j = 0, 0
    while i < len(left_onsets) and j < len(right_onsets):
        if abs(left_onsets[i] - right_onsets[j]) <= tolerance:
            concat_onsets.append(np.mean([left_onsets[i], right_onsets[j]]))
            if i < len(left_offsets) and j < len(right_offsets):
                concat_offsets.append(np.mean([left_offsets[i], right_offsets[j]]))
            i += 1
            j += 1
        elif left_onsets[i] < right_onsets[j]:
            i += 1
        else:
            j += 1

    blinks["blink_onset"] = concat_onsets
    blinks["blink_offset"] = concat_offsets

    return blinks
