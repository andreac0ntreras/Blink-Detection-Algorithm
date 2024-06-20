# Introduction
The purpose of this project is to develop an algorithm for blink detection to be used for the analysis of 
data from eye trackers such as Tobii products.

Data from any eyetracker can be analyzed using the scripts in this repository, however, the algorithm was built 
for Tobii products specifically, so be advised that different brands of eyetrackers have different noise
patterns. 

## Cloning the repository
To run this program, enter the following in your command prompt
```commandline
git clone https://github.com/andreac0ntreras/Blink-Detection-Algorithm.git
```

### Navigate to the codebase
Move into the project directory
```commandline
cd ./EOGData
```

### Install the required packages
```commandline
pip install -r requirements.txt
```

Within the project directory, there are three directories. The data directory (_data/raw/XDF Files_) contains the raw data from the
OWDM experiment, which we will be analyzing. The raw data is in the form of XDF files, where one of the streams 
contains the eye tracker timeseries data and the eyetracker specs. 

The hardware used during this experiment is the Tobii Spectrum Pro with a sampling frequency of 600Hz.

