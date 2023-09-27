# Multi-Modal V2V Beam Prediction Challenge 2023 Baseline & Starter Scripts
Baseline for the [DeepSense6G Beam Prediction Challenge 2023](https://deepsense6g.net/beam_prediction_challenge_2023/) based on V2V datasets. 

## Requirements to run the code

The Python environment used is exported in ```deepsense-env.txt```, but most of them are not 
necessary. All that is necessary to run the scripts is a basic Python installation and the
modules: 
- NumPy
- SciPy
- Pandas
- tqdm
- Pickle
- Matplotlib

Here are the recommended steps to setup a Python environment that can run this code:

1. Install Mambaforge -> https://github.com/conda-forge/miniforge#mambaforge
2. Open Miniforge Prompt and create a new environment
```mamba create -n deepsense-challenge```
3. Install the required packages
```mamba install numpy scipy pandas tqdm pickle matplotlib```

## Problem and Code Explanation

[![PROBLEM & CODE EXPLANATION](code_explanation_video_thumbnail.png)](https://youtu.be/1D3PAe5uKVM)
(*click the image to watch the video*)

Outline of the video:
1. Problem & provided data: 3 pieces of information in 4 files
2. Necessary data and folder structure:
	- necessary data for competition -> GPS, optionally RGB, optionally CSVs/pre-loaded dicts
	- necessary data for benchmark   -> GPS and PWR and pre-loaded dicts
3. Pyhon environment setup
4. Two different ways of loading the data: fast for training and slow for testing 
5. Loading & Display images
6. Baseline approach and code
7. Output results in submission format
8. Evaluation metric/score

**Important changes**:
- Evaluation metric/score changed! Instead of the average beam distance, it is the average power loss. The current code has the right metric, but the video will still show the old metric. For more information about this metric, refer to the competition page: [Beam Prediction Challenge 2023](https://deepsense6g.net/beam_prediction_challenge_2023/)

# FAQs

## Problems unzipping data

1. Make sure you read the section “How to Access Scenario Data?” in DeepSense Scenarios 36-39 and watch the video
2. For MacOS/Linux: install 7zip (https://www.7-zip.org/download.html) and run `7z x my_zip.zip.001`
    (For MacOS, it can also be installed with `brew install p7zip`)
    
## Problems downloading data? “Link expired”?

1. Make sure you are trying to download the files individually, not all the folders at once
2. Make sure you read the section “How to Access Scenario Data?” in DeepSense Scenarios 36-39 and watch the video
3. Try to get a better internet connection by moving locations (e.g., to the university) or by connecting an Ethernet cable (you can test your internet speed in speedtest.net - speeds under 10 Mbps often struggle to complete the download)
4. If you get a “link expired” error, then try to download the data via an alternative shared folder from Dropbox: 
    https://www.dropbox.com/sh/5zpmzzbkp93w5zl/AADbaPG1aSfQEBgpr8xy6BnTa?dl=1
    Note: *when using this link*, the download size is limited depending on your account. Basic and trial accounts can only download 20GB per day. Premium accounts can do 400GB a day. 
5. If none of the above steps work, contact us, and we’ll figure something out.

## What is the scenario36.p file? 

It’s a pickle file. It contains the same content as the CSV file, but the powers are already pre-loaded, making it faster to use if you use Python. The equivalent version for Matlab is scenario36.mat. 

Take a look at the video at the top of this GitHub page for more information. 

