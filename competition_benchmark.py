# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 08:17:57 2023

@author: Joao

This script reads a training CSV, outputs predictions for the data, and applies
the score metric to compute the final score. 

It also computes a few other known metrics, like top-k accuracy and average power loss.


"""
import scipy
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

X_SIZE = 5      # 5 input samples
N_GPS = 2       # 2 GPSs (unit1 and unit2)
N_GPS_COORD = 2 # 2 GPS coords (latitude & longitude)
N_ARR = 4       # 4 arrays
N_BEAMS = 64    # 64 beams per array


def norm_2pi(x):
    # -pi to pi
    x_normed = np.empty_like(x)
    x_normed[:] = x
    for i in range(len(x)):
        if abs(x[i]) >= np.pi:
            x_normed[i] = x[i] % (2*np.pi)

            if x[i] >= np.pi:
                x_normed[i] -= 2*np.pi

        while x_normed[i] < -np.pi:
            x_normed[i] += 2*np.pi

        while x_normed[i] > np.pi:
            x_normed[i] -= 2*np.pi

        if x_normed[i] < -np.pi:
            print(f'{i}, {x_normed[i]}')

    return x_normed


def compute_ori_from_pos_delta(lat_deltas, lon_deltas, thres_lat=0, thres_lon=0):
    """
    Returns orientation in [-pi, pi]
    # 0 = East
    # pi/2 = North
    # pi / -pi = West
    # -pi/2 = South

    Thresholds do: if delta_lat < thres_lat -> delta_lat = 0

    If lats and lons are N x 2, it computes the difference between the two columns.
    If lats and lons are N x 1, it computes differences in consecutive samples 
    (uses two positions at different times to get orientation).

    """
    n_samples = len(lat_deltas)
    pose = np.zeros(n_samples)

    for i in range(n_samples):

        delta_lon = 0 if abs(lat_deltas[i-1]) < thres_lon else lat_deltas[i-1]
        delta_lat = 0 if abs(lon_deltas[i-1]) < thres_lat else lon_deltas[i-1]

        if delta_lon == 0:
            if delta_lat == 0:
                pose[i] = 0 # pose[i-1]
                continue
            elif delta_lat > 0:
                slope = np.pi / 2
            elif delta_lat < 0:
                slope = -np.pi / 2
        else:
            slope = np.arctan(delta_lat / delta_lon)
            if delta_lat == 0:
                slope = np.pi if delta_lon < 0 else 0
            elif delta_lat < 0 and delta_lon < 0:
                slope = -np.pi + slope
            elif delta_lon < 0 and delta_lat > 0:
                slope = np.pi + slope

        pose[i] = slope

    return pose


def estimate_positions(input_positions, delta_input, delta_output):

    n_samples = input_positions.shape[0]
    out_pos = np.zeros((n_samples, 2))

    x_size = input_positions.shape[1]
    x = delta_input * np.arange(x_size)
    for sample_idx in tqdm(range(n_samples), desc='Estimating input positions'):
        input_pos = input_positions[sample_idx]

        f_lat = scipy.interpolate.interp1d(
            x, input_pos[:, 0], fill_value='extrapolate')
        f_lon = scipy.interpolate.interp1d(
            x, input_pos[:, 1], fill_value='extrapolate')

        out_pos[sample_idx, 0] = f_lat(x[-1] + delta_output)
        out_pos[sample_idx, 1] = f_lon(x[-1] + delta_output)

    return out_pos


def predict_beam_uniformly_from_aoa(aoa):
    """
    Computes distance of each datapoint to each predictor point and returns
    ordered list of indices of the closest predictor point. 

    aoa is N x 1, beam_ori is K x 1 (k = no. total beams), beam_pred is N x K
    """
    beam_predictions = np.zeros_like(aoa)

    beam_ori = np.arange(N_BEAMS * N_ARR) / (N_BEAMS * N_ARR - 1) * 2*np.pi - np.pi

    angl_diff_to_each_beam = aoa.reshape((-1, 1)) - beam_ori

    beam_predictions = np.argsort(abs(angl_diff_to_each_beam), axis=1)

    return beam_predictions


def circular_distance(a, b, l=256, sign=False):
    """
    Computes the distance between two beam indices, <a> and <b> in a circular
    way. I.e., if all numbers are written in a circle with <l> numbers,
    this computes the shortest distance between any two numbers. 
    E.g., assuming l=256
        (a = 0, b = 5) -> dist = 5
        (a = 0, b = 255) -> dist = 1
        (a = 0, b = 250) -> dist = 6
        (a = 0, b = 127) = (a = 0, b = 129) = 127
        
    If <sign> = True -> a = predicted & b = truth    
    
    """
    while a < 0:
        a = l - abs(a)
    while b < 0:
        b = l - abs(b)
        
    a = a % l if a >= l else a
    b = b % l if b >= l else b
    
    dist = a - b

    if abs(dist) > l/2:
        dist = l - abs(dist)

    return dist if sign else abs(dist)


def compute_acc(all_beams, only_best_beam, top_k=[1, 3, 5]):
    
    """ 
    Computes top-k accuracy given prediction and ground truth labels.

    Note that it works bidirectionally. 
    <all_beams> is (N_SAMPLES, N_BEAMS) but it can represent:
        a) the ground truth beams sorted by receive power
        b) the predicted beams sorted by algorithm's confidence of being the best

    <only_best_beam> is (N_SAMPLES, 1) and can represent (RESPECTIVELY!):
        a) the predicted optimal beam index
        b) the ground truth optimal beam index

    For the competition, we will be using the function with inputs described in (a).

    """
    n_top_k = len(top_k)
    total_hits = np.zeros(n_top_k)

    n_test_samples = len(only_best_beam)
    if len(all_beams) != n_test_samples:
        raise Exception(
            'Number of predicted beams does not match number of labels.')

    # For each test sample, count times where true beam is in k top guesses
    for samp_idx in range(len(only_best_beam)):
        for k_idx in range(n_top_k):
            hit = np.any(all_beams[samp_idx, :top_k[k_idx]] == only_best_beam[samp_idx])
            total_hits[k_idx] += 1 if hit else 0

    # Average the number of correct guesses (over the total samples)
    return np.round(total_hits / len(only_best_beam), 4)


# %% Read CSV and Load dataset
csv_train = 'D:/BENCHMARKS/deepsense_challenge2023_trainset.csv'
csv_dict_path = 'D:/BENCHMARKS/scenario36/scenario36.p'

with open(csv_dict_path, 'rb') as fp:
    csv_dict = pickle.load(fp)

#%% Example of loading and displaying RGB180 images

df_train = pd.read_csv(csv_train)
sample_idx = 0

csv_train_folder = '/'.join(csv_train.split('/')[:-1])
scen_folder = 'scenario' + str(df_train['scenario'][sample_idx])
img1_path = csv_train_folder + '/' + scen_folder + '/' + csv_dict['unit1_rgb5'][sample_idx]
img2_path = csv_train_folder + '/' + scen_folder + '/' + csv_dict['unit1_rgb6'][sample_idx]
img1 = plt.imread(img1_path)
img2 = plt.imread(img2_path)
fig, axs = plt.subplots(2,1, figsize=(16,9), dpi=200)
axs[0].imshow(img1) # front
axs[1].imshow(img2) # back

# %% (Fast Loading) Load Training positions and ground truth positions (for scenario 36 only)

# Load all positions
samples_of_scen36 = np.where(df_train['scenario'] == 36)[0]
n_samples = len(samples_of_scen36)

loaded_positions = set()
train_positions = np.zeros((n_samples, X_SIZE, N_GPS, N_GPS_COORD))

y_pos1 = np.zeros((n_samples, N_GPS_COORD))
y_pos2 = np.zeros((n_samples, N_GPS_COORD))
y_pwrs = np.zeros((n_samples, N_ARR, N_BEAMS))
for sample_idx in tqdm(range(n_samples), desc='Loading data'):
    train_sample = samples_of_scen36[sample_idx]
    for x_idx in range(X_SIZE):
        abs_idx_relative_index = (csv_dict['abs_index'] == df_train[f'x{x_idx+1}_abs_index'][train_sample])
        train_positions[train_sample, x_idx, 0, :] = csv_dict['unit1_gps1'][abs_idx_relative_index]
        train_positions[train_sample, x_idx, 1,:] = csv_dict['unit2_gps1'][abs_idx_relative_index]

    # Positions of the output to compare with our position estimation approach
    y_idx = (csv_dict['abs_index'] == df_train['y1_abs_index'][train_sample])
    y_pos1[train_sample] = csv_dict['unit1_gps1'][y_idx]
    y_pos2[train_sample] = csv_dict['unit2_gps1'][y_idx]
    for arr_idx in range(N_ARR):
        y_pwrs[train_sample, arr_idx] = csv_dict[f'unit1_pwr{arr_idx+1}'][y_idx]

y_true_beams = df_train['y1_unit1_overall-beam'].values[samples_of_scen36]

# array 1 (0-63), array 2 (64-127), array 3 (128-191), array 4 (192-255)
y_pwrs_reshaped = y_pwrs.reshape((n_samples, -1))
all_true_beams = np.flip(np.argsort(y_pwrs_reshaped, axis=1), axis=1)

# %% (Slow loading) Example of data loading for the testset

csv_test = csv_train
df_test = pd.read_csv(csv_test)
n_samples = 1000 # example size of testset
folder = '/'.join(csv_test.split('/')[:-1])
input_pos = np.zeros((n_samples, X_SIZE, N_GPS, N_GPS_COORD))

for sample_idx in tqdm(range(n_samples), desc='Loading data'):
    for x_idx in range(X_SIZE):
        input_pos[sample_idx, x_idx, 0, :] = \
            np.loadtxt(folder + '/' + df_test[f'x{x_idx+1}_unit1_gps1'][sample_idx])
        input_pos[sample_idx, x_idx, 1, :] = \
            np.loadtxt(folder + '/' + df_test[f'x{x_idx+1}_unit2_gps1'][sample_idx])

# %% Step 1: Estimate positions in the new timestamp (linear interpolation)
delta_input = 0.2  # time difference between input samples [s]
delta_output = 0.5  # time difference from last input to output [s]

gps1_est_pos = estimate_positions(train_positions[:, :, 0, :], delta_input, delta_output)
gps2_est_pos = estimate_positions(train_positions[:, :, 1, :], delta_input, delta_output)

# Compare estimated with real
if True:
    plt.figure(figsize=(10, 6), dpi=200)
    n = np.arange(300)
    plt.plot(y_pos1[n, 0], -y_pos1[n, 1], alpha=.5,
             label='True positions', marker='o', markersize=1)
    plt.plot(gps1_est_pos[n, 0], -gps1_est_pos[n, 1], alpha=.5, 
             label='Estimated positions', c='r', marker='o', markersize=1)
    plt.title('Position Estimation')
    plt.xlabel('Latitude')
    plt.ylabel('Longitude')
    plt.legend()
    plt.show()
    # annotate start and end points

# %% Step 2: With the estimated positions, estimate orientation

# 2.1 - Determine heading of vehicles using last available location and the new estimated location
lat_deltas = gps1_est_pos[:, 0] - train_positions[:, -1, 0, 0]
lon_deltas = gps1_est_pos[:, 1] - train_positions[:, -1, 0, 1]
heading = compute_ori_from_pos_delta(lat_deltas, lon_deltas)

# 2.2 - Determine relative position (converted to orientation) between vehicles
lat_deltas = gps1_est_pos[:, 0] - gps2_est_pos[:, 0]
lon_deltas = gps1_est_pos[:, 1] - gps2_est_pos[:, 1]

ori_rel = compute_ori_from_pos_delta(lat_deltas, lon_deltas)

# referenced to beam 0 of front array
aoa_estimation = norm_2pi(ori_rel - heading + np.pi/4)

# check if they correlate enough for an accurate prediction
if True:
    beams_angle = y_true_beams/255*2*np.pi - np.pi
    x = np.arange(len(aoa_estimation))
    plt.figure(figsize=(10, 6), dpi=200)
    plt.scatter(x, aoa_estimation, alpha=.5, s=3, zorder=2, label='aoa est')
    plt.scatter(x, beams_angle, alpha=.5, s=3, label='beams angle')
    plt.title('Position Estimation')
    plt.xlabel('Sample index')
    plt.ylabel('Angle [rad]')
    plt.legend()
    plt.show()
    
# %% Step 3: From orientation, estimate beam (assume uniform beam distribution)

beam_pred_all = predict_beam_uniformly_from_aoa(aoa_estimation)
best_beam_pred = beam_pred_all[:, 0]

# After analysis, we were often 1 beam short. 
pred_diff = np.array([circular_distance(a, b, sign=True)
                      for a, b in zip(best_beam_pred, y_true_beams)])

# The box sometimes is slightly rotated around it's Z axis, so we can shift our
# Beam predictions a constant offset to get better performance. Admit offsets up to 2.
# Note: this adjustment is only for the training phase
shift = -round(np.mean(pred_diff[abs(pred_diff) < 2]))
print(f'estimated_shift = {shift}')
beam_pred_all += shift
best_beam_pred += shift

# Check if the prediction is good
if True:
    x = np.arange(len(aoa_estimation))
    plt.figure(figsize=(10, 6), dpi=200)
    plt.scatter(x, best_beam_pred, alpha=.5, s=3, zorder=2, label='pred')
    plt.scatter(x, y_true_beams, alpha=.5, s=3, label='true')
    plt.legend()
    plt.show()

#%% Output results for competition

df_out = pd.DataFrame()
df_out['prediction'] = best_beam_pred
df_out.to_csv('SUBMISSION-EXAMPLE_prediction.csv', index=False)
# put group name instead of "SUBMISSION-EXAMPLE"

# %% Compute Scores

# Note: only best_beam_pred and all_true_beams needed -> 
# Our evaluation script will have access to the ground truth beams of the testset
# and you submit the best_beam_pred in a csv

pred_diff_abs = np.array([circular_distance(a, b)
                             for a, b in zip(best_beam_pred, all_true_beams[:,0])])
total_score = np.mean(pred_diff_abs) # lower is better!

n_beams_under_3_of_best = len(np.where(pred_diff_abs < 3)[0])
print('No. beams under distance of 3 of the best = '
      f'{n_beams_under_3_of_best} ({n_beams_under_3_of_best / n_samples * 100:.2f} %)')

n_beams_under_5_of_best = len(np.where(pred_diff_abs < 5)[0])
print('No. beams under distance of 5 of the best = '
      f'{n_beams_under_5_of_best} ({n_beams_under_5_of_best / n_samples * 100:.2f} %)')

print(f'Score (average distance to optimal beam) = {total_score:.2f}')


# Visualize score
if True:
    plt.figure(figsize=(10, 6), dpi=200)
    plt.scatter(np.arange(len(pred_diff_abs)), pred_diff_abs, s=2)
    plt.hlines(y=total_score, xmin=0, xmax=len(pred_diff_abs), color='r',alpha=.5)

# "Probability of the prediction of the best beam being in the set of best k ground truth beams"
top_k = compute_acc(all_true_beams, best_beam_pred, top_k=[1, 3, 5])
print(f'Top-k = {top_k}')


# "Probability of the ground truth best beam being in the set of most likely k predicted beams"
top_k = compute_acc(beam_pred_all, all_true_beams[:, 0], top_k=[1, 3, 5])
print(f'(usual) Top-k = {top_k}')

# To make it practical for submissions, we implement way 1 and only require the
# best predicted beam. 

