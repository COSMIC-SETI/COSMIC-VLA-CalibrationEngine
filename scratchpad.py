import sys
import os
import argparse
import time
import json
import numpy as np
import pandas as pd
import redis
import shutil
from cosmic.redis_actions import redis_obj, redis_hget_keyvalues, redis_publish_dict_to_hash
from matplotlib import pyplot as plt
import pyuvdata.utils as uvutils
from pyuvdata import UVData
from calib_util import gaincal_cpu, gaincal_gpu, applycal, flag_complex_vis_smw, flag_complex_vis_medf
from sliding_rfi_flagger import flag_rfi_real
import glob

actual_non_cal_change = 0.9964189260194121
actual_cal_change = 0.05712315922132594

#From a folder, read all json files and extract the field "freqs_hz" and "proposed_gain_grade" from each file. Then stitch together
#the freqs_hz sections as the x-axis and the proposed_gain_grade sections as the y-axis. Plot the gain grade vs frequency. It should
#look like a step function as the same grade is for multiple freqs_hz
def plot_gain_grade_vs_freqs(files):
    freqs = []
    prop_grades = []
    grades = []
    for file in files:
        with open(file, 'r') as f:
            data = json.load(f)
            nested_data = data[next(iter(data))]  # Access the dictionary under the first key
            freqs.append(nested_data["freqs_hz"])
            print(f"File {file} has proposed grade: {nested_data['proposed_gain_grade']}")
            print(f"File {file} has grade: {nested_data['grade']}")
            prop_grades.append(nested_data["proposed_gain_grade"])
            grades.append(nested_data["grade"])
    #assume all filenames are the same:
    obs_id = ".".join(os.path.basename(file).split(".")[:-3])

    freqs = np.concatenate(freqs)
    prop_grades = np.array(prop_grades)
    grades = np.array(grades)
    print(prop_grades.shape)
    print(np.mean(prop_grades))
    sort_indices = np.argsort(freqs)
    freqs = freqs[sort_indices]
    
    # Pad out the grade values to be the same size as freqs
    prop_grades = np.repeat(prop_grades, len(freqs) // len(prop_grades))
    grades = np.repeat(grades, len(freqs) // len(grades))
    prop_grades = prop_grades[sort_indices]
    grades = grades[sort_indices]
    
    plt.step(freqs, prop_grades, where='post',color='black',label='Proposed Grade per subband')
    plt.step(freqs, grades, where='post',color='cyan',label='Grade per subband')

    prop_gain_mean = sum(prop_grades) / len(prop_grades)
    gain_mean = sum(grades) / len(grades)
    print(len(prop_grades))
    print(prop_gain_mean)
    print(gain_mean)
    plt.axhline(y=prop_gain_mean, color='r', linestyle='--',label='Proposed Next Observation Grade')
    plt.axhline(y=gain_mean, color='green', linestyle='--',label='Observation Grade')
    plt.axhline(y=actual_cal_change, color='g', linestyle='--',label='Next Observation Actual Grade')
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Gain Grade (flagged)")
    # plt.ylabel("Gain Grade")
    # plt.title(f"Gain Grade vs Frequency for observation:\n{obs_id}\nMean Gain Grade: {gain_mean}")
    plt.title(f"Gain Grade (flagged) vs Frequency for observation:\n{obs_id}\nMean Gain Grade: {gain_mean}")
    plt.legend()
    plt.savefig(f'flagged_gain_and_prop_grade_vs_freqs_{obs_id}.png')
    # plt.savefig(f'gain_grade_vs_freqs_{obs_id}.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot gain grade vs frequency')
    parser.add_argument('files', nargs='*', type=str, help='Folder or list of json files')
    args = parser.parse_args()
    if len(args.files) == 1 and os.path.isdir(args.files[0]):
        # If there's only one argument and it's a directory, get all json files in the directory
        folder = args.files[0]
        files = glob.glob(os.path.join(folder, '*.json'))
    else:
        # Otherwise, treat the arguments as individual file paths
        files = args.files
    plot_gain_grade_vs_freqs(files)