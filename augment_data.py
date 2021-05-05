import argparse
import os
import glob
import librosa
from imblearn.over_sampling import SMOTE
import numpy as np
import soundfile
from constants import *
from common import *

def load_raw_data(raw_normal_path, raw_abnormal_path):
    print_log("Loading raw data")
    normal_sounds = load_wavs_from_directory(raw_normal_path)
    abnormal_sounds = load_wavs_from_directory(raw_abnormal_path)
    print_log("Finished loading raw data")

    return normal_sounds, abnormal_sounds

def apply_smote(normal_sounds, abnormal_sounds):
    print_log("Combining data for SMOTE")
    print_log("normal_sounds.shape={}".format(normal_sounds.shape))
    print_log("abnormal_sounds.shape={}".format(abnormal_sounds.shape))
    labels = np.array([])
    sounds = np.array([])
    if NORMAL_VALUE == 0 and ABNORMAL_VALUE == 1:
        labels = np.append(np.ones(len(abnormal_sounds)), np.zeros(len(normal_sounds)))
        sounds = np.concatenate([abnormal_sounds, normal_sounds])
    else:
        raise NotImplementedError("Only the case where ABNORMAL=1 and NORMAL=0 has been implemented")
    print_log("Data for SMOTE has been combined")
    print_log("labels.shape={}".format(labels.shape))
    print_log("sounds.shape={}".format(sounds.shape))
    
    print_log("Applying SMOTE")
    smt = SMOTE(sampling_strategy=800.0/1101.0)
    resampled_sounds, resampled_labels = smt.fit_resample(sounds, labels)
    print_log("SMOTE has been applied")

    print_log("resampled_sounds.shape={}".format(resampled_sounds.shape))
    print_log("resampled_labels.shape={}".format(resampled_labels.shape))

    return resampled_sounds, resampled_labels

def save_resampled_sounds(resampled_sounds, resampled_labels, augmented_normal_path, augmented_abnormal_path):
    normal_counter = 0
    abnormal_counter = 0

    print_log("Saving resampled sounds")
    print_log("resampled_labels.shape={}".format(resampled_labels.shape))
    print_log("resampled_sounds.shape={}".format(resampled_sounds.shape))
    print_log("augmented_normal_path={}".format(augmented_normal_path))
    print_log("augmented_abnormal_path={}".format(augmented_abnormal_path))

    for i in range(len(resampled_labels)):
        if resampled_labels[i] == NORMAL_VALUE:
            path = os.path.join(augmented_normal_path, "{}.wav".format(normal_counter))
            soundfile.write(path, resampled_sounds[i], SAMPLE_RATE_OF_RAW_DATA)
            normal_counter += 1
            
            if normal_counter % 100 == 0:
                print_log("Heartbeat -- Just saved {}".format(path))
        elif resampled_labels[i] == ABNORMAL_VALUE:
            path = os.path.join(augmented_abnormal_path, "{}.wav".format(abnormal_counter))
            soundfile.write(path, resampled_sounds[i], SAMPLE_RATE_OF_RAW_DATA)
            abnormal_counter += 1

            if abnormal_counter % 100 == 0:
                print_log("Heartbeat -- Just saved {}".format(path))
        else:
            raise RuntimeError("Something wrong occurred!")

    print_log("Resampled sounds have been saved")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_normal_path", type=existing_dir_path)
    parser.add_argument("raw_abnormal_path", type=existing_dir_path)
    parser.add_argument("augmented_normal_path", type=dir_path)
    parser.add_argument("augmented_abnormal_path", type=dir_path)
    args = parser.parse_args()

    raw_normal_path = args.raw_normal_path
    raw_abnormal_path = args.raw_abnormal_path
    augmented_normal_path = args.augmented_normal_path
    augmented_abnormal_path = args.augmented_abnormal_path

    print_log("raw_normal_path={}".format(raw_normal_path))
    print_log("raw_abnormal_path={}".format(raw_abnormal_path))
    print_log("augmented_normal_path={}".format(augmented_normal_path))
    print_log("augmented_abnormal_path={}".format(augmented_abnormal_path))

    normal_sounds, abnormal_sounds = load_raw_data(raw_normal_path, raw_abnormal_path)

    resampled_sounds, resampled_labels = apply_smote(normal_sounds, abnormal_sounds)

    resampled_sounds, resampled_labels = unison_shuffled_copies(resampled_sounds, resampled_labels)

    save_resampled_sounds(resampled_sounds, resampled_labels, augmented_normal_path, augmented_abnormal_path)

if __name__ == "__main__":
    main()
