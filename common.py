
import os 
import librosa 
import numpy as np
import glob
import soundfile
from constants import *

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def print_log(string):
    print("[LOG] {}".format(string))

def existing_dir_path(string):
    directory = os.path.abspath(string)
    if os.path.isdir(directory):
        print_log("Found {}".format(directory))
        return directory
    else:
        raise NotADirectoryError(string)

def dir_path(string):
    directory = os.path.abspath(string)

    if not os.path.isdir(directory):
        print_log("Creating {}".format(directory))
        os.makedirs(directory)
        print_log("Created {}".format(directory))
    
    return existing_dir_path(directory)

def load_wavs_from_directory(directory):
    files = []
    y = 0
    sr = 0

    for file in glob.glob(os.path.join(directory, "*.wav")):
        y, sr = librosa.load(file, sr=SAMPLE_RATE_OF_RAW_DATA, mono=True)
        y = y[:NUM_OF_SAMPLES_PER_WAV_FILE]
        files.append(y)

    return np.array(files)

def save_wavs_to_directory(directory, files):
    print_log("Will save files into {}".format(directory))
    print_log("Number of files: {}".format(len(files)))
    counter = 0
    for file in files:
        path = os.path.join(directory, "{}.wav".format(counter))
        soundfile.write(path, file, SAMPLE_RATE_OF_RAW_DATA)
        counter += 1
        
        if counter % 100 == 0:
            print_log("Heartbeat -- Just saved {}".format(path))
    print_log("Finished saving files into {}".format(directory))
