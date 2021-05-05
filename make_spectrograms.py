
import argparse
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from constants import *
from common import *

def save_spectrogram(data, output_dir, count):
    y = data 
    filepath = os.path.join(output_dir, "{}.jpg".format(count))

    D = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    # Some tricks to get rid of all white-space are needed
    plt.figure(figsize=(0.432, 0.288), dpi=100)
    librosa.display.specshow(S_db, sr=SAMPLE_RATE_OF_RAW_DATA, hop_length=HOP_LENGTH, x_axis='time', y_axis='linear', cmap="gray")
    plt.ioff() # Avoids a memory leak!
    plt.axis('off') 
    plt.gca().set_axis_off()
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])
    plt.savefig(filepath, dpi=1000, bbox_inches=None, pad_inches=0)
    plt.cla() 
    plt.clf() 
    plt.close('all')

    if count % 100 == 0:
        print_log("Heartbeat -- Saved {}".format(filepath))

def save_spectrograms(input_data, output_dir):
    count = 0
    for data in input_data:
        save_spectrogram(data, output_dir, count)
        count += 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=dir_path)
    parser.add_argument("output_dir", type=dir_path)
    args = parser.parse_args()

    input_dir = args.input_dir 
    output_dir = args.output_dir 

    print_log("input_dir={}".format(input_dir))
    print_log("output_dir={}".format(output_dir))

    input_data = load_wavs_from_directory(input_dir) 
    save_spectrograms(input_data, output_dir)

if __name__ == "__main__":
    main()
