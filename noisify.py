import argparse
import numpy as np
import math
from constants import *
from common import *

def get_noise_at_snr(signal, noise, SNR):

    RMS_of_signal = math.sqrt(np.mean(signal ** 2))
    RMS_of_noise_desired = math.sqrt(RMS_of_signal ** 2 / (pow(10, SNR / 10)))
    RMS_of_noise_actual = math.sqrt(np.mean(noise ** 2))
    noise *= (RMS_of_noise_desired / RMS_of_noise_actual)
    
    return noise

def add_noise_to_signal(signal, noise, SNR):
    return signal + get_noise_at_snr(signal, noise, SNR)

def add_noise_to_data(input_data, noise, SNR):
    print_log("Adding noise to data")
    noisy_sounds = [] 

    for i in range(len(input_data)):
        clean_sound = input_data[i]
        noisy_sound = add_noise_to_signal(clean_sound, noise[i % len(noise)], SNR) 
        noisy_sounds.append(noisy_sound)

        if i % 100 == 0:
            print_log("Heartbeat i={}".format(i))
    print_log("Finished adding noise to data")
    return np.array(noisy_sounds) 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_noise_dir", type=existing_dir_path)
    parser.add_argument("input_dir", type=dir_path)
    parser.add_argument("output_dir", type=dir_path)
    parser.add_argument("SNR", type=int)
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir 
    raw_noise_dir = args.raw_noise_dir
    SNR = args.SNR

    print_log("input_dir={}".format(input_dir))
    print_log("output_dir={}".format(output_dir))
    print_log("raw_noise_dir={}".format(raw_noise_dir))
    print_log("SNR={}".format(SNR))

    input_data = load_wavs_from_directory(input_dir)
    noise = load_wavs_from_directory(raw_noise_dir)

    output_data = add_noise_to_data(input_data, noise, SNR)

    save_wavs_to_directory(output_dir, output_data)

if __name__ == "__main__":
    main()
