import pyaudio
import wave
import argparse
import os
import pathlib

import sys
sys.path.append(os.path.join(os.getcwd(), ".."))
from constants import *
from common import *

FORMAT = pyaudio.paInt16

def positive_int(x):
    return abs(int(x))

def get_channel_input(num_devices):
    print_log("Please select a channel")
    print_log("Enter Index of Input Device > ")
    while True:
        try:
            index = int(input())

            if index < 0 or index >= num_devices:
                raise RuntimeError("Index {} for num_devices {}".format(index, num_devices))

            return index
        except:
            print_log("Try again > ")
            continue

def check_if_file_exists(filepath):
    return os.path.exists(filepath)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("save_dir", type=dir_path)
    parser.add_argument("num_samples", type=positive_int)
    args = parser.parse_args()

    save_dir = args.save_dir 
    num_samples = args.num_samples 

    print_log("save_dir={}".format(save_dir))
    print_log("num_samples={}".format(num_samples))

    audio = pyaudio.PyAudio()

    print_log("----------------------record device list---------------------")
    info = audio.get_host_api_info_by_index(0)
    num_devices = info.get('deviceCount')
    for i in range(0, num_devices):
        if (audio.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
            print_log("Input Device id {} - {}".format(i, audio.get_device_info_by_host_api_device_index(0, i).get('name')))

    print_log("-------------------------------------------------------------")

    index = get_channel_input(num_devices) 

    
    print_log("Recording via index {}".format(index))

    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True, input_device_index=index,
                        frames_per_buffer=CHUNK)

    file_num = 0

    for j in range(0, num_samples):
        print_log("Started recording sample {}".format(j))
        record_frames = []
        stream.start_stream()
    
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            record_frames.append(data)
        print_log("Finished recording sample {}".format(j))
    
        stream.stop_stream()

        
        filepath = os.path.join(save_dir, "{}.wav".format(file_num)) 
        while (check_if_file_exists(filepath)):
            file_num += 1
            filepath = os.path.join(save_dir, "{}.wav".format(file_num))

        print_log("Sample {} will be saved under {}".format(j, filepath))
        waveFile = wave.open(filepath, 'wb')
        waveFile.setnchannels(CHANNELS)
        waveFile.setsampwidth(audio.get_sample_size(FORMAT))
        waveFile.setframerate(RATE)
        waveFile.writeframes(b''.join(record_frames))
        waveFile.close()
        print_log("Sample {} was saved under {}".format(j, filepath))

        file_num += 1

    stream.close()
    audio.terminate()

if __name__ == "__main__":
    main()