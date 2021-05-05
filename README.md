
You will need:
- Python3 with venv and pip (Has been tested on Python 3.8.0 and 3.8.4): "apt-get install python3 python3-venv"
- unzip: "apt-get install unzip"
- libsndfile1: "apt-get install libsndfile1"
- libjpeg: "apt-get install libjpeg-dev"


Instructions:
- You should be in this directory
- Execute run_tutorial.sh ('bash run_tutorial.sh')

How to record data:
- Data for the demo is located in raw_data/normal/ and raw_data/abnormal/ 
- anomaly_detection/recording/record.py is a script that records audio using pyaudio
- Please read anomaly_detection/recording/README.md

How to run on Icicle Kit:
- After run_tutorial.sh has finished, transfer icicle/ into the icicle kit (e.g.: scp -r icicle root@IP:/home/root/)
- On the icicle kit, navigate to icicle/example/sim-c/ and follow the instructions on the README.md located in that directory

Has been tested on: 
- Windows Subsystem for Linux 1 (WSL1), under Ubuntu Bionic 18.04.5 LTS
- A Linux Machine, under Ubuntu Bionic 18.04.5 LTS

Notes for Umair/Nancy:
- Place sdk.zip in this directory
- Place icicle-kit-simulator.zip in this directory
- Place raw_data/ and raw_noise/ in this directory

raw_data/ and raw_noise/ aren't in the repo simply cause they're big and you can get off google drive

sdk.zip and icicle-kit-simulator.zip are not public releases yet, so there's no git repo that we can mark as a submodule nor a permanent mirror to download from
