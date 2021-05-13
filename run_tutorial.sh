#!/bin/bash
set -e
set -x

echo "Removing previous results"
rm -rf anomaly garbage augmented augmented_noise noisified spectrograms data final_images trained_model vnnx sdk __pycache__ log.txt icicle
BASE_DIR=$(pwd)

echo -e "\n"

{
    echo "Will unzip VBX sdk and install its dependencies"
    unzip sdk 
    cd sdk
    bash install_dependencies.sh
    cd $BASE_DIR
    echo -e "\n"

    echo "Creating environment"
    python3 -m venv anomaly
    echo -e "\n"

    echo "Activating environment"
    source anomaly/bin/activate
    echo -e "\n"

    echo "Installing all dependencies"
    pip3 install --upgrade pip
    python3 -m pip install -r requirements.txt 
    echo -e "\n"

    # raw_data and raw_noise -> augmented and augmented_noise
    echo "Augmenting data"
    python3 augment_data.py raw_data/normal/ raw_data/abnormal/ augmented/normal/ augmented/abnormal/
    python3 augment_data.py raw_noise/ raw_data/normal/ augmented_noise/ garbage/
    rm -rf garbage/ 
    echo -e "\n"

    # augmented -> noisified
    echo "Noisifying data"
    python3 noisify.py augmented_noise/ augmented/normal/ noisified/normal/ "-6"
    python3 noisify.py augmented_noise/ augmented/abnormal/ noisified/abnormal/ "-6"
    echo -e "\n"

    # noisified -> spectrograms
    echo "Making spectrograms"
    python3 make_spectrograms.py noisified/normal/ spectrograms/normal/
    python3 make_spectrograms.py noisified/abnormal/ spectrograms/abnormal/
    echo -e "\n"

    # final_images -> trained_model
    echo "Training"
    python3 train.py trained_model/ spectrograms/normal/ spectrograms/abnormal/ final_images/
    echo -e "\n"

    # trained_model and final_images -> vnnx and run tutorial
    echo "Prepare to convert to *.vnnx format"
    echo "Copy final_images into vnnx/data/ and model into vnnx/model/"
    mkdir vnnx
    mkdir vnnx/data
    mkdir vnnx/model
    cp -r final_images/test/normal/ vnnx/data/
    cp -r final_images/test/abnormal/ vnnx/data/ 
    cp -r trained_model/model_checkpoint/. vnnx/model/
    echo -e "\n"

    echo "Activate the VBX Environment"
    cd sdk
    source setup_vars.sh 
    cd $BASE_DIR
    echo -e "\n"

    echo "Copy (modified) tutorial scripts"
    cp sdk/tutorials/onnx/uw_2d_tuned/README.md vnnx/ 
    cp sdk/tutorials/onnx/uw_2d_tuned/uw_tutorial_tuned.sh vnnx/ 
    cp sdk/tutorials/onnx/uw_2d_tuned/uw_run_model.py vnnx/ 
    echo -e "\n"

    echo "Copy Sample Data"
    mkdir vnnx/sample_data
    find "$(pwd)/final_images/test/normal/"| grep ".*\.jpg"| shuf --head-count=48 | xargs cp --backup=numbered  -t vnnx/sample_data/
    find "$(pwd)/final_images/test/abnormal/"| grep ".*\.jpg"| shuf --head-count=48 | xargs cp --backup=numbered  -t vnnx/sample_data/
    echo -e "\n"

    echo "Run the tutorial"
    cd vnnx 
    bash uw_tutorial_tuned.sh 
    cd $BASE_DIR  
    echo -e "\n"

    echo "Now, we will prepare the archives for the icicle kit"
    unzip icicle-kit-simulator
    mv icicle-kit-simulator icicle 
    cp simulator_scripts/check_all.py icicle/example/sim-c/
    cp simulator_scripts/sim-run-model.cpp icicle/example/sim-c/
    cp simulator_scripts/README.md icicle/example/sim-c/ 
    curl -s https://codeload.github.com/PetteriAimonen/libfixmath/zip/master > icicle/example/sim-c/libfixmath-master.zip
    cp -r vnnx/example.tuned.vnnx icicle/example/sim-c/graph.vnnx
    cp -r vnnx/data/ icicle/example/sim-c/ 
    
    echo "Transferring the files to the icicle-kit"
    scp -r icicle root@192.168.0.103:/home/root/

    echo "Finally, let us run the VBX simulator (desktop version) on the test data"
    cp -r vnnx/data/ sdk/example/sim-c/ 
    cp vnnx/example.tuned.vnnx sdk/example/sim-c/graph.vnnx 
    cp simulator_scripts/check_all.py sdk/example/sim-c/
    cp simulator_scripts/sim-run-model.cpp sdk/example/sim-c/
    cd sdk/example/sim-c/ 
    python3 check_all.py graph.vnnx data/normal/ data/abnormal/
    
} 2>&1 | tee -a "log.txt"

echo "Log has been saved into log.txt"
