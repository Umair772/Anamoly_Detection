import glob
import os
import sys 
import argparse
import pathlib

NORMAL_VALUE = 0
ABNORMAL_VALUE = 1

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

def vnnx_file(string):
    filepath = os.path.abspath(string)
    if os.path.exists(filepath) and pathlib.Path(filepath).suffix == ".vnnx":
        return filepath 
    
    raise FileNotFoundError("Could not find {}".format(string))

def text_file(string):
    filepath = os.path.abspath(string)
    if os.path.exists(filepath) and pathlib.Path(filepath).suffix == ".txt":
        return filepath 
    
    raise FileNotFoundError("Could not find {}".format(string))

def execute_cmd(cmd):
    print_log("Executing {}".format(cmd))
    os.system(cmd)

def get_predictions(log, correct_value):
    logfile = open(log, "r")
    lines = logfile.readlines()

    correct = 0
    total = 0

    for line in lines:
        if "[LOG SCORE]=" in line:
            score = float(line.split("=")[-1])
            
            if correct_value == NORMAL_VALUE:
                if score < 0.5:
                    correct += 1
            else:
                if score >= 0.5:
                    correct += 1
            total += 1
    
    return correct, total

def get_f_measure(TN, FP, FN, TP):
    recall = TP / (TP+FN)
    precision = TP / (TP+FP)
    F_measure = 2*recall*precision / (recall + precision)
    FNR = FN / (FN + TP)
    FPR = FP / (FP + TN)
    
    return recall, precision, F_measure, FNR, FPR

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=vnnx_file)
    parser.add_argument("normal_dir", type=existing_dir_path)
    parser.add_argument("abnormal_dir", type=existing_dir_path)
    args = parser.parse_args()

    model_dir = args.model_dir
    normal_dir = args.normal_dir
    abnormal_dir = args.abnormal_dir

    print_log("model_dir={}".format(model_dir))
    print_log("normal_dir={}".format(normal_dir))
    print_log("abnormal_dir={}".format(abnormal_dir))

    normal_files = []
    for file in glob.glob(os.path.join(normal_dir, "*.jpg")):
        normal_files.append(file)

    abnormal_files = []
    for file in glob.glob(os.path.join(abnormal_dir, "*.jpg")):
        abnormal_files.append(file)

    execute_cmd(r"""./build.sh""")

    execute_cmd(r"""rm -rf abnormal_log.txt""")
    execute_cmd(r"""rm -rf normal_log.txt""")

    for file in normal_files:
        execute_cmd(r"""./sim-run {} {} >> normal_log.txt""".format(model_dir, file))

    for file in abnormal_files:
        execute_cmd(r"""./sim-run {} {} >> abnormal_log.txt""".format(model_dir, file))

    normal_log = text_file("normal_log.txt")
    abnormal_log = text_file("abnormal_log.txt")
    correct_normal, total_normal = get_predictions(normal_log, NORMAL_VALUE)
    correct_abnormal, total_abnormal = get_predictions(abnormal_log, ABNORMAL_VALUE)

    TP = correct_abnormal
    TN = correct_normal 

    FP = total_normal - correct_normal 
    FN = total_abnormal - correct_abnormal 

    print_log("confusion_matrix=\n[[{}, {}],\n [{}, {}]]".format(TN, FP, FN, TP))

    recall, precision, F_measure, FNR, FPR = get_f_measure(TN, FP, FN, TP)

    print_log("TP={}".format(TP))
    print_log("FP={}".format(FP))
    print_log("TN={}".format(TN))
    print_log("FN={}".format(FN))
    print_log("recall={}".format(recall))
    print_log("precision={}".format(precision))
    print_log("F_measure={}".format(F_measure))
    print_log("FNR={}".format(FNR))
    print_log("FPR={}".format(FPR))
    print_log("Accuracy={}".format((correct_normal + correct_abnormal) / (total_normal + total_abnormal)))



if __name__ == "__main__":
    main()