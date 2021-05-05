
import matplotlib as mpl 
mpl.use("Agg")
import matplotlib.pyplot as plt

import sklearn
from shutil import copyfile
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
import os
import argparse
from pathlib import Path
import pickle
import math

from constants import *
from common import *

DATA_FOLDER = os.path.join(os.getcwd(), "data")
TRAIN_FOLDER = os.path.join(DATA_FOLDER, "train")
TEST_FOLDER = os.path.join(DATA_FOLDER, "test")
VALIDATION_FOLDER = os.path.join(DATA_FOLDER, "validation")

def read_spectrogram_paths_and_shuffle(normal_dir, abnormal_dir):
    spectrogram_paths = []
    labels = []

    print_log("Loading spectrogram paths from {}".format(normal_dir))
    for path in Path(normal_dir).rglob('*.jpg'):
        spectrogram_paths.append(path)
        labels.append(NORMAL_VALUE)
    print_log("Loaded spectrogram paths from {}".format(normal_dir))
    print_log("len(spectrogram_paths)={}".format(len(spectrogram_paths)))
    print_log("len(labels)={}".format(len(labels)))

    print_log("Loading spectrogram paths from {}".format(abnormal_dir))
    for path in Path(abnormal_dir).rglob('*.jpg'):
        spectrogram_paths.append(path)
        labels.append(ABNORMAL_VALUE)
    print_log("Loaded spectrogram paths from {}".format(abnormal_dir))
    print_log("len(spectrogram_paths)={}".format(len(spectrogram_paths)))
    print_log("len(labels)={}".format(len(labels)))

    spectrogram_paths = np.array(spectrogram_paths)
    labels = np.array(labels)
    print_log("spectrogram_paths.shape={}".format(spectrogram_paths.shape))
    print_log("labels.shape={}".format(labels.shape))

    print_log("Will shuffle!")
    spectrogram_paths, labels = unison_shuffled_copies(spectrogram_paths, labels)
    print_log("Shuffled!")
    print_log("spectrogram_paths.shape={}".format(spectrogram_paths.shape))
    print_log("labels.shape={}".format(labels.shape))
    
    return spectrogram_paths, labels

def do_split(spectrogram_paths, labels):
    print_log("Will split, using train/val/test splits of {}/{}/{}".format(TRAINING_SPLIT, VALIDATION_SPLIT, TESTING_SPLIT))
    spectrogram_paths_train, spectrogram_paths_test, labels_train, labels_test = sklearn.model_selection.train_test_split(
        spectrogram_paths, labels, 
        train_size=TRAINING_SPLIT+VALIDATION_SPLIT,
        shuffle=True
    )

    spectrogram_paths_train, spectrogram_paths_val, labels_train, labels_val = sklearn.model_selection.train_test_split(
        spectrogram_paths_train, labels_train,
        train_size= TRAINING_SPLIT / (TRAINING_SPLIT+VALIDATION_SPLIT),
        shuffle=True
    )

    print_log("spectrogram_paths_train.shape={}".format(spectrogram_paths_train.shape))
    print_log("labels_train.shape={}".format(labels_train.shape))

    print_log("spectrograms_paths_val.shape={}".format(spectrogram_paths_val.shape))
    print_log("labels_val.shape={}".format(labels_val.shape))
    
    print_log("spectrogram_paths_test.shape={}".format(spectrogram_paths_test.shape))
    print_log("labels_test.shape={}".format(labels_test.shape))

    return spectrogram_paths_train, spectrogram_paths_test, labels_train, labels_test, spectrogram_paths_val, labels_val

def from_spectrograms_to_data(normal_dir, abnormal_dir):
    spectrogram_paths, labels = read_spectrogram_paths_and_shuffle(normal_dir, abnormal_dir)

    spectrogram_paths_train, spectrogram_paths_test, labels_train, labels_test, spectrogram_paths_val, labels_val = do_split(
        spectrogram_paths, labels
    )

    copy_split_to_data(spectrogram_paths_train, spectrogram_paths_test, spectrogram_paths_val, 
                          labels_train, labels_test, labels_val)

def copy_split_to_data(spectrogram_paths_train, spectrogram_paths_test, spectrogram_paths_val, 
                      labels_train, labels_test, labels_val):
    clear_previous_data()
    copy_split_to_folders(spectrogram_paths_train, labels_train, TRAIN_FOLDER)
    copy_split_to_folders(spectrogram_paths_test, labels_test, TEST_FOLDER)
    copy_split_to_folders(spectrogram_paths_val, labels_val, VALIDATION_FOLDER)

def copy_split_to_folders(spectrogram_paths, labels, folder):
    print_log("Copying splits")
    for i in range(len(spectrogram_paths)):
        assert labels.shape == spectrogram_paths.shape

        if labels[i] == NORMAL_VALUE:
            dest = os.path.join(folder, "normal", "{}.jpg".format(i))
        else:
            dest = os.path.join(folder, "abnormal", "{}.jpg".format(i))

        copyfile(spectrogram_paths[i], dest)

        if i % 100 == 0:
            print_log("Heartbeat -- Copied {} to {}".format(spectrogram_paths[i], dest))

    print_log("Finished copying splits")

def get_iterators():
    # Use generators, since we care for our memory system :D 
    datagen = ImageDataGenerator(preprocessing_function=model_1_my_preprocessing_func)

    print_log("Creating iterators (Train/Val/Test)")
    model_1_train_it = datagen.flow_from_directory(
        TRAIN_FOLDER, 
        class_mode='binary', 
        batch_size=IMAGE_GENERATOR_BATCH_SIZE, 
        target_size=SPECTROGRAM_SIZE,
        color_mode="grayscale",
        classes=["normal", "abnormal"]
    )

    model_1_val_it = datagen.flow_from_directory(
        VALIDATION_FOLDER, 
        class_mode='binary', 
        batch_size=IMAGE_GENERATOR_BATCH_SIZE, 
        target_size=SPECTROGRAM_SIZE,
        color_mode="grayscale",
        classes=["normal", "abnormal"]
    )

    model_1_test_it = datagen.flow_from_directory(
        TEST_FOLDER, 
        class_mode='binary', 
        batch_size=IMAGE_GENERATOR_BATCH_SIZE, 
        target_size=SPECTROGRAM_SIZE,
        color_mode="grayscale",
        classes=["normal", "abnormal"]
    )

    print_log("Finished creating iterators")
    
    return model_1_train_it, model_1_val_it, model_1_test_it

def model_1_my_preprocessing_func(img):
    image = np.array(img)
    
    xImageMax = max(image.flatten())
    xImageMin = min(image.flatten())
    
    image = (image - xImageMin) / (xImageMax - xImageMin)
    
    return image

def clear_previous_data():
    print_log("Clearing *.jpg files from {}".format(DATA_FOLDER))
    count = 0
    for path in Path(DATA_FOLDER).rglob('*.jpg'):
        os.remove(path)

        if count % 100 == 0:
            print_log("Heartbeat -- removed {}".format(path))
        
        count += 1

    print_log("*.jpg files have been cleared from {}".format(DATA_FOLDER))

def get_model():
    ## Model #1
    model_1 = models.Sequential()
    model_1.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(288, 432, 1)))
    model_1.add(layers.MaxPooling2D((2, 2)))
    model_1.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model_1.add(layers.MaxPooling2D((2, 2)))
    model_1.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model_1.add(layers.Flatten())
    model_1.add(layers.Dense(64, activation='relu'))
    model_1.add(layers.Dense(1, activation='sigmoid'))
    model_1.summary()
    
    opt = tf.keras.optimizers.Adam()
    
    model_1.compile(optimizer=opt,
                    loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                    metrics=["accuracy", "AUC"])
    
    return model_1

def create_dirs():
    dir_path(DATA_FOLDER)

    dir_path(TRAIN_FOLDER)
    dir_path(os.path.join(TRAIN_FOLDER, "normal"))
    dir_path(os.path.join(TRAIN_FOLDER, "abnormal"))
    
    dir_path(TEST_FOLDER)
    dir_path(os.path.join(TEST_FOLDER, "normal"))
    dir_path(os.path.join(TEST_FOLDER, "abnormal"))

    dir_path(VALIDATION_FOLDER)
    dir_path(os.path.join(VALIDATION_FOLDER, "normal"))
    dir_path(os.path.join(VALIDATION_FOLDER, "abnormal"))

def summarize_training_results(history, output_model_dir):
    print_log("History: {}".format(history.history))

    plt.subplots(3, 1, figsize=(10, 10))
    plt.subplot(3, 1, 1)
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.plot(history.history["accuracy"], label="train accuracy")
    plt.plot(history.history["val_accuracy"], label="validation accuracy")
    plt.legend()
    plt.grid()

    plt.subplot(3, 1, 2)
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("loss")
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="validation loss")
    plt.legend()
    plt.grid()
    
    plt.subplot(3, 1, 3)
    plt.title("AUC")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.plot(history.history["auc"], label="auc")
    plt.plot(history.history["val_auc"], label="val_auc")
    plt.legend()
    plt.grid()
    
    plt.tight_layout()
    save_path = os.path.join(output_model_dir, "training.jpg")
    plt.savefig(save_path)

def save_object(object, filepath):
    with open(filepath, 'wb') as handle:
        pickle.dump(object, handle)

def save_history(output_model_dir, history):
    save_path = os.path.join(output_model_dir, "history.pckl")
    save_object(history, save_path)

def save_test_results(output_model_dir, model_1_results):
    save_path = os.path.join(output_model_dir, "test_results.pckl")
    save_object(model_1_results, save_path)

def save_confusion_matrix(output_model_dir, confusion_matrix):
    save_path = os.path.join(output_model_dir, "confusion_matrix.pckl")
    save_object(confusion_matrix, save_path)

def save_conf_matrix_metrics(output_model_dir, metrics):
    save_path = os.path.join(output_model_dir, "conf_matrix_metrics.pckl") 
    save_object(metrics, save_path)

def save_final_images(model, iterator, final_images_dir, batch_size=IMAGE_GENERATOR_BATCH_SIZE):
    print_log("Will collect Final Images -- Needed Later for the VNNX Model")
    n = batch_size
    labels = []
    images = []

    number_of_examples = len(iterator.filenames)
    number_of_generator_calls = math.ceil(number_of_examples / (1.0 * n))
        
    for i in range(0, int(number_of_generator_calls)):
        batchx, batchy = iterator.next()
        labels.extend(np.array(batchy))
        images.extend(np.array(batchx))

    labels = np.array(labels)
    images = np.array(images)

    print_log("labels.shape={}".format(labels.shape))
    print_log("images.shape={}".format(images.shape))
    print_log("Will start saving final images at {}".format(final_images_dir))

    for i in range(len(images)):
        image = images[i]
        label = labels[i]

        path = "example.jpg"
        if label == NORMAL_VALUE:
            path = os.path.join(final_images_dir, "normal", "{}.jpg".format(i))
        elif label == ABNORMAL_VALUE:
            path = os.path.join(final_images_dir, "abnormal", "{}.jpg".format(i))
        else:
            raise RuntimeError("A label is not normal not abnormal!")

        tf.keras.preprocessing.image.save_img(path, image)

        if i % 100 == 0:
            print_log("Heartbeat -- Just Saved image of shape {} at {}".format(image.shape, path))


def get_confusion_matrix(model, iterator, batch_size=IMAGE_GENERATOR_BATCH_SIZE):
    n = batch_size
    number_of_examples = len(iterator.filenames)
    number_of_generator_calls = math.ceil(number_of_examples / (1.0 * n)) 
    
    test_labels = []
    test_images = []

    for i in range(0,int(number_of_generator_calls)):
        batchx, batchy = iterator.next()
        test_labels.extend(np.array(batchy))
        test_images.extend(np.array(batchx))
    
    test_labels = np.array(test_labels)
    test_images = np.array(test_images)

    print_log(test_labels.shape)
    print_log(test_images.shape)

    test_predictions = model.predict(test_images)

    test_labels = test_labels > 0.5
    test_predictions = test_predictions > 0.5

    matrix = tf.math.confusion_matrix(test_labels, test_predictions)
    return matrix

def get_f_measure(confusion_matrix):
    TP = confusion_matrix[1][1]
    FP = confusion_matrix[0][1]
    TN = confusion_matrix[0][0]
    FN = confusion_matrix[1][0]
        
    recall = TP / (TP+FN)
    precision = TP / (TP+FP)
    F_measure = 2*recall*precision / (recall + precision)
    FNR = FN / (FN + TP)
    FPR = FP / (FP + TN)
    
    return TP, FP, TN, FN, recall, precision, F_measure, FNR, FPR

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_model_dir", type=dir_path)
    parser.add_argument("normal_dir", type=existing_dir_path)
    parser.add_argument("abnormal_dir", type=existing_dir_path)
    parser.add_argument("final_images_dir", type=dir_path)
    args = parser.parse_args()

    output_model_dir = args.output_model_dir 
    normal_dir = args.normal_dir 
    abnormal_dir = args.abnormal_dir 
    final_images_dir = args.final_images_dir

    print_log("output_model_dir={}".format(output_model_dir))
    print_log("normal_dir={}".format(normal_dir))
    print_log("abnormal_dir={}".format(abnormal_dir))
    print_log("final_images_dir={}".format(final_images_dir))

    create_dirs()

    from_spectrograms_to_data(normal_dir, abnormal_dir)

    model_1_train_it, model_1_val_it, model_1_test_it = get_iterators()
    
    model_1 = get_model() 
    final_images_dir_train = dir_path(os.path.join(final_images_dir, "train"))
    dir_path(os.path.join(final_images_dir, "train", "normal"))
    dir_path(os.path.join(final_images_dir, "train", "abnormal"))

    final_images_dir_test = dir_path(os.path.join(final_images_dir, "test"))
    dir_path(os.path.join(final_images_dir, "test", "normal"))
    dir_path(os.path.join(final_images_dir, "test", "abnormal"))

    final_images_dir_val = dir_path(os.path.join(final_images_dir, "validation"))
    dir_path(os.path.join(final_images_dir, "validation", "normal"))
    dir_path(os.path.join(final_images_dir, "validation", "abnormal"))

    save_final_images(model_1, model_1_train_it, final_images_dir_train, batch_size=IMAGE_GENERATOR_BATCH_SIZE)
    save_final_images(model_1, model_1_test_it, final_images_dir_test, batch_size=IMAGE_GENERATOR_BATCH_SIZE)
    save_final_images(model_1, model_1_val_it, final_images_dir_val, batch_size=IMAGE_GENERATOR_BATCH_SIZE)

    checkpoint_path = os.path.join(output_model_dir, "model_checkpoint")
    os.system(r"""mkdir {}""".format(checkpoint_path))

    early_stopping = EarlyStopping(patience=10, monitor='val_loss', verbose=2, mode='auto', restore_best_weights=True)
    checkpoint = ModelCheckpoint(   filepath=checkpoint_path, 
                                    monitor='val_loss', verbose=2, save_best_only=True, 
                                    save_best_weights_only=False, mode='auto', save_freq='epoch')

    callbacks = [early_stopping, checkpoint]

    model_1_history = model_1.fit(  model_1_train_it,
                                    validation_data=model_1_val_it,
                                    verbose=2,
                                    shuffle=True,
                                    epochs=50,
                                    callbacks=callbacks)
    summarize_training_results(model_1_history, output_model_dir)

    model_1_results = model_1.evaluate(model_1_test_it)
    
    confusion_matrix = np.array(get_confusion_matrix(model_1, model_1_test_it, IMAGE_GENERATOR_BATCH_SIZE))
    print_log("Confusion Matrix is \n{}".format(confusion_matrix))
    
    TP, FP, TN, FN, recall, precision, F_measure, FNR, FPR = get_f_measure(confusion_matrix)
    print_log("TP={}\nFP={}\nTN={}\nFN={}\nrecall={}\nprecision={}\nF_measure={}\nFNR={}\nFPR={}".format(
                    TP, FP, TN, FN, recall, precision, F_measure, FNR, FPR))
    
    os.system(r"""mkdir {}""".format(os.path.join(output_model_dir, "model_saved")))
    model_1.save(os.path.join(output_model_dir, "model_saved"))

    save_test_results(output_model_dir, model_1_results)
    save_confusion_matrix(output_model_dir, confusion_matrix)
    save_conf_matrix_metrics(output_model_dir, np.array([TP, FP, TN, FN, recall, precision, F_measure, FNR, FPR]))
    save_history(output_model_dir, model_1_history.history)

    predictions = model_1.predict(model_1_test_it)
    print_log("Predictions for the test iterator: {}".format(predictions))
    print_log("max(predictions)={}".format(max(predictions)))
    print_log("min(predictions)={}".format(min(predictions)))

if __name__ == "__main__":
    main()