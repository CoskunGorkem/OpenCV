'''
Script to train a classification model for traffic sign detection with German Traffic Sign Dataset.
    -achieves ~98% accuracy on test dataset
    -achieves ~99.90% accuracy on training dataset

Please download the data from here:https://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset
    -Downloaded data directory will be used as "data_dir" variable.

If you have more than 1 GPU, you can choose which one you would like to use for the training, if all need to be used
then comment os.environ["CUDA_VISIBLE_DEVICES"]


Example usage: python 02-German_traffic_sign_training.py --data_dir /Users/xx/Downloads/GTSRB --epochs 20
'''

import os
########################################
# Choose the GPU index to be used
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
########################################

import cv2
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import Image
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout



def main(data_dir, epochs):
    # Reading the input images and putting them into a numpy array
    data = []
    labels = []

    height = 30
    width = 30
    channels = 3
    classes = 43
    n_inputs = height * width * channels

    for i in range(classes):
        path = data_dir + "/Train/{0}/".format(i)
        Class = os.listdir(path)
        for a in Class:
            try:
                image = cv2.imread(path + a)
                image_from_array = Image.fromarray(image, 'RGB')
                size_image = image_from_array.resize((height, width))
                data.append(np.array(size_image))
                labels.append(i)
            except AttributeError:
                print("Error")

    Cells = np.array(data)
    labels = np.array(labels)

    # Randomize the order of the input images
    s = np.arange(Cells.shape[0])
    np.random.seed(43)
    np.random.shuffle(s)
    Cells = Cells[s]
    labels = labels[s]

    # Spliting the images into train and validation sets
    (X_train, X_val) = Cells[(int)(0.2 * len(labels)):], Cells[:(int)(0.2 * len(labels))]
    X_train = X_train.astype('float32') / 255
    X_val = X_val.astype('float32') / 255
    (y_train, y_val) = labels[(int)(0.2 * len(labels)):], labels[:(int)(0.2 * len(labels))]

    # Using one hot encoding for the train and validation labels

    y_train = to_categorical(y_train, 43)
    y_val = to_categorical(y_val, 43)

    # Definition of the DNN model
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=X_train.shape[1:]))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(43, activation='softmax'))

    # Compilation of the model
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    # using ten epochs for the training and saving the accuracy for each epoch

    history = model.fit(X_train, y_train, batch_size=32, epochs=epochs,
                        validation_data=(X_val, y_val))

    # Display of the accuracy and the loss values

    plt.figure(0)
    plt.plot(history.history['accuracy'], label='training accuracy')
    plt.plot(history.history['val_accuracy'], label='val accuracy')
    plt.title('Accuracy')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend()

    plt.figure(1)
    plt.plot(history.history['loss'], label='training loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.title('Loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()

    # Predicting with the test data
    y_test_path = os.path.join(data_dir, "Test.csv")
    y_test = pd.read_csv(y_test_path)

    labels = y_test['Path'].values
    y_test = y_test['ClassId'].values

    data = []

    for f in labels:
        label_path = os.path.join(data_dir, f)
        image = cv2.imread(label_path)
        image_from_array = Image.fromarray(image, 'RGB')
        size_image = image_from_array.resize((height, width))
        data.append(np.array(size_image))

    X_test = np.array(data)
    X_test = X_test.astype('float32') / 255
    pred = model.predict_classes(X_test)

    print("ACCURACY:", accuracy_score(y_test, pred))

    model_name = "gt_acc_" + str(accuracy_score(y_test, pred))[2:6] + "_epoch_" + str(epochs) + "_.h5"

    model.save(model_name)

    print("Training completed")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--epochs', type=int, required=True)

    args = parser.parse_args()
    main(args.data_dir, args.epochs)
