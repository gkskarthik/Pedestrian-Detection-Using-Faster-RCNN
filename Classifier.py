from __future__ import absolute_import, division, print_function, unicode_literals
import cv2
import os
import numpy as np
import re
from sklearn.model_selection import train_test_split
from tensorflow import keras
import datetime


IM_height = 640
IM_width = 480
RE_height = 224
RE_width = 224
output_dirname = "TUD_DATASET_OUTPUT"
input_dirname = "TUD_DATASET"


def read_idl_paths(path):
    """
        Returns a list paths for idl file inside the DATASET folder.
    """
    files = []
    for r, d, f in os.walk(path):
        for file in f:
            if '.idl' in file:
                files.append(os.path.join(r, file))
    return files


def load_images_path_from_folder(path):
    """
        Returns a list paths for images inside the DATASET folder.
    """
    files = []
    for r, d, f in os.walk(path):
        for file in f:
            if '.png' in file:
                files.append(os.path.join(r, file))
    return files


def read_idl(folder):
    """
        Returns a list of tuples. The first element of a tuple is the full path of the image. The second element of the tuple is 0, if there are no people in the image. If there are people in the image, then the second element of the tuple is a list of tuples. Each tuple is of length 4 (Xmin, Ymin, Xmax, Ymax). So, number of elements in the list would be equal to the number of people annotated in the image.
    """
    paths = read_idl_paths(folder)
    output = []
    noann_regex = re.compile('"(.+)"[;.]')
    name_regex = re.compile('"(.+)":')
    tud_regex = re.compile('(\d+),\s+(\d+),\s+(\d+),\s+(\d+)')

    for i in paths:
        fid = open(i, 'r')
        for line in fid:
            line = line.strip()
            if re.match(noann_regex, line):
                name = list(map(str, re.findall(noann_regex, line)))[0]
                output.append((name, 0))
            else:
                name = list(map(str, re.findall(name_regex, line)))[0]
                annotations = re.findall(tud_regex, line)
                annotations = list(map(lambda x: tuple(map(int, list(x))), annotations))
                output.append((name, annotations))
    return output


def input_output_data(in_folder):
    """
        Returns input and output dataset as batch size, image width, image height and pixels.
    """
    file_path_list = []
    data_dict = read_idl(in_folder)
    BATCH_SIZE = len(data_dict)
    input_images = np.empty((BATCH_SIZE, RE_width, RE_height, 3), dtype='uint8')
    output_label = np.zeros((BATCH_SIZE, 1))

    for row in data_dict:
        file_name = row[0]
        temp_file = os.path.join(in_folder, file_name)
        file_path_list.append(temp_file)

    for n in range(0, BATCH_SIZE):
        img = cv2.imread(file_path_list[n], cv2.IMREAD_COLOR)
        if img.shape != (RE_width, RE_height):
            img = cv2.resize(img, (RE_height, RE_width))
        input_images[n] = img

    count = 0
    for row in data_dict:
        file_name = row[0]
        list_tuple = row[1]
        if list_tuple != 0:
            y = (plot_output_box(file_name, list_tuple, IM_width, IM_height))
            cv2.imwrite(os.path.join(output_dirname, file_name), y)
            output_label[count] = 1
            count = count + 1
        else:
            temp_file = os.path.join(in_folder, file_name)
            y = cv2.imread(temp_file, cv2.IMREAD_COLOR)
            cv2.imwrite(os.path.join(output_dirname, file_name), y)
            output_label[count] = 0
            count = count + 1

    return input_images, output_label


def plot_output_box(file, tuple_boxes_list, inp_width, inp_height):
    """
       Plots the boxes using the coordinates of the image.
    """
    temp_file = os.path.join(input_dirname, file)
    output_image_file = cv2.imread(temp_file, cv2.IMREAD_COLOR)
    w_scale = inp_width / IM_width
    h_scale = inp_height / IM_height
    for boxes in tuple_boxes_list:
        Xmin = int(boxes[0])
        Ymin = int(boxes[1])
        Xmax = int(boxes[2])
        Ymax = int(boxes[3])
        pt1 = (int(Xmin * w_scale), int(Ymin * h_scale))
        pt2 = (int(Xmax * w_scale), int(Ymax * h_scale))
        cv2.rectangle(output_image_file, pt1, pt2, (255, 0, 0), 2)

    return output_image_file


def classification_train_model():
    """
        Trains the model for classification.
    """
    model_class = keras.models.Sequential()

    # Layer 1 convolution, max-pooling
    model_class.add(keras.layers.Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), padding='valid', activation='relu', input_shape=(224, 224, 3)))
    model_class.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    # Layer 2 convolution, max-pooling
    model_class.add(keras.layers.Conv2D(filters=256, kernel_size=(11, 11), strides=(1, 1), padding='valid', activation='relu'))
    model_class.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    # Layer 3 convolution, max-pooling
    model_class.add(keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu'))

    # Layer 4 convolution, max-pooling
    model_class.add(keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu'))

    # Layer 5 convolution, max-pooling
    model_class.add(keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu'))
    model_class.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    # Flattening the layer
    model_class.add(keras.layers.Flatten())

    model_class.add(keras.layers.Dense(4096, activation='relu', input_shape=(224*224*3,)))
    model_class.add(keras.layers.Dropout(0.4))

    model_class.add(keras.layers.Dense(4096, activation='relu'))
    model_class.add(keras.layers.Dropout(0.4))

    model_class.add(keras.layers.Dense(1000, activation='relu'))
    model_class.add(keras.layers.Dropout(0.4))

    model_class.add(keras.layers.Dense(2, activation='softmax'))

    model_class.summary()

    return model_class


def classifier(Xtrain, ytrain, Xtest, ytest):
    """
        Runs the classifier for classifying the classes.
    """
    model_classifier = classification_train_model()
    rms = keras.optimizers.RMSprop(lr=0.001, decay=1e-6)
    model_classifier.compile(optimizer=rms, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    log_dir = os.path.join(
        "logs",
        "fit",
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
    )
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    model_classifier.fit(Xtrain, ytrain, epochs=25, batch_size=50, callbacks=[tensorboard_callback], validation_data=(Xtest, ytest))

    # Importing the model to json file for future use
    model_json = model_classifier.to_json()
    with open("model_class.json", "w") as json_file:
        json_file.write(model_json)
    model_classifier.save_weights("model_class.h5")


output_images_hm, output_label_hm = input_output_data(input_dirname)
X_train, X_test, y_train, y_test = train_test_split(output_images_hm, output_label_hm, test_size=0.3, shuffle=False)
X_train, X_test, = X_train / 255.0, X_test / 255.0
classifier(X_train, y_train, X_test, y_test)
