from __future__ import absolute_import, division, print_function, unicode_literals
import cv2
import os
import numpy as np
import re
from sklearn.model_selection import train_test_split
from tensorflow import keras
from sklearn import metrics
import matplotlib.pyplot as plt

IM_height = 640
IM_width = 480
RE_height = 224
RE_width = 224
output_dirname = "TUD_DATASET_OUTPUT"
input_dirname = "TUD_DATASET"
result_dirname = "RESULTS"
compare_dirname = "COMPARE"


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


def input_output_data(in_folder, out_folder):
    """
        Returns input and output dataset as batch size, image width, image height and pixels.
    """
    data_dict = read_idl(in_folder)
    BATCH_SIZE = len(data_dict)
    output_images = np.empty((BATCH_SIZE, RE_width, RE_height, 3), dtype='uint8')
    output_label = np.zeros((BATCH_SIZE, 4))

    output_image_paths = load_images_path_from_folder(out_folder)

    for n in range(0, BATCH_SIZE):
        img = cv2.imread(output_image_paths[n], cv2.IMREAD_COLOR)
        if img.shape != (RE_width, RE_height):
            img = cv2.resize(img, (RE_height, RE_width))
        output_images[n] = img

    count = 0
    for row in data_dict:
        list_tuple = row[1]
        if list_tuple != 0:
            for val in list_tuple:
                output_label[count][0] = val[0]
                output_label[count][1] = val[1]
                output_label[count][2] = val[2]
                output_label[count][3] = val[3]
            count = count + 1
        else:
            output_label[count][0] = 0
            output_label[count][1] = 0
            output_label[count][2] = 0
            output_label[count][3] = 0
            count = count + 1

    return output_images, output_label


def plot_box(img, Xmin_res, Ymin_res, Xmax_res, Ymax_res, color):
    """
           Plots the boxes using the coordinates of the image.
    """
    pt1 = (Xmin_res, Ymin_res)
    pt2 = (Xmax_res, Ymax_res)
    output_image_file = cv2.rectangle(img, pt1, pt2, color, 2)
    return output_image_file


output_images_hm, output_labels_hm = input_output_data(input_dirname, output_dirname)
X_train, X_test, y_train, y_test = train_test_split(output_images_hm, output_labels_hm, test_size=0.3, shuffle=False)
X_train, X_test, = X_train / 255.0, X_test / 255.0

json_file = open('model_regress.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = keras.models.model_from_json(loaded_model_json)
model.load_weights('model_regress.h5')

results = model.predict(X_test)
for iter_i in range(0, results.shape[0]):
    color_r = (225, 0, 0)
    im = X_test[iter_i]
    xmin_res = results[iter_i][0]
    ymin_res = results[iter_i][1]
    xmax_res = results[iter_i][2]
    ymax_res = results[iter_i][3]
    y = plot_box(im, xmin_res, ymin_res, xmax_res, ymax_res, color_r)
    filename = "Result%s.png" % iter_i
    cv2.imwrite(os.path.join(result_dirname, filename), y)

o_files = load_images_path_from_folder(result_dirname)

for iter_i in range(0, results.shape[0]):
    color_c = (0, 255, 0)
    scale1 = RE_height / IM_height
    scale2 = RE_width / IM_width
    xmin = int(y_test[iter_i][0] * scale1)
    ymin = int(y_test[iter_i][1] * scale2)
    xmax = int(y_test[iter_i][2] * scale1)
    ymax = int(y_test[iter_i][3] * scale2)
    im = cv2.imread(o_files[iter_i], cv2.IMREAD_COLOR)
    y = plot_box(im, xmin, ymin, xmax, ymax, color_c)
    filename = "Compare%s.png" % iter_i
    cv2.imwrite(os.path.join(compare_dirname, filename), y)

