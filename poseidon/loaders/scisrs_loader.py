import json
import os
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

# For testing
#from ml_collections.config_flags import config_flags
from configs.scisrs import get_config

#config_flags.DEFINE_config_file("config", "configs/scisrs.py")

#FLAGS = flags.FLAGS


def parse_annotation_file(annotation_filename:str) -> Tuple[np.ndarray,np.ndarray]:


    # First entry in a line is the label, other entries are bbox coordinates
    labels = []
    bboxes = []
    with open(annotation_filename) as f:
        for line in f:
            line = line.split()
            #array.append([float(x) for x in line.split()]) 
            labels.append(int(line[0]))
            bboxes.append([float(x) for x in line[1:]])

    return (np.array(bboxes),np.array(labels))


def load_scisrs_dataset(config, csv_path):
    
    base_path = config.data_path # "data/scisrs/sig_images/yolo_images_dataset"

    # Load the given csv that contains image and annotation filenames
    df = pd.read_csv(csv_path)

    # Add complete paths to image and annotation files
    df["image_path"] = base_path + "/" + df["image_filename"].astype(str)
    df["annotation_path"] = base_path + "/" + df["annotation_filename"].astype(str)

    image_paths = df["image_path"]
    annotation_paths = df["annotation_path"]

    # testing
    # img = cv2.imread(image_paths[0])
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    # img /= 255.0
    # cv2.imshow('test',img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # annotation = parse_annotation_file(annotation_paths[0])
    # yolo_bboxes = annotation[0]
    # labels = annotation[1]

    # bboxes = np.zeros(yolo_bboxes.shape)
    # print(yolo_bboxes,labels,bboxes)
    # print(img.shape,bboxes.shape,labels.shape)

    def dataset_generator():
        for image_path, annotation_path in zip(image_paths,annotation_paths):
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
            #img /= 255.0 # It loooks like this might be done in a preprocessing stage

            annotation = parse_annotation_file(annotation_path)
            yolo_bboxes = annotation[0]
            labels = annotation[1]

            bboxes = np.zeros(yolo_bboxes.shape)
            bboxes[:,0] = yolo_bboxes[:,1] - yolo_bboxes[:,3]/2 # y1 = y_center - h/2
            bboxes[:,1] = yolo_bboxes[:,0] - yolo_bboxes[:,2]/2 # x1 = x_center - w/2
            bboxes[:,2] = yolo_bboxes[:,1] + yolo_bboxes[:,3]/2 # y2 = y_center + h/2
            bboxes[:,3] = yolo_bboxes[:,0] + yolo_bboxes[:,2]/2 # x2 = x_center + w/2

            yield (img, bboxes, labels)

    return tf.data.Dataset.from_generator(
        dataset_generator,
        output_signature=(
            tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
        ),
    ), len(df)

            




# Parse annotation file for bboxes and labels

# Get dataframe with columns that contain image and annotation filenames

# Loop through all filenames with zip

# Get the image

# Parse the annotations for bboxes and labels

# Convert bboxes from [x_center,y_center,w,h] to [y,x,y2,x2] (normalized)

# yield (img, bbox, label)


if __name__ == "__main__":
    annotation_filename = "/home/wcsng-30/Documents/poseidon/data/scisrs/sig_images/yolo_images_dataset/yolo_data_20220404_1_1.txt"
    print(parse_annotation_file(annotation_filename))

    config = get_config()
    csv_path = config.custom_path + "/train.csv"
    print(config)
    print(csv_path)

    load_scisrs_dataset(config,csv_path)

    