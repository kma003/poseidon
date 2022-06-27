import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
#import wandb
from absl import app
from absl import flags
from absl import logging
from ml_collections.config_flags import config_flags

import poseidon
from poseidon.utils import convert_to_corners
from poseidon.utils import convert_to_xywh
from poseidon.utils import swap_xy

from reef_net.loaders.scisrs_loader import load_scisrs_dataset

FLAGS = flags.FLAGS


config_flags.DEFINE_config_file("config", "configs/scisrs.py")


def visualize_bounding_boxes(img, bbox, category):
    bbox = swap_xy(bbox)  # Swap_xy makes this go Nan as of now I suppose
    image_shape = img.shape

    bbox = tf.stack(
        [
            bbox[:, 0] * image_shape[1],
            bbox[:, 1] * image_shape[0],
            bbox[:, 2] * image_shape[1],
            bbox[:, 3] * image_shape[0],
        ],
        axis=-1,
    )
    bbox = convert_to_xywh(bbox)
    bbox = convert_to_corners(bbox)
    bbox = bbox.numpy()
    for annotation in bbox:
        x1, y1, x2, y2 = annotation.astype(int)
        img = cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
        #img = cv2.rectangle(img, (x1, y1), (x2, y2), (1, 0, 0), 3)

    return img

def load_dataset(config):
    if config.dataset == 'scisrs':
        return poseidon.loaders.load_scisrs_dataset(config, f"{config.custom_path}/train.csv")
    if config.dataset == 'tensorflow-great-barrier-reef':
        return poseidon.loaders.load_reef_dataset(config, f"custom_csv/train.csv")
    else:
        raise ValueError(f"unsupported dataset {config.dataset}")

def main(args):
    config = FLAGS.config
    ds, dataset_size = load_dataset(config)
    ds = ds.shuffle(20)

    (image, bounding_boxes, category) = next(iter(ds.take(100)))
    image, bounding_boxes, category = (
        image.numpy(),
        bounding_boxes.numpy(),
        category.numpy(),
    )
    plt.imshow(image / 255.0)
    #plt.imshow(image)
    plt.axis("off")
    plt.show()
    print("Category", category)
    print("Image size", image.shape)
    print(category)

    image = visualize_bounding_boxes(image, bounding_boxes, category)
    plt.imshow(image / 255.0)
    #plt.imshow(image)
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    app.run(main)
