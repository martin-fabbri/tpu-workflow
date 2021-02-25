import argparse
import json
import re

import numpy as np
import tensorflow as tf


def get_dataset_path(gcs_path, image_size):
    flowers_datasets = {
        192: f"{gcs_path}/tfrecords-jpeg-192x192/*.tfrec",
        224: f"{gcs_path}/tfrecords-jpeg-224x224/*.tfrec",
        331: f"{gcs_path}/tfrecords-jpeg-331x331/*.tfrec",
        512: f"{gcs_path}/tfrecords-jpeg-512x512/*.tfrec",
    }
    return flowers_datasets[image_size]


def count_data_items(filenames):
    n = [
        int(re.compile(r"-([0-9]*)\.").search(filename).group(1))
        for filename in filenames
    ]
    return int(np.sum(n))


def split(
    gcs_path,
    train_split_path,
    val_split_path,
    images_count_path,
    validation_split,
    image_size,
):
    print("split", gcs_path)

    dataset_path = get_dataset_path(gcs_path, image_size)
    # dataset_path = gcs_path
    filenames = tf.io.gfile.glob(dataset_path)
    split = len(filenames) - int(len(filenames) * validation_split)
    training_filenames = filenames[:split]
    validation_filenames = filenames[split:]
    with open(train_split_path, "w") as train_split_json, open(
        val_split_path, "w"
    ) as val_split_json:
        json.dump(training_filenames, train_split_json)
        json.dump(validation_filenames, val_split_json)

    images_count = {
        "num_training_images": count_data_items(training_filenames),
        "num_validation_images": count_data_items(validation_filenames),
    }
    print("images count:", images_count["num_training_images"])
    print("validation images: ", images_count["num_validation_images"])
    with open(images_count_path, "w") as images_count_json:
        json.dump(images_count, images_count_json)

    print("training split saved at: data/interim/train_split.json")
    print("validation split saved at: data/interim/val_split.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gcs-path",
        dest="gcs_path",
        required=True,
        help="GCP bucket containing the dataset",
    )
    parser.add_argument(
        "--train-split-path",
        dest="train_split_path",
        required=True,
        help="Train split relative path",
    )
    parser.add_argument(
        "--val-split-path",
        dest="val_split_path",
        required=True,
        help="Validation split relative path",
    )
    parser.add_argument(
        "--images-count-path",
        dest="images_count_path",
        required=True,
        help="Training and Validation images count",
    )
    parser.add_argument(
        "--validation-split",
        dest="validation_split",
        type=float,
        required=False,
        help="Validation split",
        default=0.19,
    )
    parser.add_argument(
        "--image-size",
        dest="image_size",
        type=int,
        required=False,
        help="Dataset samples image size",
        default=512,
    )
    args = parser.parse_args()
    split(
        args.gcs_path,
        args.train_split_path,
        args.val_split_path,
        args.images_count_path,
        args.validation_split,
        args.image_size,
    )
