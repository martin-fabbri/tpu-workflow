import argparse
import json

import tensorflow as tf


def get_dataset_path(gcs_path, image_size):
    flowers_datasets = {
        192: f"{gcs_path}/tfrecords-jpeg-192x192/*.tfrec",
        224: f"{gcs_path}/tfrecords-jpeg-224x224/*.tfrec",
        331: f"{gcs_path}/tfrecords-jpeg-331x331/*.tfrec",
        512: f"{gcs_path}/tfrecords-jpeg-512x512/*.tfrec",
    }
    return flowers_datasets[image_size]


def split(gcs_path, train_split_path, val_split_path, validation_split, image_size):
    print("split", gcs_path)

    dataset_path = get_dataset_path(gcs_path, image_size)
    filenames = tf.io.gfile.glob(dataset_path)
    split = len(filenames) - int(len(filenames) * validation_split)
    training_filenames = filenames[:split]
    validation_filenames = filenames[split:]
    with open(train_split_path, "w") as train_split_json, open(
        val_split_path, "w"
    ) as val_split_json:
        json.dump(training_filenames, train_split_json)
        json.dump(validation_filenames, val_split_json)
    # print("TRAINING IMAGES: ", count_data_items(TRAINING_FILENAMES), ", STEPS PER EPOCH: ", TRAIN_STEPS)
    # print("VALIDATION IMAGES: ", count_data_items(VALIDATION_FILENAMES))

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
        args.validation_split,
        args.image_size,
    )
