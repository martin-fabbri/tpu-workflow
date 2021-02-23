import argparse
import tensorflow as tf
import json
from kaggle.api.kaggle_api_extended import KaggleApi


def get_dataset_path(gcs_path, image_resolution):
    flowers_datasets = {
        192: f"{gcs_path}/tfrecords-jpeg-192x192/*.tfrec",
        224: f"{gcs_path}/tfrecords-jpeg-224x224/*.tfrec",
        331: f"{gcs_path}/tfrecords-jpeg-331x331/*.tfrec",
        512: f"{gcs_path}/tfrecords-jpeg-512x512/*.tfrec",
    }
    return flowers_datasets[image_resolution]


def split(gcs_path, validation_split, image_resolution):
    print("split", gcs_path)
    api = KaggleApi()
    api.authenticate()

    dataset_path = get_dataset_path(gcs_path, image_resolution)
    filenames = tf.io.gfile.glob(dataset_path)
    split = len(filenames) - int(len(filenames) * validation_split)
    training_filenames = filenames[:split]
    validation_filenames = filenames[split:]
    with open("train_split.json", "w") as train_split_json, open(
        "val_split.json", "w"
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
        "--batch-size",
        dest="batch_size",
        required=True,
        help="Batch size",
        default=16 * 8,
    )
    parser.add_argument(
        "--validation-split",
        dest="validation_split",
        required=False,
        help="Validation split",
        default=0.19,
    )
    parser.add_argument(
        "--image-resolution",
        dest="image_resolution",
        required=False,
        help="Image resolution",
        default=512,
    )
    args = parser.parse_args()
    split(args.gcs_path, args.validation_split, args.image_resolution)
