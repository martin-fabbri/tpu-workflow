if __name__ == "__main__":
    import os
    import sys
    from pathlib import Path

    path = Path(os.getcwd())
    src_path = str(path / "src")
    sys.path.append(src_path)

import argparse
import json

import numpy as np
import tensorflow as tf
from models.xception_ft_model import create_xception_ft_model
from tensorflow.keras.callbacks import LearningRateScheduler

from pipeline.train_utils import (
    MapDict,
    get_training_dataset,
    get_validation_dataset,
    load_dataset,
    lrfn,
    to_float32,
)


def initialize_tpu_connection():
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
    except ValueError:
        strategy = tf.distribute.MirroredStrategy()
    print("Number of accelerators", strategy.num_replicas_in_sync)
    return strategy


def create_model(strategy, image_size, num_classes):
    with strategy.scope():
        model = create_xception_ft_model(image_size, num_classes)
    print(model.summary())
    return model


def samples_split_files(train_split_path, val_split_path):
    train_split_path = str(path / train_split_path)
    val_split_path = str(path / val_split_path)

    with open(train_split_path, "r") as train_split_file, open(
        val_split_path, "r"
    ) as val_split_file:
        train_split_files = json.load(train_split_file)
        val_split_files = json.load(val_split_file)
    return train_split_files, val_split_files


def samples_count(images_split_count_path):
    with open(images_split_count_path, "r") as images_split_count_file:
        images_split_count = json.load(images_split_count_file)

    num_training_images = images_split_count["num_training_images"]
    num_validation_images = images_split_count["num_validation_images"]
    return num_training_images, num_validation_images


def save_training_plot_series(history, epochs, train_plot_path):
    num_epoch = [i + 1 for i in range(epochs)]
    with open(train_plot_path, "w") as fplot:
        json.dump(
            {
                "avls": [
                    {
                        "epoch": e,
                        "accuracy": a,
                        "val_accuracy": va,
                        "loss": l,
                        "val_loss": vl,
                    }
                    for e, a, va, l, vl in zip(
                        num_epoch,
                        history.history["accuracy"][1:],
                        history.history["val_accuracy"][1:],
                        history.history["loss"][1:],
                        history.history["val_loss"][1:],
                    )
                ]
            },
            fplot,
        )


def train(
    lr,
    classes_path,
    image_size,
    epochs,
    batch_size,
    train_split_path,
    val_split_path,
    images_count_path,
    saved_model_path,
    train_plot_path,
):
    strategy = initialize_tpu_connection()

    batch_size = batch_size * strategy.num_replicas_in_sync
    lr.max = lr.max * strategy.num_replicas_in_sync

    with open(classes_path, "r") as classes_file:
        classes = json.load(classes_file)
    num_classes = len(classes)
    model = create_model(strategy, image_size, num_classes)

    train_split_files, val_split_files = samples_split_files(
        train_split_path, val_split_path
    )

    training_dataset = get_training_dataset(train_split_files, image_size, batch_size)
    training_dataset = training_dataset.map(to_float32)

    validation_dataset = get_validation_dataset(val_split_files, image_size, batch_size)
    validation_dataset = validation_dataset.map(to_float32)

    num_training_images, num_validation_images = samples_count(images_count_path)
    print("Number of training images:", num_training_images)
    print("Number of validation images:", num_validation_images)

    train_steps = num_training_images // batch_size

    lr_callback = LearningRateScheduler(lambda epoch: lrfn(lr, epoch), verbose=True)

    history = model.fit(
        training_dataset,
        validation_data=validation_dataset,
        steps_per_epoch=train_steps,
        epochs=epochs,
        callbacks=[lr_callback],
    )

    model.save(saved_model_path)
    save_training_plot_series(history, epochs, train_plot_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lr-start",
        dest="lr_start",
        type=float,
        required=False,
        help="Learning rate start value",
        default=0.00001,
    )
    parser.add_argument(
        "--lr-max",
        dest="lr_max",
        type=float,
        required=False,
        help="Learning rate max factor will be affected by num of replicas",
        default=0.00005,
    )
    parser.add_argument(
        "--lr-min",
        dest="lr_min",
        type=float,
        required=False,
        help="Learning rate min factor will be affected by num of replicas",
        default=0.00001,
    )
    parser.add_argument(
        "--lr-rampup-epochs",
        dest="lr_rampup_epochs",
        type=int,
        required=False,
        help="Defines the epoch num after which the lr will start to ramp up",
        default=5,
    )
    parser.add_argument(
        "--lr-sustain-epochs",
        dest="lr_sustain_epochs",
        type=int,
        required=False,
        help="Learning rate sustain epochs",
        default=0,
    )
    parser.add_argument(
        "--lr-exp-decay",
        dest="lr_exp_decay",
        type=float,
        required=False,
        help="Learning rate exp decay",
        default=0.8,
    )
    parser.add_argument(
        "--classes-path",
        dest="classes_path",
        type=str,
        required=True,
        help="Domain classes definition path",
    )
    parser.add_argument(
        "--image-size",
        dest="image_size",
        type=int,
        required=False,
        help="Image size",
        default=331,
    )
    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        type=int,
        required=False,
        help="Batch size",
        default=10,
    )
    parser.add_argument(
        "--epochs",
        dest="epochs",
        type=int,
        required=True,
        help="Training epochs",
    )
    parser.add_argument(
        "--train-split-path",
        dest="train_split_path",
        type=str,
        required=True,
        help="Training split files",
    )
    parser.add_argument(
        "--val-split-path",
        dest="val_split_path",
        type=str,
        required=True,
        help="Validation split files",
    )
    parser.add_argument(
        "--images-count-path",
        dest="images_count_path",
        type=str,
        required=True,
        help="Number of images on train and val",
    )
    parser.add_argument(
        "--saved-model-path",
        dest="saved_model_path",
        type=str,
        required=True,
        help="Saved model path",
    )
    parser.add_argument(
        "--train-plot-path",
        dest="train_plot_path",
        type=str,
        required=True,
        help="Train plot metrics series",
    )
    args = parser.parse_args()
    lr = MapDict(
        {
            "start": args.lr_start,
            "max": args.lr_max,
            "min": args.lr_min,
            "rampup_epochs": args.lr_rampup_epochs,
            "sustain_epochs": args.lr_sustain_epochs,
            "exp_decay": args.lr_exp_decay,
        }
    )
    train(
        lr,
        args.classes_path,
        args.image_size,
        args.epochs,
        args.batch_size,
        args.train_split_path,
        args.val_split_path,
        args.images_count_path,
        args.saved_model_path,
        args.train_plot_path,
    )
