import os
import sys

import numpy as np
import yaml
from tensorflow.keras import applications
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm.keras import TqdmCallback

from config import config

with open(config.PARAMS, "r") as fd:
    params = yaml.safe_load(fd)

nb_validation_samples = params["train"]["nb_validation_samples"]
epochs = params["train"]["epochs"]
batch_size = params["train"]["batch_size"]

# dimensions of our images.
img_width, img_height = 150, 150

top_model_weights_path = "model.h5"
train_data_dir = config.TRAIN_DATA_DIR
validation_data_dir = config.VALIDATION_DATA_DIR
cats_train_path = config.CATS_TRAIN_DATA_DIR
nb_train_samples = 2 * len(
    [
        name
        for name in os.listdir(cats_train_path)
        if os.path.isfile(os.path.join(cats_train_path, name))
    ]
)


def save_bottlebeck_features():
    datagen = ImageDataGenerator(rescale=1.0 / 255)

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights="imagenet")

    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False,
    )
    bottleneck_features_train = model.predict_generator(
        generator, nb_train_samples // batch_size
    )
    np.save(open("bottleneck_features_train.npy", "wb"), bottleneck_features_train)

    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False,
    )
    bottleneck_features_validation = model.predict_generator(
        generator, nb_validation_samples // batch_size
    )
    np.save(
        open("bottleneck_features_validation.npy", "wb"), bottleneck_features_validation
    )


def train_top_model():
    train_data = np.load(open("bottleneck_features_train.npy", "rb"))
    train_labels = np.array(
        [0] * (int(nb_train_samples / 2)) + [1] * (int(nb_train_samples / 2))
    )

    validation_data = np.load(open("bottleneck_features_validation.npy", "rb"))
    validation_labels = np.array(
        [0] * (int(nb_validation_samples / 2)) + [1] * (int(nb_validation_samples / 2))
    )

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])

    model.fit(
        train_data,
        train_labels,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(validation_data, validation_labels),
        verbose=0,
        callbacks=[TqdmCallback(), CSVLogger("metrics.csv")],
    )
    model.save_weights(top_model_weights_path)


save_bottlebeck_features()
train_top_model()
