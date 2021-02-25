from enum import auto
from numpy.lib.npyio import load

import tensorflow as tf

AUTO = tf.data.experimental.AUTOTUNE


class MapDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError as k:
            raise AttributeError(k)


def lrfn(lr, epoch):
    """
    Learning rate schedule using ramp up strategy.

    Fine-tunning requires to start with low learning rates
    to avoid "erasing" the loaded weights.
    """
    if epoch < lr.rampup_epochs:
        lr = (lr.max - lr.start) / lr.rampup_epochs * epoch + lr.start
    elif epoch < lr.rampup_epochs + lr.sustain_epochs:
        lr = lr.max
    else:
        lr = (lr.max - lr.min) * lr.exp_decay ** (
            epoch - lr.rampup_epochs - lr.sustain_epochs
        ) + lr.min
    return lr


def read_tfrecords(example):
    """
    Defines the structure to load an image record.
    """
    features = {
        "image": tf.io.FixedLenFeature([], tf.string),  # tf.str = bytestring
        "class": tf.io.FixedLenFeature([], tf.int64),  # shape [] means scalar
        "one_hot_class": tf.io.VarLenFeature(tf.float32),
    }
    example = tf.io.parse_single_example(example, features)
    image = tf.image.decode_jpeg(example["image"], channels=3)
    # class_label = tf.cast(example["class"], tf.int32)  # not used
    one_hot_class = tf.sparse.to_dense(example["one_hot_class"])
    one_hot_class = tf.reshape(one_hot_class, [5])
    return image, one_hot_class


def force_image_sizes(dataset, image_size):
    """
    Explicit size needed for TPU
    """
    reshape_images = lambda image, label: (
        tf.reshape(image, [image_size, image_size, 3]),
        label,
    )
    dataset = dataset.map(reshape_images, num_parallel_calls=AUTO)
    return dataset


def load_dataset(filenames, image_size):
    """
    Read from TFREcords.

    For optimal performance read from multiple files at once disregarding
    data order. Order does not matter since we will be shuffling the data.
    """
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
    # use data as soon as it streams, rather than in its original order
    dataset = dataset.with_options(ignore_order)
    dataset = dataset.map(read_tfrecords, num_parallel_calls=AUTO)
    dataset = force_image_sizes(dataset, image_size)
    return dataset


def get_validation_dataset(validation_files_path, image_size, batch_size):
    dataset = load_dataset(validation_files_path, image_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTO)
    return dataset
