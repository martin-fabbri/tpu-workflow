import argparse

import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler

from train_utils import MapDict, lrfn


def train(lr):
    lr_callback = LearningRateScheduler(lrfn, verbose=True)
    print(type(lr))
    print(lr)
    print(lrfn(lr, 10))


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
    train(lr)
