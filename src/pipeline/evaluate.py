if __name__ == "__main__":
    import os
    import sys
    from pathlib import Path

    path = Path(os.getcwd())
    src_path = str(path / "src")
    sys.path.append(src_path)

import argparse
import json

from tensorflow.keras.models import load_model

from pipeline.train_utils import get_validation_dataset, dataset_to_numpy_util


def validation_split_files(val_split_path):
    with open(val_split_path, "r") as val_split_file:
        val_split_files = json.load(val_split_file)
    return val_split_files


def evaluate(
    saved_model_path, val_split_path, image_size, batch_size, evaluate_metrics_path
):
    val_split_files, val_split_files = validation_split_files(val_split_path)
    validation_dataset = get_validation_dataset(val_split_files, image_size, batch_size)
    eval_samples, eval_labels = dataset_to_numpy_util(validation_dataset, 160)
    reload_model = load_model(saved_model_path)
    evaluations = reload_model.evaluate(
        eval_samples, eval_labels, batch_size=batch_size
    )
    with open(evaluate_metrics_path, "w") as feval:
        json.dump({"val_loss": evaluations[0], "val_acc": evaluations[1]}, feval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
        "--val-split-path",
        dest="val_split_path",
        type=str,
        required=True,
        help="Validation split files",
    )
    parser.add_argument(
        "--saved-model-path",
        dest="saved_model_path",
        type=str,
        required=True,
        help="Saved model path",
    )
    parser.add_argument(
        "--evaluate-metrics-path",
        dest="evaluate_metrics_path",
        type=str,
        required=True,
        help="Evaluation metrics path",
    )
    args = parser.parse_args()
    evaluate(
        args.saved_model_path,
        args.val_split_path,
        args.image_size,
        args.batch_size,
        args.evaluate_metrics_path,
    )
