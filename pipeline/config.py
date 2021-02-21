from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    RANDOM_SEED = 47
    ARTIFACTS_PATH = Path("./artifacts").absolute()
    DATASET_PATH = Path("./data").absolute()
    TRAIN_DATA_DIR = str(DATASET_PATH / "train")
    CATS_TRAIN_DATA_DIR = str(DATASET_PATH / "train" / "cats")
    VALIDATION_DATA_DIR = str(DATASET_PATH / "validation")
    TEST_SCORES = str(ARTIFACTS_PATH / "test_scores.json")

    # dataset
    DOWNLOADED_DATA = "data.zip"

    # params
    PARAMS = "params.yaml"


config = Config()
