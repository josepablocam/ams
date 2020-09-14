import os
import pandas as pd

ROOT_DATA_FOLDER = os.environ.get(
    "DATA", os.path.join(os.path.dirname(__file__), "../data/")
)

assert ROOT_DATA_FOLDER is not None
DATASETS_FOLDER = os.path.abspath(
    os.path.join(ROOT_DATA_FOLDER, "benchmarks-datasets")
)

if not os.path.exists(DATASETS_FOLDER):
    raise Exception("Missing benchmarks-datasets folder:", DATASETS_FOLDER)


def fetch_data(name):
    dataset_path = os.path.join(DATASETS_FOLDER, name + ".tsv")
    dataset = pd.read_csv(dataset_path, sep='\t')
    X = dataset.drop('target', axis=1).values
    y = dataset['target'].values
    return X, y
