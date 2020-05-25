#!/usr/bin/env python3

from argparse import ArgumentParser
import os

import pmlb

DATASETS = [
    "Hill_Valley_without_noise",
    "Hill_Valley_with_noise",
    "breast-cancer-wisconsin",
    "car-evaluation",
    "glass",
    "ionosphere",
    "spambase",
    "wine-quality-red",
    "wine-quality-white",
]

DEFAULT_LOCAL_CACHE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../data/benchmarks-datasets/")
)
# overwrite this if $DATA environment variable set
if "DATA" in os.environ:
    data_dir = os.environ["DATA"]
    cache_dir = os.path.join(data_dir, "benchmarks-datasets")
    DEFAULT_LOCAL_CACHE_DIR = cache_dir

print(
    "Setting default benchmarks-datasets cache directory to: {}".
    format(DEFAULT_LOCAL_CACHE_DIR)
)


def get_args():
    parser = ArgumentParser(description="Download datasets to local cache")
    parser.add_argument("--output", type=str, help="Path to local cache")
    return parser.parse_args()


def main():

    args = get_args()
    local_cache_dir = DEFAULT_LOCAL_CACHE_DIR
    if args.output is not None:
        local_cache_dir = args.output
    if not os.path.exists(local_cache_dir):
        print("Creating", local_cache_dir)
        os.makedirs(local_cache_dir, exist_ok=True)

    for d in DATASETS:
        print("Downloading", d)
        pmlb.fetch_data(
            d,
            return_X_y=True,
            local_cache_dir=local_cache_dir,
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        import pdb
        pdb.post_mortem()
