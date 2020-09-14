#!/usr/bin/env python3

from argparse import ArgumentParser
import os
import subprocess
import sys


def get_args():
    parser = ArgumentParser(
        description=
        "Wrapper for building info deriver from corpus with varying sample rates"
    )
    parser.add_argument(
        "--orig_data_folder",
        type=str,
        help="Orig folder pointed to by $DATA",
    )
    parser.add_argument(
        "--sampled_data_folder",
        type=str,
        help="Folder to place anything derived from sampled corpus"
    )
    parser.add_argument(
        "--sample_rate", type=float, help="Ratio of corpus to sample"
    )
    parser.add_argument("--seed", type=int, help="RNG seed")
    return parser.parse_args()


def run_proc(cmd):
    cmd_str = " ".join(cmd)
    status = subprocess.call(cmd_str, shell=True)
    if status != 0:
        print("Failed to run ", cmd_str)
        sys.exit(1)


def main():
    args = get_args()
    orig_data_folder = args.orig_data_folder
    sampled_data_folder = args.sampled_data_folder
    sample_rate = args.sample_rate
    seed = args.seed

    run_proc([
        "python",
        "-m",
        "core.extract_kaggle_scripts",
        "--input",
        os.path.join(orig_data_folder, "meta-kaggle"),
        "--output",
        os.path.join(sampled_data_folder, "meta_kaggle_python.pkl"),
        "--api",
        "sklearn",
        "--sample",
        str(sample_rate),
        "--seed",
        str(seed),
    ])

    run_proc([
        "python",
        "-m",
        "core.extract_parameters",
        "--input",
        os.path.join(sampled_data_folder, "meta_kaggle_python.pkl"),
        "--output",
        os.path.join(
            sampled_data_folder, "meta_kaggle_python_with_params.pkl"
        ),
        "--api",
        os.path.join(orig_data_folder, "sklearn_api.pkl"),
    ])

    run_proc([
        "python",
        "-m",
        "core.summarize_parameters",
        "--input",
        os.path.join(
            sampled_data_folder, "meta_kaggle_python_with_params.pkl"
        ),
        "--output",
        os.path.join(sampled_data_folder, "params.pkl"),
        "--api",
        os.path.join(orig_data_folder, "sklearn_api.pkl"),
    ])

    run_proc([
        "python",
        "-m",
        "core.mine_association_rules",
        "--input",
        os.path.join(
            sampled_data_folder, "meta_kaggle_python_with_params.pkl"
        ),
        "--output",
        os.path.join(sampled_data_folder, "meta_kaggle_association_rules.pkl"),
    ])


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        import pdb
        pdb.post_mortem()
