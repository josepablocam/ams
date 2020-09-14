#!/usr/bin/env python3
from argparse import ArgumentParser
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from core.extract_sklearn_api import (
    APICollection,
    APIClass,
    APIClassParameter,
)
from core.mine_association_rules import APIRulePredictor
from core.utils import get_component_constructor

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tqdm


def extract_info_from_path(folder):
    name = folder.split("/")[-1]
    corpus_size = float(name.split("-")[2])
    iter_num = int(name.split("-")[-1])
    return {"corpus_size": corpus_size, "iter": iter_num}


# decrease in size
def hyperparam_summary(full_info, sample_info):
    full_ks = full_info["keys"]
    sample_ks = sample_info["keys"]

    ks = pd.merge(
        full_ks,
        sample_ks,
        how="left",
        on=["func", "key"],
        suffixes=("_full", "_sample"),
    )
    ks["is_missing"] = pd.isnull(ks["key_ct_sample"])

    missing_ks = ks.groupby("func")["is_missing"].mean().to_frame(
        name="hyperparam_reduction_fraction"
    ).reset_index()
    # sort by which is most commonly used
    ct_ks = ks.groupby("func")["key_ct_full"].sum().to_frame(name="total_ct"
                                                             ).reset_index()
    missing_ks = pd.merge(missing_ks, ct_ks, how="left", on="func")
    missing_ks = missing_ks.sort_values("total_ct", ascending=True)
    missing_ks["ix"] = np.arange(0, missing_ks.shape[0])
    return missing_ks


def hypervalue_summary(full_info, sample_info):
    full_vs = full_info["values"]
    sample_vs = sample_info["values"]
    vs = pd.merge(
        full_vs,
        sample_vs,
        how="left",
        on=["func", "key"],
        suffixes=("_full", "_sample"),
    )
    vs["is_missing"] = pd.isnull(vs["default_value_sample"])
    vs = vs[~vs["is_missing"]]
    vs["num_values_full"] = vs["value_full"].map(len)
    vs["num_values_sample"] = vs["value_sample"].map(len)
    vs["num_values_reduction"
       ] = vs["num_values_full"] - vs["num_values_sample"]
    return vs


def association_rules_summary(full_rules_obj, sampled_rules_obj):
    full_rules = full_rules_obj.df_rules
    sampled_rules = sampled_rules_obj.df_rules

    get_pairs = lambda df: [
        frozenset(es) for es in zip(df["comp1"], df["comp2"])
    ]
    full_rules_pairs = set(get_pairs(full_rules))
    sampled_rules_pairs = set(get_pairs(sampled_rules))

    jaccard_sim = len(full_rules_pairs.intersection(sampled_rules_pairs)
                      ) / len(full_rules_pairs.union(sampled_rules_pairs))

    results = {}
    results["jaccard_sim"] = jaccard_sim
    results["num_sampled_rules"] = sampled_rules.shape[0]
    results["reduction_num_rules"
            ] = 1 - (sampled_rules.shape[0] / full_rules.shape[0])
    return pd.DataFrame([results])


def run_analysis(full_folder, sampled_folders, file_name, summary_fun):
    results = []
    full_info = pd.read_pickle(os.path.join(full_folder, file_name))
    for folder in tqdm.tqdm(sampled_folders):
        sampled_info = pd.read_pickle(os.path.join(folder, file_name))
        path_info = extract_info_from_path(folder)
        summary = summary_fun(full_info, sampled_info)
        summary["corpus_size"] = path_info["corpus_size"]
        summary["iter"] = path_info["iter"]
        results.append(summary)
    results_df = pd.concat(results, axis=0)
    return results_df


def hyperparam_analysis(full_folder, sampled_folders, output_folder):
    df_params = run_analysis(
        full_folder,
        sampled_folders,
        "params.pkl-summary",
        hyperparam_summary,
    )
    fig, ax = plt.subplots(1)
    ax = sns.barplot(
        data=df_params,
        x="corpus_size",
        y="hyperparam_reduction_fraction",
        ax=ax,
    )
    ax.set_xlabel("Corpus Sampling Ratio")
    ax.set_ylabel("Fraction of Hyperparameters Missing per Component")
    fig.savefig(os.path.join(output_folder, "hyperparameters.pdf"))

    df_values = run_analysis(
        full_folder,
        sampled_folders,
        "params.pkl-summary",
        hypervalue_summary,
    )

    fig, ax = plt.subplots(1)
    ax = sns.barplot(
        data=df_values,
        x="corpus_size",
        y="num_values_reduction",
        ax=ax,
    )
    ax.set_xlabel("Corpus Sampling Ratio")
    ax.set_ylabel("Reduction in Hyperparameter Values")
    fig.savefig(os.path.join(output_folder, "hyperparameter_values.pdf"))

    df_params.to_csv(
        os.path.join(output_folder, "hyperparams_summary.csv"), index=False
    )

    df_values.to_csv(
        os.path.join(output_folder, "hypervalues_summary.csv"), index=False
    )


def association_rules_analysis(full_folder, sampled_folders, output_folder):
    df = run_analysis(
        full_folder,
        sampled_folders,
        "meta_kaggle_association_rules.pkl",
        association_rules_summary,
    )

    df["pct_reduction_num_rules"] = 100.0 * df["reduction_num_rules"]
    fig, ax = plt.subplots(1)
    ax = sns.barplot(
        data=df,
        x="corpus_size",
        y="pct_reduction_num_rules",
        ax=ax,
    )
    ax.set_xlabel("Corpus Sampling Ratio")
    ax.set_ylabel("% Reduction in Total Number of Mined Rules")
    fig.savefig(os.path.join(output_folder, "num_mined_rules.pdf"))

    fig, ax = plt.subplots(1)
    ax = sns.barplot(
        data=df,
        x="corpus_size",
        y="jaccard_sim",
        ax=ax,
    )
    ax.set_xlabel("Corpus Sampling Ratio")
    ax.set_ylabel("Rules' Jaccard Similarity w.r.t Full Corpus Rules")
    fig.savefig(os.path.join(output_folder, "jaccard_mined_rules.pdf"))

    df.to_csv(
        os.path.join(output_folder, "association_rules_summary.csv"),
        index=False
    )


def mining_analysis(full_folder, sampled_folders, output_folder):
    hyperparam_analysis(full_folder, sampled_folders, output_folder)
    association_rules_analysis(full_folder, sampled_folders, output_folder)


def get_args():
    parser = ArgumentParser(
        description="Run analysis of corpus size changes impact"
    )
    parser.add_argument(
        "--full_data_folder", type=str, help="Folder with full data"
    )
    parser.add_argument(
        "--sampled_data_folder",
        type=str,
        nargs="+",
        help="Folder(s) with sampled data"
    )
    parser.add_argument("--output", type=str, help="Directory for output")
    return parser.parse_args()


def main():
    args = get_args()
    if not os.path.exists(args.output):
        print("Creating", args.output)
        os.makedirs(args.output)

    mining_analysis(
        args.full_data_folder, args.sampled_data_folder, args.output
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        import pdb
        pdb.post_mortem()
