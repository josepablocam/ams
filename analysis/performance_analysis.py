#!/usr/bin/env python3
from argparse import ArgumentParser
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from analysis import utils

SYSTEM_NAMES = {
    "sys": "AMS + Search",
    "spec": "Weak Spec.",
    "manual": "Expert + Search",
    "manual_components_only": "Weak Spec. + Search",
}
SYSTEM_ORDER = [
    "Weak Spec.", "Weak Spec. + Search", "Expert + Search", "AMS + Search"
]


def relabel_systems(df):
    df = df.copy()
    df["name"] = df["name"].map(SYSTEM_NAMES)
    return df


# table of number of wins (include total at the bottom)
def table_of_wins(perf_df, min_win_diff):
    ct_df = utils.count_top_scores(
        perf_df, min_diff=min_win_diff
    ).reset_index()
    ct_df["folder_num"] = ct_df["folder_num"].map(str)
    ct_df = ct_df.rename(columns={"folder_num": "Specification"})
    # add in a cumulative
    cum_total = ct_df.cumsum().iloc[-1:]
    cum_total["Specification"] = "Total"
    ct_df = pd.concat([ct_df, cum_total])
    return ct_df


def plot_of_wins(perf_df, min_win_diff):
    ct_df = utils.count_top_scores(
        perf_df, min_diff=min_win_diff
    ).reset_index()
    ct_df = ct_df.drop(columns=["folder_num"])
    cum_ct_df = ct_df.sum().to_frame(name="wins")
    cum_ct_df = cum_ct_df.reset_index()
    cum_ct_df["ix"] = cum_ct_df["name"].map(
        lambda x: SYSTEM_ORDER.index(x)
        if x in SYSTEM_ORDER else len(SYSTEM_ORDER)
    )
    cum_ct_df = cum_ct_df.sort_values("ix")
    ax = cum_ct_df.plot(kind="bar", x="name", y="wins")
    ax.set_xlabel("Approach")
    ax.set_ylabel("Number of Wins")
    plt.xticks(rotation=45)
    plt.tight_layout()
    return ax


def duplicate_results(df_tmp, unique_folders):
    acc = []
    for folder in unique_folders:
        df_tmp = df_tmp.copy()
        df_tmp["folder"] = folder
        df_tmp["folder_num"] = df_tmp["folder"].map(utils.folder_to_num)
        acc.append(df_tmp)
    return pd.concat(acc, axis=0).reset_index(drop=True)


def get_args():
    parser = ArgumentParser(description="Analysis for performance")
    parser.add_argument(
        "--input", type=str, nargs="+", help="Root results directory"
    )
    parser.add_argument(
        "--duplicate",
        type=str,
        nargs="+",
        help="Root for results that need to duplicate (per spec)",
    )
    parser.add_argument(
        "--systems",
        type=str,
        nargs="+",
        help="System to include",
    )
    parser.add_argument(
        "--filter_incomplete",
        action="store_true",
        help="Only consider specifications where all systems completed",
    )
    parser.add_argument(
        "--performance_metric",
        type=str,
        help="Performance metric label for plots",
        default="Performance",
    )
    parser.add_argument(
        "--min_win_diff",
        type=float,
        help="Minimum absolute difference wrt to second place to be a win",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output directory for RQ2 analysis",
    )
    return parser.parse_args()


def main():
    args = get_args()
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    df_acc = []
    for _input in args.input:
        df_tmp = utils.collect_results(_input)
        df_acc.append(df_tmp)
    df = pd.concat(df_acc, axis=0).reset_index(drop=True)

    unique_folders = df["folder"].unique()
    if args.duplicate is not None:
        df_acc = [df]
        for _input in args.duplicate:
            df_tmp = utils.collect_results(_input)
            df_tmp = duplicate_results(df_tmp, unique_folders)
            df_acc.append(df_tmp)
        df = pd.concat(df_acc, axis=0).reset_index(drop=True)

    if args.systems:
        df = df[df["name"].isin(args.systems)].reset_index(drop=True)

    if args.filter_incomplete:
        df = utils.filter_incomplete(df)

    df = relabel_systems(df)
    perf_df = utils.performance_table(df)

    wins_df = table_of_wins(perf_df, args.min_win_diff)
    wins_df.to_csv(os.path.join(args.output, "wins.csv"), index=False)
    wins_df.to_latex(os.path.join(args.output, "wins.tex"), index=False)

    wins_plot = plot_of_wins(perf_df, args.min_win_diff)
    wins_plot.get_figure().savefig(
        os.path.join(args.output, "wins.pdf"), index=False
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        import pdb
        pdb.post_mortem()
