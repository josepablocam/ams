#!/usr/bin/env python3
from argparse import ArgumentParser

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot(combined_df):
    systems = [
        "Weak Spec.", "Weak Spec. + Search", "Expert + Search", "AMS + Search"
    ]
    combined_df["order"] = combined_df["name"].map(lambda x: systems.index(x))
    combined_df = combined_df.sort_values("order")

    fig, ax = plt.subplots(1)
    ax = sns.barplot(
        data=combined_df,
        x="search",
        y="wins",
        hue="name",
        ci=None,
        ax=ax,
    )
    ax.set_xlabel("Search")
    ax.set_ylabel("Number of Wins")
    plt.legend(loc='upper center', bbox_to_anchor= (0.5, 1.3), title="Approach", ncol=2)
    plt.tight_layout()
    return ax


def combine_dfs(input_paths, search_names):
    acc = []
    for path, search in zip(input_paths, search_names):
        df = pd.read_csv(path)
        df["search"] = search
        acc.append(df)
    combined_df = pd.concat(acc, axis=0)
    combined_df = combined_df[combined_df["Specification"] == "Total"]
    combined_df = pd.melt(
        combined_df,
        id_vars=["Specification", "search"],
        var_name="name",
        value_name="wins",
    )
    return combined_df


def get_args():
    parser = ArgumentParser(description="Combine wins plots")
    parser.add_argument("--input", type=str, nargs="+", help="Wins.csv files")
    parser.add_argument(
        "--search",
        type=str,
        nargs="+",
        help="Search names aligned with input files",
    )
    parser.add_argument("--output", type=str, help="Output path")
    return parser.parse_args()


def main():
    args = get_args()
    df = combine_dfs(args.input, args.search)
    ax = plot(df)
    ax.get_figure().savefig(args.output)


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        import pdb
        pdb.post_mortem()
