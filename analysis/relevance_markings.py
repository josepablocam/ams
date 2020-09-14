from argparse import ArgumentParser
import os

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams['text.usetex'] = True
import pandas as pd
import seaborn as sns


def summarize_manual_markings(df):
    ks = [1, 5, 10]
    df = df.copy()
    df = df.sort_values(["sampled_id", "result_position"], ascending=True)
    # some times .astype can be funky, so just be explicit about 1.0/0.0
    df["relevant"] = df["relevant"].map(lambda x: 1.0 if x else 0.0)
    result_df = df[["sampled_id"]].drop_duplicates()
    for k in ks:
        top_k_df = df.groupby("sampled_id").head(k).groupby("sampled_id")[[
            "relevant"
        ]].mean()
        top_k_df = top_k_df.reset_index()
        top_k_df = top_k_df.rename(columns={"relevant": "top_{}".format(k)})
        result_df = pd.merge(result_df, top_k_df, how="left", on="sampled_id")
    return result_df


def plot_markings(combined_df):
    combined_df_flat = pd.melt(combined_df, id_vars=["sampled_id", "name"])
    combined_df_flat["variable"] = combined_df_flat["variable"].map(
        lambda x: " ".join(x.capitalize().split("_"))
    )
    ax = sns.barplot(
        data=combined_df_flat, x="variable", y="value", hue="name"
    )
    ax.set_xlabel("Cutoff")
    ax.set_ylabel("Fraction Functionally Related")
    ax.legend(title="Approach")
    return ax


def get_args():
    parser = ArgumentParser(
        description=
        "Summarize analysis of functionally related components",
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=str,
        help="Manually rated CSV with results to summarize",
    )
    parser.add_argument(
        "--names",
        nargs="+",
        type=str,
        help="Names for results",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output",
    )
    return parser.parse_args()


def main():
    args = get_args()
    acc = []
    if args.names is None:
        args.names = ["df_{}".format(i) for i, _ in enumerate(args.inputs)]
    for name, path in zip(args.names, args.inputs):
        df = pd.read_csv(path)
        summary_df = summarize_manual_markings(df)
        summary_df["name"] = name
        acc.append(summary_df)
    combined_df = pd.concat(acc, axis=0)

    ax = plot_markings(combined_df)
    if args.output:
        dir_path = os.path.dirname(args.output)
        if len(dir_path) > 0 and not os.path.exists(dir_path):
            os.makedirs(dir_path)
        ax.get_figure().savefig(args.output)


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        import pdb
        pdb.post_mortem()
