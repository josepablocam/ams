#!/usr/env/bin python3
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from argparse import ArgumentParser
from collections import Counter, defaultdict
import glob
import json

import matplotlib
# Note that for this one we can't use 'text.usetex'
# that seems to error out when using multipage pdf...
# the alternative with pdf/ps.fonttype works
# matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import numpy as np
import pandas as pd
import tqdm

from analysis import pipeline_to_tree
from analysis import utils


def extract_operators(pipeline):
    stack = [pipeline_to_tree.to_tree(pipeline)]
    ops = []
    while len(stack) > 0:
        curr = stack.pop()
        if curr.label == "root":
            stack.extend(curr.children)
        elif curr.label.startswith(("sklearn", "xgboost", "tpot")):
            if not curr.label.startswith("sklearn.pipeline.Pipeline"):
                ops.append(curr.label)
            stack.extend(curr.children)
        else:
            continue
    return ops


def operator_counters(df):
    counters = defaultdict(lambda: Counter())
    for _, row in tqdm.tqdm(df.iterrows()):
        ops = extract_operators(row.fitted_pipeline)
        counters[row.folder].update(ops)
    counters = {
        folder: [(op, ct) for op, ct in counter.most_common(len(counter))]
        for folder, counter in counters.items()
    }
    return counters


def get_short_op(op):
    return op.split(".")[-1]


def get_spec_text(spec):
    ops = [get_short_op(op) for op in spec]
    ops_str = ", ".join(ops)
    return ops_str


def load_specs(root_folder):
    files = glob.glob(os.path.join(root_folder, "*", "simple_config.json"))
    specs = {}
    for p in files:
        p_parts = p.split("/")
        if not p_parts[-2].startswith("q"):
            continue
        with open(p, "r") as fin:
            s = json.load(fin)
        folder = p_parts[-2]
        specs[folder] = s
    return specs


def plot_operator_distribution(
        counters,
        specs,
        k,
        output,
        combine=False,
        title=None,
):
    axes = []
    for folder, counts in counters.items():
        counts = [(get_short_op(op), ct) for op, ct in counts]
        # sort in descending order
        counts = sorted(counts, key=lambda x: x[1], reverse=True)
        xs, ys = zip(*counts)
        ys = np.array(ys)
        # normalize
        ys_norm = ys / np.sum(ys)
        # only keep k for plotting
        xs = xs[:k]
        ys_norm = ys_norm[:k]
        fig, ax = plt.subplots(1)
        axes.append(ax)
        plot_df = pd.DataFrame(zip(xs, ys_norm), columns=["x", "y"])
        plot_df.plot(kind="bar", x="x", y="y", ax=ax)
        ax.set_xlabel("Components")
        ax.set_ylabel("% of components")

        spec_text = None
        if specs is not None:
            spec_text = get_spec_text(specs[folder])

        if title is not None:
            extra = folder if spec_text is None else spec_text
            ax_title = title.format(extra)
        else:
            ax_title = "Distribution"

        ax.set_title(ax_title, fontsize=8)

        ax.get_legend().remove()
        plt.xticks(rotation=90)
        plt.tight_layout()
        if not combine:
            plot_path = os.path.join(output, folder + ".pdf")
            fig.savefig(plot_path)
            plt.close()

    if combine:
        pdf_path = os.path.join(output, "combined.pdf")
        pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_path)
        for ax in axes:
            pdf.savefig(ax.get_figure())
        pdf.close()
    plt.close()


def get_args():
    parser = ArgumentParser(
        description=
        "Bar plots for frequency (%) of top k operators in optimized pipelines"
    )
    parser.add_argument("--input", type=str, help="Input folder with results")
    parser.add_argument("--specs", type=str, help="Input folder with specs")
    parser.add_argument("--name", type=str, help="System name", default="sys")
    parser.add_argument("--k", type=int, help="Top K operators", default=10)
    parser.add_argument("--output", type=str, help="Output directory")
    parser.add_argument(
        "--combine",
        action="store_true",
        help="Put all plots in one pdf",
    )
    parser.add_argument("--title", type=str, help="Title format")
    return parser.parse_args()


def main():
    args = get_args()
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    specs = None
    if args.specs is not None:
        specs = load_specs(args.specs)

    df = utils.collect_results(args.input)
    print("Operator counts for", args.name)
    df = df[df["name"] == args.name].reset_index(drop=True)
    counters = operator_counters(df)

    plot_operator_distribution(
        counters,
        specs,
        args.k,
        args.output,
        combine=args.combine,
        title=args.title,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        import pdb
        pdb.post_mortem()
