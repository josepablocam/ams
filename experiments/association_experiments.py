#!/usr/env/bin python3
from argparse import ArgumentParser
import os
import sys
sys.path.append("../")
sys.path.append(".")

import itertools
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import KFold
import tqdm

from core.extract_parameters import *
from core import utils
from core.mine_association_rules import create_transactions, APIRulePredictor


def is_covered(x_pred):
    return len(x_pred) != 0


def get_precision(x_true, x_pred):
    if len(x_pred) == 0:
        return np.nan
    correct = set(x_pred).intersection(x_true)
    n = float(len(x_pred))
    return len(correct) / n


def evaluate_split(train_split, test_split, k, max_len, alpha=0.5):
    model = APIRulePredictor(train_split)
    perf = []
    for obs_ix, obs in tqdm.tqdm(enumerate(test_split)):
        len_obs = len(obs)
        n = min(len_obs - 1, max_len + 1)
        for len_query in range(1, n):
            combinations = itertools.combinations(obs, len_query)
            for partial in combinations:
                predicted = model.extend(partial, k=k, alpha=alpha)
                new_components = [c for c in predicted if c not in partial]
                cov = is_covered(new_components)
                prec = get_precision(obs, new_components)
                perf.append((
                    k, obs_ix, len_query, len_obs, len(new_components), cov,
                    prec
                ))
    columns = [
        "k_rules",
        "obs_id",
        "len_query",
        "num_original_obs",
        "num_new_components",
        "covered",
        "precision",
    ]
    return pd.DataFrame(perf, columns=columns)


def evaluate(
        transactions,
        n_splits=5,
        ks=None,
        max_len=3,
        alpha=0.5,
        random_state=None,
):
    if ks is None:
        ks = [1, 3, 5, 10]

    splitter = KFold(n_splits=n_splits, random_state=random_state)
    acc = []
    for cv_ix, ixs in tqdm.tqdm(enumerate(splitter.split(transactions))):
        train_ix, test_ix = ixs
        train_split = [transactions[ix] for ix in train_ix]
        test_split = [transactions[ix] for ix in test_ix]
        for k in ks:
            eval_df = evaluate_split(
                train_split,
                test_split,
                k,
                max_len,
                alpha=alpha,
            )
            eval_df["cv_id"] = cv_ix
            acc.append(eval_df)
    acc_df = pd.concat(acc, axis=0)
    return acc_df


def plot_eval(df):
    ax = sns.barplot(data=df, x="k_rules", y="precision")
    ax.set_xlabel("Top K Rules")
    ax.set_ylabel("Precision")
    return ax


def get_args():
    parser = ArgumentParser(description="NPMI rule evaluation")
    parser.add_argument(
        "--input",
        type=str,
        help="Pickled list of annotated Kaggle scripts",
    )
    parser.add_argument(
        "--k", type=int, nargs="+", help="Top K rules", default=[1, 3, 5]
    )
    parser.add_argument(
        "--alpha",
        type=float,
        help=
        "Weight to combine norm PMI and normalized count of support into score",
        default=0.5,
    )
    parser.add_argument("--n_splits", type=int, help="CV splits", default=5)
    parser.add_argument(
        "--max_len", type=int, help="Max length for partial query", default=3
    )
    parser.add_argument(
        "--random_state",
        type=int,
        help="RNG seed",
        default=42,
    )
    parser.add_argument("--output", type=str, help="Output directory")
    return parser.parse_args()


def main():
    args = get_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    with open(args.input, "rb") as fin:
        annotated_scripts = pickle.load(fin)

    transactions = create_transactions(annotated_scripts)
    df = evaluate(
        transactions,
        n_splits=args.n_splits,
        ks=args.k,
        max_len=args.max_len,
        alpha=args.alpha,
        random_state=args.random_state,
    )

    df_path = os.path.join(args.output, "df.pkl")
    df.to_pickle(df_path)

    plot = plot_eval(df)
    plot_path = os.path.join(args.output, "precision.pdf")
    plot.get_figure().savefig(plot_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        import pdb
        pdb.post_mortem()
