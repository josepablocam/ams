#!/usr/bin/env python3
from argparse import ArgumentParser
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import pickle

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pmlb
import seaborn as sns
import sklearn
import sklearn.base
import sklearn.model_selection
import tqdm

from core.extract_sklearn_api import (
    APICollection,
    APIClass,
    APIClassParameter,
)
from core.utils import get_component_constructor
from core.extract_parameters import UNKValue
from core.generate_search_space import generate_parameters

from analysis.utils import emp_cdf, plot_emp_cdf
from datasets.utils import fetch_data


def count_params_per_class(api_collection, exclude):
    results = {}
    for c in api_collection.classes:
        params = [p for p in c.children if p.param_name not in exclude]
        results[c.path] = len(params)
    return results


def distribution_ratio_params_tuned(df, cts_per_class, exclude,
                                    plot_steps=200):
    df = df.copy()
    df["ignore"] = df["key"].isin(exclude)
    df_ct = df.groupby(
        ["id",
         "func"])["key"].agg(lambda x: len([e for e in x if e is not None]))
    df_ct = df_ct.to_frame(name="ct").reset_index()
    df_ignore = df.groupby(
        ["id",
         "func"])["ignore"].sum().to_frame(name="ct_ignore").reset_index()
    df_ct = pd.merge(df_ct, df_ignore, how="left", on=["id", "func"])
    df_ct["ct_ignore"] = df_ct["ct_ignore"].fillna(0.0)
    df_ct["final_ct"] = df_ct["ct"] - df_ct["ct_ignore"]
    df_ct["reference_ct"] = df_ct["func"].map(cts_per_class)
    df_ct["ratio"] = df_ct["final_ct"] / df_ct["reference_ct"]
    # drop entries with zero reference
    df_ct = df_ct[df_ct["reference_ct"] > 0].reset_index()
    plot = plot_emp_cdf(df_ct["ratio"], np.linspace(0, 1.0, plot_steps))
    plot.set_xlabel("Ratio of hyperparameters tuned to available")
    plot.set_ylabel("Empirical CDF (Calls)")
    return df_ct, plot


# people tend to not tune all hyper-parameters


def jaccard_similarity(s1, s2):
    if len(s1) == 0 and len(s2) == 0:
        return 1.0
    i = s1.intersection(s2)
    u = s1.union(s2)
    return len(i) / float(len(u))


def distribution_distance_params_tuned(df,
                                       exclude,
                                       n_samples=100,
                                       plot_steps=200):
    df = df.copy()
    df = df[~df["key"].isin(exclude)]
    df = df.groupby([
        "id", "func"
    ])["key"].apply(lambda x: frozenset([e for e in x if e is not None]))
    df = df.to_frame(name="params").reset_index()
    distances = []
    for func in tqdm.tqdm(df["func"].unique()):
        df_func = df[df["func"] == func].reset_index(drop=True)
        n = df_func.shape[0]
        if n > n_samples:
            df_func = df_func.sample(n=n_samples).reset_index(drop=True)
            n = n_samples
        for i in tqdm.tqdm(range(0, n)):
            for j in range(i + 1, n):
                row1 = df_func.iloc[i]["params"]
                row2 = df_func.iloc[j]["params"]
                jacc_dist = 1.0 - jaccard_similarity(row1, row2)
                distances.append((func, jacc_dist))
    df_dist = pd.DataFrame(distances, columns=["func", "dist"])
    df_dist = df_dist.groupby("func")["dist"].mean().to_frame(name="dist")
    df_dist = df_dist.reset_index()
    plot = plot_emp_cdf(df_dist["dist"].values,
                        np.linspace(0.0, 1.0, plot_steps))
    plot.set_xlabel("Mean Jaccard Distance of Parameters Tuned Across Calls")
    plot.set_ylabel("Empirical CDF (Components)")
    return df_dist, plot


def distribution_param_values(df, exclude):
    df = df.copy()
    df = df[~df["key"].isin(exclude)]
    df_cts = df.groupby(["func", "key"])["value"].agg(lambda x: len(set(x)))
    df_cts = df_cts.to_frame(name="ct_unique_values").reset_index()
    min_val = 1
    max_val = df_cts.ct_unique_values.max()
    plot = plot_emp_cdf(df_cts.ct_unique_values.values,
                        np.arange(min_val, max_val + 1, 1.0))
    plot.set_xlabel("Unique values in calls")
    plot.set_ylabel("Empirical CDF (Component Hyperparameters)")
    return df_cts, plot


def get_top_n_classifiers(df, n):
    funcs = df["func"].unique()
    clfs = [
        f for f in funcs
        if sklearn.base.is_classifier(get_component_constructor(f))
    ]
    df_clfs = df[df["func"].isin(clfs)]
    # single entry per call
    df_clfs = df_clfs.groupby(["func", "id"]).head(1)
    df_cts = df_clfs.groupby("func").size()
    df_top = df_cts.sort_values(ascending=False).head(n)
    return df_top.index.values


def evaluate_component_possible_improvement(X,
                                            y,
                                            component,
                                            param_summary,
                                            num_params,
                                            num_values,
                                            exclude,
                                            scoring="f1_macro",
                                            cv=5,
                                            random_state=None):
    params = generate_parameters(
        component,
        param_summary,
        num_params,
        num_values,
        exclude_params=exclude,
        add_default=True,
    )
    obj = get_component_constructor(component)

    param_grid = sklearn.model_selection.ParameterGrid(params)
    acc_scores = []
    for ps in tqdm.tqdm(param_grid):
        if random_state is not None:
            np.random.seed(random_state)
        try:
            score = np.mean(
                sklearn.model_selection.cross_val_score(
                    obj(**ps),
                    X,
                    y,
                    scoring=scoring,
                    cv=cv,
                    n_jobs=-1,
                ))
            acc_scores.append(score)
        except:
            pass

    # Note: this is the max of the cross-validation score
    # we only use this to show the room for *possible* improvement
    # if we had started with the best configuration
    # This is **not** equivalent to the test score
    # since a true test score would perform CV only on the
    # training set (we instead are choosing the config that performed
    # best on the test CV splits...so this is only indicative of the
    # *possible* improvement, not actually what we would observe in
    # practice)
    assert len(acc_scores) > 0, "All params failed"
    best_score = np.max(acc_scores)

    if random_state is not None:
        np.random.seed(random_state)
    default_score = np.mean(
        sklearn.model_selection.cross_val_score(
            obj(),
            X,
            y,
            scoring=scoring,
            cv=cv,
            n_jobs=-1,
        ))
    return default_score, best_score


def performance_of_tuned_vs_default(
        datasets,
        df_params_raw,
        df_params_summary,
        exclude,
        num_clfs=5,
        num_params=3,
        num_values=3,
        scoring="f1_macro",
        cv=5,
        random_state=42,
):
    clfs = get_top_n_classifiers(df_params_raw, n=num_clfs)
    # SVMs can take a looongg time to fit
    clfs = [c for c in clfs if "svm" not in c]
    results = []
    for c in tqdm.tqdm(clfs):
        for d in datasets:
            X, y = fetch_data(d)
            default_score, best_score = evaluate_component_possible_improvement(
                X,
                y,
                c,
                df_params_summary,
                num_params,
                num_values,
                exclude,
                scoring=scoring,
                cv=cv,
                random_state=random_state,
            )
            results.append((c, d, default_score, best_score))
    df_results = pd.DataFrame(
        results,
        columns=["classifier", "dataset", "default_score", "best_score"],
    )
    df_results[
        "ratio"] = df_results["best_score"] / df_results["default_score"]
    return df_results


def summarize_performance(df_perf):
    df_perf = df_perf.copy()
    df_perf["pct"] = (df_perf["ratio"] - 1.0) * 100.0
    df_perf["classifier_basename"] = df_perf["classifier"].map(
        lambda x: x.split(".")[-1])
    fig, ax = plt.subplots(1)
    plot = sns.scatterplot(
        data=df_perf,
        x="classifier_basename",
        y="pct",
        hue="dataset",
        ax=ax,
    )
    plt.xticks(rotation=45)
    plot.set_xlabel("Classifier")
    plot.set_ylabel("Possible improvement over defaults (%)")
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", ncol=1)
    plt.tight_layout()
    return plot


def get_args():
    parser = ArgumentParser(
        description="Analysis of Kaggle hyperparameter data")
    parser.add_argument(
        "--params_raw", type=str, help="Path to pickled -raw parameters")
    parser.add_argument(
        "--params_summary",
        type=str,
        help="Path to pickled -summary parameters")
    parser.add_argument(
        "--num_params",
        type=int,
        help="Hyperparams per component",
        default=3,
    )
    parser.add_argument(
        "--num_values",
        type=int,
        help="Values per hyperparam",
        default=3,
    )
    parser.add_argument(
        "--cv",
        type=int,
        help="CV iterations",
        default=5,
    )
    parser.add_argument(
        "--scoring",
        type=str,
        help="Scoring function",
        default="f1_macro",
    )
    parser.add_argument(
        "--num_clfs",
        type=int,
        help="Top n classifiers to eval",
        default="n_clfs",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        help="Datasets",
    )
    parser.add_argument(
        "--api", type=str, help="Path to pickled API collection")
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
    df_raw = pd.read_pickle(args.params_raw)
    df_summary = pd.read_pickle(args.params_summary)

    with open(args.api, "rb") as fin:
        api_collection = pickle.load(fin)

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    np.random.seed(args.random_state)

    # nuisance hyperparameters (i.e. just implementation details)
    exclude = [
        "verbose",
        "random_state",
        "cv",
        "n_jobs",
        "prefit",
        "refit",
    ]

    reference_counts = count_params_per_class(api_collection, exclude)
    df_num_tuned, plot_num_tuned = distribution_ratio_params_tuned(
        df_raw,
        reference_counts,
        exclude,
    )
    df_num_tuned.to_csv(
        os.path.join(args.output, "num_params_tuned.csv"),
        index=False,
    )
    plot_num_tuned.get_figure().savefig(
        os.path.join(args.output, "num_params_tuned.pdf"))

    df_dist_tuned, plot_dist_tuned = distribution_distance_params_tuned(
        df_raw,
        exclude,
    )
    df_dist_tuned.to_csv(
        os.path.join(args.output, "distance_params_tuned.csv"),
        index=False,
    )
    plot_dist_tuned.get_figure().savefig(
        os.path.join(args.output, "distance_params_tuned.pdf"))

    df_dist_values, plot_dist_values = distribution_param_values(
        df_raw, exclude)
    df_dist_values.to_csv(
        os.path.join(args.output, "num_param_values.csv"),
        index=False,
    )
    plot_dist_values.get_figure().savefig(
        os.path.join(args.output, "num_param_values.pdf"))

    df_perf = performance_of_tuned_vs_default(
        args.datasets,
        df_raw,
        df_summary,
        exclude,
        num_clfs=args.num_clfs,
        num_params=args.num_params,
        num_values=args.num_values,
        scoring=args.scoring,
        cv=args.cv,
        random_state=args.random_state,
    )
    df_perf.to_csv(os.path.join(args.output, "perf.csv"), index=False)
    perf_plot = summarize_performance(df_perf)
    perf_plot.get_figure().savefig(os.path.join(args.output, "perf.pdf"))


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        import pdb
        pdb.post_mortem()
