#!/usr/bin/env python3

from argparse import ArgumentParser
import os
import pickle
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.base

from core.extract_sklearn_api import (
    APICollection,
    APIClass,
    APIClassParameter,
)
from core.mine_association_rules import APIRulePredictor
from core.utils import get_component_constructor

from analysis.utils import emp_cdf, plot_emp_cdf


def distribution_norm_pmi(df_rules):
    # compute the distribution of norm_pmi
    print("{} rules mined".format(df_rules.shape[0]))
    comps = df_rules["comp1"].values.tolist(
    ) + df_rules["comp2"].values.tolist()
    num_comps = len(set(comps))
    print("Rules use {} different components".format(num_comps))

    norm_pmi = df_rules["norm_pmi"].values
    cutoffs = np.linspace(0.0, 1.0, 100)
    ax = plot_emp_cdf(norm_pmi, cutoffs)
    ax.set_xlabel("Normalized PMI")
    ax.set_ylabel("Empirical CDF (PMI-rules)")
    return ax


def get_coverage(api_collection, df_rules):
    possible_comps = [c.path for c in api_collection.classes]
    results = []
    for c in possible_comps:
        ct = ((df_rules["comp1"] == c) | (df_rules["comp2"] == c)).sum()
        results.append((c, ct))
    df_results = pd.DataFrame(results, columns=["component", "ct"])
    cov = (df_results["ct"] > 0).mean()
    print("Coverage of {:.2f}% of API".format(cov * 100))
    return cov


def label_role(obj):
    if sklearn.base.is_regressor(obj):
        return "regressor"
    elif sklearn.base.is_classifier(obj):
        return "classifier"
    elif obj.__module__.startswith("sklearn.cluster"):
        return "cluster"
    elif obj.__module__.startswith(
        ("sklearn.decomposition", "sklearn.manifold")):
        return "decomposition"
    elif obj.__module__.startswith(
        ("sklearn.feature_extraction", "sklearn.feature_selection")):
        return "feature extraction/selection"
    else:
        return "preprocessor"


def distribution_roles(df_rules):
    roles = []
    for _, row in df_rules.iterrows():
        try:
            obj1 = get_component_constructor(row.comp1)
        except AttributeError:
            print("Comp1", row.comp1)
            obj1 = None
        try:
            obj2 = get_component_constructor(row.comp2)
        except AttributeError:
            print("Comp2", row.comp2)
            obj2 = None

        role1 = label_role(obj1)
        role2 = label_role(obj2)
        norm_pmi = row["norm_pmi"]
        roles.append((role1, role2, norm_pmi))
    df_roles = pd.DataFrame(roles, columns=["role1", "role2", "norm_pmi"])
    df_roles["pair"] = [
        tuple(sorted([r1, r2]))
        for r1, r2 in zip(df_roles["role1"], df_roles["role2"])
    ]
    df_roles = df_roles.groupby("pair")["norm_pmi"].agg([
        "count", "mean", "std"
    ])
    df_roles = df_roles.reset_index()
    df_roles["pair"] = df_roles["pair"].map(
        lambda x: x[0] if x[0] == x[1] else x
    )
    df_roles["mean"] = df_roles["mean"].map(lambda x: "{:.2f}".format(x))
    df_roles["std"] = df_roles["std"].map(
        lambda x: "{:.2f}".format(x) if not np.isnan(x) else "-"
    )
    df_roles = df_roles.rename(
        columns={
            "pair": "Rule Type",
            "count": "# Rules",
            "mean": "Mean Norm. PMI",
            "std": "SD Norm. PMI"
        }
    )
    return df_roles


def get_args():
    parser = ArgumentParser(description="Analysis of PMI-based rules")
    parser.add_argument("--model", type=str, help="PMI-rule model")
    parser.add_argument("--api", type=str, help="API collection")
    parser.add_argument("--output", type=str, help="Folder to dump results")
    return parser.parse_args()


def main():
    args = get_args()

    with open(args.model, "rb") as fin:
        model = pickle.load(fin)
    df_rules = model.df_rules

    with open(args.api, "rb") as fin:
        api = pickle.load(fin)

    pmi_plot = distribution_norm_pmi(df_rules)
    get_coverage(api, df_rules)
    df_roles = distribution_roles(df_rules)
    print(df_roles)

    if args.output is not None:
        if not os.path.exists(args.output):
            os.makedirs(args.output)

        pmi_plot.get_figure().savefig(
            os.path.join(args.output, "norm_pmi_cdf.pdf")
        )

        # format
        df_roles.to_latex(os.path.join(args.output, "roles.tex"), index=False)


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        import pdb
        pdb.post_mortem()
