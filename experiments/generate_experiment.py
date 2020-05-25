#!/usr/bin/env python3
from argparse import ArgumentParser
import copy
import json
import os

import numpy as np
import sklearn
import tpot.config

### {stub_name: ({API_name:{manual config}}, natural language name)}
classifiers = {
    "lr": (
        {
            "sklearn.linear_model.LogisticRegression": {
                "penalty": ["l1", "l2"],
                "C": [0.01, 0.1, 1.0, 10.0, 100.0],
            }
        },
        "fit logistic regression classifier",
    ),
    "rf": (
        {
            "sklearn.ensemble.RandomForestClassifier": {
                "n_estimators": [
                    100,
                    150,
                    200,
                ],
                "max_depth": [None, 5, 10],
                "criterion": ["gini", "entropy"],
            }
        },
        "fit random forest classifier",
    ),
    "dt": (
        {
            "sklearn.tree.DecisionTreeClassifier": {
                "criterion": ["gini", "entropy"],
                "splitter": ["best", "random"],
            }
        },
        "fit decision tree classifier",
    ),
}

preprocessing_feats = {
    "scale": (
        {
            "sklearn.preprocessing.MinMaxScaler": {}
        },
        "scale values",
    ),
    "poly": (
        {
            "sklearn.preprocessing.PolynomialFeatures": {
                "degree": [2, 3],
                "interaction_only": [True, False],
            }
        },
        "add polynomial features",
    ),
}

postprocessing_feats = {
    "var": (
        {
            "sklearn.feature_selection.VarianceThreshold": {
                "threshold": [0.0, 1.0],
            }
        },
        "remove low variance features",
    ),
    "pca": (
        {
            "sklearn.decomposition.PCA": {
                "n_components": [0.1, 0.25, 0.5, 0.75],
                "whiten": [True, False],
            }
        },
        "reduce number of dimensions",
    )
}

COMPONENTS = {}
COMPONENTS.update(classifiers)
COMPONENTS.update(preprocessing_feats)
COMPONENTS.update(postprocessing_feats)


def generate_nl_query(comps):
    query = []
    for comp in comps:
        query.append(COMPONENTS[comp][1])
    return ", ".join(query)


def generate_simple_config_list(comps):
    config = []
    for comp in comps:
        component_name = list(COMPONENTS[comp][0].keys())[0]
        config.append(component_name)
    return config


def get_clean_params(params):
    clean_params = {}
    for k, v in params.items():
        if isinstance(v, (range, np.ndarray)):
            v = list(v)
        clean_params[k] = v
    return clean_params


def generate_simple_config_with_hyperparams_dict(comps, use_tpot=False):
    config = {}
    for comp in comps:
        entry = COMPONENTS[comp][0]
        if use_tpot:
            entry = copy.deepcopy(entry)
            comp = list(entry.keys())[0]
            tpot_params = copy.deepcopy(
                tpot.config.classifier_config_dict[comp]
            )
            tpot_params = get_clean_params(tpot_params)
            entry[comp] = tpot_params
        config.update(entry)
    return config


def get_args():
    parser = ArgumentParser(
        description="Generate NL and simple config for an experiment",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        nargs="+",
        help="Component sequence to generate NL/config",
    )
    parser.add_argument(
        "--use_tpot_hyperparams",
        action="store_true",
        help="Use TPOT default hyperparameter configuration instead of ours",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output directory",
    )
    return parser.parse_args()


def main():
    args = get_args()
    comps = args.experiment
    nl_query = generate_nl_query(comps)
    simple_config = generate_simple_config_list(comps)
    # outputs normal config (no spec order)
    simple_config_with_params_dict = generate_simple_config_with_hyperparams_dict(
        comps,
        use_tpot=args.use_tpot_hyperparams,
    )

    if not os.path.exists(args.output):
        print("Creating directory", args.output)
        os.makedirs(args.output)

    with open(os.path.join(args.output, "nl.txt"), "w") as fout:
        fout.write(nl_query + "\n")

    with open(os.path.join(args.output, "code.txt"), "w") as fout:
        code_query = " ".join(simple_config)
        fout.write(code_query + "\n")

    with open(os.path.join(args.output, "simple_config.json"), "w") as fout:
        json.dump(simple_config, fout)

    with open(os.path.join(args.output, "simple_config_with_params_dict.json"),
              "w") as fout:
        json.dump(simple_config_with_params_dict, fout)


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        import pdb
        pdb.post_mortem()
