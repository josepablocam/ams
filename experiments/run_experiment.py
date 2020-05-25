#!/usr/bin/env python3
from argparse import ArgumentParser
import copy
from datetime import datetime
import os
import pickle
import random
import traceback
import sys

import json
import numpy as np
import pandas as pd
import pmlb
import tpot
import tqdm
import sklearn.base
import sklearn.pipeline
import sklearn.metrics
from sklearn.model_selection import StratifiedKFold, cross_validate

import download_datasets as dd
import simple_pipeline

import sys
sys.path.append("../")
sys.path.append(".")
import core.search
from core.search import (FailedOptim, RobustSearch)
from core import mp_utils

MAX_TIME_MINS_PER_PIPELINE = 1


def get_robust_tpot(
    config_dict=None,
    max_time_mins=5,
    scoring="f1_macro",
    cv=5,
    random_state=42,
    n_jobs=-1,
    check_point_folder=None,
    verbosity=2,
):
    clf = RobustSearch(
        search_model=tpot.TPOTClassifier(
            config_dict=config_dict,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
            max_time_mins=max_time_mins,
            # max on a single timeline...otherwise can blow out
            # and end up with not a single pipeline fit
            max_eval_time_mins=MAX_TIME_MINS_PER_PIPELINE,
            random_state=random_state,
            verbosity=verbosity,
            disable_update_check=True,
        )
    )
    return clf


def get_robust_random(
    config_dict,
    max_depth=4,
    max_time_mins=5,
    scoring="f1_macro",
    cv=5,
    random_state=42,
    n_jobs=-1,
):
    clf = RobustSearch(
        search_model=core.search.RandomSearch(
            config_dict=config_dict,
            max_depth=max_depth,
            scoring=scoring,
            cv=cv,
            max_time_mins=max_time_mins,
            # max on a single timeline...otherwise can blow out
            # and end up with not a single pipeline fit
            max_time_mins_per_pipeline=MAX_TIME_MINS_PER_PIPELINE,
            random_state=random_state,
        )
    )
    return clf


def get_robust_predefined_random(
    config_list,
    max_depth=4,
    max_time_mins=5,
    scoring="f1_macro",
    cv=5,
    random_state=42,
    n_jobs=-1,
):
    clf = RobustSearch(
        search_model=core.search.DefinedPipelineRandomHyperParamSearch(
            config_dict=config_list,
            max_depth=list(config_list),
            scoring=scoring,
            cv=cv,
            max_time_mins=max_time_mins,
            random_state=random_state,
        )
    )
    return clf


def get_no_hyperparams_config(config_dict):
    # drop hyperparameters from configuration
    return {k: {} for k in config_dict.keys()}


def get_simple_pipeline(config):
    return simple_pipeline.generate_pipeline(config)


def get_scoring(scoring):
    if scoring == "balanced_accuracy_score":
        return sklearn.metrics.make_scorer(
            sklearn.metrics.balanced_accuracy_score
        )
    else:
        return scoring


def get_num_pipelines_explored(model):
    if isinstance(model, sklearn.pipeline.Pipeline):
        return 1
    elif isinstance(model, tpot.TPOTClassifier):
        return len(model.evaluated_individuals_)
    elif isinstance(model, core.search.CustomSearch):
        return len(model.pipelines_tried_)
    elif isinstance(model, FailedOptim):
        return 0
    else:
        raise Exception("Unknown search model")


def limit_poly_features_in_config(config, X, max_cols=50, max_degree=2):
    # Trying to generate degrees of order 4
    # with anything more than a couple of columns
    # quickly blows up
    # copy in case we modify it
    config = copy.deepcopy(config)
    if X.shape[1] < max_cols:
        return config

    params = None

    poly_comp = "sklearn.preprocessing.PolynomialFeatures"
    if isinstance(config, dict):
        params = config.get(poly_comp, None)
    elif isinstance(config, list) and isinstance(config[0], str):
        # its a list configuration without hyperparamers
        return config
    else:
        # its a list configuration, for specified order
        entry = [
            comp_dict for comp_dict in config if poly_comp in comp_dict.keys()
        ]
        params = None if len(entry) == 0 else entry[0][poly_comp]

    if params is None or 'degree' not in params:
        # not relevant, or using default (degree=2), so good to go
        return config
    else:
        # set a max on the degree
        params['degree'] = [d for d in params['degree'] if d <= max_degree]
        return config


def fetch_data(dataset, cache_dir):
    try:
        X, y = pmlb.fetch_data(
            dataset,
            return_X_y=True,
            local_cache_dir=cache_dir,
        )
    except ValueError:
        path = os.path.join(cache_dir, dataset)
        df = pd.read_csv(path)
        y_col = "target"
        X_cols = [c for c in df.columns if c != y_col]
        X = df[X_cols].values
        y = df[y_col].values
    return X, y


def run_dataset(
    dataset,
    search,
    config=None,
    max_time_mins=5,
    max_depth=4,
    cv=10,
    scoring="f1_macro",
    n_jobs=-1,
    random_state=None,
):
    X, y = fetch_data(dataset, cache_dir=dd.DEFAULT_LOCAL_CACHE_DIR)

    cv_splitter = StratifiedKFold(
        cv,
        random_state=random_state,
        shuffle=True,
    )
    scoring_fun = get_scoring(scoring)

    config = limit_poly_features_in_config(config, X)

    if search == "tpot":
        # running search with tpot
        model = get_robust_tpot(
            config_dict=config,
            max_time_mins=max_time_mins,
            scoring=scoring_fun,
            n_jobs=n_jobs,
            random_state=random_state,
        )
    elif search == "random":
        model = get_robust_random(
            config_dict=config,
            max_depth=max_depth,
            max_time_mins=max_time_mins,
            scoring=scoring_fun,
            random_state=random_state,
            n_jobs=1,
        )
    elif search == "simple":
        model = get_simple_pipeline(config)
    elif search == "predefined-with-hyperparams":
        model = get_robust_predefined_random(
            config,
            max_time_mins=max_time_mins,
            scoring=scoring_fun,
        )
    else:
        raise TypeError(
            "configuration must be dictionary (automl) or list (simple)"
        )
    start_time = datetime.now()
    results = cross_validate(
        model,
        X,
        y,
        cv=cv_splitter,
        scoring=scoring_fun,
        return_estimator=True,
        return_train_score=True,
    )
    end_time = datetime.now()
    exec_time = (end_time - start_time).total_seconds()
    nrows = len(results['test_score'])

    if search == "simple":
        fitted_pipelines = [model] * nrows
        scores = results["test_score"]
        train_scores = results["train_score"]
        pipelines_explored = get_num_pipelines_explored(model)
        # we don't track errors for simple pipelines, since debugging them
        # is not tricky
        errors = [None] * nrows
    else:
        # TOPT and ours can fail during fitting...
        fitted_pipelines = [e.fitted_pipeline_ for e in results["estimator"]]
        # replace scores with np.nan if produced by a failed optimization
        scores = [
            score
            if not isinstance(estimator.search_model, FailedOptim) else np.nan
            for score, estimator in
            zip(results['test_score'], results["estimator"])
        ]
        # training scores
        train_scores = [
            score
            if not isinstance(estimator.search_model, FailedOptim) else np.nan
            for score, estimator in
            zip(results['train_score'], results["estimator"])
        ]
        # keep track of errors, so we can debug searches
        errors = [
            estimator.search_model
            if isinstance(estimator.search_model, FailedOptim) else None
            for estimator in results["estimator"]
        ]
        # pipelines explored
        pipelines_explored = [
            get_num_pipelines_explored(estimator.search_model)
            for estimator in results["estimator"]
        ]

    results_info = {
        "score": scores,
        "train_score": train_scores,
        "cv_iter": np.arange(0, nrows),
        "dataset": dataset,
        "scoring": scoring,
        "config_dict": [config] * nrows,
        "max_time_mins": max_time_mins,
        "fitted_pipeline": fitted_pipelines,
        "pipelines_explored": pipelines_explored,
        "avg_exec_time_secs": exec_time / nrows,
        "errors": errors,
    }
    results_df = pd.DataFrame(results_info)
    return results_df


def to_df_and_save(acc, name, output):
    acc_df = pd.concat(acc, axis=0)

    if name is not None:
        acc_df["name"] = name
    else:
        acc_df["name"] = "unk"

    if output is not None:
        acc_df.to_pickle(output)

    return acc_df


def load_config(poss_config):
    if isinstance(poss_config, str) and poss_config == "TPOT":
        return copy.deepcopy(tpot.config.classifier_config_dict)
    try:
        config = json.loads(poss_config)
        return config
    except json.JSONDecodeError:
        with open(poss_config, "r") as fin:
            return json.load(fin)


def get_args():
    parser = ArgumentParser(description="Run experiments")
    parser.add_argument(
        "--dataset",
        type=str,
        nargs="+",
        help="Name of datasets to run",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="String dictionary for config_dict or path to file",
    )
    parser.add_argument(
        "--search",
        type=str,
        help="Search strategy",
        choices=[
            "tpot",
            "random",
            "simple",
            "predefined-with-hyperparams",
        ]
    )
    parser.add_argument(
        "--cv",
        type=int,
        help="Number of CV iters",
        default=10,
    )
    parser.add_argument(
        "--scoring",
        type=str,
        help="Scoring function",
        default="f1_macro",
    )
    parser.add_argument(
        "--max_time_mins",
        type=int,
        help="Time budget for each outer cv iteration",
        default=5,
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        help="Number of cores to use",
        default=-1,
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        help="Max search depth for random search",
        default=4,
    )
    parser.add_argument(
        "--components_only",
        action="store_true",
        help="Drop hyperparameters from configuration",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        help="Seed for RNG",
        default=42,
    )
    parser.add_argument(
        "--name",
        type=str,
        help="Name for experiment",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output path for results",
    )
    return parser.parse_args()


def main():
    args = get_args()
    config = None
    if args.config is not None:
        config = load_config(args.config)

    if config is not None and args.components_only:
        print("Dropping hyper-parameters from configuration")
        config = get_no_hyperparams_config(config)

    acc = []
    if args.name is not None:
        print("Running run_experiment.py, name={}".format(args.name))

    if args.random_state:
        # adding more set seeds....something deep down
        # in tpot/sklearn not actually taking the random seed otherwise
        np.random.seed(args.random_state)
        random.seed(args.random_state)

    if args.output is not None:
        dir_path = os.path.dirname(args.output)
        if len(dir_path) > 0:
            os.makedirs(dir_path, exist_ok=True)

    for dataset in tqdm.tqdm(args.dataset):
        print("Running dataset", dataset)
        results = run_dataset(
            dataset,
            search=args.search,
            config=config,
            max_depth=args.max_depth,
            max_time_mins=args.max_time_mins,
            cv=args.cv,
            scoring=args.scoring,
            n_jobs=args.n_jobs,
            random_state=args.random_state,
        )
        acc.append(results)
        acc_df = to_df_and_save(acc, args.name, args.output)

    print(acc_df.groupby(["dataset", "name"])["score"].agg(["mean", "std"]))


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        args = get_args()
        err_path = args.output + "-error"
        with open(err_path, "wb") as fout:
            pickle.dump(err, fout)

        detailed_msg = traceback.format_exc()
        tb_path = args.output + "-tb"
        with open(tb_path, "w") as fout:
            fout.write(detailed_msg)
            fout.write("\n")

        failed_args_path = args.output + "-args"
        with open(failed_args_path, "wb") as fout:
            pickle.dump(args, fout)

        import pdb
        pdb.post_mortem()
        sys.exit(1)
