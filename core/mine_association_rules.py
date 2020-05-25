#!/usr/bin/env python3
from argparse import ArgumentParser
from collections import Counter
import itertools
import pickle
import warnings

import numpy as np
import pandas as pd
import sklearn.base
import tqdm

from core.extract_parameters import *
from core import utils


def create_transactions(annotated_scripts):
    transactions = []
    for script in annotated_scripts:
        calls = script["calls"]
        api_elems_called = {call.func for call in calls}
        if len(api_elems_called) > 0:
            transactions.append(list(api_elems_called))
    return transactions


def compute_pmi_df(transactions):
    all_pairs = [
        pair for ts in transactions
        for pair in itertools.combinations(set(ts), 2)
    ]
    pair_counts = Counter(all_pairs)
    n = float(len(all_pairs))
    bi_probs = {}

    unique_pairs = set([frozenset(p) for p in pair_counts.keys()])
    unique_pairs = [tuple(p) for p in unique_pairs]
    for pair in unique_pairs:
        rpair = (pair[1], pair[0])
        bi_ct = pair_counts.get(pair, 0) + pair_counts.get(rpair, 0)
        bi_probs[pair] = bi_ct / n

    uni_probs = {}
    single = set([e for ts in transactions for e in ts])
    for e in single:
        uni_ct = sum([ct for pair, ct in pair_counts.items() if e in pair])
        uni_probs[e] = uni_ct / n

    pmi_results = []
    for pair, p_first_second in bi_probs.items():
        first_elem, second_elem = pair
        p_first = uni_probs[first_elem]
        p_second = uni_probs[second_elem]
        res = np.log2(p_first_second / (p_first * p_second))
        # normalized PMI
        norm_res = res / (-1 * np.log2(p_first_second))
        pmi_results.append((first_elem, second_elem, p_first, p_second,
                            p_first_second, norm_res))

    df = pd.DataFrame(
        pmi_results, columns=['comp1', 'comp2', 'p1', 'p2', 'p12', 'norm_pmi'])
    df = df.sort_values('norm_pmi', ascending=False)
    df = df[df["norm_pmi"] > 0].reset_index(drop=True)
    return df


def annotate_regression(pmi_df):
    pmi_df = pmi_df.copy()
    unique_components = pmi_df["comp1"].values.tolist()
    unique_components.extend(pmi_df["comp2"].values.tolist())
    is_regressor = {}
    for comp in tqdm.tqdm(unique_components):
        try:
            obj = utils.get_component_constructor(comp)
            is_regressor[comp] = sklearn.base.is_regressor(obj)
        except Exception:
            print("Failed verifying if {} is regressor, assume".format(comp))
            # assume it is, so exclude
            is_regressor[comp] = True

    pmi_df["is_regression"] = pmi_df["comp1"].map(
        is_regressor) | pmi_df["comp2"].map(is_regressor)
    return pmi_df


def get_role(obj):
    if sklearn.base.is_regressor(obj):
        return "regressor"
    elif sklearn.base.is_classifier(obj):
        return "classifier"
    else:
        return "other"


def remove_conflicting_roles(pmi_df):
    pmi_df = pmi_df.copy()
    is_ok = []

    for _, row in tqdm.tqdm(pmi_df.iterrows()):
        try:
            obj1 = utils.get_component_constructor(row["comp1"])
            role1 = get_role(obj1)
        except Exception:
            is_ok.append(False)
            continue
        try:
            obj2 = utils.get_component_constructor(row["comp2"])
            role2 = get_role(obj2)
        except Exception:
            is_ok.append(False)
            continue

        bad_combinations = [
            ("classifier", "regressor"),
            ("regressor", "classifier"),
        ]
        if (role1, role2) in bad_combinations:
            is_ok.append(False)
        else:
            is_ok.append(True)

    print("Keeping {}/{} rules".format(np.sum(is_ok), len(is_ok)))
    pmi_df = pmi_df[is_ok].reset_index(drop=True)
    return pmi_df


def remove_bad_components(pmi_df):
    bad_endings = ("CV", "Loss")
    bad = pmi_df["comp1"].str.endswith(
        bad_endings) | pmi_df["comp2"].str.endswith(bad_endings)
    return pmi_df[~bad].reset_index(drop=True)


def query_pmi_df(comp, pmi_df, k=None, is_classification=False):
    relevant = pmi_df[(pmi_df.comp1 == comp) | (pmi_df.comp2 == comp)]
    if is_classification:
        relevant = relevant[~relevant["is_regression"]]
    if k is not None:
        relevant = relevant.sort_values("norm_pmi", ascending=False).head(k)
    return relevant


class APIRulePredictor(object):
    def __init__(self, transactions):
        self.df_rules = None
        ok_input = self.validate_transactions(transactions)
        if not ok_input:
            raise Exception(
                "Input must be iterable of iterable of transactions")
        pmi_df = compute_pmi_df(transactions)
        pmi_df = remove_bad_components(pmi_df)
        pmi_df = annotate_regression(pmi_df)
        pmi_df = remove_conflicting_roles(pmi_df)
        self.df_rules = pmi_df

    def validate_transactions(self, transactions):
        try:
            first_elem = transactions[0][0]
            assert isinstance(first_elem, str)
            return True
        except (KeyError, AssertionError) as err:
            return False

    def extend(self, transaction, alpha=0.5, k=3, is_classification=True):
        transaction = set(transaction)
        results = []
        for comp in set(transaction):
            res = query_pmi_df(
                comp,
                self.df_rules,
                is_classification=is_classification,
            )
            for _, row in res.iterrows():
                norm_pmi = row["norm_pmi"]
                new_comp = row["comp2"] if row["comp1"] == comp else row[
                    "comp1"]
                if new_comp not in transaction:
                    results.append((new_comp, norm_pmi))

        if len(results) == 0:
            # perform no extension
            return set(transaction)

        results_df = pd.DataFrame(
            results, columns=["new_component", "norm_pmi"])
        norm_pmi_df = results_df.groupby("new_component")[["norm_pmi"]].mean()
        norm_ct_df = results_df.groupby("new_component").size().to_frame(
            name="norm_ct") / len(transaction)
        comb_df = pd.merge(
            norm_pmi_df, norm_ct_df, how="left", on="new_component")
        comb_df["score"] = comb_df["norm_pmi"] * alpha + comb_df["norm_ct"] * (
            1.0 - alpha)
        comb_df = comb_df.reset_index()
        comb_df = comb_df.sort_values("score", ascending=False)

        new_transaction = set([])
        new_transaction.update(transaction)
        new_transaction.update(
            comb_df["new_component"].head(k).values.tolist())
        return new_transaction


def get_args():
    parser = ArgumentParser(
        description="Mine association rules from API calls based on PMI")
    parser.add_argument(
        "--input",
        type=str,
        help="Pickled list of annotated Kaggle scripts",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output path for pickled association rules predictor",
    )
    return parser.parse_args()


def main():
    args = get_args()
    with open(args.input, "rb") as fin:
        annotated_scripts = pickle.load(fin)
    transactions = create_transactions(annotated_scripts)
    predictor = APIRulePredictor(transactions)
    n_rules = predictor.df_rules.shape[0]
    print("Mined {} rules using PMI".format(n_rules))
    with open(args.output, "wb") as fout:
        pickle.dump(predictor, fout)


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        import pdb
        pdb.post_mortem()
