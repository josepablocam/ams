#!/usr/bin/env python3
from argparse import ArgumentParser
from collections import defaultdict
import pickle

import numpy as np
import pandas as pd
import tqdm

from core.extract_parameters import UNKValue, ObservedCall
from core.extract_sklearn_api import (
    APICollection,
    APIClass,
    APIClassParameter,
)


def tabulate(parsed_scripts):
    call_id = 0
    acc = []
    for s in parsed_scripts:
        for c in s["calls"]:
            if len(c.args) == 0 and len(c.kwargs) == 0:
                acc.append((call_id, c.func, None, None, None))

            for ix, pos_arg in enumerate(c.args):
                key = "pos_{}".format(ix)
                type_ = pos_arg[0]
                value = pos_arg[1]
                acc.append((call_id, c.func, key, type_, value))

            for kwarg in c.kwargs:
                key, type_, value = kwarg
                acc.append((call_id, c.func, key, type_, value))

            call_id += 1
    df = pd.DataFrame(acc, columns=["id", "func", "key", "type", "value"])
    return df


def summarize(df_calls, default_value_map):
    # ignore calls with *no* arguments
    df_calls = df_calls[~df_calls["key"].isnull()]
    # ignore positional arguments, we can't do anything about those
    df_calls = df_calls[~df_calls["key"].str.startswith("pos")]

    # most tuned values, based on calls where we explicitly
    # observe the user set the value of that parameter
    df_keys = df_calls.groupby(["func", "key"]).size().to_frame(name="key_ct")
    df_keys = df_keys.reset_index()
    df_keys = df_keys.sort_values(
        ["func", "key_ct", "key"],
        ascending=[True, False, True],
    )

    # values for each key
    # NOTE: avoiding pandas groupby etc
    # because different types in `value` column (e.g. dtype, function)
    # can lead to type errors in groupby due to lack of order comparators
    # func -> key -> value -> ct
    value_cts = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: 0))
    )

    for ix, row in tqdm.tqdm(df_calls.iterrows()):
        if row.value == UNKValue():
            continue
        value_cts[row.func][row.key][row.value] += 1

    value_cts_acc = []
    for func, func_dict in value_cts.items():
        for key, key_dict in func_dict.items():
            key_values = sorted(
                list(key_dict.items()), key=lambda x: x[1], reverse=True
            )
            # drop the counts
            key_values = [value for value, _ in key_values]
            value_cts_acc.append((func, key, key_values))
    df_values = pd.DataFrame(value_cts_acc, columns=["func", "key", "value"])

    # add in default values for each
    default_values = []
    for _, row in tqdm.tqdm(df_values.iterrows()):
        value = default_value_map[(row.func, row.key)]
        default_values.append(value)
    df_values["default_value"] = default_values

    summary = {
        "keys": df_keys,
        "values": df_values,
    }
    return summary


def get_default_value_map(api_collection):
    dv_map = {}
    for cls in api_collection.classes:
        path = cls.path
        for param in cls.children:
            key = param.param_name
            dv = param.default_value
            dv_map[(path, key)] = dv

    return dv_map


def get_top_k_params(comp, summary, k, exclude_params=None):
    df = summary["keys"]
    if exclude_params is not None:
        df = df[~df["key"].isin(exclude_params)]
    return df[df["func"] == comp]["key"].values[:k]


def get_top_k_values(comp, param, summary, k, add_default=True):
    df = summary["values"]
    df = df[(df["func"] == comp) & (df["key"] == param)]
    if df.shape[0] == 0:
        return None
    vals = df["value"].values[0]
    # make copy, in case append
    vals = list(vals[:k])
    if add_default:
        default_value = df["default_value"].values[0]
        if default_value not in vals:
            vals.append(default_value)
    return vals


def get_args():
    parser = ArgumentParser(description="Summarize parameters for calls")
    parser.add_argument(
        "--input", type=str, help="Path to parsed scripts/calls"
    )
    parser.add_argument("--api", type=str, help="Path to pickled API")
    parser.add_argument("--output", type=str, help="Path to dump table")
    return parser.parse_args()


def main():
    args = get_args()
    with open(args.input, "rb") as fin:
        parsed_scripts = pickle.load(fin)

    df = tabulate(parsed_scripts)
    df.to_pickle(args.output + "-raw")

    with open(args.api, "rb") as fin:
        api = pickle.load(fin)
    default_value_map = get_default_value_map(api)

    df_summary = summarize(df, default_value_map)
    with open(args.output + "-summary", "wb") as fout:
        pickle.dump(df_summary, fout)


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        import pdb
        pdb.post_mortem()
