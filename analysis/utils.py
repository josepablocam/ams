import glob
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm


def get_table_paths(root):
    paths = glob.glob(os.path.join(root, "*", "*.pkl"))
    if len(paths) == 0:
        # flat, no folders for specific queries
        paths = glob.glob(os.path.join(root, "*.pkl"))
    return paths


def folder_to_num(folder):
    return int(folder.replace("q", "")) if folder.startswith("q") else -1


def collect_results(results_root):
    all_tables = get_table_paths(results_root)
    acc = []
    all_columns = set([])
    for path in tqdm.tqdm(all_tables):
        path_parts = path.split("/")
        if len(path_parts) >= 2 and path_parts[-2].startswith("q"):
            folder_name = path_parts[-2]
        else:
            folder_name = "unk_folder"
        try:
            df = pd.read_pickle(path)
            df["folder"] = folder_name
            if len(all_columns) == 0:
                all_columns = all_columns.union(df.columns)
            else:
                all_columns = all_columns.intersection(df.columns)
            acc.append(df)
        except Exception as err:
            print(err)
            print("Failed to load {} results file".format(path))
    # make sure only include columns all have
    all_columns = list(all_columns)
    acc = [d[all_columns] for d in acc]
    df = pd.concat(acc, axis=0).reset_index(drop=True)
    df["folder_num"] = df["folder"].map(folder_to_num)
    return df


def performance_table(df):
    summary_df = df.groupby([
        "folder_num",
        "dataset",
        "name",
    ]).score.agg(["mean", "std"])
    summary_df = summary_df.reset_index()
    return summary_df.sort_values(["folder_num", "dataset", "name"])


def filter_incomplete(df):
    # remove any queries where any system failed to finish
    # should only compare cases where everyone finishes
    df = df.copy()
    df = df[~df["score"].isnull()]
    all_systems = df["name"].unique()
    num_systems = len(all_systems)
    sys_cts = df.groupby(["folder_num", "dataset"])["name"].agg(lambda x: len(set(x)))
    ok_entries = set(sys_cts[sys_cts == num_systems].index.values)
    print("Keeping {}/{} folder/dataset combinations".format(len(ok_entries), sys_cts.shape[0]))
    is_ok = [pair in ok_entries for pair in zip(df["folder_num"], df["dataset"])]
    df = df[is_ok].reset_index(drop=True)
    return df


def count_top_scores(summary_df, complete_only=False, min_diff=None):
    summary_df = summary_df.sort_values("mean", ascending=False)

    # only consider queries where all systems completed...
    if complete_only:
        summary_df = filter_incomplete(summary_df)
    # top score by folder/dataset
    top_df = summary_df.groupby(["folder_num", "dataset"]).head(1)

    if min_diff is not None:
        diffs = summary_df.groupby(
            ["folder_num",
             "dataset"])["mean"].apply(lambda x: x.values[0] - x.values[1])
        diffs = diffs >= min_diff
        diffs = diffs.to_frame(name="sat_min_diff").reset_index()
        top_df_ext = pd.merge(
            top_df, diffs, how="left", on=["folder_num", "dataset"])
        top_df = top_df_ext[top_df_ext["sat_min_diff"]].reset_index(drop=True)

    # count of system entries for that folder (i..e # of datasets where it wins)
    top_df = top_df.groupby(["folder_num", "name"]).size().to_frame(name="ct")
    top_df = top_df.reset_index()
    top_df = pd.pivot(top_df, index="folder_num", columns="name", values="ct")
    top_df = top_df.fillna(0)
    return top_df


def time_table(df):
    pv = df.groupby(["folder",
                     "name"])["avg_exec_time_secs"].agg(["mean", "std"])
    pv = pv.reset_index().rename(columns={
        "mean": "mean_secs",
        "std": "std_secs"
    })
    pv = pv.reset_index()
    return pv


def emp_cdf(vals, cutoffs=None):
    xs = []
    ys = []
    for cutoff in cutoffs:
        frac = np.mean(vals <= cutoff)
        xs.append(cutoff)
        ys.append(frac)
    return xs, ys


def plot_emp_cdf(vals, cutoffs=None):
    xs, ys = emp_cdf(vals, cutoffs)
    fig, ax = plt.subplots(1)
    ax.plot(xs, ys)
    return ax
