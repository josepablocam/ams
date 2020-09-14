#!/usr/bin/env python3
from argparse import ArgumentParser
import json
import os
import pickle
import random

import jedi
import pandas as pd
import tqdm


def imports_apis(script, apis):
    if apis is None or len(apis) == 0:
        return True
    names = jedi.names(script["source"])
    imports = [n for n in names if n.description.startswith("module")]
    for imp in imports:
        if imp.full_name.startswith(apis):
            return True
    return False


def get_source(source, is_notebook=False):
    if not is_notebook:
        return source
    # TODO: this may not work because of spaces etc
    source = json.loads(source)
    cells = source["cells"]
    code = []
    for c in cells:
        if c["cell_type"] == "code":
            c_code = c["source"]
            if isinstance(c_code, list):
                code.extend(c_code)
            else:
                code.append(c["source"])
    return "\n".join(code)


def python_language_ids(db_path):
    df_langs = pd.read_csv(os.path.join(db_path, "ScriptLanguages.csv"))
    return df_langs[df_langs["AceLanguageName"] == "python"]["Id"].values


def notebook_ids(db_path):
    df_langs = pd.read_csv(os.path.join(db_path, "ScriptLanguages.csv"))
    return df_langs[df_langs["Name"].str.startswith("IPython")]["Id"].values


def query_db_scripts(db_path):
    python_lang_ids = python_language_ids(db_path)
    scripts = pd.read_csv(os.path.join(db_path, "ScriptVersions.csv"))
    scripts = scripts[scripts["ScriptLanguageId"].isin(python_lang_ids)]
    scripts["DateCreated"] = pd.to_datetime(scripts["DateCreated"])
    scripts = scripts[scripts["IsChange"]].reset_index(drop=True)
    scripts = scripts.sort_values(["ScriptId", "DateCreated"])
    scripts = scripts.groupby("ScriptId").tail(1)
    nb_ids = notebook_ids(db_path)
    results = []
    for ix, row in tqdm.tqdm(scripts.iterrows()):
        if pd.isnull(row["ScriptContent"]):
            continue

        try:
            script_content = get_source(
                row["ScriptContent"],
                is_notebook=row["ScriptLanguageId"] in nb_ids,
            )
        except json.JSONDecodeError:
            script_content = get_source(
                row["ScriptContent"],
                is_notebook=False,
            )

        res = {
            "id": row["Id"],
            "source": script_content,
        }
        results.append(res)
    return results


def load_dict_scripts(file_path):
    with open(file_path, "rb") as fin:
        contents = pickle.load(fin)
        results = []
        for entry in tqdm.tqdm(contents):
            if pd.isnull(entry["source"]):
                continue
            try:
                # just assume its a notebook
                script_content = get_source(
                    entry["source"],
                    is_notebook=True,
                )
            except json.JSONDecodeError:
                script_content = get_source(
                    entry["source"],
                    is_notebook=False,
                )

            res = {
                "id": entry["id"],
                "source": script_content,
            }
            results.append(res)
        return results


def get_args():
    parser = ArgumentParser(
        description=
        "Get Python scripts from meta kaggle that import relevant apis")
    parser.add_argument("--input", type=str, help="Path to meta kaggle files")
    parser.add_argument(
        "--output",
        type=str,
        help="Path to dump pickled list of script contents",
    )
    parser.add_argument(
        "--api",
        type=str,
        nargs="+",
        help="Relevant APIs",
    )
    parser.add_argument(
        "--sample",
        type=float,
        help="Fraction of total scripts to sample",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="RNG seed (useful when sampling)",
    )
    return parser.parse_args()


def main():
    args = get_args()

    apis = tuple(args.api)
    if args.input.endswith(".pkl"):
        # assume it's a dictionary of the form {"id": XXX, "source": XXX}
        scripts = load_dict_scripts(args.input)
    else:
        scripts = query_db_scripts(args.input)
    scripts = [s for s in tqdm.tqdm(scripts) if imports_apis(s, apis)]

    if args.seed is not None:
        random.seed(args.seed)

    if args.sample is not None:
        random.shuffle(scripts)
        n = int(args.sample * len(scripts))
        scripts = scripts[:n]

    print("# Scripts", len(scripts))
    with open(args.output, "wb") as fout:
        pickle.dump(scripts, fout)


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        import pdb
        pdb.post_mortem()
