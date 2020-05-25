#!/usr/bin/env python3
from argparse import ArgumentParser
import ast
import astunparse
import pickle

import jedi
import tqdm

from core.extract_sklearn_api import (
    APICollection,
    APIClass,
    APIClassParameter,
)


class UNKValue(object):
    def __init__(self):
        self.str_ = "UNK"

    def __repr__(self):
        return self.str_

    def __str__(self):
        return self.str_

    def __hash__(self):
        return 42

    def __eq__(self, other):
        return isinstance(other, UNKValue)

    def __lt__(self, v):
        return 0 < v

    def __le__(self, v):
        return 0 <= v


def get_ast_subtree(src):
    try:
        return ast.parse(src).body[0]
    except SyntaxError:
        return None


def unparse(tree):
    return astunparse.unparse(tree).strip()


def is_assign(tree):
    return isinstance(tree, ast.Assign)


def is_call(tree):
    return isinstance(tree, ast.Call)


def get_func_str(tree):
    return unparse(tree.func)


def get_assigned_calls(source, api_basenames=None):
    skip = (
        "class",
        "module",
        "def",
    )
    assignments = [
        n for n in jedi.names(source) if not n.description.startswith(skip)
    ]
    trees = [get_ast_subtree(n.description) for n in assignments]
    trees = [t for t in trees if t is not None]
    calls_trees = [t.value for t in trees if is_assign(t) and is_call(t.value)]
    if api_basenames is not None:
        calls_trees = [
            c for c in calls_trees
            if get_func_str(c).split(".")[-1] in api_basenames
        ]
    return calls_trees


def arg_value(tree):
    if isinstance(tree, ast.Num):
        return tree.n
    elif isinstance(tree, ast.Str):
        return tree.s
    elif isinstance(tree, ast.NameConstant):
        return tree.value
    else:
        try:
            val = eval(astunparse.unparse(tree))
            if val is None or isinstance(val, (
                    int,
                    float,
                    bool,
                    str,
            )):
                return val
            else:
                return UNKValue()
        except:
            return UNKValue()


def type_str(e):
    return str(type(e))


class ObservedCall(object):
    def __init__(self, call_ast, api_map):
        func_str = get_func_str(call_ast)
        basename = func_str.split(".")[-1]

        self.func = api_map[basename].path
        self.code = unparse(call_ast)
        args = [arg_value(a) for a in call_ast.args]
        self.args = [(type_str(v), v) for v in args]

        # (arg name, arg type, arg value)
        kwargs = []
        observed_kwargs = {kwarg.arg: kwarg for kwarg in call_ast.keywords}
        valid_kwargs = [p.param_name for p in api_map[basename].children]
        for key in observed_kwargs:
            if key not in valid_kwargs:
                # skip if not part of current API
                continue
            val = arg_value(observed_kwargs[key].value)
            type_ = type_str(val)
            rec = (key, type_, val)
            kwargs.append(rec)

        self.kwargs = kwargs

    def __str__(self):
        acc = []
        for a in self.args:
            acc.append(str(a[1]))

        for kw in self.kwargs:
            acc.append("{}={}".format(kw[0], kw[2]))

        acc_str = ",".join(acc)
        return "{func}({args})".format(func=self.func, args=acc_str)


def collect_observed_calls(script, api_map=None):
    # only care about assigned calls
    api_basenames = set(api_map.keys())
    calls = get_assigned_calls(script, api_basenames=api_basenames)
    parsed_calls = [ObservedCall(c, api_map) for c in calls]
    return parsed_calls


def create_api_map(api_collection):
    api_map = {}
    for c in api_collection.classes:
        path = c.path
        basename = path.split(".")[-1]
        api_map[basename] = c
    return api_map


def get_args():
    parser = ArgumentParser(description="Extract calls and arguments")
    parser.add_argument("--input", type=str, help="Pickled scripts")
    parser.add_argument("--output", type=str, help="Path to save results")
    parser.add_argument("--api", type=str, help="Path to pickled API")
    return parser.parse_args()


def main():
    args = get_args()
    with open(args.input, "rb") as fin:
        scripts_info = pickle.load(fin)

    api_map = None
    with open(args.api, "rb") as fin:
        api = pickle.load(fin)
        api_map = create_api_map(api)

    scripts_info_parsed = []
    for ix, script in tqdm.tqdm(enumerate(scripts_info)):
        script = dict(script)
        obs_calls = collect_observed_calls(script["source"], api_map=api_map)
        script["calls"] = obs_calls
        scripts_info_parsed.append(script)

    with open(args.output, "wb") as fout:
        pickle.dump(scripts_info_parsed, fout)

    print(
        "Collected",
        len([s for s in scripts_info_parsed if len(s["calls"]) > 0]),
        "scripts with calls"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        import pdb
        pdb.post_mortem()
