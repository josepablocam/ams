#!/usr/bin/env python3
from argparse import ArgumentParser
import inspect
import json
import pickle
import pprint

import pandas as pd

from core import code_to_api

# imports for purposes of unpickling
from core.extract_sklearn_api import (
    APICollection,
    APIClass,
    APIClassParameter,
)
from core.extract_parameters import UNKValue
from core.mine_association_rules import APIRulePredictor
from core.summarize_parameters import (
    get_top_k_params,
    get_top_k_values,
)


def remove_regression_components(api_elems):
    return [(e, score) for e, score in api_elems if not e.is_regression_op]


def filter_api_elems(api_elems, perform_ad_hoc_filter, is_classification):
    if perform_ad_hoc_filter:
        api_elems = remove_components_with_empty_defaults(api_elems)
        api_elems = remove_components_with_cv(api_elems)
        api_elems = remove_components_loss_only(api_elems)
    if is_classification:
        api_elems = remove_regression_components(api_elems)
    return api_elems


def get_sorted_code_matches(
    matched,
    k,
    is_classification=False,
    perform_ad_hoc_filter=True,
):
    results = {}
    for code_elem, api_elems in matched.items():
        api_elems = filter_api_elems(
            api_elems,
            perform_ad_hoc_filter,
            is_classification,
        )
        api_elems = sorted(api_elems, key=lambda x: x[1], reverse=True)
        # no heuristic ranking for code version
        # drop scores
        api_elems = [e for e, _ in api_elems[:k]]
        # extract sklearn path for component
        api_elems_paths = [e.path for e in api_elems]
        results[code_elem] = api_elems_paths
    return results


def parse_include_exclude_specification(code):
    includes = []
    excludes = []
    # after removing includ/exclude spec info
    clean_code = []
    parts = [s.split(":") for s in code]
    for elem in parts:
        if len(elem) == 1:
            # no hard/soft spec
            clean_code.append(elem[0])
            continue
        comp, req_spec = elem
        if req_spec.strip() == "1":
            includes.append(comp)
        elif req_spec.strip() == "0":
            excludes.append(comp)
        else:
            raise ValueError("Unknown specification constraint: " + req_spec)
        clean_code.append(comp)
    code = clean_code
    return code, includes, excludes


def handle_include_exclude_specification(code_spec, config):
    _, includes, excludes = parse_include_exclude_specification(code_spec)
    for comp in includes:
        if comp not in config:
            config[comp] = {}
    for comp in excludes:
        if comp in config:
            config.pop(comp)
    return config


def generate_components_configuration_from_code(
    api_collection,
    code,
    k,
    is_classification=False,
    strategy="bm25",
):
    code, _, _ = parse_include_exclude_specification(code)
    matched = code_to_api.compute_matches(api_collection, code, strategy)
    sorted_matches = get_sorted_code_matches(
        matched,
        k,
        is_classification=is_classification,
    )
    components = set([])
    for code_elem, api_paths in sorted_matches.items():
        components.update(api_paths)
    config = {}
    for c in components:
        config[c] = {}
    return config


def generate_complementary_components_from_specification(
    model,
    code,
    alpha,
    k,
    is_classification=False,
):
    code, _, _ = parse_include_exclude_specification(code)
    extended = model.extend(
        code,
        k=k,
        alpha=alpha,
        is_classification=is_classification,
    )
    return {c: {} for c in extended}


def remove_components_with_empty_defaults(api_elems):
    # we don't handle classes that have required arguments
    # in constructors without defaults (i.e. they will fail
    # to build under current automl search approach)
    elems_with_defaults = []
    for (elem, score) in api_elems:
        args = elem.children
        if any(a.default_value == inspect._empty for a in args):
            continue
        elems_with_defaults.append((elem, score))
    return elems_with_defaults


def remove_components_from_config_with_empty_defaults(config, api_collection):
    api_map = {c.path: c for c in api_collection.classes}
    # add in a dummy score and map to api elem
    config_components = []
    for c in config.keys():
        if c not in api_map:
            continue
        # add dummy score
        config_components.append((api_map[c], None))
    config_components_with_defs = remove_components_with_empty_defaults(
        config_components
    )
    allowed = [c.path for c, _ in config_components_with_defs]
    # subset configuration down to this
    return {c: v for c, v in config.items() if c in allowed}


def remove_components_with_cv(api_elems):
    # some components internally apply CV, we want to exclude these
    # unnecessary
    return [(e, score) for e, score in api_elems if not e.path.endswith("CV")]


def remove_components_loss_only(api_elems):
    # some components are just loss functions, remove these
    return [(e, score) for e, score in api_elems
            if not e.path.endswith("Loss")]


def heuristic_post_rank(input_tokens, input_len, elems):
    for ix, (elem, sim_score) in enumerate(elems):
        elem_tokens = elem.filter_tokens
        overlap_ct = 0.0
        for token in input_tokens:
            if token in elem_tokens:
                overlap_ct += 1.0
            elif any(token in t for t in elem_tokens):
                # token is prefix, in, or suffix of a token in API
                overlap_ct += 1.0
            elif any(t in token for t in elem_tokens):
                # api token is prefix, in, or suffix of given NL token
                overlap_ct += 1.0
            else:
                continue
        post_rank_score = overlap_ct / input_len
        elems[ix] = (elem, (post_rank_score, sim_score))
    elems = sorted(elems, key=lambda x: x[1], reverse=True)
    return elems


def generate_parameters(
    component,
    summary,
    num_params,
    num_values,
    exclude_params=None,
    add_default=True,
):

    chosen_params = get_top_k_params(
        component,
        summary,
        num_params,
        exclude_params=exclude_params,
    )

    params_config = {}
    for p in chosen_params:
        vals = get_top_k_values(
            component,
            p,
            summary,
            num_values,
            add_default=add_default,
        )
        if vals is not None:
            params_config[p] = vals
    return params_config


def generate_hyperparameter_configuration(
    config,
    parameter_summary,
    num_params,
    num_values,
    exclude_params=None,
):
    for comp in config.keys():
        config[comp] = generate_parameters(
            comp,
            parameter_summary,
            num_params,
            num_values,
            exclude_params=exclude_params,
        )
    return config


def get_args():
    parser = ArgumentParser(description="Generate search space from NL")
    parser.add_argument(
        "--code",
        type=str,
        nargs="+",
        help="Code-based weak specification",
    )
    parser.add_argument(
        "--classification",
        action="store_true",
        help="Problem is classification, drop regression operators",
    )
    parser.add_argument(
        "--strategy",
        choices=["bm25"],
        type=str,
        help="Search strategy for related components",
        default="bm25",
    )
    parser.add_argument(
        "--num_components",
        type=int,
        help="Number of API components per NL phrase matched",
    )
    parser.add_argument(
        "--rule_mining_model", type=str, help="Association rule model"
    )
    parser.add_argument(
        "--num_association_rules",
        type=int,
        help="Top K association rules",
        default=3,
    )
    parser.add_argument(
        "--alpha",
        type=float,
        help="Weight to combine NPMI and normalized ct of support into score",
        default=0.5,
    )
    parser.add_argument(
        "--num_params",
        type=int,
        help="Number of hyperparameters per component to tune",
    )
    parser.add_argument(
        "--num_param_values",
        type=int,
        help="Number of values per hyperparameter to search",
    )
    parser.add_argument(
        "--exclude_params",
        type=str,
        nargs="+",
        help="Remove parameters we don't want to tune (e.g. random_state)",
    )
    parser.add_argument("--api", type=str, help="Pickled API Collection")
    parser.add_argument(
        "--params",
        type=str,
        help="Pickled DF summary of parameters",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="File to output JSON configuration",
    )
    return parser.parse_args()


def main():
    args = get_args()
    spec = args.code
    spec_source = "code"

    if (spec_source == "code" and spec[0].endswith(".txt")):
        print("Treating --{} as file, ends with txt".format(spec_source))
        spec = spec[0]
        with open(spec, "r") as fin:
            spec = fin.read()
        spec = spec.split()

    k = args.num_components

    with open(args.api, "rb") as fin:
        api_collection = pickle.load(fin)

    # functionally related components
    config = generate_components_configuration_from_code(
        api_collection,
        spec,
        k,
        is_classification=args.classification,
        strategy=args.strategy,
    )
    # complementary components
    if args.rule_mining_model is not None:
        with open(args.rule_mining_model, "rb") as fin:
            model = pickle.load(fin)

        comp_config = generate_complementary_components_from_specification(
            model,
            spec,
            k=args.num_association_rules,
            alpha=args.alpha,
            is_classification=args.classification,
        )
        # add to the extended
        config.update(comp_config)

    # remove/include any components according to spec exclude/include info
    config = handle_include_exclude_specification(spec, config)

    config = remove_components_from_config_with_empty_defaults(
        config,
        api_collection,
    )

    if args.params is not None:
        param_summary = pd.read_pickle(args.params)
        num_params = args.num_params
        num_values = args.num_param_values
        exclude_params = args.exclude_params
        config = generate_hyperparameter_configuration(
            config,
            param_summary,
            num_params,
            num_values,
            exclude_params=exclude_params,
        )
    if args.output is None:
        pprint.pprint(config)
    else:
        with open(args.output, "w") as fout:
            json.dump(config, fout)


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        import pdb
        pdb.post_mortem()
