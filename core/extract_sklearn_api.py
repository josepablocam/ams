#!/usr/bin/env python3
from argparse import ArgumentParser
import inspect
import importlib
import pickle
import re
import string

import numpy as np
import sklearn
import sklearn.base
import sklearn.compose
import sklearn.datasets
import sklearn.exceptions
import sklearn.dummy
import sklearn.inspection
import sklearn.metrics
import sklearn.model_selection
import sklearn.multioutput
import sklearn.pipeline
import sklearn.utils
import sklearn.tree
import sklearn.semi_supervised

from core import nlp

EXCLUDED = [
    sklearn.compose,
    sklearn.config_context,
    sklearn.datasets,
    sklearn.dummy,
    sklearn.exceptions,
    sklearn.inspection,
    sklearn.metrics,
    sklearn.model_selection,
    sklearn.multioutput,
    sklearn.pipeline,
    sklearn.tree.export_graphviz,
    sklearn.show_versions,
    sklearn.utils,
    sklearn.semi_supervised,
]


def get_short_class_desc(docstr):
    short_desc_lines = []
    accumulating = False
    for line in docstr.split("\n"):
        line = line.strip()
        if len(line) == 0 and not accumulating:
            continue
        elif (len(line) == 0 and accumulating) or line == "Parameters":
            return "\n".join(short_desc_lines)
        else:
            accumulating = True
            short_desc_lines.append(line)
    # just in case...
    return "\n".join(short_desc_lines)


def get_signature_params(class_):
    try:
        return inspect.signature(class_).parameters
    except ValueError:
        # need to fall back to parse signature differently
        print("Failed on", class_)
        return None


def get_expected_params(class_, class_elem):
    parameters = get_signature_params(class_)
    if parameters is None:
        return []
    parsed_params = []
    param_names = parameters.keys()
    parsed_param_docs = parse_param_docs(class_elem.docstring, param_names)
    for param_name in param_names:
        if param_name in ["args", "kwargs"]:
            continue
        param = parameters[param_name]
        param_elem = APIClassParameter(
            param_name,
            param,
            class_elem,
            doc_dict=parsed_param_docs,
        )
        parsed_params.append(param_elem)
    return parsed_params


def parse_param_docs(class_docs, param_names):
    in_param_section = False
    parse_this_param = False
    param_names = list(param_names)
    curr_param = None
    curr_param_docs = []
    param_docs = {}

    for line in class_docs.split("\n"):
        line = line.strip()
        if line == "Parameters":
            in_param_section = True
        elif line == "Attributes":
            # done parsing parameters
            break
        elif in_param_section:
            is_new_param_line = re.match("([a-zA-Z0-9_])+ : ", line)

            if is_new_param_line is not None:
                new_param = is_new_param_line.group(0)[:-3]

                if len(curr_param_docs) > 0:
                    param_docs[curr_param] = "\n".join(curr_param_docs)
                    curr_param_docs = []

                if new_param in param_names:
                    param_names.remove(new_param)
                    curr_param = new_param
                    parse_this_param = True
                else:
                    curr_param = None
                    parse_this_param = False

            if parse_this_param:
                if len(line) > 0:
                    curr_param_docs.append(line)
                else:
                    # hit first empty line after short description
                    # done parsing this param
                    parse_this_param = False
        else:
            continue

    if curr_param is not None and len(curr_param_docs) > 0:
        param_docs[curr_param] = "\n".join(curr_param_docs)

    return param_docs


def build_class_elem(elem, module, collected_info):
    path = module.__name__ + "." + elem.__name__
    class_elem = APIClass(elem, path)
    collected_info[path] = class_elem


def traverse_module_(stack, collected_info, expanded, root_name, exclude):
    while len(stack) > 0:
        parent = stack.pop()

        if not inspect.ismodule(parent):
            raise ValueError("Stack should only contain module types")

        parent_name = parent.__name__
        expanded.add(parent)
        if not parent_name.startswith(root_name):
            continue
        possible_children_names = set([])

        try:
            possible_children_names.update(parent.__all__)
        except AttributeError:
            continue

        for child_name in possible_children_names:
            imp_path = parent_name + "." + child_name
            try:
                print("Trying to import", imp_path)
                child = importlib.import_module(imp_path)
            except ModuleNotFoundError:
                print("Failed to import", imp_path)
                pass

            try:
                child = getattr(parent, child_name)
            except AttributeError:
                continue

            try:
                hash(child)
            except TypeError:
                # can't hash, so don't care about it
                # can't compare to things already expanded or excluded
                continue

            if child in exclude or child in expanded:
                # don't want it or already collected it
                continue

            if inspect.isclass(child):
                try:
                    child()
                except NotImplementedError:
                    # abstract classes, don't want these
                    continue
                except TypeError:
                    # have required args, that's fine
                    pass
                build_class_elem(child, parent, collected_info)
                expanded.add(child)
            elif inspect.ismodule(child):
                stack.append(child)
            else:
                continue


def traverse_module(module):
    stack = [module]
    collected_info = {}
    expanded = set([])
    traverse_module_(
        stack,
        collected_info,
        expanded,
        module.__name__,
        EXCLUDED,
    )
    return collected_info


def split_camel_case(token):
    matches = re.findall(r'[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))', token)
    if len(matches) == 0:
        return [token]
    else:
        return matches


def split_snake_case(token):
    if "_" in token:
        return token.split("_")
    else:
        return [token]


def split_api_name_subtokens(basename):
    result_tokens = set()
    subtokens = split_snake_case(basename)
    result_tokens.update(subtokens)
    for tk in subtokens:
        result_tokens.update(split_camel_case(tk))
    return set([t.lower() for t in result_tokens])


def extend_class_description(path):
    extra_tokens = []
    for token in path.split("."):
        extra_tokens.extend(split_api_name_subtokens(token))
        extra_tokens.append(token)
    extra_tokens = set(extra_tokens)
    return " ".join([t.lower() for t in extra_tokens])


class APIClass(object):
    def __init__(self, class_, path):
        self.docstring = class_.__doc__
        self.description = get_short_class_desc(self.docstring)
        self.path = path
        self.children = get_expected_params(class_, self)
        self.arg_names = [c.param_name for c in self.children]
        ext_tokens = extend_class_description(path)
        self.embedded_text = ext_tokens + " " + self.description.lower()
        self.vector = nlp.vectorize(
            self.embedded_text,
            remove_stop_words=True,
        )
        self.filter_tokens = self.get_filter_tokens()
        self.is_regression_op = sklearn.base.is_regressor(class_)
        self.is_classification_op = sklearn.base.is_classifier(class_)

    def get_filter_tokens(self):
        basename = self.path.split(".")[-1]
        subtokens = split_api_name_subtokens(basename)
        subtokens.add(basename.lower())
        # lemmatize subtokens
        lemmas = [nlp.lemmatize(t) for t in subtokens]
        subtokens.update(lemmas)
        return subtokens

    def __str__(self):
        return "APIClass({})".format(self.path)

    def __repr__(self):
        return str(self)


class APIClassParameter(object):
    def __init__(self, param_name, param, class_elem, doc_dict=None):
        if doc_dict is None:
            doc_dict = parse_param_docs(class_elem, [param_name])
        self.param_name = param_name
        self.description = doc_dict.get(param_name, "")
        self.path = class_elem.path + ":" + param_name
        self.parent = class_elem
        self.default_value = param.default
        self.vector = nlp.vectorize(
            self.description.lower() + param_name,
            remove_stop_words=True,
        )
        self.filter_tokens = self.get_filter_tokens()

    def get_filter_tokens(self):
        param_name = self.path.split(":")[-1]
        subtokens = split_api_name_subtokens(param_name)
        subtokens.add(param_name)
        lemmas = [nlp.lemmatize(t) for t in subtokens]
        subtokens.update(lemmas)
        return subtokens

    def __str__(self):
        return "APIClassParameter({})".format(self.path)

    def __repr__(self):
        return str(self)


class APICollection(object):
    def __init__(self, root_module):
        self.module_info = traverse_module(root_module)
        self.classes = list(self.module_info.values())
        # remove Base (abstract) clases
        self.classes = [
            c for c in self.classes
            if not c.path.split(".")[-1].startswith("Base")
        ]
        self.params = [p for c in self.classes for p in c.children]
        self.all_elems = self.classes + self.params
        self.paths = [e.path for e in self.all_elems]
        self.matrix = None
        self.filter_tokens = self.get_filter_tokens()
        self.basenames = [c.path.split(".")[-1] for c in self.classes]

    def get_filter_tokens(self):
        subtokens = set([])
        for e in self.classes:
            # remove numeric
            tokens = [t for t in e.filter_tokens if t not in string.digits]
            subtokens.update(tokens)
        return subtokens

    def build_matrix(self):
        num_rows = len(self.classes)
        num_cols = self.classes[0].vector.shape[0]
        matrix = np.zeros((num_rows, num_cols), dtype=np.float32)
        for ix, elem in enumerate(self.classes):
            matrix[ix, :] = elem.vector
        self.matrix = matrix

    def get_matrix(self):
        if self.matrix is None:
            self.build_matrix()
        return self.matrix


def get_args():
    parser = ArgumentParser(description="Extract sklearn API")
    parser.add_argument(
        "--output",
        type=str,
        help="Path to dump pickled results",
    )
    return parser.parse_args()


def main():
    args = get_args()
    api_collection = APICollection(sklearn)
    with open(args.output, "wb") as fout:
        pickle.dump(api_collection, fout)


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        import pdb
        pdb.post_mortem()
