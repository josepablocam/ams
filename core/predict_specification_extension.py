#!/usr/bin/env python3
from argparse import ArgumentParser
import json
import pickle

from core.mine_association_rules import APIRulePredictor


def get_args():
    parser = ArgumentParser(
        description=
        "Make specification extensions with PMI association rule model", )
    parser.add_argument("--model", type=str, help="Association rule model")
    parser.add_argument("--input", type=str, nargs="+", help="Specification")
    parser.add_argument("--k", type=int, help="Top K rules", default=3)
    parser.add_argument(
        "--alpha",
        type=float,
        help=
        "Weight to combine norm PMI and normalized count of support into score",
        default=0.5,
    )
    parser.add_argument(
        "--classification", action="store_true", help="Is classification task")
    parser.add_argument("--output", type=str, help="Output file (if any)")
    return parser.parse_args()


def main():
    args = get_args()
    with open(args.model, "rb") as fin:
        model = pickle.load(fin)

    if len(args.input) == 1 and args.input[0].endswith(("txt", "json")):
        input_file = args.input[0]
        print("Treating {} as an input file".format(input_file))
        with open(input_file, "r") as fin:
            _input = fin.read()
            if input_file.endswith("json"):
                _input = json.loads(_input)
            else:
                _input = _input.split()
    else:
        _input = args.input

    extended = model.extend(
        _input,
        k=args.k,
        alpha=args.alpha,
        is_classification=args.classification,
    )
    print(extended)
    if args.output is not None:
        with open(args.output, "w") as fout:
            fout.write(" ".join(extended))
            fout.write("\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        import pdb
        pdb.post_mortem()
