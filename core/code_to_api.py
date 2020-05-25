from gensim.summarization import bm25
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity

from core import nlp
from core import utils


class BM25Index(object):
    def __init__(self, api):
        self.api = api
        self._build()

    def _preprocess_text(self, txt):
        return " ".join([t.text for t in nlp.remove_stop_words_from_nl(txt)])

    def _build(self):
        # extract classes
        self.classes = self.api.classes
        self.class_docs = {
            c.path: self._preprocess_text(c.embedded_text).split(" ")
            for c in self.classes
        }
        self.index = bm25.BM25([self.class_docs[c.path] for c in self.classes])

    def score(self, query):
        doc = self.class_docs[query]
        scores = self.index.get_scores(doc)
        return list(zip(self.classes, scores))


def simplify_path(path, obj):
    path_steps = path.split(".")
    module_steps, obj_name = path_steps[:-1], path_steps[-1]
    simplified_path = None
    for ix in range(1, len(module_steps)):
        poss_module = ".".join(module_steps[:ix])
        try:
            retrieved_obj = utils.get_component_constructor(
                poss_module + "." + obj_name
            )
            if isinstance(obj, retrieved_obj):
                simplified_path = poss_module + "." + obj_name
                break
        except AttributeError:
            continue
    return simplified_path


def get_component_path(obj):
    path = obj.__class__.__module__ + "." + obj.__class__.__name__
    simplified_path = simplify_path(path, obj)
    return path if simplified_path is None else simplified_path


def extract_code_matchable_portions(pipeline):
    assert isinstance(pipeline, Pipeline)
    components = []
    for _, step in pipeline.steps:
        comp = get_component_path(step)
        components.append(comp)
    return components


def embeddings_retrieval(api_collection, components):
    api_matrix = api_collection.get_matrix()
    # retrieve vectors for specific components
    paths = [c.path for c in api_collection.classes]
    indices = [paths.index(c) for c in components]
    code_matrix = api_collection.matrix[indices, :]
    cosine_matrix = cosine_similarity(code_matrix, api_matrix)

    matched = {}
    for row_ix in range(cosine_matrix.shape[0]):
        # larger first
        top_n_ixs = np.argsort(-cosine_matrix[row_ix, :])
        matches = [
            (api_collection.classes[col_ix], cosine_matrix[row_ix, col_ix])
            for col_ix in top_n_ixs
        ]
        matched[components[row_ix]] = matches
    return matched


def bm25_retrieval(api_collection, components):
    searcher = BM25Index(api_collection)
    matched = {c: searcher.score(c) for c in components}
    return matched


def compute_matches(api_collection, components, strategy):
    if strategy == "embeddings":
        return embeddings_retrieval(api_collection, components)
    elif strategy == "bm25":
        return bm25_retrieval(api_collection, components)
    else:
        raise ValueError("Unknown strategy: " + strategy)
