import os

import numpy as np
import spacy
from spacy_transformers import (
    TransformersWordPiecer,
    TransformersTok2Vec,
)

# experiments showed comparable to BM25 and BM25 much cheaper
USE_SCIBERT = False

# scispacy and scibert
spacy_nlp = spacy.load("en_core_sci_lg")

if USE_SCIBERT:
    path = os.path.join(
        os.path.dirname(__file__), "../..", "resources",
        "scibert_scivocab_uncased"
    )

    spacy_nlp.add_pipe(
        TransformersWordPiecer.from_pretrained(spacy_nlp.vocab, path)
    )
    spacy_nlp.add_pipe(
        TransformersTok2Vec.from_pretrained(spacy_nlp.vocab, path)
    )


def parse(doc):
    if isinstance(doc, str):
        doc = spacy_nlp(doc)
    return doc


def remove_stop_words_from_nl(doc):
    doc = parse(doc)
    return [t for t in doc if not t.is_stop]


def vectorize_tokens(tokens):
    # TODO: agh, hack, but want same vector
    # for same tokens, so going to reparse
    combined_str = " ".join([t.text for t in tokens])
    tokens = parse(combined_str)
    vectors = [t.vector for t in tokens]
    return np.mean(vectors, axis=0)


def vectorize(doc, remove_stop_words=False):
    doc = parse(doc)
    if remove_stop_words:
        tokens = remove_stop_words_from_nl(doc)
        if len(tokens) > 0:
            return vectorize_tokens(tokens)
        else:
            return np.zeros(doc.vector.shape)
    else:
        return doc.vector


def tokens_to_text(tokens, sep=" "):
    return sep.join([t.text for t in tokens])


def augmented_noun_chunks(doc, n=2):
    doc = parse(doc)
    noun_chunks = list(doc.noun_chunks)
    start_ends = [(nc.start, nc.end) for nc in noun_chunks]
    tokenized = [list(nc) for nc in noun_chunks]
    for ix, nc in enumerate(noun_chunks):
        root = nc.root
        for k in range(n):
            root = root.head
            root_position = root.i
            if any(root_position >= s and root_position <= e
                   for s, e in start_ends):
                # already part of different noun chunk
                continue
            else:
                tokenized[ix].insert(0, root)
            if root.pos_ == "VERB":
                # verbs are blocking
                # assume each noun chunk associated with at most one verb
                # semantically, so once we encounter, stop traversing backwards
                break
    return tokenized


def lemmatize(token):
    return parse(token)[0].lemma_


def unique_tokens_and_lemmas(doc, return_count_tokens=False):
    doc = parse(doc)
    unique_tokens = set([token.text.lower() for token in doc])
    unique_tokens.update(token.lemma_.lower() for token in doc)
    if return_count_tokens:
        return unique_tokens, len(doc)
    else:
        return unique_tokens


def visualize(parsed_doc, style="dep"):
    spacy.displacy.serve(parsed_doc, style=style)
