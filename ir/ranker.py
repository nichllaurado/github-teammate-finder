import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .preprocessing import preprocess
from .evidence import generate_explanation

# How many overlapping evidence terms to surface per candidate
EVIDENCE_LIMIT = 8


def rank_candidates(query, candidates):
    """
    Rank a list of candidate dicts by TF-IDF cosine similarity to `query`.

    Each candidate must have a `document` field (plain text string).
    Returns the same list sorted descending by score, with two new fields:
      - `score`    float  cosine similarity in [0, 1]
      - `evidence` list   top matched terms explaining the score
    """
    if not candidates:
        return []

    documents = [c["document"] for c in candidates]

    # Preprocess query and all candidate documents
    processed_query = preprocess(query)
    processed_docs  = [preprocess(doc) for doc in documents]

    # Fit TF-IDF on the full corpus (query + candidates)
    vectorizer = TfidfVectorizer(
        sublinear_tf=True,   # log(1+tf) dampens high-frequency terms
        min_df=1,
        ngram_range=(1, 2),  # unigrams + bigrams catch compound terms
    )
    all_texts = [processed_query] + processed_docs
    matrix = vectorizer.fit_transform(all_texts)

    query_vec     = matrix[0]
    candidate_vecs = matrix[1:]

    scores = cosine_similarity(query_vec, candidate_vecs).flatten()

    feature_names = vectorizer.get_feature_names_out()
    query_term_set = set(processed_query.split())

    ranked = []
    for i, candidate in enumerate(candidates):
        score    = float(scores[i])
        evidence = _extract_evidence(
            candidate_vecs[i], feature_names, query_term_set
        )
        scored = {**candidate, "score": round(score, 4), "evidence": evidence}
        scored["explanation"] = generate_explanation(scored)
        ranked.append(scored)

    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked


def _extract_evidence(candidate_vec, feature_names, query_term_set, limit=EVIDENCE_LIMIT):
    """
    Return up to `limit` terms that are:
      1. Present in both the query and the candidate (overlap terms first)
      2. Then highest-weight candidate terms to fill remaining slots

    Bigrams are included only when both component words appear in the query
    to avoid surfacing noisy compound terms.
    """
    weights = candidate_vec.toarray().flatten()
    # Indices sorted by TF-IDF weight, highest first
    ranked_indices = weights.argsort()[::-1]

    overlap = []
    other   = []

    for idx in ranked_indices:
        if weights[idx] == 0:
            break
        term = feature_names[idx]
        parts = term.split()  # unigram → [term], bigram → [w1, w2]

        if all(p in query_term_set for p in parts):
            overlap.append(term)
        else:
            other.append(term)

        if len(overlap) + len(other) >= limit * 2:
            break

    evidence = overlap[:limit]
    if len(evidence) < limit:
        evidence += other[: limit - len(evidence)]

    return evidence
