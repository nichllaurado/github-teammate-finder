import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .preprocessing import preprocess
from .evidence import generate_explanation

# How many overlapping evidence terms to surface per candidate
EVIDENCE_LIMIT = 8

# Weights for each signal field (must sum to 1.0)
_W_DESCRIPTION = 0.50  # repo descriptions, topics, bio
_W_FUNCTIONS   = 0.30  # function/method names extracted from source code
_W_COMMENTS    = 0.20  # inline comments and docstrings


def rank_candidates(query, candidates):
    """
    Rank candidates using field-weighted TF-IDF cosine similarity.

    Three signals are scored independently against the query and combined:
      - descriptions_text  (repo descriptions, topics, bio)    weight 0.50
      - functions_text     (function/method names from code)   weight 0.30
      - comments_text      (inline comments / docstrings)      weight 0.20

    Falls back to the flat `document` field for evidence extraction.
    Returns candidates sorted descending by score with `score`, `evidence`,
    and `explanation` fields added.
    """
    if not candidates:
        return []

    processed_query = preprocess(query)

    # Build per-field text lists in lock-step with candidates
    desc_texts = [preprocess(c.get("descriptions_text") or c.get("document", "")) for c in candidates]
    func_texts = [preprocess(c.get("functions_text", "")) for c in candidates]
    comm_texts = [preprocess(c.get("comments_text", "")) for c in candidates]

    # Fit one shared TF-IDF vocabulary over the entire corpus so IDF is consistent
    all_texts = [processed_query] + desc_texts + func_texts + comm_texts
    vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        min_df=1,
        ngram_range=(1, 2),
    )
    matrix = vectorizer.fit_transform(all_texts)

    n = len(candidates)
    query_vec  = matrix[0]
    desc_vecs  = matrix[1       : 1 + n]
    func_vecs  = matrix[1 + n   : 1 + 2 * n]
    comm_vecs  = matrix[1 + 2 * n : 1 + 3 * n]

    sim_desc = cosine_similarity(query_vec, desc_vecs).flatten()
    sim_func = cosine_similarity(query_vec, func_vecs).flatten()
    sim_comm = cosine_similarity(query_vec, comm_vecs).flatten()

    # For evidence extraction, use the flat document field
    doc_texts = [preprocess(c.get("document", "")) for c in candidates]
    doc_matrix = vectorizer.transform(doc_texts)

    feature_names  = vectorizer.get_feature_names_out()
    query_term_set = set(processed_query.split())

    ranked = []
    for i, candidate in enumerate(candidates):
        score = round(
            _W_DESCRIPTION * float(sim_desc[i])
            + _W_FUNCTIONS   * float(sim_func[i])
            + _W_COMMENTS    * float(sim_comm[i]),
            4,
        )
        evidence = _extract_evidence(doc_matrix[i], feature_names, query_term_set)
        scored = {**candidate, "score": score, "evidence": evidence}
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
