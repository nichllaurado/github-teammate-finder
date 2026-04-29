def generate_explanation(candidate):
    """
    Build a human-readable explanation of why a candidate matched a query.

    Combines score, matched evidence terms, languages, and top repos into
    a short paragraph shown directly on the candidate card.
    """
    score    = candidate.get("score", 0)
    evidence = candidate.get("evidence") or []
    languages = (candidate.get("languages") or [])[:3]
    repos     = (candidate.get("repos") or [])[:5]

    # ── Score qualifier ──────────────────────────────────────────────────────
    if score >= 0.4:
        qualifier = "Strong match"
    elif score >= 0.25:
        qualifier = "Good match"
    elif score >= 0.1:
        qualifier = "Moderate match"
    else:
        qualifier = "Low match"

    sentences = [f"{qualifier} ({score * 100:.0f}% similarity)."]

    # ── Language expertise ───────────────────────────────────────────────────
    if languages:
        lang_str = _join_list(languages)
        sentences.append(f"Works primarily in {lang_str}.")

    # ── Relevant repos ───────────────────────────────────────────────────────
    # Prefer repos whose names or descriptions share words with the evidence
    evidence_words = {w for term in evidence for w in term.split()}
    relevant, fallback = [], []

    for r in repos:
        name = r.get("name", "")
        readable_name = name.replace("-", " ").replace("_", " ")
        name_words = set(readable_name.lower().split())
        desc_words = set((r.get("description") or "").lower().split())

        if name_words & evidence_words or desc_words & evidence_words:
            relevant.append(readable_name)
        else:
            fallback.append(readable_name)

    repo_names = (relevant or fallback)[:3]
    if repo_names:
        repo_list = _join_list([f'"{n}"' for n in repo_names])
        sentences.append(f"Relevant repos: {repo_list}.")

    # ── Evidence signals ─────────────────────────────────────────────────────
    # Show up to 5 clean terms (skip pure numbers)
    signals = [t for t in evidence if not t.isdigit()][:5]
    if signals:
        sentences.append(f"Matched on: {', '.join(signals)}.")

    return " ".join(sentences)


def _join_list(items):
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"
