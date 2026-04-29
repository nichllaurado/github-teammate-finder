import json
import os
import re
from datetime import datetime

from .client import GitHubClient

# Max README chars included in the document text (keeps tokens manageable)
README_CHAR_LIMIT = 2000
# Number of top repos (by stars) to fetch READMEs for
README_FETCH_LIMIT = 3
# Max repos included in the candidate's repo list
REPO_LIST_LIMIT = 20

# Path to outputs directory (two levels up from this file: github_client/ -> project root)
_OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")

# Common English stopwords to skip when broadening a query
_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "that", "this", "is", "it", "be", "as", "by", "i",
    "want", "build", "create", "make", "use", "using", "uses",
}


def _broaden_query(query: str) -> str:
    """
    Convert a natural-language query into a GitHub search query by extracting
    meaningful keywords and joining them with OR.
    e.g. "AI event planning app that uses calendars" -> "AI OR event OR planning OR calendars"
    """
    words = re.findall(r"[A-Za-z0-9]+", query)
    keywords = [w for w in words if len(w) > 2 and w.lower() not in _STOPWORDS]
    if not keywords:
        return query
    return " OR ".join(keywords)


def _save_search_output(query: str, search_results: dict, broadened: bool = False):
    """Write raw search results to outputs/ for inspection during testing."""
    os.makedirs(_OUTPUTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"search_results_{timestamp}.json"
    filepath = os.path.join(_OUTPUTS_DIR, filename)
    payload = {
        "query": query,
        "broadened": broadened,
        "total_count": search_results.get("total_count"),
        "returned": len(search_results.get("items", [])),
        "items": search_results.get("items", []),
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"[search] results saved -> {filepath}")


def build_candidates_from_search(query, client: GitHubClient, per_page=30, readme_limit=README_FETCH_LIMIT):
    """
    Search GitHub for repos matching `query`, collect unique owners,
    and build a candidate document for each one.

    If the initial search returns no results, automatically broadens the query
    by extracting keywords and joining with OR.

    Returns a list of candidate dicts, one per unique GitHub user.
    """
    search_results = client.search_repositories(query, per_page=per_page)
    broadened = False

    if not search_results.get("items"):
        broad_query = _broaden_query(query)
        if broad_query != query:
            print(f"[search] No results for '{query}', retrying with broadened query: '{broad_query}'")
            search_results = client.search_repositories(broad_query, per_page=per_page)
            broadened = True

    _save_search_output(query if not broadened else _broaden_query(query), search_results, broadened)

    repos = search_results.get("items", [])

    # Deduplicate owners while preserving order
    seen = set()
    unique_owners = []
    for repo in repos:
        owner = repo.get("owner", {}).get("login")
        if owner and owner not in seen:
            seen.add(owner)
            unique_owners.append(owner)

    candidates = []
    for username in unique_owners:
        try:
            candidate = build_candidate(username, client, readme_limit=readme_limit)
            candidates.append(candidate)
        except Exception:
            # Skip users whose profiles are inaccessible (orgs, deleted accounts, etc.)
            continue

    return candidates


def build_candidate(username, client: GitHubClient, readme_limit=README_FETCH_LIMIT):
    """
    Build a candidate document for a single GitHub user.

    Fetches:
      - public profile
      - all public repos (up to 100)
      - READMEs for the top `readme_limit` repos by star count

    Returns a dict with structured metadata and a flat `document` text
    field suitable for TF-IDF / BM25 vectorization.
    """
    profile = client.get_user(username)
    repos = client.get_user_repos(username, per_page=100)

    # Sort by stars descending for README prioritization
    repos_sorted = sorted(repos, key=lambda r: r.get("stargazers_count", 0), reverse=True)

    # Attach README text to top repos
    for repo in repos_sorted[:readme_limit]:
        repo["readme"] = client.get_readme(username, repo["name"])

    # Aggregate languages and topics across all repos
    languages = _unique_ordered([r.get("language") for r in repos_sorted if r.get("language")])
    topics = _unique_ordered([t for r in repos_sorted for t in r.get("topics", [])])

    document = _build_document(profile, repos_sorted)

    return {
        "username": username,
        "name": profile.get("name"),
        "bio": profile.get("bio"),
        "location": profile.get("location"),
        "profile_url": profile.get("html_url"),
        "avatar_url": profile.get("avatar_url"),
        "followers": profile.get("followers"),
        "public_repos": profile.get("public_repos"),
        "languages": languages,
        "topics": topics,
        "repos": [_slim_repo(r) for r in repos_sorted[:REPO_LIST_LIMIT]],
        "document": document,
    }


def _build_document(profile, repos):
    """
    Concatenate all text signals into a single string for vectorization.
    Order: profile fields → repo names/descriptions/topics → README snippets.
    """
    parts = []

    for field in ("name", "bio", "location"):
        val = profile.get(field)
        if val:
            parts.append(val)

    for repo in repos:
        name = repo.get("name", "")
        # Normalize delimiters so "event-planner" becomes "event planner"
        parts.append(name.replace("-", " ").replace("_", " "))

        if repo.get("description"):
            parts.append(repo["description"])

        if repo.get("language"):
            parts.append(repo["language"])

        for topic in repo.get("topics", []):
            parts.append(topic.replace("-", " "))

        readme = repo.get("readme", "")
        if readme:
            parts.append(readme[:README_CHAR_LIMIT])

    return " ".join(filter(None, parts))


def _slim_repo(r):
    return {
        "name": r.get("name"),
        "description": r.get("description"),
        "language": r.get("language"),
        "stars": r.get("stargazers_count", 0),
        "topics": r.get("topics", []),
        "html_url": r.get("html_url"),
        "readme_snippet": r.get("readme", "")[:300] if r.get("readme") else "",
    }


def _unique_ordered(items):
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
