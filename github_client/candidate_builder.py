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

# Maps lowercase aliases/names found in queries to GitHub's language: qualifier value
_LANGUAGE_ALIASES: dict[str, str] = {
    "python": "Python",
    "javascript": "JavaScript",
    "js": "JavaScript",
    "typescript": "TypeScript",
    "ts": "TypeScript",
    "java": "Java",
    "cpp": "C++",
    "c++": "C++",
    "csharp": "C#",
    "c#": "C#",
    "ruby": "Ruby",
    "go": "Go",
    "golang": "Go",
    "rust": "Rust",
    "swift": "Swift",
    "kotlin": "Kotlin",
    "php": "PHP",
    "scala": "Scala",
    "shell": "Shell",
    "bash": "Shell",
    "r": "R",
    "matlab": "MATLAB",
    "dart": "Dart",
    "flutter": "Dart",
    "haskell": "Haskell",
    "elixir": "Elixir",
    "clojure": "Clojure",
    "lua": "Lua",
    "perl": "Perl",
    "sql": "SQL",
}

# Maps regex patterns (matched against the full query) to GitHub topic: qualifier values
_TOPIC_PATTERNS: list[tuple[str, str]] = [
    (r"\bmachine[\s\-]?learning\b", "machine-learning"),
    (r"\bdeep[\s\-]?learning\b", "deep-learning"),
    (r"\bneural[\s\-]?network", "neural-network"),
    (r"\bnlp\b|natural[\s\-]?language[\s\-]?processing", "nlp"),
    (r"\bcomputer[\s\-]?vision\b", "computer-vision"),
    (r"\breinforcement[\s\-]?learning\b", "reinforcement-learning"),
    (r"\bdata[\s\-]?science\b", "data-science"),
    (r"\bdata[\s\-]?analysis\b", "data-analysis"),
    (r"\bweb[\s\-]?scraping\b", "web-scraping"),
    (r"\brest[\s\-]?api\b|\brestful\b", "rest-api"),
    (r"\bgraphql\b", "graphql"),
    (r"\bcli\b|command[\s\-]?line", "cli"),
    (r"\bgui\b|desktop[\s\-]?app", "gui"),
    (r"\bsqlite\b", "sqlite"),
    (r"\bpostgres\b|postgresql", "postgresql"),
    (r"\bmongodb\b", "mongodb"),
    (r"\bdocker\b", "docker"),
    (r"\bkubernetes\b|\bk8s\b", "kubernetes"),
    (r"\bgame[\s\-]?dev\b|game[\s\-]?engine", "game-development"),
    (r"\bchatbot\b|chat[\s\-]?bot", "chatbot"),
    (r"\bblockchain\b", "blockchain"),
    (r"\bcryptocurrency\b|\bcrypto\b", "cryptocurrency"),
    (r"\bcybersecurity\b|security[\s\-]?tool", "security"),
    (r"\bspeech[\s\-]?recognition\b|\bvoice\b", "speech-recognition"),
    (r"\bimage[\s\-]?processing\b", "image-processing"),
    (r"\bautomation\b", "automation"),
    (r"\bdevops\b", "devops"),
    (r"\bmobile[\s\-]?app\b", "mobile"),
    (r"\bandroid\b", "android"),
    (r"\bios\b|\bswiftui\b", "ios"),
]


def _extract_qualifier_parts(query: str) -> tuple[str, set[str]]:
    """
    Detect language/topic qualifiers and the source words consumed by them.
    Returns ('language:Python topic:machine-learning', {'python', 'machine', 'learning'}).
    """
    q_lower = query.lower()
    parts = []
    consumed_words: set[str] = set()

    # Language: match whole words/tokens in the query
    for alias, gh_name in _LANGUAGE_ALIASES.items():
        pattern = r"\b" + re.escape(alias) + r"\b"
        if re.search(pattern, q_lower):
            qualifier = f"language:{gh_name}"
            if qualifier not in parts:
                parts.append(qualifier)
            consumed_words.update(re.findall(r"[A-Za-z0-9]+", alias.lower()))

    # Topics: match phrase patterns
    for pattern, topic in _TOPIC_PATTERNS:
        match = re.search(pattern, q_lower)
        if match:
            qualifier = f"topic:{topic}"
            if qualifier not in parts:
                parts.append(qualifier)
            consumed_words.update(re.findall(r"[A-Za-z0-9]+", match.group(0).lower()))

    return " ".join(parts), consumed_words


def _extract_qualifiers(query: str) -> str:
    """
    Detect language and topic qualifiers from a natural-language query.
    Returns a string like 'language:Python topic:machine-learning topic:sqlite'
    to be appended to each search chunk.
    """
    qualifiers, _ = _extract_qualifier_parts(query)
    return qualifiers


def _extract_keywords(query: str, excluded_words: set[str] | None = None) -> list[str]:
    """Extract meaningful keywords from a natural-language query."""
    excluded_words = excluded_words or set()
    words = re.findall(r"[A-Za-z0-9]+", query)
    seen = set()
    keywords = []
    for w in words:
        w_lower = w.lower()
        if (
            len(w) > 2
            and w_lower not in _STOPWORDS
            and w_lower not in excluded_words
            and w_lower not in seen
        ):
            seen.add(w_lower)
            keywords.append(w)
    return keywords


_GH_QUERY_LIMIT = 256


def _chunk_keywords(keywords: list[str], reserved: int = 0) -> list[str]:
    """
    Pack keywords into OR-joined chunks that each fit within _GH_QUERY_LIMIT.
    `reserved` is the number of characters already spoken for by qualifiers
    (space + qualifier string) so each chunk leaves room for them.
    Returns a list of keyword-only query strings, one per chunk.
    """
    limit = _GH_QUERY_LIMIT - reserved
    chunks = []
    current: list[str] = []
    for kw in keywords:
        trial = " OR ".join(current + [kw])
        if current and len(trial) > limit:
            chunks.append(" OR ".join(current))
            current = [kw]
        else:
            current.append(kw)
    if current:
        chunks.append(" OR ".join(current))
    return chunks


def _iterative_search(query: str, client: "GitHubClient", per_page: int) -> dict:
    """
    Extract keywords and qualifiers from the query, pack keywords into
    OR-joined chunks that fit within GitHub's query limit, and append
    language:/topic: qualifiers to each chunk before sending.
    Returns a dict with 'total_count' and 'items' (deduplicated by full_name).
    """
    qualifiers, qualifier_words = _extract_qualifier_parts(query)
    keywords = _extract_keywords(query, excluded_words=qualifier_words)
    qualifier_suffix = f" {qualifiers}" if qualifiers else ""

    if qualifiers:
        print(f"[search] Qualifiers detected: {qualifiers}")

    reserved = len(qualifier_suffix)
    if keywords:
        chunks = _chunk_keywords(keywords, reserved=reserved)
    elif qualifiers:
        chunks = [""]
    else:
        chunks = [query[:_GH_QUERY_LIMIT]]

    seen_repos = {}  # full_name -> item
    for chunk in chunks:
        full_query = (chunk + qualifier_suffix) if chunk else qualifiers or query[:_GH_QUERY_LIMIT]
        print(f"[search] Searching chunk (len={len(full_query)}): '{full_query}'")
        try:
            result = client.search_repositories(full_query, per_page=per_page)
        except Exception as e:
            print(f"[search] Skipping chunk: {e}")
            continue
        for item in result.get("items", []):
            full_name = item.get("full_name")
            if full_name and full_name not in seen_repos:
                seen_repos[full_name] = item

    items = list(seen_repos.values())
    return {"total_count": len(items), "items": items}


def _save_search_output(query: str, search_results: dict):
    """Write raw search results to outputs/ for inspection during testing."""
    os.makedirs(_OUTPUTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"search_results_{timestamp}.json"
    filepath = os.path.join(_OUTPUTS_DIR, filename)
    payload = {
        "query": query,
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
    search_results = _iterative_search(query, client, per_page=per_page)
    _save_search_output(query, search_results)

    repos = search_results.get("items", [])

    # Deduplicate owners while preserving order; skip organizations
    seen = set()
    unique_owners = []
    for repo in repos:
        owner = repo.get("owner", {})
        login = owner.get("login")
        if login and login not in seen and owner.get("type") == "User":
            seen.add(login)
            unique_owners.append(login)

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
    if profile.get("type") == "Organization":
        raise ValueError(f"{username} is an organization, not an individual user")
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
