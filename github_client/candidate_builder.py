import json
import os
import re
from datetime import datetime

import requests

from .client import GitHubClient

# Max README chars included in the document text (keeps tokens manageable)
README_CHAR_LIMIT = 2000
# Number of repos to fetch READMEs for
README_FETCH_LIMIT = 3
# Max repos included in the candidate's repo list
REPO_LIST_LIMIT = 20

# Code extraction limits (keeps API calls manageable)
_CODE_REPOS_LIMIT = 2       # repos to extract code signals from per candidate
_CODE_FILES_PER_REPO = 4    # source files to sample per repo
_CODE_FILE_SIZE_LIMIT = 80_000  # bytes; skip files larger than this
_CODE_COMMENT_LIMIT = 30    # max comment snippets to collect

# File extensions treated as source code
_CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx",
    ".java", ".go", ".cpp", ".c", ".cs",
    ".rb", ".rs", ".swift", ".kt",
}

# Patterns that identify function/method definitions across common languages
_FUNC_PATTERNS = [
    r"\bdef\s+([a-zA-Z_][a-zA-Z0-9_]+)",                          # Python, Ruby
    r"\bfn\s+([a-zA-Z_][a-zA-Z0-9_]+)",                           # Rust
    r"\bfunc\s+(?:\([^)]*\)\s+)?([a-zA-Z_][a-zA-Z0-9_]+)",       # Go
    r"\bfunction\s+([a-zA-Z_$][a-zA-Z0-9_$]+)",                   # JS/TS
    r"(?:const|let|var)\s+([a-zA-Z_$][a-zA-Z0-9_$]+)\s*=\s*(?:async\s*)?\(",  # JS arrow fns
    r"\bfun\s+([a-zA-Z_][a-zA-Z0-9_]+)",                          # Kotlin/Swift
    # Java/C#/C++: modifier(s) + return type + name + (
    r"(?:public|private|protected|static|async|override|virtual|void|int|long|bool|string|String|float|double)\s+([a-zA-Z_][a-zA-Z0-9_]+)\s*\(",
]

# Patterns that extract comment text
_COMMENT_PATTERNS = [
    (r"#\s*(.+)",          0),   # Python/Ruby/Shell line comments
    (r"//\s*(.+)",         0),   # JS/TS/Java/Go/C++ line comments
    (r"/\*+\s*(.*?)\s*\*+/", re.DOTALL),  # /* block */ comments
]


def _split_identifier(name: str) -> str:
    """Convert snake_case or camelCase identifiers to space-separated words."""
    name = re.sub(r"_+", " ", name)
    name = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    return name.lower().strip()


def _extract_code_signals(code: str) -> tuple[str, str]:
    """
    Extract function/method names and comment text from source code.
    Returns (functions_text, comments_text) as whitespace-joined strings.
    """
    func_words = []
    for pattern in _FUNC_PATTERNS:
        for m in re.finditer(pattern, code):
            name = m.group(1)
            # Skip very short names and private/dunder names
            if len(name) > 2 and not name.startswith("__"):
                func_words.append(_split_identifier(name))

    comments = []
    for pattern, flags in _COMMENT_PATTERNS:
        for m in re.finditer(pattern, code, flags):
            text = m.group(1).strip()
            if len(text) > 5:
                comments.append(text[:200])
            if len(comments) >= _CODE_COMMENT_LIMIT:
                break

    return " ".join(func_words), " ".join(comments)

# Path to outputs directory (two levels up from this file: github_client/ -> project root)
_OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")

# Common English stopwords to skip when extracting search keywords
_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "that", "this", "is", "it", "be", "as", "by", "i",
    "want", "build", "create", "make", "use", "using", "uses", "online",
}

_SHORT_QUERY_LIMIT = 8
_OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
_OPENAI_QUERY_MODEL = os.getenv("OPENAI_QUERY_MODEL", "gpt-5.4-mini")

_DOMAIN_EXPANSIONS: list[tuple[set[str], tuple[str, ...]]] = [
    ({"chess"}, ("chess engine", "minimax chess")),
    ({"game", "multiplayer"}, ("multiplayer game", "game server")),
    ({"matchmaking"}, ("matchmaking game",)),
    ({"calendar"}, ("calendar integration", "calendar app")),
    ({"event"}, ("event planner", "event recommendation")),
    ({"recommendation", "recommender"}, ("recommendation system", "recommender system")),
    ({"ai"}, ("ai opponent",)),
]

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


_TOKEN_PATTERN = r"[A-Za-z0-9+#]+"


def _alias_pattern(alias: str) -> str:
    escaped = re.escape(alias)
    if re.search(r"[^A-Za-z0-9_]", alias):
        return rf"(?<![A-Za-z0-9+#]){escaped}(?![A-Za-z0-9+#])"
    return rf"\b{escaped}\b"


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
        pattern = _alias_pattern(alias)
        if re.search(pattern, q_lower):
            qualifier = f"language:{gh_name}"
            if qualifier not in parts:
                parts.append(qualifier)
            consumed_words.update(re.findall(_TOKEN_PATTERN, alias.lower()))

    # Topics: match phrase patterns
    for pattern, topic in _TOPIC_PATTERNS:
        match = re.search(pattern, q_lower)
        if match:
            qualifier = f"topic:{topic}"
            if qualifier not in parts:
                parts.append(qualifier)
            consumed_words.update(re.findall(_TOKEN_PATTERN, match.group(0).lower()))

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
    words = re.findall(_TOKEN_PATTERN, query)
    seen = set()
    keywords = []
    for w in words:
        w_lower = w.lower()
        if (
            (len(w) > 2 or w_lower in {"ai", "ui", "ml"})
            and w_lower not in _STOPWORDS
            and w_lower not in excluded_words
            and w_lower not in seen
        ):
            seen.add(w_lower)
            keywords.append(w)
    return keywords


_GH_QUERY_LIMIT = 256
_KEYWORD_IN_QUALIFIER = "in:name,description,readme"


def _add_unique_query(queries: list[str], seen: set[str], query: str):
    query = " ".join(query.split())
    if query and query.lower() not in seen:
        seen.add(query.lower())
        queries.append(query)


def _build_short_keyword_queries(keywords: list[str]) -> list[str]:
    """Build compact GitHub search phrases from the extracted prompt keywords."""
    queries: list[str] = []
    seen: set[str] = set()
    normalized = [kw.lower() for kw in keywords]

    # Adjacent keyword phrases preserve the user's wording: "multiplayer chess",
    # "chess game", etc. A few weak transitions are left to domain expansions.
    for left, right in zip(keywords, keywords[1:]):
        left_lower = left.lower()
        right_lower = right.lower()
        if (
            left_lower in {"app", "game"}
            and right_lower in {"matchmaking", "recommendation", "recommender"}
        ) or (
            left_lower in {"matchmaking", "recommendation", "recommender"}
            and right_lower in {"ai", "ml"}
        ):
            continue
        _add_unique_query(queries, seen, f"{left} {right}")

    keyword_set = set(normalized)
    for triggers, expansions in _DOMAIN_EXPANSIONS:
        if triggers & keyword_set:
            for expansion in expansions:
                _add_unique_query(queries, seen, expansion)

    # Pair central nouns with surrounding capabilities when possible.
    anchors = [kw for kw in keywords if kw.lower() in {"app", "game", "server", "engine"}]
    descriptors = [
        kw for kw in keywords
        if kw.lower() not in {"app", "game", "server", "engine"}
    ]
    for anchor in anchors:
        for descriptor in descriptors:
            _add_unique_query(queries, seen, f"{descriptor} {anchor}")

    if not queries and keywords:
        _add_unique_query(queries, seen, " ".join(keywords[:3]))

    return queries[:_SHORT_QUERY_LIMIT]


def _openai_query_generation_enabled() -> bool:
    disabled_values = {"0", "false", "no", "off"}
    setting = os.getenv("OPENAI_QUERY_GENERATION", "1").strip().lower()
    return bool(os.getenv("OPENAI_API_KEY")) and setting not in disabled_values


def _extract_response_text(response_json: dict) -> str:
    if isinstance(response_json.get("output_text"), str):
        return response_json["output_text"]

    parts = []
    for item in response_json.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text" and isinstance(content.get("text"), str):
                parts.append(content["text"])
    return "\n".join(parts)


def _clean_model_query_chunk(query: str) -> str:
    query = re.sub(r"\bin:name,description,readme\b", " ", query, flags=re.IGNORECASE)
    query = re.sub(r"\blanguage:(\"[^\"]+\"|'[^']+'|[^\s]+)", " ", query, flags=re.IGNORECASE)
    query = re.sub(r"\btopic:(\"[^\"]+\"|'[^']+'|[^\s]+)", " ", query, flags=re.IGNORECASE)
    return " ".join(query.split()).strip()


def _parse_model_query_chunks(response_text: str) -> list[str]:
    try:
        payload = json.loads(response_text)
    except json.JSONDecodeError:
        return []

    if not isinstance(payload, dict) or not isinstance(payload.get("queries"), list):
        return []

    queries: list[str] = []
    seen: set[str] = set()
    for raw_query in payload["queries"]:
        if not isinstance(raw_query, str):
            continue
        cleaned = _clean_model_query_chunk(raw_query)
        if len(cleaned) > _GH_QUERY_LIMIT:
            cleaned = cleaned[:_GH_QUERY_LIMIT].rsplit(" ", 1)[0].strip()
        _add_unique_query(queries, seen, cleaned)
        if len(queries) >= _SHORT_QUERY_LIMIT:
            break
    return queries


def _build_openai_query_chunks(query: str, keywords: list[str], qualifiers: str) -> list[str]:
    """
    Ask OpenAI to create compact GitHub repository search phrases.
    The returned phrases are later expanded with this app's GitHub field and
    language/topic qualifiers, so this function strips those qualifiers if the
    model includes them.
    """
    if not _openai_query_generation_enabled():
        return []

    api_key = os.getenv("OPENAI_API_KEY")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": _OPENAI_QUERY_MODEL,
        "instructions": (
            "Generate concise GitHub repository search query phrases for iterative search. "
            "Return only JSON that matches the schema. Queries should be short, diverse, "
            "and likely to find repositories relevant to the user's project. Include domain "
            "terms, implementation terms, and technology terms when useful. Do not include "
            "GitHub qualifiers such as in:, language:, stars:, or topic:."
        ),
        "input": (
            f"User project request: {query}\n"
            f"Extracted keywords: {', '.join(keywords) if keywords else '(none)'}\n"
            f"Detected GitHub qualifiers that will be appended later: {qualifiers or '(none)'}\n\n"
            "For a chess project, good query phrases might include: chess game, chess engine, "
            "multiplayer chess, cpp chess, c++ game server, minimax chess, matchmaking game."
        ),
        "text": {
            "format": {
                "type": "json_schema",
                "name": "github_search_queries",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "queries": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1,
                            "maxItems": _SHORT_QUERY_LIMIT,
                        }
                    },
                    "required": ["queries"],
                    "additionalProperties": False,
                },
            }
        },
    }

    try:
        response = requests.post(
            _OPENAI_RESPONSES_URL,
            headers=headers,
            json=payload,
            timeout=15,
        )
        response.raise_for_status()
        chunks = _parse_model_query_chunks(_extract_response_text(response.json()))
        if chunks:
            print(f"[search] OpenAI generated query chunks: {chunks}")
        return chunks
    except Exception as e:
        print(f"[search] OpenAI query generation skipped: {e}")
        return []


def _iterative_search(query: str, client: "GitHubClient", per_page: int) -> dict:
    """
    Extract keywords and qualifiers from the query, pack keywords into
    space-separated chunks that fit within GitHub's query limit, and append
    language:/topic: qualifiers to each chunk before sending.
    Returns a dict with 'total_count' and 'items' (deduplicated by full_name).
    """
    qualifiers, qualifier_words = _extract_qualifier_parts(query)
    keywords = _extract_keywords(query, excluded_words=qualifier_words)
    qualifier_suffix = f" {qualifiers}" if qualifiers else ""

    if qualifiers:
        print(f"[search] Qualifiers detected: {qualifiers}")

    keyword_field_suffix = f" {_KEYWORD_IN_QUALIFIER}"
    if keywords:
        chunks = _build_openai_query_chunks(query, keywords, qualifiers)
        if not chunks:
            chunks = _build_short_keyword_queries(keywords)
    elif qualifiers:
        chunks = [""]
    else:
        chunks = [query[:_GH_QUERY_LIMIT]]

    seen_repos = {}  # full_name -> item
    sent_queries = []
    for chunk in chunks:
        full_query = (
            chunk + keyword_field_suffix + qualifier_suffix
            if chunk
            else qualifiers or query[:_GH_QUERY_LIMIT]
        )
        print(f"[search] Searching chunk (len={len(full_query)}): '{full_query}'")
        try:
            sent_queries.append(full_query)
            result = client.search_repositories(full_query, per_page=per_page)
        except Exception as e:
            print(f"[search] Skipping chunk: {e}")
            continue
        for item in result.get("items", []):
            full_name = item.get("full_name")
            if full_name and full_name not in seen_repos:
                seen_repos[full_name] = item

    items = list(seen_repos.values())
    return {"total_count": len(items), "items": items, "queries": sent_queries}


def _save_search_output(search_results: dict):
    """Write raw search results to outputs/ for inspection during testing."""
    os.makedirs(_OUTPUTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"search_results_{timestamp}.json"
    filepath = os.path.join(_OUTPUTS_DIR, filename)
    payload = {
        "queries": search_results.get("queries", []),
        "total_count": search_results.get("total_count"),
        "returned": len(search_results.get("items", [])),
        "items": search_results.get("items", []),
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"[search] results saved -> {filepath}")


def build_candidates_from_search(
    query,
    client: GitHubClient,
    per_page=30,
    readme_limit=README_FETCH_LIMIT,
    return_search_queries=False,
):
    """
    Search GitHub for repos matching `query`, collect unique owners,
    and build a candidate document for each one.

    Returns a list of candidate dicts, one per unique GitHub user.
    """
    search_results = _iterative_search(query, client, per_page=per_page)
    _save_search_output(search_results)

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

    if return_search_queries:
        return candidates, search_results.get("queries", [])

    return candidates


def build_candidate(username, client: GitHubClient, readme_limit=README_FETCH_LIMIT):
    """
    Build a candidate document for a single GitHub user.

    Fetches:
      - public profile
      - all public repos (up to 100)
      - READMEs for the first `readme_limit` repos returned by GitHub

    Returns a dict with structured metadata and a flat `document` text
    field suitable for TF-IDF / BM25 vectorization.
    """
    profile = client.get_user(username)
    if profile.get("type") == "Organization":
        raise ValueError(f"{username} is an organization, not an individual user")
    repos = client.get_user_repos(username, per_page=100)

    # get_user_repos already returns owner repos by recent update; do not reorder by stars.
    repos_sorted = repos

    # Attach README text to top repos
    for repo in repos_sorted[:readme_limit]:
        repo["readme"] = client.get_readme(username, repo["name"])

    # Extract function names and comments from source files in top repos
    all_functions: list[str] = []
    all_comments: list[str] = []
    for repo in repos_sorted[:_CODE_REPOS_LIMIT]:
        tree = client.get_repo_tree(username, repo["name"])
        code_paths = [
            node["path"]
            for node in tree
            if node.get("type") == "blob"
            and os.path.splitext(node["path"])[1].lower() in _CODE_EXTENSIONS
            and node.get("size", 0) < _CODE_FILE_SIZE_LIMIT
        ][:_CODE_FILES_PER_REPO]

        for path in code_paths:
            content = client.get_file_content(username, repo["name"], path)
            if content:
                fnames, fcomments = _extract_code_signals(content)
                if fnames:
                    all_functions.append(fnames)
                if fcomments:
                    all_comments.append(fcomments)

    # Aggregate languages and topics across all repos
    languages = _unique_ordered([r.get("language") for r in repos_sorted if r.get("language")])
    topics = _unique_ordered([t for r in repos_sorted for t in r.get("topics", [])])

    document = _build_document(profile, repos_sorted)
    descriptions_text = _build_descriptions_text(profile, repos_sorted)
    functions_text = " ".join(all_functions)
    comments_text = " ".join(all_comments)

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
        "descriptions_text": descriptions_text,
        "functions_text": functions_text,
        "comments_text": comments_text,
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


def _build_descriptions_text(profile, repos):
    """
    Build a text field focused on repo descriptions, topics, and profile bio.
    Used as the primary signal field for cosine similarity scoring.
    """
    parts = []
    for field in ("bio",):
        val = profile.get(field)
        if val:
            parts.append(val)
    for repo in repos:
        if repo.get("description"):
            parts.append(repo["description"])
        for topic in repo.get("topics", []):
            parts.append(topic.replace("-", " "))
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
