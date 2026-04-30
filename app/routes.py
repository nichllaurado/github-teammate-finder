import json

from flask import Blueprint, Response, jsonify, request, stream_with_context
from github_client import GitHubClient, build_candidates_from_search, build_candidate
from ir import rank_candidates
import requests as http_requests

bp = Blueprint("api", __name__)
client = GitHubClient()


@bp.route("/search/repos")
def search_repos():
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({"error": "Missing query parameter 'q'"}), 400

    sort = request.args.get("sort")
    order = request.args.get("order")
    per_page = min(int(request.args.get("per_page", 30)), 100)
    page = int(request.args.get("page", 1))

    try:
        data = client.search_repositories(query, sort=sort, order=order, per_page=per_page, page=page)
        return jsonify({
            "total_count": data.get("total_count"),
            "items": [_slim_repo(r) for r in data.get("items", [])],
        })
    except http_requests.HTTPError as e:
        return jsonify({"error": str(e)}), e.response.status_code


@bp.route("/users/<username>")
def get_user(username):
    try:
        data = client.get_user(username)
        return jsonify(_slim_user(data))
    except http_requests.HTTPError as e:
        return jsonify({"error": str(e)}), e.response.status_code


@bp.route("/users/<username>/repos")
def get_user_repos(username):
    per_page = min(int(request.args.get("per_page", 100)), 100)
    page = int(request.args.get("page", 1))

    try:
        repos = client.get_user_repos(username, per_page=per_page, page=page)
        return jsonify([_slim_repo(r) for r in repos])
    except http_requests.HTTPError as e:
        return jsonify({"error": str(e)}), e.response.status_code


@bp.route("/candidates")
def get_candidates():
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({"error": "Missing query parameter 'q'"}), 400

    per_page = min(int(request.args.get("per_page", 20)), 30)
    readme_limit = min(int(request.args.get("readme_limit", 3)), 5)

    try:
        candidates, queries = build_candidates_from_search(
            query,
            client,
            per_page=per_page,
            readme_limit=readme_limit,
            return_search_queries=True,
        )
        ranked = rank_candidates(query, candidates)
        return jsonify({"queries": queries, "candidates": ranked})
    except http_requests.HTTPError as e:
        return jsonify({"error": str(e)}), e.response.status_code


@bp.route("/candidates/stream")
def stream_candidates():
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({"error": "Missing query parameter 'q'"}), 400

    per_page = min(int(request.args.get("per_page", 20)), 30)
    readme_limit = min(int(request.args.get("readme_limit", 3)), 5)

    def event(event_name, payload):
        return f"event: {event_name}\ndata: {json.dumps(payload)}\n\n"

    @stream_with_context
    def generate():
        try:
            candidates, queries = build_candidates_from_search(
                query,
                client,
                per_page=per_page,
                readme_limit=readme_limit,
                return_search_queries=True,
            )
            yield event("ranking", {
                "message": "Candidates fetched - ranking candidates...",
                "queries": queries,
                "candidate_count": len(candidates),
            })

            ranked = rank_candidates(query, candidates)
            yield event("complete", {"queries": queries, "candidates": ranked})
        except http_requests.HTTPError as e:
            status_code = e.response.status_code if e.response is not None else 500
            yield event("error", {"error": str(e), "status_code": status_code})
        except Exception as e:
            yield event("error", {"error": str(e), "status_code": 500})

    return Response(generate(), mimetype="text/event-stream")


@bp.route("/rate-limit")
def rate_limit():
    try:
        return jsonify(client.get_rate_limit())
    except http_requests.HTTPError as e:
        return jsonify({"error": str(e)}), e.response.status_code


def _slim_user(u):
    return {
        "login": u.get("login"),
        "name": u.get("name"),
        "bio": u.get("bio"),
        "location": u.get("location"),
        "html_url": u.get("html_url"),
        "avatar_url": u.get("avatar_url"),
        "public_repos": u.get("public_repos"),
        "followers": u.get("followers"),
    }


def _slim_repo(r):
    return {
        "id": r.get("id"),
        "name": r.get("name"),
        "full_name": r.get("full_name"),
        "owner": r.get("owner", {}).get("login"),
        "html_url": r.get("html_url"),
        "description": r.get("description"),
        "language": r.get("language"),
        "stargazers_count": r.get("stargazers_count"),
        "topics": r.get("topics", []),
        "updated_at": r.get("updated_at"),
    }
