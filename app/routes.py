from flask import Blueprint, jsonify, request
from github_client import GitHubClient
import requests as http_requests

bp = Blueprint("api", __name__)
client = GitHubClient()


@bp.route("/search/repos")
def search_repos():
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({"error": "Missing query parameter 'q'"}), 400

    sort = request.args.get("sort", "stars")
    order = request.args.get("order", "desc")
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
