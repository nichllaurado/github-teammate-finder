import os
import requests
from dotenv import load_dotenv

load_dotenv()

GITHUB_API_BASE = "https://api.github.com"


class GitHubClient:
    def __init__(self):
        token = os.getenv("GITHUB_TOKEN")
        self.headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if token:
            self.headers["Authorization"] = f"Bearer {token}"

    def _get(self, path, params=None):
        response = requests.get(
            f"{GITHUB_API_BASE}{path}",
            headers=self.headers,
            params=params,
            timeout=10,
        )
        response.raise_for_status()
        return response.json()

    def search_repositories(self, query, sort="stars", order="desc", per_page=30, page=1):
        """Search GitHub repositories by query string."""
        return self._get("/search/repositories", params={
            "q": query,
            "sort": sort,
            "order": order,
            "per_page": per_page,
            "page": page,
        })

    def get_user(self, username):
        """Fetch a GitHub user's public profile."""
        return self._get(f"/users/{username}")

    def get_user_repos(self, username, per_page=100, page=1):
        """Fetch a GitHub user's public repositories."""
        return self._get(f"/users/{username}/repos", params={
            "type": "owner",
            "sort": "updated",
            "per_page": per_page,
            "page": page,
        })

    def get_rate_limit(self):
        """Check current API rate limit status."""
        return self._get("/rate_limit")
