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

    def search_repositories(self, query, sort=None, order=None, per_page=30, page=1):
        """Search GitHub repositories by query string."""
        params = {
            "q": query,
            "per_page": per_page,
            "page": page,
        }
        if sort:
            params["sort"] = sort
        if order:
            params["order"] = order
        return self._get("/search/repositories", params=params)

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

    def get_readme(self, owner, repo):
        """Fetch the README for a repository. Returns raw text or empty string."""
        import base64
        try:
            data = self._get(f"/repos/{owner}/{repo}/readme")
            content = data.get("content", "")
            if data.get("encoding") == "base64":
                return base64.b64decode(content).decode("utf-8", errors="ignore")
            return content
        except Exception:
            return ""

    def get_repo_tree(self, owner, repo):
        """
        Fetch the recursive file tree for a repository.
        Returns a list of tree node dicts with 'path', 'type', 'size'.
        """
        try:
            data = self._get(
                f"/repos/{owner}/{repo}/git/trees/HEAD",
                params={"recursive": "1"},
            )
            return data.get("tree", [])
        except Exception:
            return []

    def get_file_content(self, owner, repo, path):
        """Fetch raw text content of a single file. Returns empty string on failure."""
        import base64
        try:
            data = self._get(f"/repos/{owner}/{repo}/contents/{path}")
            content = data.get("content", "")
            if data.get("encoding") == "base64":
                return base64.b64decode(content).decode("utf-8", errors="ignore")
            return content
        except Exception:
            return ""

    def get_rate_limit(self):
        """Check current API rate limit status."""
        return self._get("/rate_limit")
