import unittest

from github_client.client import GitHubClient


class GitHubClientTests(unittest.TestCase):
    def test_search_repositories_omits_sort_by_default(self):
        class FakeGitHubClient(GitHubClient):
            def __init__(self):
                self.calls = []

            def _get(self, path, params=None):
                self.calls.append((path, params))
                return {}

        client = FakeGitHubClient()

        client.search_repositories("calendar app")

        self.assertEqual(client.calls[0][0], "/search/repositories")
        self.assertNotIn("sort", client.calls[0][1])
        self.assertNotIn("order", client.calls[0][1])


if __name__ == "__main__":
    unittest.main()
