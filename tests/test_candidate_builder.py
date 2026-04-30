import unittest
from unittest.mock import mock_open, patch

from github_client.candidate_builder import (
    build_candidate,
    build_candidates_from_search,
    _save_search_output,
    _build_openai_query_chunks,
    _build_short_keyword_queries,
    _extract_keywords,
    _extract_qualifier_parts,
    _iterative_search,
)


class CandidateBuilderQueryTests(unittest.TestCase):
    def setUp(self):
        patcher = patch("github_client.candidate_builder._build_openai_query_chunks", return_value=[])
        self.openai_chunks_patcher = patcher
        self.mock_openai_chunks = patcher.start()
        self.addCleanup(self._stop_openai_chunks_patcher)

    def _stop_openai_chunks_patcher(self):
        if self.mock_openai_chunks is not None:
            self.openai_chunks_patcher.stop()
            self.mock_openai_chunks = None

    def test_qualifier_words_are_excluded_from_keywords(self):
        qualifiers, qualifier_words = _extract_qualifier_parts(
            "Build a Python machine learning calendar app"
        )

        self.assertEqual(qualifiers, "language:Python topic:machine-learning")
        self.assertEqual(
            _extract_keywords("Build a Python machine learning calendar app", qualifier_words),
            ["calendar", "app"],
        )

    def test_qualifier_only_query_does_not_reuse_original_query(self):
        class FakeClient:
            def __init__(self):
                self.queries = []

            def search_repositories(self, query, per_page):
                self.queries.append(query)
                return {"items": []}

        client = FakeClient()

        _iterative_search("Python machine learning", client, per_page=10)

        self.assertEqual(client.queries, ["language:Python topic:machine-learning"])

    def test_iterative_search_returns_queries_sent_to_github(self):
        class FakeClient:
            def search_repositories(self, query, per_page):
                return {"items": []}

        result = _iterative_search("Python calendar app", FakeClient(), per_page=10)

        self.assertEqual(
            result["queries"],
            [
                "calendar app in:name,description,readme language:Python",
                "calendar integration in:name,description,readme language:Python",
            ],
        )

    def test_saved_search_output_uses_queries_list(self):
        search_results = {
            "queries": ["calendar app in:name,description,readme"],
            "total_count": 0,
            "items": [],
        }

        with patch("github_client.candidate_builder.os.makedirs"), patch(
            "github_client.candidate_builder.open",
            mock_open(),
        ) as opened, patch("github_client.candidate_builder.json.dump") as dumped:
            _save_search_output(search_results)

        opened.assert_called_once()
        payload = dumped.call_args.args[0]
        self.assertEqual(payload["queries"], ["calendar app in:name,description,readme"])
        self.assertNotIn("query", payload)

    def test_build_candidates_can_return_search_queries(self):
        class FakeClient:
            def search_repositories(self, query, per_page):
                return {"items": []}

        with patch("github_client.candidate_builder._save_search_output"):
            candidates, queries = build_candidates_from_search(
                "Python calendar app",
                FakeClient(),
                return_search_queries=True,
            )

        self.assertEqual(candidates, [])
        self.assertEqual(
            queries,
            [
                "calendar app in:name,description,readme language:Python",
                "calendar integration in:name,description,readme language:Python",
            ],
        )

    def test_builds_short_keyword_queries_from_prompt(self):
        qualifiers, qualifier_words = _extract_qualifier_parts(
            "I want to build a multiplayer chess game in C++ "
            "with online matchmaking and an AI opponent"
        )
        keywords = _extract_keywords(
            "I want to build a multiplayer chess game in C++ "
            "with online matchmaking and an AI opponent",
            qualifier_words,
        )

        self.assertEqual(qualifiers, "language:C++")
        self.assertEqual(
            _build_short_keyword_queries(keywords),
            [
                "multiplayer chess",
                "chess game",
                "AI opponent",
                "chess engine",
                "minimax chess",
                "multiplayer game",
                "game server",
                "matchmaking game",
            ],
        )

    def test_openai_query_chunks_use_structured_response(self):
        self._stop_openai_chunks_patcher()

        class FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {"output_text": '{"queries":["chess game","cpp chess","language:C++ minimax chess"]}'}

        with patch.dict("github_client.candidate_builder.os.environ", {"OPENAI_API_KEY": "test-key"}), patch(
            "github_client.candidate_builder.requests.post", return_value=FakeResponse()
        ) as posted:
            chunks = _build_openai_query_chunks(
                "Build a multiplayer chess game in C++",
                ["multiplayer", "chess", "game"],
                "language:C++",
            )

        self.assertEqual(chunks, ["chess game", "cpp chess", "minimax chess"])
        self.assertEqual(posted.call_args.kwargs["headers"]["Authorization"], "Bearer test-key")
        self.assertEqual(
            posted.call_args.kwargs["json"]["text"]["format"]["type"],
            "json_schema",
        )

    def test_iterative_search_uses_openai_generated_chunks_when_available(self):
        self.mock_openai_chunks.return_value = ["chess game", "cpp chess"]

        class FakeClient:
            def __init__(self):
                self.queries = []

            def search_repositories(self, query, per_page):
                self.queries.append(query)
                return {"items": []}

        client = FakeClient()

        _iterative_search("Build a chess game in C++", client, per_page=10)

        self.assertEqual(
            client.queries,
            [
                "chess game in:name,description,readme language:C++",
                "cpp chess in:name,description,readme language:C++",
            ],
        )

    def test_keyword_queries_search_name_description_and_readme(self):
        class FakeClient:
            def __init__(self):
                self.queries = []

            def search_repositories(self, query, per_page):
                self.queries.append(query)
                return {"items": []}

        client = FakeClient()

        _iterative_search("Python calendar app", client, per_page=10)

        self.assertEqual(
            client.queries,
            [
                "calendar app in:name,description,readme language:Python",
                "calendar integration in:name,description,readme language:Python",
            ],
        )

    def test_build_candidate_does_not_sort_repos_by_stars(self):
        class FakeClient:
            def __init__(self):
                self.readmes = []

            def get_user(self, username):
                return {
                    "type": "User",
                    "name": "Test User",
                    "bio": None,
                    "location": None,
                    "html_url": f"https://github.com/{username}",
                    "avatar_url": None,
                    "followers": 0,
                    "public_repos": 2,
                }

            def get_user_repos(self, username, per_page=100):
                return [
                    {
                        "name": "recent-zero-star",
                        "description": None,
                        "language": "Python",
                        "stargazers_count": 0,
                        "topics": [],
                        "html_url": "https://github.com/test/recent-zero-star",
                    },
                    {
                        "name": "older-popular",
                        "description": None,
                        "language": "Python",
                        "stargazers_count": 99,
                        "topics": [],
                        "html_url": "https://github.com/test/older-popular",
                    },
                ]

            def get_readme(self, owner, repo):
                self.readmes.append(repo)
                return f"{repo} readme"

        client = FakeClient()

        candidate = build_candidate("test", client, readme_limit=1)

        self.assertEqual(client.readmes, ["recent-zero-star"])
        self.assertEqual(
            [repo["name"] for repo in candidate["repos"]],
            ["recent-zero-star", "older-popular"],
        )


if __name__ == "__main__":
    unittest.main()
