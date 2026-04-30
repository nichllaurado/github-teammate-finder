import unittest

from github_client.candidate_builder import (
    _extract_keywords,
    _extract_qualifier_parts,
    _iterative_search,
)


class CandidateBuilderQueryTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
