# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

import unittest

from tools.github_readme_sync.future_work_metadata import (
    METADATA_DOC_URL,
    is_future_work_doc_path,
    render_future_work_metadata,
)


class TestFutureWorkMetadata(unittest.TestCase):
    def test_is_future_work_doc_path(self):
        self.assertTrue(
            is_future_work_doc_path("docs/future-work/learning-module-improvements")
        )
        self.assertTrue(is_future_work_doc_path("docs/future-work"))
        self.assertFalse(is_future_work_doc_path("docs/contributing/documentation"))
        self.assertFalse(is_future_work_doc_path("docs/glossary.md"))

    def test_render_future_work_metadata_returns_empty_without_fields(self):
        result = render_future_work_metadata(
            {"title": "Example", "body": "Body text", "slug": "example"}
        )
        self.assertEqual(result, "")

    def test_render_future_work_metadata_includes_all_fields(self):
        doc = {
            "slug": "deal-with-incomplete-models",
            "estimated-scope": "large",
            "improved-metric": "learning",
            "output-type": "prototype, monty-feature, PR",
            "skills": "python, research, monty",
            "contributor": "vkakerbeck",
            "status": "open",
            "rfc": "required",
        }

        result = render_future_work_metadata(doc)

        self.assertIn("Scope", result)
        self.assertIn("large", result)
        self.assertIn("Metric", result)
        self.assertIn("learning", result)
        self.assertIn("Output", result)
        self.assertIn("prototype", result)
        self.assertIn("Skills", result)
        self.assertIn("python", result)
        self.assertIn("Status", result)
        self.assertIn("open", result)
        self.assertIn("RFC", result)
        self.assertIn("required", result)
        self.assertIn("vkakerbeck.png", result)
        self.assertNotIn("background-color", result)
        self.assertNotIn("<span", result)
        self.assertIn("display:flex", result)
        self.assertNotIn("<table", result)
        self.assertLess(result.index("Status"), result.index("Scope"))
        self.assertLess(result.index("Scope"), result.index("Output"))
        self.assertLess(result.index("Skills"), result.index("Metric"))
        self.assertLess(result.index("Metric"), result.index("RFC"))
        self.assertLess(result.index("Status"), result.index("Skills"))
        self.assertLess(result.index("Scope"), result.index("Metric"))
        self.assertLess(result.index("Output"), result.index("RFC"))
        self.assertIn(METADATA_DOC_URL, result)

    def test_render_future_work_metadata_joins_multiple_values_with_commas(self):
        doc = {"output-type": "prototype, monty-feature, PR"}
        result = render_future_work_metadata(doc)

        self.assertIn("prototype, monty-feature, PR", result)
        self.assertNotIn("<span", result)
        self.assertNotIn("background-color", result)

    def test_render_future_work_metadata_status_plain_text(self):
        doc = {"status": "in-progress"}
        result = render_future_work_metadata(doc)

        self.assertIn("in-progress", result)
        self.assertNotIn("background-color", result)
        self.assertNotIn("<span", result)

    def test_render_future_work_metadata_links_http_rfc(self):
        doc = {
            "slug": "example",
            "rfc": "https://github.com/thousandbrainsproject/tbp.monty/blob/main/rfcs/0015_future_work.md",
        }

        result = render_future_work_metadata(doc)

        self.assertIn("0015_future_work.md", result)
        self.assertIn(">RFC</a>", result)


if __name__ == "__main__":
    unittest.main()
