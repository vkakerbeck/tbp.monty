# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import html
import tempfile
from pathlib import Path

from tools.github_readme_sync.md import process_markdown
from tools.github_readme_sync.readme import ReadMe

md_path = Path(
    "docs/future-work/learning-module-improvements/dynamic-adjustment-of-hypothesis-space-size.md"
)
doc = process_markdown(md_path.read_text(encoding="utf-8"), md_path.stem)
file_path = "docs/future-work/learning-module-improvements"
body = ReadMe("preview").process_markdown(doc["body"], file_path, doc["slug"], doc=doc)

with tempfile.NamedTemporaryFile(
    mode="w",
    suffix=".html",
    prefix="future-work-preview-",
    delete=False,
    encoding="utf-8",
) as preview_file:
    preview_file.write(
        f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{html.escape(doc["title"])}</title></head>
<body style="font-family:sans-serif;max-width:800px;margin:2rem auto;padding:0 1rem;">
<h1>{html.escape(doc["title"])}</h1>
<p><em>{html.escape(doc.get("description", ""))}</em></p>
{body}
</body></html>"""
    )
    out = Path(preview_file.name)

print(f"Open {out}")
