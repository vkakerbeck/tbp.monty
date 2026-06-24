# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Any

FUTURE_WORK_SECTION = "future-work"

FUTURE_WORK_METADATA_KEYS = frozenset(
    {
        "estimated-scope",
        "improved-metric",
        "output-type",
        "skills",
        "contributor",
        "status",
        "rfc",
    }
)

GITHUB_AVATAR_URL = "https://github.com"
GITHUB_USERNAME_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9-]{0,38}$")

BADGE_STYLE = (
    "padding:2px 4px;border-radius:4px;font-size:0.85em;"
    "display:inline-block;margin:2px 4px 2px 0;"
)
DEFAULT_BADGE_STYLE = f"{BADGE_STYLE}background-color:#e8e8f0;color:#2f2b5c;"
STATUS_BADGE_STYLE = (
    f"{BADGE_STYLE}background-color:#ffffff;color:#666;border:1px solid #999;"
)
SCOPE_STYLES = {
    "small": f"{BADGE_STYLE}background-color:#f0f0f0;color:#666;",
    "medium": f"{BADGE_STYLE}background-color:#00a0df;color:#ffffff;",
    "large": f"{BADGE_STYLE}background-color:#2f2b5c;color:#ffffff;",
    "unknown": DEFAULT_BADGE_STYLE,
}
METADATA_CONTAINER_STYLE = (
    "border:1px solid #ddd;border-radius:8px;padding:12px 16px;"
    "margin-bottom:16px;background-color:#fafafa;"
)


def is_future_work_doc_path(file_path: str) -> bool:
    """Check whether a document path is within the future work docs section.

    Returns:
        True when the path is under the future work section.
    """
    return FUTURE_WORK_SECTION in Path(file_path).as_posix().split("/")


def _has_metadata_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return any(str(item).strip() for item in value)
    return True


def _split_values(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if not value:
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _badge(text: str, style: str = DEFAULT_BADGE_STYLE) -> str:
    return f'<span style="{style}">{html.escape(text)}</span>'


def _scope_badge(scope: str) -> str:
    scope_key = scope.strip().lower()
    return _badge(scope, SCOPE_STYLES.get(scope_key, DEFAULT_BADGE_STYLE))


def _label_row(label: str, content: str) -> str:
    if not content:
        return ""
    return (
        f'<div style="margin-bottom:6px;">'
        f"<strong>{html.escape(label)}:</strong> {content}"
        f"</div>"
    )


def render_future_work_metadata(doc: dict[str, Any]) -> str:
    """Render future work frontmatter fields as an HTML metadata block.

    Args:
        doc: Processed markdown document containing optional metadata fields.

    Returns:
        HTML metadata block, or an empty string when no metadata is present.
    """
    fields = {
        key: doc[key]
        for key in FUTURE_WORK_METADATA_KEYS
        if key in doc and _has_metadata_value(doc[key])
    }
    if not fields:
        return ""

    rows: list[str] = []

    status_parts: list[str] = []
    if "status" in fields:
        status_parts.append(_badge(str(fields["status"]), STATUS_BADGE_STYLE))
    if "contributor" in fields:
        for username in _split_values(fields["contributor"]):
            if GITHUB_USERNAME_PATTERN.match(username):
                avatar_url = f"{GITHUB_AVATAR_URL}/{html.escape(username)}.png"
                status_parts.append(
                    f'<img src="{avatar_url}" alt="{html.escape(username)}" '
                    f'title="{html.escape(username)}" '
                    f'style="width:24px;height:24px;border-radius:50%;'
                    f'vertical-align:middle;margin-right:4px;" />'
                )
    if status_parts:
        rows.append(_label_row("Status", " ".join(status_parts)))

    if "estimated-scope" in fields:
        rows.append(_label_row("Scope", _scope_badge(str(fields["estimated-scope"]))))

    for key, label in (
        ("improved-metric", "Metric"),
        ("output-type", "Output Type"),
        ("skills", "Skills"),
    ):
        if key in fields:
            badges = " ".join(_badge(value) for value in _split_values(fields[key]))
            rows.append(_label_row(label, badges))

    if "rfc" in fields:
        rfc = str(fields["rfc"]).strip()
        if rfc.lower().startswith(("http://", "https://")):
            rfc_content = (
                f'<a href="{html.escape(rfc)}" target="_blank" '
                f'rel="noopener noreferrer">RFC</a>'
            )
        else:
            rfc_content = html.escape(rfc)
        rows.append(_label_row("RFC", rfc_content))

    return f'<div style="{METADATA_CONTAINER_STYLE}">{"".join(rows)}</div>'
