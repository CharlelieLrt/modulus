# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Count source lines between ``# LOC-START`` and ``# LOC-END`` markers.

Blank lines and comment-only lines are excluded so the count reflects the
necessary training/inference code rather than whitespace.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

_START_RE = re.compile(r"^\s*#\s*LOC-START\b")
_END_RE = re.compile(r"^\s*#\s*LOC-END\b")
_COMMENT_ONLY_RE = re.compile(r"^\s*(#.*)?$")


def _function_span(source: str, function_name: str) -> tuple[int, int] | None:
    """Return ``(start_line, end_line)`` (0-indexed, end exclusive) of the
    named top-level function, or None if not found.

    Uses ``ast`` to handle multi-line signatures, decorators, and nested defs
    correctly.
    """

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    for node in tree.body:
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == function_name
        ):
            start = node.lineno - 1
            end = node.end_lineno or start + 1
            return start, end
    return None


def count_marked_loc(
    file_path: str | Path,
    *,
    function_name: str | None = None,
) -> int | None:
    """Count non-blank, non-comment-only lines between LOC-START / LOC-END.

    If ``function_name`` is given the search is restricted to that function's
    body. Returns ``None`` if no marker is found.
    """

    text = Path(file_path).read_text()
    lines = text.splitlines()
    if function_name is not None:
        span = _function_span(text, function_name)
        if span is None:
            return None
        lines = lines[span[0] : span[1]]

    count = 0
    inside = False
    matched = False
    for line in lines:
        if _START_RE.search(line):
            inside = True
            matched = True
            continue
        if _END_RE.search(line):
            inside = False
            continue
        if inside and not _COMMENT_ONLY_RE.match(line):
            count += 1
    return count if matched else None


def loc_table(
    file_path: str | Path,
    function_names: list[str],
) -> dict[str, int | None]:
    """Convenience: count LoC for several functions in the same file."""

    return {fn: count_marked_loc(file_path, function_name=fn) for fn in function_names}
