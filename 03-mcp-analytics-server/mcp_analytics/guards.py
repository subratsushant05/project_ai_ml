"""SQL statement validation: allow a single read-only SELECT, reject the rest.

This is the first of three defensive layers (validator, SQLite authorizer
callback, read-only ``mode=ro`` connection). The validator works on a copy of
the statement with string literals, quoted identifiers, and comments blanked
out, so keywords smuggled inside literals ("SELECT 'drop table'") never cause
false positives and keywords hidden behind comments never slip through.
"""

from __future__ import annotations

import re

__all__ = ["QueryValidationError", "validate_select"]

_BANNED_KEYWORDS: frozenset[str] = frozenset(
    {
        "insert", "update", "delete", "drop", "alter", "create", "replace",
        "pragma", "attach", "detach", "vacuum", "reindex", "analyze",
        "begin", "commit", "rollback", "savepoint", "release",
        "grant", "revoke", "truncate", "merge", "upsert",
    }
)

_ALLOWED_LEADING = ("select", "with", "values", "explain")
_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class QueryValidationError(ValueError):
    """Raised when a SQL statement fails the read-only policy."""


def _blank_literals_and_comments(sql: str) -> str:
    """Replace literals and comments with spaces, preserving structure.

    Handles single-quoted strings (with ``''`` escapes), double-quoted and
    backtick-quoted identifiers, ``[bracketed]`` identifiers, ``--`` line
    comments, and ``/* */`` block comments.

    Args:
        sql: Raw SQL text.

    Returns:
        The SQL with literal/comment interiors replaced by spaces.

    Raises:
        QueryValidationError: If a literal or comment is left unterminated.
    """
    out: list[str] = []
    i, n = 0, len(sql)
    while i < n:
        ch = sql[i]
        if ch == "'" or ch == '"' or ch == "`":
            closer, j = ch, i + 1
            while j < n:
                if sql[j] == closer:
                    if j + 1 < n and sql[j + 1] == closer:  # escaped quote
                        j += 2
                        continue
                    break
                j += 1
            if j >= n:
                raise QueryValidationError("Unterminated quoted literal in query.")
            out.append(" " * (j + 1 - i))
            i = j + 1
        elif ch == "[":
            j = sql.find("]", i + 1)
            if j == -1:
                raise QueryValidationError("Unterminated [identifier] in query.")
            out.append(" " * (j + 1 - i))
            i = j + 1
        elif ch == "-" and sql[i : i + 2] == "--":
            j = sql.find("\n", i)
            j = n if j == -1 else j
            out.append(" " * (j - i))
            i = j
        elif ch == "/" and sql[i : i + 2] == "/*":
            j = sql.find("*/", i + 2)
            if j == -1:
                raise QueryValidationError("Unterminated block comment in query.")
            out.append(" " * (j + 2 - i))
            i = j + 2
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def validate_select(sql: str) -> str:
    """Validate that ``sql`` is one read-only SELECT-family statement.

    Args:
        sql: Raw SQL text supplied by the client.

    Returns:
        The trimmed statement (trailing semicolon removed), safe to execute
        on a read-only connection.

    Raises:
        QueryValidationError: If the statement is empty, contains multiple
            statements, does not start with SELECT/WITH/VALUES, or references
            a mutating / administrative keyword outside string literals.
    """
    if not isinstance(sql, str) or not sql.strip():
        raise QueryValidationError("Query is empty.")

    blanked = _blank_literals_and_comments(sql)
    # Work out the trimmed statement using the blanked text as a map.
    stripped_blanked = blanked.rstrip().rstrip(";").rstrip()
    if not stripped_blanked.strip():
        raise QueryValidationError("Query contains no executable statement.")
    if ";" in stripped_blanked:
        raise QueryValidationError(
            "Multiple SQL statements are not allowed; submit one SELECT."
        )

    words = [w.lower() for w in _WORD_RE.findall(stripped_blanked)]
    if not words or words[0] not in _ALLOWED_LEADING:
        raise QueryValidationError(
            "Only read-only SELECT statements are allowed "
            "(must start with SELECT, WITH, or VALUES)."
        )

    banned_hits = sorted(set(words) & _BANNED_KEYWORDS)
    if banned_hits:
        raise QueryValidationError(
            f"Statement rejected: disallowed keyword(s) {', '.join(banned_hits)}. "
            "This server only executes read-only queries."
        )

    return sql[: len(stripped_blanked)].strip()
