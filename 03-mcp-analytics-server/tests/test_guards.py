"""Unit tests for the SELECT-only statement validator."""

from __future__ import annotations

import pytest

from mcp_analytics.guards import QueryValidationError, validate_select


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT 1",
        "select * from customers",
        "  SELECT name FROM products ORDER BY price DESC LIMIT 5;  ",
        "WITH t AS (SELECT 1 AS x) SELECT x FROM t",
        "VALUES (1, 2), (3, 4)",
        "SELECT 'drop table customers' AS scary_string",
        'SELECT "delete" FROM customers',  # quoted identifier, not a keyword
        "SELECT 1 -- trailing comment with DROP TABLE",
        "SELECT /* insert update */ 42",
    ],
)
def test_valid_statements_pass(sql: str) -> None:
    assert validate_select(sql)


@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO customers VALUES (999, 'x', 'x@x.com', 'USA', '2024-01-01')",
        "UPDATE customers SET name = 'hacked'",
        "DELETE FROM orders",
        "DROP TABLE customers",
        "CREATE TABLE evil (x)",
        "ALTER TABLE customers ADD COLUMN pwned TEXT",
        "PRAGMA writable_schema = 1",
        "ATTACH DATABASE '/tmp/evil.db' AS evil",
        "VACUUM",
        "REPLACE INTO customers VALUES (1, 'x', 'y', 'z', 'w')",
        "BEGIN; DROP TABLE customers; COMMIT",
    ],
)
def test_mutating_statements_rejected(sql: str) -> None:
    with pytest.raises(QueryValidationError):
        validate_select(sql)


def test_multi_statement_rejected() -> None:
    with pytest.raises(QueryValidationError, match="[Mm]ultiple"):
        validate_select("SELECT 1; SELECT 2")


def test_piggybacked_mutation_rejected() -> None:
    with pytest.raises(QueryValidationError):
        validate_select("SELECT * FROM customers; DROP TABLE customers")


def test_comment_hidden_mutation_rejected() -> None:
    """A semicolon+DROP after a comment must still be caught."""
    with pytest.raises(QueryValidationError):
        validate_select("SELECT 1 /* c */ ; DELETE FROM orders")


def test_case_insensitive_rejection() -> None:
    with pytest.raises(QueryValidationError):
        validate_select("dRoP tAbLe customers")


@pytest.mark.parametrize("sql", ["", "   ", ";;", "-- just a comment"])
def test_empty_input_rejected(sql: str) -> None:
    with pytest.raises(QueryValidationError):
        validate_select(sql)


def test_unterminated_literal_rejected() -> None:
    with pytest.raises(QueryValidationError, match="[Uu]nterminated"):
        validate_select("SELECT 'oops")


def test_trailing_semicolon_stripped() -> None:
    assert validate_select("SELECT 1;") == "SELECT 1"
