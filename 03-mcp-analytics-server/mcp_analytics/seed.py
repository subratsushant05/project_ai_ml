"""Deterministic synthetic e-commerce dataset generator.

Creates a small SQLite database (customers, products, orders, order_items)
with roughly 500 rows. The generator is seeded, so every invocation produces
byte-for-byte identical data -- important for reproducible demos and tests.

Usage:
    python -m mcp_analytics.seed [--db PATH] [--force]
"""

from __future__ import annotations

import argparse
import logging
import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

from .config import get_settings

logger = logging.getLogger(__name__)

RANDOM_SEED = 42
N_CUSTOMERS = 60
N_PRODUCTS = 40
N_ORDERS = 130

_FIRST_NAMES = [
    "Ava", "Ben", "Chloe", "Dev", "Elena", "Farid", "Grace", "Hugo", "Iris",
    "Jonas", "Kira", "Liam", "Maya", "Noor", "Owen", "Priya", "Quinn", "Rosa",
    "Sam", "Tara", "Umar", "Vera", "Wes", "Xin", "Yara", "Zane",
]
_LAST_NAMES = [
    "Anders", "Brooks", "Costa", "Diaz", "Ellis", "Fischer", "Gupta", "Haddad",
    "Ivanov", "Jensen", "Khan", "Lopez", "Meyer", "Nakamura", "Okafor", "Patel",
    "Quist", "Rossi", "Silva", "Tanaka",
]
_COUNTRIES = ["USA", "Germany", "India", "Brazil", "Japan", "UK", "Canada"]
_CATEGORIES = ["Electronics", "Books", "Home", "Sports", "Toys"]
_ADJECTIVES = ["Compact", "Deluxe", "Eco", "Pro", "Smart", "Ultra", "Classic", "Mini"]
_NOUNS = {
    "Electronics": ["Headphones", "Keyboard", "Monitor", "Speaker", "Webcam"],
    "Books": ["Novel", "Cookbook", "Atlas", "Biography", "Guide"],
    "Home": ["Lamp", "Kettle", "Blanket", "Organizer", "Planter"],
    "Sports": ["Backpack", "Water Bottle", "Yoga Mat", "Jersey", "Tracker"],
    "Toys": ["Puzzle", "Blocks", "Robot Kit", "Board Game", "Plushie"],
}
_STATUSES = ["delivered", "delivered", "delivered", "shipped", "pending", "cancelled"]

SCHEMA_SQL = """
CREATE TABLE customers (
    customer_id  INTEGER PRIMARY KEY,
    name         TEXT NOT NULL,
    email        TEXT NOT NULL UNIQUE,
    country      TEXT NOT NULL,
    signup_date  TEXT NOT NULL
);
CREATE TABLE products (
    product_id  INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    category    TEXT NOT NULL,
    price       REAL NOT NULL CHECK (price > 0),
    stock       INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE orders (
    order_id     INTEGER PRIMARY KEY,
    customer_id  INTEGER NOT NULL REFERENCES customers(customer_id),
    order_date   TEXT NOT NULL,
    status       TEXT NOT NULL,
    total_amount REAL NOT NULL
);
CREATE TABLE order_items (
    order_item_id INTEGER PRIMARY KEY,
    order_id      INTEGER NOT NULL REFERENCES orders(order_id),
    product_id    INTEGER NOT NULL REFERENCES products(product_id),
    quantity      INTEGER NOT NULL CHECK (quantity > 0),
    unit_price    REAL NOT NULL
);
CREATE INDEX idx_orders_customer ON orders(customer_id);
CREATE INDEX idx_items_order ON order_items(order_id);
"""


def seed_database(db_path: Path, force: bool = False) -> Path:
    """Create and populate the sample database.

    Args:
        db_path: Target SQLite file. Parent directories are created.
        force: If True, overwrite an existing database file.

    Returns:
        The path of the created database.

    Raises:
        FileExistsError: If the file exists and ``force`` is False.
    """
    if db_path.exists():
        if not force:
            raise FileExistsError(f"Database already exists: {db_path}")
        db_path.unlink()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(RANDOM_SEED)
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(SCHEMA_SQL)
        _insert_customers(conn, rng)
        _insert_products(conn, rng)
        _insert_orders(conn, rng)
        conn.commit()
    finally:
        conn.close()
    logger.info("Seeded database at %s", db_path)
    return db_path


def _insert_customers(conn: sqlite3.Connection, rng: random.Random) -> None:
    """Insert ``N_CUSTOMERS`` synthetic customers."""
    base = date(2023, 1, 1)
    rows = []
    for cid in range(1, N_CUSTOMERS + 1):
        first, last = rng.choice(_FIRST_NAMES), rng.choice(_LAST_NAMES)
        signup = base + timedelta(days=rng.randint(0, 700))
        email = f"{first.lower()}.{last.lower()}{cid}@example.com"
        rows.append((cid, f"{first} {last}", email, rng.choice(_COUNTRIES), signup.isoformat()))
    conn.executemany("INSERT INTO customers VALUES (?,?,?,?,?)", rows)


def _insert_products(conn: sqlite3.Connection, rng: random.Random) -> None:
    """Insert ``N_PRODUCTS`` synthetic products."""
    rows = []
    for pid in range(1, N_PRODUCTS + 1):
        category = rng.choice(_CATEGORIES)
        name = f"{rng.choice(_ADJECTIVES)} {rng.choice(_NOUNS[category])} #{pid}"
        price = round(rng.uniform(4.99, 399.99), 2)
        rows.append((pid, name, category, price, rng.randint(0, 250)))
    conn.executemany("INSERT INTO products VALUES (?,?,?,?,?)", rows)


def _insert_orders(conn: sqlite3.Connection, rng: random.Random) -> None:
    """Insert ``N_ORDERS`` orders, each with 1-4 line items."""
    prices = dict(conn.execute("SELECT product_id, price FROM products").fetchall())
    base = date(2024, 1, 1)
    item_id = 0
    for oid in range(1, N_ORDERS + 1):
        n_items = rng.randint(1, 4)
        product_ids = rng.sample(sorted(prices), n_items)
        items, total = [], 0.0
        for pid in product_ids:
            item_id += 1
            qty = rng.randint(1, 3)
            unit = prices[pid]
            total += qty * unit
            items.append((item_id, oid, pid, qty, unit))
        conn.execute(
            "INSERT INTO orders VALUES (?,?,?,?,?)",
            (
                oid,
                rng.randint(1, N_CUSTOMERS),
                (base + timedelta(days=rng.randint(0, 540))).isoformat(),
                rng.choice(_STATUSES),
                round(total, 2),
            ),
        )
        conn.executemany("INSERT INTO order_items VALUES (?,?,?,?,?)", items)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for seeding the database."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Seed the sample e-commerce database.")
    parser.add_argument("--db", type=Path, default=None, help="Target SQLite file path.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing file.")
    args = parser.parse_args(argv)
    target = args.db or get_settings().db_path
    seed_database(target, force=args.force)
    print(f"Seeded sample database: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
