"""
Builds the deterministic SQLite fixture (shop.db) used by the
sql_generation domain. Re-run with `python3 build_db.py` to regenerate
shop.db from scratch if the schema or seed data below ever changes -- the
committed shop.db in this directory is the actual fixture tools.py and
evaluator.py read; this script exists so the fixture is reproducible and
auditable rather than an opaque binary.

All data is hand-seeded with a fixed random seed. No network access.
"""

from __future__ import annotations

import random
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "shop.db"

SCHEMA = """
CREATE TABLE customers (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    city TEXT NOT NULL,
    signup_date TEXT NOT NULL,
    loyalty_tier TEXT NOT NULL
);

CREATE TABLE products (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    price REAL NOT NULL
);

CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(id),
    order_date TEXT NOT NULL,
    status TEXT NOT NULL
);

CREATE TABLE order_items (
    id INTEGER PRIMARY KEY,
    order_id INTEGER NOT NULL REFERENCES orders(id),
    product_id INTEGER NOT NULL REFERENCES products(id),
    quantity INTEGER NOT NULL
);
"""

CITIES = ["Austin", "Denver", "Seattle", "Boston", "Portland", "Chicago", "Miami", "Raleigh"]
TIERS = ["bronze", "silver", "gold"]
FIRST_NAMES = [
    "Alice", "Ben", "Cora", "Derek", "Elena", "Farid", "Grace", "Hassan",
    "Ines", "Jamal", "Kira", "Leo", "Mira", "Noel", "Priya", "Quinn",
    "Rosa", "Sam", "Tara", "Umar",
]
LAST_NAMES = ["Nguyen", "Smith", "Okoro", "Garcia", "Kim", "Patel", "Rossi", "Dubois"]

CATEGORIES = ["electronics", "outdoor", "home", "kitchen", "fitness"]
PRODUCT_NAMES = [
    ("Wireless Earbuds", "electronics", 89.99), ("4K Monitor", "electronics", 249.00),
    ("Bluetooth Speaker", "electronics", 59.50), ("Laptop Stand", "electronics", 34.99),
    ("Camp Tent", "outdoor", 129.00), ("Trail Backpack", "outdoor", 74.99),
    ("Insulated Bottle", "outdoor", 19.99), ("Hiking Boots", "outdoor", 109.00),
    ("Throw Blanket", "home", 29.99), ("Ceramic Mug Set", "home", 22.50),
    ("Desk Organizer", "home", 17.99), ("Table Lamp", "home", 44.00),
    ("Stand Mixer", "kitchen", 189.00), ("Chef Knife", "kitchen", 64.99),
    ("Cast Iron Pan", "kitchen", 39.99), ("Espresso Maker", "kitchen", 149.00),
    ("Yoga Mat", "fitness", 24.99), ("Adjustable Dumbbells", "fitness", 159.00),
]

STATUSES = ["completed", "completed", "completed", "pending", "cancelled", "returned"]


def build():
    rng = random.Random(42)

    if DB_PATH.exists():
        DB_PATH.unlink()
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA)

    # customers: id 1..20
    customers = []
    for i in range(1, 21):
        name = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
        city = rng.choice(CITIES)
        year = rng.choice([2023, 2024, 2025])
        month = rng.randint(1, 12)
        day = rng.randint(1, 28)
        signup_date = f"{year:04d}-{month:02d}-{day:02d}"
        tier = rng.choices(TIERS, weights=[3, 2, 1])[0]
        customers.append((i, name, city, signup_date, tier))
    conn.executemany("INSERT INTO customers VALUES (?, ?, ?, ?, ?)", customers)

    # products: id 1..18
    products = [(i + 1, name, cat, price) for i, (name, cat, price) in enumerate(PRODUCT_NAMES)]
    conn.executemany("INSERT INTO products VALUES (?, ?, ?, ?)", products)

    # orders: id 1..70, skew towards recent, some customers order more than others
    orders = []
    for i in range(1, 71):
        customer_id = rng.choices(range(1, 21), weights=[3 if c % 3 == 0 else 1 for c in range(1, 21)])[0]
        year = rng.choice([2024, 2025])
        month = rng.randint(1, 12)
        day = rng.randint(1, 28)
        order_date = f"{year:04d}-{month:02d}-{day:02d}"
        status = rng.choice(STATUSES)
        orders.append((i, customer_id, order_date, status))
    conn.executemany("INSERT INTO orders VALUES (?, ?, ?, ?)", orders)

    # order_items: 1-4 line items per order
    order_items = []
    item_id = 1
    for order_id, *_ in orders:
        num_items = rng.randint(1, 4)
        chosen_products = rng.sample(range(1, 19), k=num_items)
        for product_id in chosen_products:
            qty = rng.randint(1, 5)
            order_items.append((item_id, order_id, product_id, qty))
            item_id += 1
    conn.executemany("INSERT INTO order_items VALUES (?, ?, ?, ?)", order_items)

    conn.commit()
    conn.close()
    print(f"built {DB_PATH} with {len(customers)} customers, {len(products)} products, "
          f"{len(orders)} orders, {len(order_items)} order_items")


if __name__ == "__main__":
    build()
