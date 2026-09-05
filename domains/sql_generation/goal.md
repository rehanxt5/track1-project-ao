# SQL Generation

You are given a question about a small local SQLite database ("shop.db")
for a fictional retailer, and must produce the SQL query that answers it.

## Schema

- `customers(id, name, city, signup_date, loyalty_tier)` -- `loyalty_tier`
  is one of `'bronze'`, `'silver'`, `'gold'`.
- `products(id, name, category, price)`
- `orders(id, customer_id, order_date, status)` -- `status` is one of
  `'completed'`, `'pending'`, `'cancelled'`, `'returned'`. `order_date` is
  `YYYY-MM-DD` text.
- `order_items(id, order_id, product_id, quantity)`

Use `list_tables` / `describe_table` if you want to confirm column names,
and `run_query` to test a candidate query before finalizing your answer --
it runs against the real database, so you can check your query actually
returns something sensible.

Read each question carefully: "revenue" and "orders" usually mean
`status = 'completed'` unless the question says otherwise (cancelled and
returned orders should not count as revenue), and dates are plain text
comparisons (`order_date >= '2025-01-01'` etc. work directly).

## Output format

Respond with exactly one JSON object and nothing else:

```json
{"sql": "SELECT ..."}
```

The query must be a single SELECT statement. It will be executed for real
against the database and your result set will be compared against the
gold result set -- so it is graded by what it actually returns, not by
matching any particular SQL text.
