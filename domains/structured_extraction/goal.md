# Structured Extraction

You will be given a single, messy piece of free-text customer-support
correspondence (an email or chat transcript, exactly as a customer typed
it -- typos, run-on sentences, and irrelevant asides included). Extract
the fields below into one strict JSON object.

## Target schema

- `customer_name` (string): the customer's full name as it appears in the text.
- `order_id` (string or null): the order identifier the customer's CURRENT
  request is about, formatted like `"ORD-12345"`. Some messages mention a
  second, older order in passing (e.g. an order that already arrived fine,
  or was already resolved) -- do not confuse it with the order the
  customer is actually writing about now. `null` if no order id appears
  anywhere in the text.
- `product` (string or null): the specific product name the current issue
  concerns. `null` if no product is named.
- `issue_type` (enum): exactly one of `"defective"`, `"wrong_item"`,
  `"missing_item"`, `"late_delivery"`, `"billing"`, `"other"`.
- `priority` (enum): exactly one of `"low"`, `"medium"`, `"high"`,
  `"urgent"`. This is almost never stated directly -- infer it from tone
  and urgency language (repeated contact attempts, deadlines, threats to
  cancel/chargeback, all-caps, "ASAP"/"immediately" vs. "whenever you get
  a chance"). Messages with no urgency language at all are `"medium"`.
- `requested_action` (enum): exactly one of `"refund"`, `"replacement"`,
  `"repair"`, `"information"`, `"cancellation"`.
- `contact_method` (string or null): exactly one of `"email"`, `"phone"`,
  `"chat"`, or `null` if the text never says how the customer wants to be
  contacted back.

## Tools available

- `get_schema`: returns this schema in machine-readable form (field names,
  types, allowed enum values). Use it to double check exact enum spelling.
- `validate_extraction`: takes a candidate JSON string and reports which
  fields are missing, malformed, or use an invalid enum value. It does
  **not** tell you the correct value -- it only helps you catch structural
  mistakes (typo'd enum, missing required field) before you finalize.

## Output format

Respond with exactly one JSON object matching the schema above and nothing
else -- no markdown code fences, no explanation before or after it.
