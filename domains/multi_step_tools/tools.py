"""
Deterministic, offline tool implementations for the multi_step_tools
domain (a small fictional retailer, "Orbit Retail").

Convention (relied on by the agent runner, not by metaagent/evaluation.py):
TOOLS maps each tool name declared in tools.json to a callable that accepts
the tool's declared parameters as keyword arguments and returns a
JSON-serializable value. Every function returns {"error": "..."} for an
unknown id instead of raising, so a runner can feed the result straight
back to the agent as a tool message.

None of these functions have side effects or hidden state -- they are
pure lookups over the fixture data below, so evaluation is fully
deterministic and reproducible offline.
"""

from __future__ import annotations

EMPLOYEES = {
    "EMP-01": {"employee_id": "EMP-01", "name": "Dana Whitfield", "role": "support_rep", "manager_id": "EMP-10"},
    "EMP-02": {"employee_id": "EMP-02", "name": "Carlos Mendez", "role": "support_rep", "manager_id": "EMP-10"},
    "EMP-03": {"employee_id": "EMP-03", "name": "Priya Ramesh", "role": "support_rep", "manager_id": "EMP-11"},
    "EMP-04": {"employee_id": "EMP-04", "name": "Tom Halligan", "role": "support_rep", "manager_id": "EMP-11"},
    "EMP-10": {"employee_id": "EMP-10", "name": "Sana Iqbal", "role": "support_manager", "manager_id": "EMP-20"},
    "EMP-11": {"employee_id": "EMP-11", "name": "Ben Okafor", "role": "support_manager", "manager_id": "EMP-20"},
    "EMP-20": {"employee_id": "EMP-20", "name": "Rita Alvarado", "role": "director", "manager_id": None},
}

CUSTOMERS = {
    "CUST-01": {"customer_id": "CUST-01", "name": "Harper Lin", "region": "west", "loyalty_tier": "gold"},
    "CUST-02": {"customer_id": "CUST-02", "name": "Owen Baptiste", "region": "east", "loyalty_tier": "silver"},
    "CUST-03": {"customer_id": "CUST-03", "name": "Nina Souza", "region": "midwest", "loyalty_tier": "bronze"},
    "CUST-04": {"customer_id": "CUST-04", "name": "Faisal Rahman", "region": "south", "loyalty_tier": "gold"},
    "CUST-05": {"customer_id": "CUST-05", "name": "Greta Voss", "region": "west", "loyalty_tier": "bronze"},
    "CUST-06": {"customer_id": "CUST-06", "name": "Julien Marchand", "region": "east", "loyalty_tier": "silver"},
    "CUST-07": {"customer_id": "CUST-07", "name": "Aiko Suzuki", "region": "midwest", "loyalty_tier": "gold"},
    "CUST-08": {"customer_id": "CUST-08", "name": "Leo Ferreira", "region": "south", "loyalty_tier": "silver"},
}

PRODUCTS = {
    "SKU-100": {"sku": "SKU-100", "name": "Trail Backpack", "unit_price": 89.99, "category": "outdoor"},
    "SKU-101": {"sku": "SKU-101", "name": "Insulated Bottle", "unit_price": 24.50, "category": "outdoor"},
    "SKU-102": {"sku": "SKU-102", "name": "Camp Stove", "unit_price": 64.00, "category": "outdoor"},
    "SKU-200": {"sku": "SKU-200", "name": "Wireless Earbuds", "unit_price": 129.99, "category": "electronics"},
    "SKU-201": {"sku": "SKU-201", "name": "Portable Charger", "unit_price": 39.99, "category": "electronics"},
    "SKU-202": {"sku": "SKU-202", "name": "Smart Watch", "unit_price": 199.00, "category": "electronics"},
    "SKU-300": {"sku": "SKU-300", "name": "Ceramic Mug Set", "unit_price": 18.00, "category": "home"},
    "SKU-301": {"sku": "SKU-301", "name": "Throw Blanket", "unit_price": 34.99, "category": "home"},
    "SKU-302": {"sku": "SKU-302", "name": "Desk Organizer", "unit_price": 22.49, "category": "home"},
}

ORDERS = {
    "ORD-2001": {"order_id": "ORD-2001", "customer_id": "CUST-01", "handled_by": "EMP-01", "status": "delivered",
                 "items": [{"sku": "SKU-100", "qty": 1}, {"sku": "SKU-101", "qty": 2}]},
    "ORD-2002": {"order_id": "ORD-2002", "customer_id": "CUST-02", "handled_by": "EMP-02", "status": "delivered",
                 "items": [{"sku": "SKU-200", "qty": 1}]},
    "ORD-2003": {"order_id": "ORD-2003", "customer_id": "CUST-03", "handled_by": "EMP-03", "status": "shipped",
                 "items": [{"sku": "SKU-300", "qty": 4}, {"sku": "SKU-301", "qty": 1}]},
    "ORD-2004": {"order_id": "ORD-2004", "customer_id": "CUST-04", "handled_by": "EMP-04", "status": "delivered",
                 "items": [{"sku": "SKU-202", "qty": 1}, {"sku": "SKU-201", "qty": 1}]},
    "ORD-2005": {"order_id": "ORD-2005", "customer_id": "CUST-05", "handled_by": "EMP-01", "status": "delivered",
                 "items": [{"sku": "SKU-102", "qty": 1}]},
    "ORD-2006": {"order_id": "ORD-2006", "customer_id": "CUST-06", "handled_by": "EMP-02", "status": "delivered",
                 "items": [{"sku": "SKU-302", "qty": 3}]},
    "ORD-2007": {"order_id": "ORD-2007", "customer_id": "CUST-07", "handled_by": "EMP-03", "status": "shipped",
                 "items": [{"sku": "SKU-200", "qty": 2}, {"sku": "SKU-300", "qty": 1}]},
    "ORD-2008": {"order_id": "ORD-2008", "customer_id": "CUST-08", "handled_by": "EMP-04", "status": "delivered",
                 "items": [{"sku": "SKU-101", "qty": 3}, {"sku": "SKU-102", "qty": 1}]},
    "ORD-2009": {"order_id": "ORD-2009", "customer_id": "CUST-01", "handled_by": "EMP-02", "status": "delivered",
                 "items": [{"sku": "SKU-201", "qty": 2}]},
    "ORD-2010": {"order_id": "ORD-2010", "customer_id": "CUST-03", "handled_by": "EMP-01", "status": "delivered",
                 "items": [{"sku": "SKU-202", "qty": 1}, {"sku": "SKU-301", "qty": 2}]},
    "ORD-2011": {"order_id": "ORD-2011", "customer_id": "CUST-05", "handled_by": "EMP-03", "status": "shipped",
                 "items": [{"sku": "SKU-100", "qty": 2}]},
    "ORD-2012": {"order_id": "ORD-2012", "customer_id": "CUST-06", "handled_by": "EMP-04", "status": "delivered",
                 "items": [{"sku": "SKU-300", "qty": 2}, {"sku": "SKU-101", "qty": 1}]},
    "ORD-2013": {"order_id": "ORD-2013", "customer_id": "CUST-08", "handled_by": "EMP-01", "status": "delivered",
                 "items": [{"sku": "SKU-200", "qty": 1}, {"sku": "SKU-201", "qty": 1}, {"sku": "SKU-300", "qty": 1}]},
    "ORD-2014": {"order_id": "ORD-2014", "customer_id": "CUST-07", "handled_by": "EMP-02", "status": "delivered",
                 "items": [{"sku": "SKU-302", "qty": 1}]},
    "ORD-2015": {"order_id": "ORD-2015", "customer_id": "CUST-04", "handled_by": "EMP-03", "status": "shipped",
                 "items": [{"sku": "SKU-102", "qty": 2}, {"sku": "SKU-100", "qty": 1}]},
}

SHIPPING_RATES = {"west": 12.50, "east": 9.00, "midwest": 7.50, "south": 10.25}
LOYALTY_DISCOUNTS = {"bronze": 0.0, "silver": 0.05, "gold": 0.12}


def get_order(order_id: str) -> dict:
    order = ORDERS.get(order_id)
    if order is None:
        return {"error": f"no such order_id: {order_id}"}
    return dict(order)


def get_customer(customer_id: str) -> dict:
    customer = CUSTOMERS.get(customer_id)
    if customer is None:
        return {"error": f"no such customer_id: {customer_id}"}
    return dict(customer)


def get_product(sku: str) -> dict:
    product = PRODUCTS.get(sku)
    if product is None:
        return {"error": f"no such sku: {sku}"}
    return dict(product)


def get_employee(employee_id: str) -> dict:
    employee = EMPLOYEES.get(employee_id)
    if employee is None:
        return {"error": f"no such employee_id: {employee_id}"}
    return dict(employee)


def get_shipping_rate(region: str) -> dict:
    if region not in SHIPPING_RATES:
        return {"error": f"no such region: {region}"}
    return {"region": region, "flat_rate": SHIPPING_RATES[region]}


def get_loyalty_discount(tier: str) -> dict:
    if tier not in LOYALTY_DISCOUNTS:
        return {"error": f"no such loyalty tier: {tier}"}
    return {"tier": tier, "discount_pct": LOYALTY_DISCOUNTS[tier]}


TOOLS = {
    "get_order": get_order,
    "get_customer": get_customer,
    "get_product": get_product,
    "get_employee": get_employee,
    "get_shipping_rate": get_shipping_rate,
    "get_loyalty_discount": get_loyalty_discount,
}
