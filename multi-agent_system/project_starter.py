import pandas as pd
import numpy as np
import os
import time
import dotenv
import ast
from sqlalchemy.sql import text
from datetime import datetime, timedelta
from typing import Dict, List, Union
from sqlalchemy import create_engine, Engine
from smolagents import (
    ToolCallingAgent,
    OpenAIServerModel,
    tool,
)


dotenv.load_dotenv(dotenv_path=".env")
openai_api_key = os.getenv("UDACITY_OPENAI_API_KEY")

model = OpenAIServerModel(
    model_id="gpt-4o-mini",
    api_base="https://openai.vocareum.com/v1",
    api_key=openai_api_key,
)

# Create an SQLite database
db_engine = create_engine("sqlite:///munder_difflin.db")

# List containing the different kinds of papers 
paper_supplies = [
    # Paper Types (priced per sheet unless specified)
    {"item_name": "A4 paper",                         "category": "paper",        "unit_price": 0.05},
    {"item_name": "Letter-sized paper",              "category": "paper",        "unit_price": 0.06},
    {"item_name": "Cardstock",                        "category": "paper",        "unit_price": 0.15},
    {"item_name": "Colored paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Glossy paper",                     "category": "paper",        "unit_price": 0.20},
    {"item_name": "Matte paper",                      "category": "paper",        "unit_price": 0.18},
    {"item_name": "Recycled paper",                   "category": "paper",        "unit_price": 0.08},
    {"item_name": "Eco-friendly paper",               "category": "paper",        "unit_price": 0.12},
    {"item_name": "Poster paper",                     "category": "paper",        "unit_price": 0.25},
    {"item_name": "Banner paper",                     "category": "paper",        "unit_price": 0.30},
    {"item_name": "Kraft paper",                      "category": "paper",        "unit_price": 0.10},
    {"item_name": "Construction paper",               "category": "paper",        "unit_price": 0.07},
    {"item_name": "Wrapping paper",                   "category": "paper",        "unit_price": 0.15},
    {"item_name": "Glitter paper",                    "category": "paper",        "unit_price": 0.22},
    {"item_name": "Decorative paper",                 "category": "paper",        "unit_price": 0.18},
    {"item_name": "Letterhead paper",                 "category": "paper",        "unit_price": 0.12},
    {"item_name": "Legal-size paper",                 "category": "paper",        "unit_price": 0.08},
    {"item_name": "Crepe paper",                      "category": "paper",        "unit_price": 0.05},
    {"item_name": "Photo paper",                      "category": "paper",        "unit_price": 0.25},
    {"item_name": "Uncoated paper",                   "category": "paper",        "unit_price": 0.06},
    {"item_name": "Butcher paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Heavyweight paper",                "category": "paper",        "unit_price": 0.20},
    {"item_name": "Standard copy paper",              "category": "paper",        "unit_price": 0.04},
    {"item_name": "Bright-colored paper",             "category": "paper",        "unit_price": 0.12},
    {"item_name": "Patterned paper",                  "category": "paper",        "unit_price": 0.15},

    # Product Types (priced per unit)
    {"item_name": "Paper plates",                     "category": "product",      "unit_price": 0.10},  # per plate
    {"item_name": "Paper cups",                       "category": "product",      "unit_price": 0.08},  # per cup
    {"item_name": "Paper napkins",                    "category": "product",      "unit_price": 0.02},  # per napkin
    {"item_name": "Disposable cups",                  "category": "product",      "unit_price": 0.10},  # per cup
    {"item_name": "Table covers",                     "category": "product",      "unit_price": 1.50},  # per cover
    {"item_name": "Envelopes",                        "category": "product",      "unit_price": 0.05},  # per envelope
    {"item_name": "Sticky notes",                     "category": "product",      "unit_price": 0.03},  # per sheet
    {"item_name": "Notepads",                         "category": "product",      "unit_price": 2.00},  # per pad
    {"item_name": "Invitation cards",                 "category": "product",      "unit_price": 0.50},  # per card
    {"item_name": "Flyers",                           "category": "product",      "unit_price": 0.15},  # per flyer
    {"item_name": "Party streamers",                  "category": "product",      "unit_price": 0.05},  # per roll
    {"item_name": "Decorative adhesive tape (washi tape)", "category": "product", "unit_price": 0.20},  # per roll
    {"item_name": "Paper party bags",                 "category": "product",      "unit_price": 0.25},  # per bag
    {"item_name": "Name tags with lanyards",          "category": "product",      "unit_price": 0.75},  # per tag
    {"item_name": "Presentation folders",             "category": "product",      "unit_price": 0.50},  # per folder

    # Large-format items (priced per unit)
    {"item_name": "Large poster paper (24x36 inches)", "category": "large_format", "unit_price": 1.00},
    {"item_name": "Rolls of banner paper (36-inch width)", "category": "large_format", "unit_price": 2.50},

    # Specialty papers
    {"item_name": "100 lb cover stock",               "category": "specialty",    "unit_price": 0.50},
    {"item_name": "80 lb text paper",                 "category": "specialty",    "unit_price": 0.40},
    {"item_name": "250 gsm cardstock",                "category": "specialty",    "unit_price": 0.30},
    {"item_name": "220 gsm poster paper",             "category": "specialty",    "unit_price": 0.35},
]


# Given below are some utility functions you can use to implement your multi-agent system

def generate_sample_inventory(paper_supplies: list, coverage: float = 0.4, seed: int = 137) -> pd.DataFrame:
    """
    Generate inventory for exactly a specified percentage of items from the full paper supply list.

    This function randomly selects exactly `coverage` × N items from the `paper_supplies` list,
    and assigns each selected item:
    - a random stock quantity between 200 and 800,
    - a minimum stock level between 50 and 150.

    The random seed ensures reproducibility of selection and stock levels.

    Args:
        paper_supplies (list): A list of dictionaries, each representing a paper item with
                               keys 'item_name', 'category', and 'unit_price'.
        coverage (float, optional): Fraction of items to include in the inventory (default is 0.4, or 40%).
        seed (int, optional): Random seed for reproducibility (default is 137).

    Returns:
        pd.DataFrame: A DataFrame with the selected items and assigned inventory values, including:
                      - item_name
                      - category
                      - unit_price
                      - current_stock
                      - min_stock_level
    """
    # Ensure reproducible random output
    np.random.seed(seed)

    # Calculate number of items to include based on coverage
    num_items = int(len(paper_supplies) * coverage)

    # Randomly select item indices without replacement
    selected_indices = np.random.choice(
        range(len(paper_supplies)),
        size=num_items,
        replace=False
    )

    # Extract selected items from paper_supplies list
    selected_items = [paper_supplies[i] for i in selected_indices]

    # Construct inventory records
    inventory = []
    for item in selected_items:
        inventory.append({
            "item_name": item["item_name"],
            "category": item["category"],
            "unit_price": item["unit_price"],
            "current_stock": np.random.randint(200, 800),  # Realistic stock range
            "min_stock_level": np.random.randint(50, 150)  # Reasonable threshold for reordering
        })

    # Return inventory as a pandas DataFrame
    return pd.DataFrame(inventory)

def init_database(db_engine: Engine, seed: int = 137) -> Engine:    
    """
    Set up the Munder Difflin database with all required tables and initial records.

    This function performs the following tasks:
    - Creates the 'transactions' table for logging stock orders and sales
    - Loads customer inquiries from 'quote_requests.csv' into a 'quote_requests' table
    - Loads previous quotes from 'quotes.csv' into a 'quotes' table, extracting useful metadata
    - Generates a random subset of paper inventory using `generate_sample_inventory`
    - Inserts initial financial records including available cash and starting stock levels

    Args:
        db_engine (Engine): A SQLAlchemy engine connected to the SQLite database.
        seed (int, optional): A random seed used to control reproducibility of inventory stock levels.
                              Default is 137.

    Returns:
        Engine: The same SQLAlchemy engine, after initializing all necessary tables and records.

    Raises:
        Exception: If an error occurs during setup, the exception is printed and raised.
    """
    try:
        # ----------------------------
        # 1. Create an empty 'transactions' table schema
        # ----------------------------
        transactions_schema = pd.DataFrame({
            "id": [],
            "item_name": [],
            "transaction_type": [],  # 'stock_orders' or 'sales'
            "units": [],             # Quantity involved
            "price": [],             # Total price for the transaction
            "transaction_date": [],  # ISO-formatted date
        })
        transactions_schema.to_sql("transactions", db_engine, if_exists="replace", index=False)

        # Set a consistent starting date
        initial_date = datetime(2025, 1, 1).isoformat()

        # ----------------------------
        # 2. Load and initialize 'quote_requests' table
        # ----------------------------
        quote_requests_df = pd.read_csv("quote_requests.csv")
        quote_requests_df["id"] = range(1, len(quote_requests_df) + 1)
        quote_requests_df.to_sql("quote_requests", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 3. Load and transform 'quotes' table
        # ----------------------------
        quotes_df = pd.read_csv("quotes.csv")
        quotes_df["request_id"] = range(1, len(quotes_df) + 1)
        quotes_df["order_date"] = initial_date

        # Unpack metadata fields (job_type, order_size, event_type) if present
        if "request_metadata" in quotes_df.columns:
            quotes_df["request_metadata"] = quotes_df["request_metadata"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
            quotes_df["job_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("job_type", ""))
            quotes_df["order_size"] = quotes_df["request_metadata"].apply(lambda x: x.get("order_size", ""))
            quotes_df["event_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("event_type", ""))

        # Retain only relevant columns
        quotes_df = quotes_df[[
            "request_id",
            "total_amount",
            "quote_explanation",
            "order_date",
            "job_type",
            "order_size",
            "event_type"
        ]]
        quotes_df.to_sql("quotes", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 4. Generate inventory and seed stock
        # ----------------------------
        inventory_df = generate_sample_inventory(paper_supplies, seed=seed)

        # Seed initial transactions
        initial_transactions = []

        # Add a starting cash balance via a dummy sales transaction
        initial_transactions.append({
            "item_name": None,
            "transaction_type": "sales",
            "units": None,
            "price": 50000.0,
            "transaction_date": initial_date,
        })

        # Add one stock order transaction per inventory item
        for _, item in inventory_df.iterrows():
            initial_transactions.append({
                "item_name": item["item_name"],
                "transaction_type": "stock_orders",
                "units": item["current_stock"],
                "price": item["current_stock"] * item["unit_price"],
                "transaction_date": initial_date,
            })

        # Commit transactions to database
        pd.DataFrame(initial_transactions).to_sql("transactions", db_engine, if_exists="append", index=False)

        # Save the inventory reference table
        inventory_df.to_sql("inventory", db_engine, if_exists="replace", index=False)

        return db_engine

    except Exception as e:
        print(f"Error initializing database: {e}")
        raise

def create_transaction(
    item_name: str,
    transaction_type: str,
    quantity: int,
    price: float,
    date: Union[str, datetime],
) -> int:
    """
    This function records a transaction of type 'stock_orders' or 'sales' with a specified
    item name, quantity, total price, and transaction date into the 'transactions' table of the database.

    Args:
        item_name (str): The name of the item involved in the transaction.
        transaction_type (str): Either 'stock_orders' or 'sales'.
        quantity (int): Number of units involved in the transaction.
        price (float): Total price of the transaction.
        date (str or datetime): Date of the transaction in ISO 8601 format.

    Returns:
        int: The ID of the newly inserted transaction.

    Raises:
        ValueError: If `transaction_type` is not 'stock_orders' or 'sales'.
        Exception: For other database or execution errors.
    """
    try:
        # Convert datetime to ISO string if necessary
        date_str = date.isoformat() if isinstance(date, datetime) else date

        # Validate transaction type
        if transaction_type not in {"stock_orders", "sales"}:
            raise ValueError("Transaction type must be 'stock_orders' or 'sales'")

        # Prepare transaction record as a single-row DataFrame
        transaction = pd.DataFrame([{
            "item_name": item_name,
            "transaction_type": transaction_type,
            "units": quantity,
            "price": price,
            "transaction_date": date_str,
        }])

        # Insert the record into the database
        transaction.to_sql("transactions", db_engine, if_exists="append", index=False)

        # Fetch and return the ID of the inserted row
        result = pd.read_sql("SELECT last_insert_rowid() as id", db_engine)
        return int(result.iloc[0]["id"])

    except Exception as e:
        print(f"Error creating transaction: {e}")
        raise


def get_all_inventory(as_of_date: str) -> Dict[str, int]:
    """
    Retrieve a snapshot of available inventory as of a specific date.

    This function calculates the net quantity of each item by summing 
    all stock orders and subtracting all sales up to and including the given date.

    Only items with positive stock are included in the result.

    Args:
        as_of_date (str): ISO-formatted date string (YYYY-MM-DD) representing the inventory cutoff.

    Returns:
        Dict[str, int]: A dictionary mapping item names to their current stock levels.
    """
    # SQL query to compute stock levels per item as of the given date
    query = """
        SELECT
            item_name,
            SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END) as stock
        FROM transactions
        WHERE item_name IS NOT NULL
        AND transaction_date <= :as_of_date
        GROUP BY item_name
        HAVING stock > 0
    """

    # Execute the query with the date parameter
    result = pd.read_sql(query, db_engine, params={"as_of_date": as_of_date})

    # Convert the result into a dictionary {item_name: stock}
    return dict(zip(result["item_name"], result["stock"]))


def get_stock_level(item_name: str, as_of_date: Union[str, datetime]) -> pd.DataFrame:
    """
    Retrieve the stock level of a specific item as of a given date.

    This function calculates the net stock by summing all 'stock_orders' and 
    subtracting all 'sales' transactions for the specified item up to the given date.

    Args:
        item_name (str): The name of the item to look up.
        as_of_date (str or datetime): The cutoff date (inclusive) for calculating stock.

    Returns:
        pd.DataFrame: A single-row DataFrame with columns 'item_name' and 'current_stock'.
    """
    # Convert date to ISO string format if it's a datetime object
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # SQL query to compute net stock level for the item
    stock_query = """
        SELECT
            item_name,
            COALESCE(SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END), 0) AS current_stock
        FROM transactions
        WHERE item_name = :item_name
        AND transaction_date <= :as_of_date
    """

    # Execute query and return result as a DataFrame
    return pd.read_sql(
        stock_query,
        db_engine,
        params={"item_name": item_name, "as_of_date": as_of_date},
    )


def get_supplier_delivery_date(input_date_str: str, quantity: int) -> str:
    """
    Estimate the supplier delivery date based on the requested order quantity and a starting date.

    Delivery lead time increases with order size:
        - ≤10 units: same day
        - 11–100 units: 1 day
        - 101–1000 units: 4 days
        - >1000 units: 7 days

    Args:
        input_date_str (str): The starting date in ISO format (YYYY-MM-DD).
        quantity (int): The number of units in the order.

    Returns:
        str: Estimated delivery date in ISO format (YYYY-MM-DD).
    """
    # Debug log (comment out in production if needed)
    print(f"FUNC (get_supplier_delivery_date): Calculating for qty {quantity} from date string '{input_date_str}'")

    # Attempt to parse the input date
    try:
        input_date_dt = datetime.fromisoformat(input_date_str.split("T")[0])
    except (ValueError, TypeError):
        # Fallback to current date on format error
        print(f"WARN (get_supplier_delivery_date): Invalid date format '{input_date_str}', using today as base.")
        input_date_dt = datetime.now()

    # Determine delivery delay based on quantity
    if quantity <= 10:
        days = 0
    elif quantity <= 100:
        days = 1
    elif quantity <= 1000:
        days = 4
    else:
        days = 7

    # Add delivery days to the starting date
    delivery_date_dt = input_date_dt + timedelta(days=days)

    # Return formatted delivery date
    return delivery_date_dt.strftime("%Y-%m-%d")


def get_cash_balance(as_of_date: Union[str, datetime]) -> float:
    """
    Calculate the current cash balance as of a specified date.

    The balance is computed by subtracting total stock purchase costs ('stock_orders')
    from total revenue ('sales') recorded in the transactions table up to the given date.

    Args:
        as_of_date (str or datetime): The cutoff date (inclusive) in ISO format or as a datetime object.

    Returns:
        float: Net cash balance as of the given date. Returns 0.0 if no transactions exist or an error occurs.
    """
    try:
        # Convert date to ISO format if it's a datetime object
        if isinstance(as_of_date, datetime):
            as_of_date = as_of_date.isoformat()

        # Query all transactions on or before the specified date
        transactions = pd.read_sql(
            "SELECT * FROM transactions WHERE transaction_date <= :as_of_date",
            db_engine,
            params={"as_of_date": as_of_date},
        )

        # Compute the difference between sales and stock purchases
        if not transactions.empty:
            total_sales = transactions.loc[transactions["transaction_type"] == "sales", "price"].sum()
            total_purchases = transactions.loc[transactions["transaction_type"] == "stock_orders", "price"].sum()
            return float(total_sales - total_purchases)

        return 0.0

    except Exception as e:
        print(f"Error getting cash balance: {e}")
        return 0.0


def generate_financial_report(as_of_date: Union[str, datetime]) -> Dict:
    """
    Generate a complete financial report for the company as of a specific date.

    This includes:
    - Cash balance
    - Inventory valuation
    - Combined asset total
    - Itemized inventory breakdown
    - Top 5 best-selling products

    Args:
        as_of_date (str or datetime): The date (inclusive) for which to generate the report.

    Returns:
        Dict: A dictionary containing the financial report fields:
            - 'as_of_date': The date of the report
            - 'cash_balance': Total cash available
            - 'inventory_value': Total value of inventory
            - 'total_assets': Combined cash and inventory value
            - 'inventory_summary': List of items with stock and valuation details
            - 'top_selling_products': List of top 5 products by revenue
    """
    # Normalize date input
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # Get current cash balance
    cash = get_cash_balance(as_of_date)

    # Get current inventory snapshot
    inventory_df = pd.read_sql("SELECT * FROM inventory", db_engine)
    inventory_value = 0.0
    inventory_summary = []

    # Compute total inventory value and summary by item
    for _, item in inventory_df.iterrows():
        stock_info = get_stock_level(item["item_name"], as_of_date)
        stock = stock_info["current_stock"].iloc[0]
        item_value = stock * item["unit_price"]
        inventory_value += item_value

        inventory_summary.append({
            "item_name": item["item_name"],
            "stock": stock,
            "unit_price": item["unit_price"],
            "value": item_value,
        })

    # Identify top-selling products by revenue
    top_sales_query = """
        SELECT item_name, SUM(units) as total_units, SUM(price) as total_revenue
        FROM transactions
        WHERE transaction_type = 'sales' AND transaction_date <= :date
        GROUP BY item_name
        ORDER BY total_revenue DESC
        LIMIT 5
    """
    top_sales = pd.read_sql(top_sales_query, db_engine, params={"date": as_of_date})
    top_selling_products = top_sales.to_dict(orient="records")

    return {
        "as_of_date": as_of_date,
        "cash_balance": cash,
        "inventory_value": inventory_value,
        "total_assets": cash + inventory_value,
        "inventory_summary": inventory_summary,
        "top_selling_products": top_selling_products,
    }


def search_quote_history(search_terms: List[str], limit: int = 5) -> List[Dict]:
    """
    Retrieve a list of historical quotes that match any of the provided search terms.

    The function searches both the original customer request (from `quote_requests`) and
    the explanation for the quote (from `quotes`) for each keyword. Results are sorted by
    most recent order date and limited by the `limit` parameter.

    Args:
        search_terms (List[str]): List of terms to match against customer requests and explanations.
        limit (int, optional): Maximum number of quote records to return. Default is 5.

    Returns:
        List[Dict]: A list of matching quotes, each represented as a dictionary with fields:
            - original_request
            - total_amount
            - quote_explanation
            - job_type
            - order_size
            - event_type
            - order_date
    """
    conditions = []
    params = {}

    # Build SQL WHERE clause using LIKE filters for each search term
    for i, term in enumerate(search_terms):
        param_name = f"term_{i}"
        conditions.append(
            f"(LOWER(qr.response) LIKE :{param_name} OR "
            f"LOWER(q.quote_explanation) LIKE :{param_name})"
        )
        params[param_name] = f"%{term.lower()}%"

    # Combine conditions; fallback to always-true if no terms provided
    where_clause = " AND ".join(conditions) if conditions else "1=1"

    # Final SQL query to join quotes with quote_requests
    query = f"""
        SELECT
            qr.response AS original_request,
            q.total_amount,
            q.quote_explanation,
            q.job_type,
            q.order_size,
            q.event_type,
            q.order_date
        FROM quotes q
        JOIN quote_requests qr ON q.request_id = qr.id
        WHERE {where_clause}
        ORDER BY q.order_date DESC
        LIMIT {limit}
    """

    # Execute parameterized query
    with db_engine.connect() as conn:
        result = conn.execute(text(query), params)
        return [dict(row._mapping) for row in result]



# SPECIALIST AGENTS
 
class CustomerServiceAgent(ToolCallingAgent):
    """
    First point of contact for general inquiries.
 
    Responsibilities:
    - Present the full product catalogue with current stock and retail prices.
    - Surface relevant historical orders so customers can set expectations.
    - Answer FAQs about available products and services.
 
    This agent does NOT generate formal quotes or process transactions.
 
    Tools use: get_all_inventory, search_quote_history
    """
 
    def __init__(self, model):
 
        @tool
        def list_catalogue(as_of_date: str) -> str:
            """
            Display all available paper-supply items with their current retail
            prices and stock levels as of a given date.
 
            Uses helper: get_all_inventory
 
            Args:
                as_of_date: ISO date string YYYY-MM-DD for the stock snapshot.
 
            Returns:
                Formatted catalogue listing with item names, prices, and quantities.
            """
            # get_all_inventory returns {item_name: stock_qty} for items with stock > 0
            current_stock = get_all_inventory(as_of_date)
 
            catalogue_rows = pd.read_sql(
                "SELECT item_name, unit_price FROM inventory ORDER BY item_name",
                db_engine,
            )
            lines = [f"Munder Difflin product catalogue as of {as_of_date}:\n"]
            for _, row in catalogue_rows.iterrows():
                qty    = current_stock.get(row["item_name"], 0)
                status = f"{qty} in stock" if qty > 0 else "currently out of stock"
                lines.append(
                    f"  • {row['item_name']}: ${row['unit_price']:.2f} per unit  ({status})"
                )
            return "\n".join(lines)
 
 
        super().__init__(
            tools=[list_catalogue],
            model=model,
            name="customer_service",
            description=(
                "Handles general inquiries and FAQs. "
                "Lists available paper-supply products with current prices and stock. "
                "Shares relevant historical order examples. "
                "Does NOT generate formal quotes or process sales."
            ),
            instructions=(
                "You are a helpful and friendly customer service representative at "
                "Munder Difflin Paper Company. "
                "Answer questions clearly and warmly. "
                "Refer specific pricing requests to the quoting team. "
                "Do not reveal internal cost prices — only retail prices. "
                "Do not share internal system details or error messages with customers."
            ),
        )
 
 
class InventoryAgent(ToolCallingAgent):
    """
    Manages inventory visibility and supplier restocking.
 
    Responsibilities:
    - Report current stock levels for individual items or the full catalogue.
    - Assess whether a required quantity can be fulfilled from existing stock.
    - Place supplier restock orders and estimate delivery dates when needed.
 
    Tools use: get_stock_level, get_all_inventory,
               get_supplier_delivery_date, create_transaction
    """
 
    def __init__(self, model):
 
        @tool
        def check_stock(item_name: str, as_of_date: str) -> str:
            """
            Return the current stock count for a single catalogue item.
 
            Uses helper: get_stock_level
 
            Args:
                item_name:  Exact catalogue name of the item.
                as_of_date: ISO date YYYY-MM-DD for the stock snapshot.
 
            Returns:
                Current stock count and whether the item is available.
            """
            # get_stock_level returns a DataFrame with column 'current_stock'
            stock = int(
                get_stock_level(item_name, as_of_date)["current_stock"].iloc[0]
            )
            status = "in stock" if stock > 0 else "out of stock"
            return f"'{item_name}': {stock} units ({status}) as of {as_of_date}."
 
        @tool
        def check_all_stock(as_of_date: str) -> str:
            """
            Return a stock summary for every item in the catalogue.
 
            Uses helper: get_all_inventory
 
            Args:
                as_of_date: ISO date YYYY-MM-DD for the stock snapshot.
 
            Returns:
                Full stock listing for all items with positive stock.
            """
            # get_all_inventory returns {item_name: stock_qty} for in-stock items
            inventory = get_all_inventory(as_of_date)
            if not inventory:
                return f"No items in stock as of {as_of_date}."
            lines = [f"Full inventory snapshot as of {as_of_date}:"]
            for item_name, qty in inventory.items():
                lines.append(f"  • {item_name}: {qty} units")
            return "\n".join(lines)
 
        @tool
        def get_delivery_estimate(order_date: str, quantity: int) -> str:
            """
            Estimate the supplier delivery date for a given order quantity.
 
            Uses helper: get_supplier_delivery_date
 
            Lead times: <=10 units = same day; 11–100 = +1 day;
                        101–1 000 = +4 days; >1 000 = +7 days.
 
            Args:
                order_date: Date the order would be placed (YYYY-MM-DD).
                quantity:   Number of units being ordered.
 
            Returns:
                Estimated delivery date as plain text.
            """
            # get_supplier_delivery_date calculates lead time from quantity
            delivery_date = get_supplier_delivery_date(order_date, quantity)
            return (
                f"For an order of {quantity} units placed on {order_date}, "
                f"estimated delivery: {delivery_date}."
            )
 
        @tool
        def restock_item(item_name: str, quantity: int, order_date: str) -> str:
            """
            Place a supplier restock order for an item and record it as a
            stock_orders transaction in the database.
 
            Use this when current stock is insufficient to fulfil a customer order.
 
            Uses helpers: get_stock_level, get_supplier_delivery_date,
                          create_transaction
 
            Args:
                item_name:  Exact catalogue name of the item to restock.
                quantity:   Number of units to order from the supplier.
                order_date: ISO date the restock order is placed (YYYY-MM-DD).
 
            Returns:
                Confirmation of the restock order with estimated delivery date,
                or an error message if the item is not in the catalogue.
            """
            # Look up retail price to use as the purchase price for this restock
            inv_row = pd.read_sql(
                "SELECT unit_price FROM inventory WHERE item_name = :n",
                db_engine, params={"n": item_name},
            )
            if inv_row.empty:
                # List available items so the caller can correct the name
                catalogue = pd.read_sql(
                    "SELECT item_name FROM inventory ORDER BY item_name", db_engine
                )
                available = ", ".join(catalogue["item_name"].tolist())
                return (
                    f"Item '{item_name}' not found in catalogue. "
                    f"Available items: {available}"
                )
 
            unit_price = float(inv_row["unit_price"].iloc[0])
            total_cost = unit_price * quantity
 
            # create_transaction records the stock_orders entry and returns the row ID
            create_transaction(item_name, "stock_orders", quantity, total_cost, order_date)
 
            # get_supplier_delivery_date estimates when the order will arrive
            delivery_date = get_supplier_delivery_date(order_date, quantity)
 
            return (
                f"Restock order placed for {quantity} units of '{item_name}'. "
                f"Estimated delivery: {delivery_date}."
            )
 
        super().__init__(
            tools=[check_stock, check_all_stock, get_delivery_estimate, restock_item],
            model=model,
            name="inventory",
            description=(
                "Manages inventory: checks current stock levels for individual items "
                "or the full catalogue, estimates supplier delivery dates, and places "
                "restock orders when supply is insufficient."
            ),
            instructions=(
                "You are an inventory specialist at Munder Difflin Paper Company. "
                "When asked about an item's availability: "
                "(1) call check_stock to get the current quantity, "
                "(2) if the requested quantity exceeds stock, call restock_item "
                "    to place a supplier order and report the delivery estimate. "
                "Use exact catalogue item names. "
                "Do not share internal cost prices or transaction IDs with customers."
            ),
        )
 
 
class QuotingAgent(ToolCallingAgent):
    """
    Generates competitive, itemised price quotes for customer orders.
 
    Responsibilities:
    - Search historical quotes to calibrate pricing for similar jobs.
    - Look up current retail prices and stock availability.
    - Calculate itemised totals with bulk discounts applied.
    - Save finalised quotes to the database for future reference.
 
    Bulk discount schedule (applied to total unit count across all line items):
      >= 5 000 units → 20% off
      >= 1 000 units → 15% off
      >=   500 units → 10% off
      >=   100 units →  5% off
      <    100 units → no discount
 
    Tools use: search_quote_history, get_stock_level, create_transaction (via save_quote)
    """
 
    def __init__(self, model):
 
        @tool
        def search_past_quotes(keywords: str, limit: int = 5) -> str:
            """
            Search historical quotes for similar jobs to inform current pricing.
 
            Uses helper: search_quote_history
 
            Args:
                keywords: Comma-separated search terms (event type, job, items).
                          Example: "ceremony,cardstock,office"
                limit:    Maximum number of results to return (default 5).
 
            Returns:
                Formatted list of matching historical quotes with amounts and notes.
            """
            terms = [t.strip() for t in keywords.split(",") if t.strip()]
            # search_quote_history performs a full-text search across quote records
            results = search_quote_history(terms, limit=limit)
            if not results:
                return "No matching historical quotes found for those keywords."
            lines = [f"Found {len(results)} historical quote(s):"]
            for i, q in enumerate(results, 1):
                lines.append(
                    f"\n  [{i}] Event: {q.get('event_type', 'N/A')} | "
                    f"Job: {q.get('job_type', 'N/A')} | "
                    f"Size: {q.get('order_size', 'N/A')}\n"
                    f"       Total: ${q.get('total_amount', 0):.2f} | "
                    f"Date: {q.get('order_date', 'N/A')}\n"
                    f"       Notes: {str(q.get('quote_explanation', ''))[:120]}..."
                )
            return "\n".join(lines)
 
        @tool
        def get_item_prices(item_names: str, as_of_date: str) -> str:
            """
            Look up retail unit prices and current stock for requested items.
 
            Uses helper: get_stock_level
 
            Args:
                item_names: Comma-separated exact catalogue item names.
                            Example: "A4 glossy paper,heavy cardstock (white)"
                as_of_date: ISO date YYYY-MM-DD for the stock snapshot.
 
            Returns:
                Price and availability table for each requested item.
            """
            names = [n.strip() for n in item_names.split(",") if n.strip()]
            price_rows = pd.read_sql(
                "SELECT item_name, unit_price FROM inventory", db_engine
            )
            lines = [f"Prices and availability as of {as_of_date}:"]
            for name in names:
                match = price_rows[price_rows["item_name"] == name]
                if not match.empty:
                    unit_price = match["unit_price"].iloc[0]
                    # get_stock_level returns current net stock for this item
                    stock = int(
                        get_stock_level(name, as_of_date)["current_stock"].iloc[0]
                    )
                    availability = f"{stock} units in stock" if stock > 0 else "OUT OF STOCK"
                    lines.append(
                        f"  • {name}: ${unit_price:.2f}/unit  |  {availability}"
                    )
                else:
                    lines.append(f"  • {name}: not found in catalogue")
            return "\n".join(lines)
 
        @tool
        def save_quote(
            request_text: str,
            request_date: str,
            total_amount: float,
            explanation:  str,
            job_type:     str,
            order_size:   str,
            event_type:   str,
        ) -> str:
            """
            Persist a finalised quote to the database for audit and future reference.
 
            Uses helpers: create_transaction (indirectly via quote_requests insert)
 
            Args:
                request_text: Original customer request verbatim.
                request_date: ISO date of the request (YYYY-MM-DD).
                total_amount: Final quoted amount in dollars, after discounts.
                explanation:  Full itemised breakdown including discount rationale.
                job_type:     Requester category (e.g. 'office manager').
                order_size:   Scale label — 'small', 'medium', or 'large'.
                event_type:   Type of event (e.g. 'ceremony', 'conference').
 
            Returns:
                Customer-facing confirmation with the quoted total.
            """
            # Insert the raw request into quote_requests and get its ID
            with db_engine.connect() as conn:
                result = conn.execute(
                    text(
                        "INSERT INTO quote_requests (response, request_date) "
                        "VALUES (:r, :d)"
                    ),
                    {"r": request_text, "d": request_date},
                )
                conn.commit()
                request_id = result.lastrowid
 
            # Save the computed quote linked to the request
            pd.DataFrame([{
                "request_id":        request_id,
                "total_amount":      total_amount,
                "quote_explanation": explanation,
                "job_type":          job_type,
                "order_size":        order_size,
                "event_type":        event_type,
                "order_date":        request_date,
            }]).to_sql("quotes", db_engine, if_exists="append", index=False)
 
            # Return a customer-friendly confirmation without internal IDs
            return (
                f"Your quote has been prepared. "
                f"Total: ${total_amount:.2f} for your {event_type}. "
                f"This quote is valid for 30 days."
            )
 
        super().__init__(
            tools=[search_past_quotes, get_item_prices, save_quote],
            model=model,
            name="quoting",
            description=(
                "Generates competitive, itemised price quotes for paper-supply orders. "
                "Searches historical quotes for pricing context, looks up current "
                "retail prices, calculates totals with automatic bulk discounts, "
                "and saves each finalised quote to the database."
            ),
            instructions=(
                "You are a professional quoting specialist at Munder Difflin Paper Company. "
                "To produce a quote, follow these steps in order:\n"
                "  (1) Call search_past_quotes with the event type and job type as keywords "
                "      to find relevant historical pricing benchmarks.\n"
                "  (2) Call get_item_prices to look up current retail prices and stock.\n"
                "  (3) Calculate the itemised total yourself:\n"
                "      - Multiply each item's quantity by its unit price.\n"
                "      - Sum all line items to get the subtotal.\n"
                "      - Count the total units across all line items.\n"
                "      - Apply the bulk discount: 5% for 100+ units, 10% for 500+, "
                "        15% for 1000+, 20% for 5000+.\n"
                "      - Subtract the discount from the subtotal to get the final total.\n"
                "  (4) Call save_quote with the full itemised breakdown.\n\n"
                "Customer-facing response must include:\n"
                "  - Itemised list (item, quantity, unit price, line total)\n"
                "  - Subtotal before discount\n"
                "  - Discount rate and amount saved (if any)\n"
                "  - Final total\n"
                "  - Explanation of why a discount applies (e.g. bulk order size)\n\n"
                "Do NOT reveal internal cost prices or profit margins. "
                "If an item is out of stock, note it clearly and suggest the inventory "
                "team can arrange restocking."
            ),
        )
 
 
class OrderingAgent(ToolCallingAgent):
    """
    Finalises confirmed customer sales and provides financial reporting.
 
    Responsibilities:
    - Record confirmed sale transactions in the database.
    - Reject sales when stock is insufficient and provide a clear reason.
    - Report the current cash balance after transactions.
    - Generate a full financial summary on request.
 
    Tools use: get_stock_level, create_transaction,
               get_cash_balance, generate_financial_report
    """
 
    def __init__(self, model):
 
        @tool
        def record_sale(
            item_name:  str,
            quantity:   int,
            unit_price: float,
            sale_date:  str,
        ) -> str:
            """
            Record a confirmed sale transaction and deduct units from inventory.
 
            Uses helpers: get_stock_level, create_transaction
 
            Args:
                item_name:  Exact catalogue name of the item being sold.
                quantity:   Number of units sold.
                unit_price: Agreed retail price per unit in dollars.
                sale_date:  ISO date of the sale (YYYY-MM-DD).
 
            Returns:
                Customer-facing order confirmation, or a clear reason if the
                sale cannot be processed due to insufficient stock.
            """
            # get_stock_level checks the current net stock for this item
            current_stock = int(
                get_stock_level(item_name, sale_date)["current_stock"].iloc[0]
            )
            if current_stock < quantity:
                # Customer-facing message: explains why the order cannot proceed
                # without revealing internal stock system details
                return (
                    f"We are unable to fulfil the order for {quantity} units of "
                    f"'{item_name}' at this time — current availability does not "
                    f"cover this quantity. Please contact us to arrange restocking "
                    f"or consider a smaller order."
                )
 
            order_total = quantity * unit_price
            # create_transaction records the sale and returns the internal row ID
            create_transaction(item_name, "sales", quantity, order_total, sale_date)
 
            # Return a clean customer-facing confirmation (no internal IDs)
            return (
                f"Order confirmed: {quantity} unit(s) of '{item_name}' "
                f"at ${unit_price:.2f} each — total ${order_total:.2f}. "
                f"Order date: {sale_date}. Thank you for your business!"
            )
 
        @tool
        def get_cash_balance_now(as_of_date: str) -> str:
            """
            Return the current net cash balance (sales revenue minus stock costs).
 
            Uses helper: get_cash_balance
 
            Args:
                as_of_date: ISO date YYYY-MM-DD.
 
            Returns:
                Cash balance as plain text (internal use only — not for customers).
            """
            # get_cash_balance computes total_sales - total_stock_purchases
            balance = get_cash_balance(as_of_date)
            return f"Net cash balance as of {as_of_date}: ${balance:.2f}"
 
        @tool
        def get_financial_summary(as_of_date: str) -> str:
            """
            Generate a complete financial snapshot: cash balance, inventory
            value, total assets, and top-selling products.
 
            Uses helper: generate_financial_report
 
            Args:
                as_of_date: ISO date YYYY-MM-DD for the report.
 
            Returns:
                Formatted financial summary (internal use — not for customers).
            """
            # generate_financial_report builds a comprehensive financial dict
            report = generate_financial_report(as_of_date)
            lines  = [
                f"Financial summary as of {as_of_date}:",
                f"  Cash balance    : ${report['cash_balance']:.2f}",
                f"  Inventory value : ${report['inventory_value']:.2f}",
                f"  Total assets    : ${report['total_assets']:.2f}",
            ]
            if report["top_selling_products"]:
                lines.append("  Top sellers:")
                for p in report["top_selling_products"]:
                    lines.append(
                        f"    • {p['item_name']}: "
                        f"{p['total_units']} units sold, "
                        f"${p['total_revenue']:.2f} revenue"
                    )
            return "\n".join(lines)
 
        super().__init__(
            tools=[record_sale, get_cash_balance_now, get_financial_summary],
            model=model,
            name="ordering",
            description=(
                "Finalises confirmed customer sales by recording each transaction. "
                "Rejects orders with insufficient stock and explains why. "
                "Provides cash balance and full financial summary reports."
            ),
            instructions=(
                "You are an order fulfilment specialist at Munder Difflin Paper Company. "
                "For each line item in a confirmed order, call record_sale. "
                "After all items are processed, call get_cash_balance_now to check the "
                "updated balance. "
                "If an order is rejected due to low stock, explain this clearly to the "
                "customer without revealing internal system details. "
                "Do not share internal transaction IDs, raw profit margins, or system "
                "error messages with customers."
            ),
        )
 
 

# ORCHESTRATOR
 
class Orchestrator(ToolCallingAgent):
    """
    Top-level orchestrator that routes each incoming request to the appropriate
    specialist agent via @tool routing closures.
 
    The LLM sees four routing tools and decides which specialist to delegate to
    and what task description to pass. This keeps the orchestrator's routing
    logic transparent and the specialist agents independently testable.
 
    Standard workflow for a new customer request:
      1. handle_inventory_check  — verify stock, restock if needed
      2. handle_quote_request    — produce an itemised quote with bulk discounts
      3. handle_order            — only if the customer explicitly confirms purchase
    """
 
    def __init__(self, model):
        self.model = model
 
        # Instantiate all four specialist agents
        self.customer_service = CustomerServiceAgent(model)
        self.inventory        = InventoryAgent(model)
        self.quoting          = QuotingAgent(model)
        self.ordering         = OrderingAgent(model)
 
        # ── Routing tool closures ─────────────────────────────────────────────
 
        @tool
        def handle_inquiry(task: str) -> str:
            """
            Route a general question, FAQ, or catalogue request to the
            Customer Service agent.
 
            Use for: "What paper do you stock?", "Do you carry recycled cardstock?",
                     "What's currently available?"
 
            Args:
                task: Full customer question including any relevant context.
 
            Returns:
                Helpful, customer-friendly response.
            """
            return self.customer_service.run(task)
 
        @tool
        def handle_inventory_check(task: str) -> str:
            """
            Route a stock availability check or restock request to the
            Inventory agent.
 
            Use for: "Do you have 500 sheets of A4 glossy paper?",
                     "We need 300 units of heavy cardstock by April 15.",
                     "Restock poster boards — we need 200 more."
 
            Args:
                task: Item names, required quantities, and relevant date.
 
            Returns:
                Stock status and, if a reorder was placed, the delivery estimate.
            """
            return self.inventory.run(task)
 
        @tool
        def handle_quote_request(task: str) -> str:
            """
            Route a pricing or quote request to the Quoting agent. The agent
            searches historical quotes, looks up prices, applies bulk discounts,
            and saves the finalised quote.
 
            Use for: "Quote for 200 sheets A4 glossy and 100 cardstock.",
                     "How much for 10,000 sheets of A4 paper for a conference?"
 
            Args:
                task: Full request with item names, quantities, event type,
                      job type, order size, and date.
 
            Returns:
                Itemised quote with bulk discount and confirmation of quote saved.
            """
            return self.quoting.run(task)
 
        @tool
        def handle_order(task: str) -> str:
            """
            Route a confirmed purchase to the Ordering agent to record the sale.
 
            Only use this when the customer has explicitly confirmed they want
            to proceed with a purchase. Do NOT use for quotes or enquiries.
 
            Use for: "Process the order: 200 A4 glossy @ $0.12 each, 2025-04-15.",
                     "Customer confirmed — finalise 100 cardstock at $0.18/unit."
 
            Args:
                task: Confirmed item names, quantities, agreed unit prices, and date.
 
            Returns:
                Order confirmation receipt or clear explanation if unfulfillable.
            """
            return self.ordering.run(task)
 
        super().__init__(
            tools=[
                handle_inquiry,
                handle_inventory_check,
                handle_quote_request,
                handle_order,
            ],
            model=model,
            name="orchestrator",
            description=(
                "Orchestrates Munder Difflin's paper-supply system by routing each "
                "customer request to the appropriate specialist agent: "
                "customer service, inventory, quoting, or ordering."
            ),
            instructions="""You are the orchestrator for Munder Difflin Paper Company.
Your job is to understand each incoming request and delegate it to the correct specialist.
 
Routing rules:
  handle_inquiry          -> general questions, product catalogue, FAQs
  handle_inventory_check  -> stock checks, delivery ETA queries, restock requests
  handle_quote_request    -> price quotes, cost estimates, bulk discount enquiries
  handle_order            -> confirmed purchases only (customer has agreed to buy)
 
Standard workflow for a new supply request:
  Step 1: handle_inventory_check  — verify stock for each requested item;
          the inventory agent will restock automatically if needed.
  Step 2: handle_quote_request    — produce a full itemised quote with bulk discounts.
  Step 3: handle_order            — ONLY if the customer explicitly confirms they
          want to proceed with the purchase.
 
When passing a task to a specialist, include the full context:
  - Exact item names and quantities requested
  - Event type and job type (e.g. 'conference', 'office manager')
  - Order size category ('small', 'medium', 'large')
  - The request date (YYYY-MM-DD)
 
Do not expose internal system details, cost prices, or transaction IDs to customers.
Summarise the specialist's response clearly and helpfully in your final reply.
""",
        )
 


# Run your test scenarios by writing them here. Make sure to keep track of them.

def run_test_scenarios():
    
    print("Initializing Database...")
    init_database(db_engine)
    try:
        quote_requests_sample = pd.read_csv("quote_requests_sample.csv")
        quote_requests_sample["request_date"] = pd.to_datetime(
            quote_requests_sample["request_date"], format="%m/%d/%y", errors="coerce"
        )
        quote_requests_sample.dropna(subset=["request_date"], inplace=True)
        quote_requests_sample = quote_requests_sample.sort_values("request_date")
    except Exception as e:
        print(f"FATAL: Error loading test data: {e}")
        return

    # Get initial state
    initial_date = quote_requests_sample["request_date"].min().strftime("%Y-%m-%d")
    report = generate_financial_report(initial_date)
    current_cash = report["cash_balance"]
    current_inventory = report["inventory_value"]

    # INITIALIZE MULTI AGENT SYSTEM

    orchestrator = Orchestrator(model)


    results = []
    for idx, row in quote_requests_sample.iterrows():
        request_date = row["request_date"].strftime("%Y-%m-%d")

        print(f"\n=== Request {idx+1} ===")
        print(f"Context: {row['job']} organizing {row['event']}")
        print(f"Request Date: {request_date}")
        print(f"Cash Balance: ${current_cash:.2f}")
        print(f"Inventory Value: ${current_inventory:.2f}")

        # Process request
        request_with_date = f"{row['request']} (Date of request: {request_date})"

        try:
            response = orchestrator.run(request_with_date)
        except Exception as e:
            response = f"[ERROR] {e}"

        # Update state
        report = generate_financial_report(request_date)
        current_cash = report["cash_balance"]
        current_inventory = report["inventory_value"]

        print(f"Response: {response}")
        print(f"Updated Cash: ${current_cash:.2f}")
        print(f"Updated Inventory: ${current_inventory:.2f}")

        results.append(
            {
                "request_id": idx + 1,
                "request_date": request_date,
                "cash_balance": current_cash,
                "inventory_value": current_inventory,
                "response": response,
            }
        )

        time.sleep(1)

    # Final report
    final_date = quote_requests_sample["request_date"].max().strftime("%Y-%m-%d")
    final_report = generate_financial_report(final_date)
    print("\n===== FINAL FINANCIAL REPORT =====")
    print(f"Final Cash: ${final_report['cash_balance']:.2f}")
    print(f"Final Inventory: ${final_report['inventory_value']:.2f}")

    # Save results
    pd.DataFrame(results).to_csv("test_results.csv", index=False)
    return results


if __name__ == "__main__":
    results = run_test_scenarios()
