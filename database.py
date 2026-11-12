"""Small SQLite persistence layer for scan history and demo alternatives."""
from pathlib import Path
import sqlite3
import pandas as pd

DB_PATH = Path(__file__).resolve().parent / "food_scans.db"


def get_connection():
    return sqlite3.connect(DB_PATH)


def initialize_database():
    with get_connection() as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scanned_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            product_name TEXT NOT NULL,
            price REAL NOT NULL,
            calories REAL NOT NULL, sugar REAL NOT NULL, protein REAL NOT NULL,
            fat REAL NOT NULL, sodium REAL NOT NULL, carbohydrates REAL NOT NULL,
            fiber REAL NOT NULL, prediction TEXT NOT NULL,
            confidence REAL, ingredients TEXT
        )""")
        conn.execute("""CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL,
            price REAL NOT NULL, calories REAL NOT NULL, sugar REAL NOT NULL,
            protein REAL NOT NULL, fiber REAL NOT NULL, category TEXT NOT NULL
        )""")


def seed_products():
    examples = [
        ("Plain oats", 120, 389, 1.0, 16.9, 10.6, "Healthy"),
        ("Unsalted roasted chana", 80, 360, 5.0, 20.0, 12.0, "Healthy"),
        ("Plain yogurt", 45, 60, 4.0, 3.5, 0.0, "Healthy"),
    ]
    with get_connection() as conn:
        count = conn.execute("SELECT COUNT(*) FROM products").fetchone()[0]
        if count == 0:
            conn.executemany(
                "INSERT INTO products(name,price,calories,sugar,protein,fiber,category) VALUES(?,?,?,?,?,?,?)",
                examples,
            )


def save_scan(data):
    columns = ["name", "price", "calories", "sugar", "protein", "fat", "sodium", "carbohydrates", "fiber", "prediction", "confidence", "ingredients"]
    values = [data.get(key) for key in columns]
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO scans(product_name,price,calories,sugar,protein,fat,sodium,carbohydrates,fiber,prediction,confidence,ingredients)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""", values
        )


def get_history():
    with get_connection() as conn:
        return pd.read_sql_query("SELECT * FROM scans ORDER BY scanned_at DESC, id DESC", conn)


def get_alternatives(max_price):
    with get_connection() as conn:
        return pd.read_sql_query(
            "SELECT name,price,calories,sugar,protein,fiber FROM products WHERE category='Healthy' AND price <= ? ORDER BY price",
            conn, params=(float(max_price),)
        )
