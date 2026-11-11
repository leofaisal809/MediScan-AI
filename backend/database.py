
import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "medical.db")


def create_database():

    connection = sqlite3.connect(DB_PATH)
    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS xray_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            prediction TEXT,
            confidence REAL
        )
    """)

    connection.commit()
    connection.close()

    print("Database created successfully!")

def save_xray_record(filename, prediction, confidence):

    connection = sqlite3.connect(DB_PATH)
    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO xray_records (filename, prediction, confidence)
        VALUES (?, ?, ?)
        """,
        (filename, prediction, confidence)
    )

    connection.commit()
    connection.close()

    print("X-ray prediction saved!")
    
def get_all_records():

    connection = sqlite3.connect(DB_PATH)
    cursor = connection.cursor()

    cursor.execute("""
        SELECT id, filename, upload_date, prediction, confidence
        FROM xray_records
        ORDER BY id DESC
    """)

    records = cursor.fetchall()
    connection.close()

    return records
