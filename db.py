import sqlite3

# DB connect (auto create ho jayega)
conn = sqlite3.connect("traffic.db")
cursor = conn.cursor()

# table create (agar exist nahi hai)
cursor.execute("""
CREATE TABLE IF NOT EXISTS violations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plate_number TEXT,
    violation_type TEXT,
    image_path TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")

conn.commit()

def save_to_db(plate, vtype, img_path):
    query = "INSERT INTO violations (plate_number, violation_type, image_path) VALUES (?, ?, ?)"
    values = (plate, vtype, img_path)

    cursor.execute(query, values)
    conn.commit()