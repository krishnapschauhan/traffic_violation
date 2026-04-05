import sqlite3

conn = sqlite3.connect("traffic.db")  # same DB name
cursor = conn.cursor()

# table ka data delete
cursor.execute("DELETE FROM violations")

conn.commit()
conn.close()

print("✅ All data cleared!")