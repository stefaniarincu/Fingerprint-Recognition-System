import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

# Connect to PostgreSQL database
conn = psycopg2.connect(
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT")
)

# Create a cursor object
cursor = conn.cursor()

# Execute SQL queries
cursor.execute("select * from FINGERPRINT;")
datapoints = cursor.fetchall()
cols = [desc[0] for desc in cursor.description]

print(cols)

# Close connection
cursor.close()
conn.close()
