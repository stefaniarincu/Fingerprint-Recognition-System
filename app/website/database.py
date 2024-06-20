import os
import numpy as np
import psycopg2
from dotenv import load_dotenv

class Database:
    def __init__(self):
        load_dotenv()

        self.conn = psycopg2.connect(
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT")
        )

        self.cursor = self.conn.cursor()

    def close_connection(self):
        self.cursor.close()
        self.conn.close()

    def insert_into_table(self, id, file_path, fingercode, fingercode_clear):
        sql = "INSERT INTO FINGERPRINT (id, file_path, fingercode, fingercode_clear) VALUES (%s, %s, %s, %s);"
        self.cursor.execute(sql, (id, file_path, fingercode, fingercode_clear))
        self.conn.commit()

    def get_all(self):
        self.cursor.execute("SELECT * FROM FINGERPRINT;")
        datapoints = self.cursor.fetchall()

        return datapoints

    def test_inserted(self, id):
        self.cursor.execute("SELECT * FROM FINGERPRINT WHERE id = %s;", (id,))
        row = self.cursor.fetchone()

        fingercode = row[2] 

        if isinstance(fingercode, memoryview):
            fingercode = bytes(fingercode)
            fingercode_clear = np.frombuffer(row[3], dtype=np.float64)

            return fingercode, fingercode_clear 

