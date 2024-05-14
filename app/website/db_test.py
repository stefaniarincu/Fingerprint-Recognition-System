import psycopg2
import os
import base64
from dotenv import load_dotenv

def set_connection():
    load_dotenv()

    conn = psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT")
    )

    cursor = conn.cursor()

    return conn, cursor

def end_connection(conn, cursor):
    cursor.close()
    conn.close()

def insert_into_table(fingercode):
    conn, cursor = set_connection()

    sql = "INSERT INTO FINGERPRINT (id, fingercode) VALUES (%s, %s);"
    cursor.execute(sql, (1, fingercode))

    conn.commit() 
        
    end_connection(conn, cursor)

def test_col_descr():
    conn, cursor = set_connection()

    cursor.execute("SELECT * FROM FINGERPRINT;")
    datapoints = cursor.fetchall()
    cols = [desc[0] for desc in cursor.description]

    print(cols)

    end_connection(conn, cursor)

def test_inserted():
    conn, cursor = set_connection()

    cursor.execute("SELECT * FROM FINGERPRINT;")
    datapoints = cursor.fetchall()
    for row in datapoints:
        for value in row:
            if isinstance(value, memoryview):
                value = bytes(value)

                if type(value) == bytes:
                    value = base64.b64encode(value)
                with open("enc_FV_DB.txt", "wb") as f:
                    f.write(value)
            else:
                print(value) 

    end_connection(conn, cursor)

