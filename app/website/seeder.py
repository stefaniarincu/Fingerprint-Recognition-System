import os
import glob
import cv2 as cv
from fingerprint_feature_extraction import process_image
import psycopg2
from dotenv import load_dotenv
import encrypt

load_dotenv()

conn = psycopg2.connect(
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT")
)

cursor = conn.cursor()

def insert_into_table(id, fingercode):
    sql = "INSERT INTO FINGERPRINT (id, fingercode) VALUES (%s, %s);"
    cursor.execute(sql, (id, fingercode))

    conn.commit() 

director_path = "D:/Licenta/74034_3_En_4_MOESM1_ESM/FVC2004/Dbs/DB1_A"
image_extension = '*.tif'

images_path = os.path.join(director_path, image_extension)
#trebuie vazut daca apar ordonate sau nu
files = sorted(glob.glob(images_path))

cnt = 0

for i in range(len(files)):
    print('Procesare imagine nr. %d...' % i)
    img = cv.imread(files[i], cv.IMREAD_GRAYSCALE)

    encrypted_fingerprints, fingercodes = process_image(img)

    for enc_fingerprint in encrypted_fingerprints:
        insert_into_table(cnt, enc_fingerprint)
       
        cnt += 1

cursor.close()
conn.close()


