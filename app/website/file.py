import os
import glob
import cv2 as cv
from fingerprint_feature_extraction import process_image
import psycopg2
from dotenv import load_dotenv
import numpy as np
import encrypt
import base64

load_dotenv()

conn = psycopg2.connect(
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT")
)

cursor = conn.cursor()

def select_from_table():
    sql = "SELECT id, fingercode FROM FINGERPRINT WHERE id=0;"
    cursor.execute(sql)
    row = cursor.fetchone()
    
    ind = row[0]
    fingercode = row[1] 

    if isinstance(fingercode, memoryview):
        fingercode = bytes(fingercode)

    return ind, fingercode

def write(fingercode):
    if type(fingercode) == bytes:
        fingercode = base64.b64encode(fingercode) 
    
    with open("enc_FV_DB.txt", "wb") as f:
        f.write(fingercode)

director_path = "D:/Licenta/74034_3_En_4_MOESM1_ESM/FVC2004/Dbs/DB1_A"
image_extension = '*.tif'

images_path = os.path.join(director_path, image_extension)
files = sorted(glob.glob(images_path))

img = cv.imread(files[0], cv.IMREAD_GRAYSCALE)

encrypted_fingerprints1, fingercodes = process_image(img)

f1 = fingercodes[0]
print(files[0])
encrypt.write_data("enc_FV_DB1.txt", encrypted_fingerprints1[0])
print("aaaa")

img = cv.imread(files[2], cv.IMREAD_GRAYSCALE)

encrypted_fingerprints2, fingercodes = process_image(img)

f2 = fingercodes[0]
print("bbbb")

img1_embedding_arr = np.array(f1)
img2_embedding_arr = np.array(f2)
euclidean_squared_plain = np.sum(np.square(img1_embedding_arr -  img2_embedding_arr))
print(euclidean_squared_plain)

ind, fingercode2 = select_from_table()
write(fingercode2)
print("cccc")

v = encrypt.calculate_euclidean_dist(encrypted_fingerprints2[0], fingercode2)
print(v)

cursor.close()
conn.close()


