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

def insert_into_table(id, filename, fingercode):
    sql = "INSERT INTO FINGERPRINT_TEST (id, image_name, fingercode) VALUES (%s, %s, %s);"
    cursor.execute(sql, (id, filename, fingercode))

    conn.commit() 

director_path = "D:/Amprente_test/CrossMatch_Sample_DB - Copy - Copy"
image_extension = '*.tif'

images_path = os.path.join(director_path, image_extension)
files = sorted(glob.glob(images_path))

for i in range(len(files)):
    print('Procesare imagine nr. %d...' % i)
    img = cv.imread(files[i], cv.IMREAD_GRAYSCALE)

    filename = os.path.basename(files[i])

    clear_fingercode, enc_fingercode = process_image(img)

    if len(clear_fingercode) != 0:
        insert_into_table(i, filename, enc_fingercode)

        '''if i == 0:
            encrypt.write_data("nume1.txt", enc_fingerprint)
            print(files[i])

            enc2 = encrypt.ecrypt_fingercode(fingercodes[0])
            encrypt.write_data("nume2.txt", enc2)'''

cursor.close()
conn.close()


