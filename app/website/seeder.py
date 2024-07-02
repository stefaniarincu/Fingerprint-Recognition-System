import os
import glob
import cv2 as cv
from FeatureExtractor import FeatureExtractor
from EncryptionScheme import EncryptionScheme
from Database import Database

enc_scheme = EncryptionScheme()
db = Database()
feature_extract = FeatureExtractor()

director_path = "D:/Fingerprints"#"C:/Users/admin/Downloads/CrossMatch_Sample_DB - Copy - Copy""D:/Amprente_test/CrossMatch_Sample_DB - Copy - Copy"
image_extension = '*.tif'
images_path = os.path.join(director_path, image_extension)
files = sorted(glob.glob(images_path))

for i in range(len(files)):
    print('Procesare imagine nr. %d...' % i)
    img = cv.imread(files[i], cv.IMREAD_GRAYSCALE)

    clear_fingercode = feature_extract.process_image(img)
    enc_fingercode = enc_scheme.encrypt_fingercode(clear_fingercode)

    if len(clear_fingercode) != 0:
        db.insert_into_table(i, files[i], enc_fingercode, clear_fingercode.tobytes())

db.close_connection()

