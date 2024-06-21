from FeatureExtractor import FeatureExtractor
from EncryptionScheme import EncryptionScheme
from Database import Database
import numpy as np
import os

class FingRecognitionSystem:
    def __init__(self):
        self.enc_scheme = EncryptionScheme()
        self.db = Database()
        self.feature_extractor = FeatureExtractor()
        self.cropped_roi = None
        self.clear_fingercode = None
        self.enc_fingercode = None

    def find_center_point(self, fingerprint_img):
        self.feature_extractor.find_reference_point(fingerprint_img)

    def determine_cropped_roi(self, fingerprint_img):
        self.cropped_roi = self.feature_extractor.get_cropped_roi(fingerprint_img)
        
    def extract_fingercode_app(self):
        if len(self.cropped_roi) != 0:
            self.clear_fingercode = self.feature_extractor.continue_process()
            self.enc_fingercode = self.enc_scheme.encrypt_fingercode(self.clear_fingercode)

    def match(self):
        all_rows = self.db.get_all()

        for row in all_rows:
            saved_enc_fingercode = bytes(row[2]) 
            enc_dist = self.enc_scheme.calculate_euclidean_dist(self.enc_fingercode, saved_enc_fingercode)
            
            if self.enc_scheme.apply_threshold(enc_dist):
                self.db.close_connection()

                saved_clear_fingercode = np.frombuffer(row[3], dtype=np.float64)
                clear_dist = np.sum(np.square(self.clear_fingercode - saved_clear_fingercode))

                #saved_fingercode_image = self.feature_extractor.create_fingercode_image(self.feature_extractor.create_fingercodes_image(saved_clear_fingercode))
                saved_fingercodes_image = self.feature_extractor.create_fingercode_image(saved_clear_fingercode)
                selected_fingercodes_image = self.feature_extractor.create_fingercode_image(self.clear_fingercode)

                return row[1], saved_fingercodes_image, selected_fingercodes_image, clear_dist, enc_dist 
            
        self.db.close_connection()    
        return '', 0, 0
    
    def compare_all(self):
        all_rows = self.db.get_all()
        print(len(all_rows))

        with open("encrypted.txt", "w") as f_enc, open("clear.txt", "w") as f_clear:
            for i in range(len(all_rows)):
                for j in range(i+1, len(all_rows)):
                    print(f'{i} {j}')
                    saved_enc_fingercode_1 = bytes(all_rows[i][2]) 
                    saved_enc_fingercode_2 = bytes(all_rows[j][2])
                    enc_dist = self.enc_scheme.calculate_euclidean_dist(saved_enc_fingercode_1, saved_enc_fingercode_2)
                    f_enc.write(f"{os.path.basename(all_rows[i][1])}, {os.path.basename(all_rows[j][1])}, {enc_dist} \n")

                    saved_clear_fingercode_1 = np.frombuffer(all_rows[i][3], dtype=np.float64)
                    saved_clear_fingercode_2 = np.frombuffer(all_rows[j][3], dtype=np.float64)
                    clear_dist = np.sum(np.square(saved_clear_fingercode_1 - saved_clear_fingercode_2))
                    f_clear.write(f"{os.path.basename(all_rows[i][1])}, {os.path.basename(all_rows[j][1])}, {clear_dist} \n")

        self.db.close_connection()    