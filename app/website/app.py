from flask import Flask, render_template, request
from database import init_db, insert_into_table, search_in_table
from fingerprint_feature_extraction import process_image
import numpy as np
import cv2
import encrypt

app = Flask(__name__)
app.config.from_object("config")

init_db(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if request.method == 'POST':
        if 'image' not in request.files:
            return 'No file part'
        
        image = request.files['image']
        img = cv2.imdecode(np.fromstring(image.read(), np.uint8), cv2.IMREAD_GRAYSCALE)

        processed_img = process_image(img)
        fingerprint_data = encrypt.ecrypt_fingercode(processed_img[0][0])

        match_found = search_in_table(fingerprint_data)

        if match_found:
            return 'Match found!'
        else:
            return 'No match found.'


if __name__ == "__main__":
    app.run(debug=True)
