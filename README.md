# Secured Fingerprint Recognition System Using Homomorphic Encryption

## Description
This project presents a biometric system based on **fingerprint recognition** that ensures user confidentiality by using **homomorphic encryption**. The system extracts features from fingerprints using a set of **Gabor filters**, creating a compact and expressive representation, called **FingerCode**. These feature vectors are then homomorphically encrypted, which allows fingerprints to be compared and matched securely without needing to decrypt the data.

<p align="center">
  <img src="./readme images/used_biometric_system.png" width="500" alt="Used biometric system" />
</p>
<p align="center">
  <i>The secured fingerprint recognition system presented in this project</i>
</p>

## Key Features
- **Pattern Recognition**: Automatic identification based on fingerprint structure.
- **Fingerprint Feature Extraction**: The system uses Gabor filters to capture both local and global fingerprint details, converting them into a fixed-size vector called FingerCode.
- **Homomorphic Encryption**: The extracted FingerCode is encrypted using homomorphic techniques, allowing operations (e.g. addition) to be performed directly on encrypted data.
- **Efficient Matching**: Euclidean distance is calculated between encrypted vectors, ensuring fast and secure comparison using a predefined threshold.

## Fingerprint Feature Extraction Algorithm

<p align="center">
  <img src="./readme images/flow_extract_fingercode.png" width="500" alt="Flow extract fingercode" />
</p>
<p align="center">
  <i>Flow of the algorithm used to extract the FingerCode from a fingerprint</i>
</p>

The presented system uses a well-structured algorithm to extract fingerprint features, transforming each fingerprint image into a compact representation called **FingerCode**. Below is a detailed breakdown of the feature extraction process:
1. **Image Preprocessing**

      To reduce noise and enhance contrast in the fingerprint images, making the patterns clearer, I applied a **Gaussian filter** followed by the **CLAHE algorithm** (Contrast Limited Adaptive Histogram Equalization). These steps are important in preparing the images for feature extraction.

2.  **Gradient Calculation**

      After preprocessing, the system calculates the **gradients** of the fingerprint image in both the **horizontal and vertical directions**. These gradients are essential for detecting the **ridges and valleys** that uniquely identify each fingerprint.

3.  **Central Point Detection**

      The next step is identifying the **central point** of the fingerprint, also referred to as the **reference point**. In this project, I considered the reference point to be the location of the **maximum curvature of the ridges**. This point serves as a key reference for dividing the fingerprint into sectors for feature extraction, ensuring stability throughout the entire extraction process.

4.  **Region of Interest Segmentation**

      The region of the fingerprint image that contains relevant data is isolated, leaving out the background and other irrelevant areas. This **segmentation** step ensures that the feature extraction focuses only on the key parts of the fingerprint.

5.  **Division into Sectors**

      Around the reference point detected above, I drew $6$ **concentric circles**. Since the innermost circle around the reference point contains very few pixels, it is not used for extracting distinctive features, the number of bands I used is $n_b = 5$. These circles represent the region of interest from which the **local discriminative information** will be extracted. In the image below, the contours of the sectors are marked with red lines, while the yellow **X** indicates the central point of the fingerprint. The distance between each two circles is $d = 20$, and each circle is divided into $n_r = 16$ regions.

      <p align="center">
            <img src="./readme images/center_and_sectors.png" width="150" alt="Fingerprint with center and sectors" />
      </p>

      The region of interest is defined by a collection of sectors $S_i$. The formula for each sector is:

       $\ \ \ \ \ \ \ \ \ \ \ \ \ S_i = \lbrace (x, y) \mid d(R_i + 1) \leq \sqrt{(x - x_c)^2 + (y - y_c)^2} < d(R_i + 2), \theta_i \leq \tan^{-1} \left( \frac{y - y_c}{x - x_c} \right) < \theta_{i+1} \rbrace$

      where $R_i = \left\lfloor \frac{i}{n_r} \right\rfloor$ and $\theta_i = \frac{2\pi}{n_r} \cdot (i \mod n_r)$. Here, $\left\lfloor \frac{i}{n_r} \right\rfloor$ is the integer part of dividing $i$ by $n_r$, and $i \mod n_r$ represents the remainder of this division. Therefore, In this project, I used a total of $S = n_b \cdot n_r = 5 \cdot 16 = 80$ sectors for feature extraction.

7.  **Normalization**

      Each sector is then **normalized** by adjusting the grayscale intensity of the pixels. This ensures uniformity across the different regions of the fingerprint, preventing variations in brightness from affecting the feature extraction process.

8.  **Gabor Filter Application**

      I applied **Gabor filters** to each sector of the fingerprint, as they are well-known for their ability to **capture texture information**, such as ridges and valleys in various orientations. In this project, I used **8 Gabor filters**, each oriented in a different direction. This process results in a series of filtered images that highlight distinct patterns in the fingerprint's structure.

9.  **Feature Vector Creation (FingerCode)**

      For each sector, the algorithm computes statistical features, such as the mean deviation from the average intensity. his generates a **640-dimensional vector** (based on 8 orientations and 80 sectors), forming the **FingerCode**. This compact representation captures the most distinctive and discriminative features of the fingerprint.

## Homomorphic Encryption
The resulting feature vector is **homomorphically encrypted**, and the system performs matching in the encrypted domain by calculating the **squared Euclidean distance** between encrypted vectors.

## Results
To test the performance of the system, I used a dataset of 205 fingerprint images selected from the archive available [here](https://neurotechnology.com/download/CrossMatch_Sample_DB.zip).

For the proposed system, I achieved a **0% False Acceptance Rate (FAR)** and **1.29% False Rejection Rate (FRR)**.

Additionally, the maximum difference in the distances obtained between two fingerprints in the clear and encrypted domains was **1.32**.

## Future Directions
- Improve fingerprint rotation handling in the feature extraction process.
- Optimize the reference point detection through machine learning techniques, such as training a linear classifier or other advanced models.

## How to run
To install all required libraries, you can find the Anaconda environment file in `frs.yaml`. Simply download it, import it into Anaconda, and select it as the interpreter when running the project. 

Alternatively, iif you prefer not to use Anaconda, you will need to manually install **tenseal**, **opencv**, **numpy**, **python-dotenv**, **tkinter** and **psycopg2**.

### Database Setup

To create the database, ensure you have Docker installed. First, create an `.env` file and place it inside the `app` folder. Then, run the following commands to set up the database inside a Docker container using the configurations specified in the [`app/docker-compose.yml`](app/docker-compose.yml), [`app/init.sql`](app/init.sql), and `app/.env` files:
```
cd app
docker-compose build db; docker-compose up db
```
