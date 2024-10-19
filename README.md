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
### 1. Core Point Detection

The first step is identifying the **core point** of the fingerprint, also known as the **reference point**. In this project, I considered the reference point to be the location of the **maximum curvature of the ridges**. This step is important because the fingerprint features are extracted with respect to this central point. To detect the core point, the following substeps are performed:

#### Image Preprocessing

To reduce noise and enhance contrast in the fingerprint images, making the patterns clearer, I applied a **Gaussian filter** followed by the **CLAHE algorithm** (Contrast Limited Adaptive Histogram Equalization). These steps are important in preparing the images for feature extraction.

#### Fingerprint Contour Detection

Knowing that a fingerprint is characterized by the uniform pattern of intersecting ridges and valleys, I used the properties of the **gradients** to capture the quick transitions from white (valleys) to black (ridges) within the fingerprint image, highlighting the area of interest. 

<p align="center">
  <img src="./readme images/gradient_x.png" width="200" alt="Horizontal gradient of a fingerprint image" />
  <img src="./readme images/gradient_y.png" width="200" alt="Vertical gradient of a fingerprint image" />
</p>
<p align="center">
  <i>The horizontal and vertical gradients of a fingerprint image</i>
</p>

After the preprocessing step, I calculated the **gradients** of the fingerprint image in both the **horizontal** ($G_x$) and **vertical** ($G_y$) directions. Using these values, I computed the **orientation field** with complex values ($A = (G_x + i \cdot G_y)^2$) and its absolute magnitude ($B = |A|$) by transforming the data into the frequency domain.

To determine the contour of the fingerprint, I created a mask equal in size to the input image, initialized with values of 1. Then, I divided each element in matrix $A$ by its corresponding element in $B$, except when the pixel value in $B$ is 0 (to avoid division errors). 

Finally, I applied a filter in the **frequency domain**, which combines a **Gaussian component** with a **complex component**. This filter is then applied to the fingerprint contour using a **Fourier transform** convolution.

#### Region of Interest Segmentation

To delimitate the region of interest, I divided the original image into non-overlapping blocks of size $L \times L$. Subsequently, I calculated the **variance** of the grayscale tones for each block, highlighting local differences in pixel intensity. This variance calculation aids in detecting areas where ridges intersect with valleys, indicated by numerous transitions from white to black. Areas with high variance appear lighter, closer to white, emphasizing structures with significant details, such as the fingerprint itself.

Then, I established a minimum variance **threshold** to retain only the significant blocks, labeling them with a value of 1 (white) in a binary mask, while the remaining blocks were assigned a value of 0 (black).

To standardize the contour of the fingerprint and eliminate residual noise in the background, I performed several **morphological transformations** on the resulting binary mask. First, I applied a **morphological closing operator** to eliminate remaining gaps within the fingerprint, represented by black pixels, without altering its original contour. Then, I performed an **erosion operation** to remove background noise, represented by white pixels. Knowing that the central point of a fingerprint typically does not extend to its outer edges, the erosion operator also serves to thin the fingerprint contour.

<p align="center">
  <img src="./readme images/applied_morphological_operators.png" width="400" alt="Applied morphological operators" />

By overlaying the variance mask obtained after applying all morphological operators over the filtered image presented at the previous step, I produced the final segmented fingerprint image.

To determine the central point, defined as the point of maximum curvature, I searched for the pixel position with the maximum amplitude.

### 2. Region of Interest Cropping

Since the algorithm implemented in this project uses only the information from the vicinity of the central point,  retaining the entire image is unnecessary. I defined a variable $l$ representing the side length of a square centered around the reference point ($l$ must be an odd number).

The way the fingerprint is positioned on the scanner's surface during the capture module affects the algorithm's performance. So, if cropping the square of side length $l$ exceeds the height or width of the original image, the system will stop the image processing and display a message indicating the impossibility of extracting the necessary information. 

### 3. Division into Sectors

Around the reference point detected above, I drew $6$ **concentric circles**. Since the innermost circle around the reference point contains very few pixels, it is not used for extracting distinctive features, the number of bands I used is $n_b = 5$. These circles represent the region of interest from which the **local discriminative information** will be extracted. In the image below, the contours of the sectors are marked with red lines, while the yellow **X** indicates the central point of the fingerprint. The distance between each two circles is $d = 20$, and each circle is divided into $n_r = 16$ regions.

<p align="center">
      <img src="./readme images/center_and_sectors.png" width="150" alt="Fingerprint with center and sectors" />
</p>

The region of interest is defined by a collection of sectors $S_i$. The formula for each sector is:

$S_i = \lbrace (x, y) \mid d(R_i + 1) \leq \sqrt{(x - x_c)^2 + (y - y_c)^2} < d(R_i + 2), \theta_i \leq \tan^{-1} \left( \frac{y - y_c}{x - x_c} \right) < \theta_{i+1} \rbrace$

where $R_i = \left\lfloor \frac{i}{n_r} \right\rfloor$ and $\theta_i = \frac{2\pi}{n_r} \cdot (i \mod n_r)$. Here, $\left\lfloor \frac{i}{n_r} \right\rfloor$ is the integer part of dividing $i$ by $n_r$, and $i \mod n_r$ represents the remainder of this division. Therefore, In this project, I used a total of $S = n_b \cdot n_r = 5 \cdot 16 = 80$ sectors for feature extraction.

### 4. Normalization

Before applying the set of Gabor filters, I applied a **local normalization technique**, calculating the mean ($M_i$) and variance ($V_i$) of the grayscale levels for each sector $S_i$. I defined a global mean ($M_0=100$) and standard variance ($V_0=100$).

A pixel's intensity in the image, denoted $I(x, y)$, is normalized based on the formula:

$$
N_i(x, y) = 
\begin{cases}
M_0 + C_i(x, y) & \text{if } I(x, y) > M_i \\
M_0 - C_i(x, y) & \text{otherwise} 
\end{cases} 
$$

where 

$$
C_i(x, y) = \sqrt{\frac{V_0 \cdot [I(x, y) - M_i]^2}{V_i}}
$$

To prevent errors, the algorithm handles cases where $V_i = 0$ by setting $N_i(x, y) = M_0$. 

### 5. Gabor Filter Application

I applied **Gabor filters** to each sector of the fingerprint, as they are well-known for their ability to **capture texture information**, such as ridges and valleys in various orientations. In this project, I used $n_f = 8$ Gabor filters, each oriented in a different direction ($\theta_i \in \lbrace 0^\circ, 22.5^\circ, 45^\circ, 67.5^\circ, 90^\circ, 112.5^\circ, 135^\circ, 157.5^\circ \rbrace$). This process results in a series of filtered images that highlight distinct patterns in the fingerprint's structure.

<p align="center">
      <img src="./readme images/img_gabor.png" width="400" alt="Gabor filters" />
</p>

### 6. Creating the Feature Vector (FingerCode)

For each of the eight filtered images obtained in the previous step, I calculated the **mean absolute deviation from the average grayscale levels**, denoted as $D_i$ for each sector $S_i$. Let $n_{p_i}$ be the number of pixels in sector $S_i$. For each filtered image $Img_f$ and pixel intensity $I_f(x, y)$, the mean absolute deviation is computed as follows:

$$
D_i = \frac{1}{n_{p_i}} \cdot \sum_{(x, y) \in Img_f} \left|I_f(x, y) - M_i\right| 
$$

where $M_i$ is the mean intensity for sector $i$. Each $D_i$ value will be a component of the feature vector. Ultimately, this vector will contain $n_f \cdot (n_b \cdot n_r) = n_f \cdot n_s = 8 \cdot 80 = 640$ elements, where $n_f$ is the number of filters, $n_b$ is the number of concentric bands, $n_r$ is the number of regions in each band, and $n_s$ is the number of sectors.

<p align="center">
      <img src="./readme images/fingercodes.png" width="400" alt="FingerCodes" />
</p>

In the image above, the eight *FingerCodes* obtained are intuitively represented by coloring an entire sector based on the mean absolute deviation corresponding to each in the feature vector. This compact representation allowed the use of homomorphic encryption, enabling the matching process in the encrypted domain through Euclidean distance calculation.

## Homomorphic Encryption
The resulting feature vector is **homomorphically encrypted**, allowing the system to perform matching directly in the encrypted domain. Since the **CKKS scheme** cannot compute the square root of a number, I used the **square of the Euclidean distance** as the similarity metric. 

For two n-dimensional vectors $\vec{a} = (a_1, a_2, \ldots, a_n)$ and $\vec{b} = (b_1, b_2, \ldots, b_n)$, the squared Euclidean distance is calculated using the formula:
 
$$
DE(a, b) = \sum_{i=1}^{n}{(a_i - b_i)^2}
$$

## Results
To demonstrate that using homomorphic encryption does not impact the performance of a fingerprint recognition system, I implemented a secure demonstrative system, presented in this project. To test its performance, I used a dataset of 205 fingerprint images from 43 different people, selected from the archive available [here](https://neurotechnology.com/download/CrossMatch_Sample_DB.zip). For each fingerprint, I extracted the feature vector based on the algorithm described above and to determine the similarity between two FingerCodes, I used the squared Euclidean distance as the metric. 

A crucial step in developing a biometric system is choosing the decision threshold. This threshold determines if the similarity between two fingerprints is low enough to conclude they belong to the same person. For security reasons, I chose a threshold that minimizes unauthorized access, even if it risks denying access to an authorized entity. In the image below I present the results obtained after testing the proposed system. The algorithm performed $C_{205}^{2} = 20910$ comparisons to assess the similarity between each pair of fingerprints, using a threshold of $\textbf{805000}$. This value was determined to maintain a **false acceptance rate (FAR)** of $0\%$. The image also shows that the number of false negatives (when two fingerprints from the same person are incorrectly identified as different) is very low compared to the correct matches. Specifically, the **false rejection rate (FRR)** is calculated as:

$$
\text{FRR} = \frac{\text{false rejected}}{\text{false rejected} \ + \ \text{correct accepted}} \cdot 100 = \frac{6}{6+457} \cdot 100 = 1.29\%
$$

<p align="center">
      <img src="./readme images/confusion_matrix.png" width="300" alt="Confusion matrix" />
</p>

In the image below, I presented the distribution of the squared Euclidean distance in the encrypted domain. The pink bars represent distances between fingerprints of the same individual, while the blue bars indicate distances between different individuals, along with the threshold affecting the system's decision. 

<p align="center">
      <img src="./readme images/distrib_dist.png" width="300" alt="Distances distribution" />
</p>

The next image provides a clearer view of the categories of fingerprint similarity. The dotted line marks the chosen threshold ($805000$) to achieve a $0\%$ FAR. I also highlighted the six instances where two fingerprints were incorrectly classified as belonging to different people.

<p align="center">
      <img src="./readme images/false_neg_int.png" width="300" alt="Intersection of false negatives" />
</p>

To showcase the properties of homomorphic encryption, I computed the similarity between all $205$ feature vectors, both in the **encrypted domain** and the **clear domain**. The differences in distances obtained from both domains are minimal, resulting in almost identical histograms. Thus, in the image below, I presented the distribution of the differences between the distances in the encrypted and clear domains, specifically showing $dist_{encrypted} - dist_{clear}$.

<p align="center">
      <img src="./readme images/distrib_diff.png" width="300" alt="Distribution of differences" />
</p>

As seen, the differences between distances from the encrypted and clear domains are negligible compared to the actual values of the distances. It is important to note that using very large numbers in the CKKS scheme can negatively impact precision due to increased noise. However, this maximum difference of $1.32$, relative to $99 \cdot 10^6$ (the maximum distance obtained in the encrypted domain), does not affect the performance of the biometric system presented. Therefore, it is evident that homomorphic encryption ensures the security of a biometric system while maintaining its accuracy and reliability.


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

### Environment File Template
The `.env` file must contain the following variables:
- **CONTAINER_NAME**: Name of the container
- **DB_USER**: Database user
- **DB_PASSWORD**: Database password
- **DB_NAME**: Database name
- **DB_HOST**: Database host (usually: localhost)
- **DB_PORT**: Database port (default: 5432 for PostgreSQL)
- **DB_PORT2**: Second database port (if applicable; default: 5432 for PostgreSQL)
- **PUBLIC_CONTEXT_FILE**: Path to the public context file (for homomorphic encryption)
- **SECRET_CONTEXT_FILE**: Path to the secret context file (for homomorphic encryption)

```
CONTAINER_NAME=your_container_name
DB_USER=your_database_user
DB_PASSWORD=your_database_password
DB_NAME=your_database_name

DB_HOST=localhost
DB_PORT=5432
DB_PORT2=5432

PUBLIC_CONTEXT_FILE=public.txt
SECRET_CONTEXT_FILE=secret.txt
```