from cgi import print_form
from pickletools import float8
import numpy as np
import cv2 as cv
import os
import glob
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
import math
from scipy.signal import fftconvolve
from matplotlib.patches import Circle

# Variabile globale
nr_filters = 8
nr_bands = 5
nr_sectors_band = 16 #k
band_width = 20 #b
nr_sectors = nr_bands * nr_sectors_band
h_roi = 10 + 2 * ((nr_bands + 1) * band_width) - 1 #ca sa fie impar => mijlocul e centrul


def find_reference_point(param_img):
    grid_x, grid_y = np.meshgrid(np.arange(-16, 17), np.arange(-16, 17))
    
    exponent = np.exp(-(grid_x ** 2 + grid_y ** 2) / (2 * np.sqrt(55) ** 2))
    core_filter = exponent * (grid_x + 1j * grid_y)

    param_img = param_img.astype(complex)
    gradient_x, gradient_y = np.gradient(param_img)
    nominator = (gradient_x + 1j * gradient_y) ** 2
    denominator = np.abs((gradient_x + 1j * gradient_y) ** 2)

    grad_field = np.ones_like(param_img, dtype=complex)
    for i in range(param_img.shape[0]):
        for j in range(param_img.shape[1]):
            if denominator[i][j] != 0:
                grad_field[i][j] = nominator[i][j] / denominator[i][j]

    block_size = 8    
    variance_thresh = 20
    k_size_close = 10
    k_size_erode = 44
    num_rows, num_cols = param_img.shape

    mirrored = np.pad(grad_field.copy(), ((20, 20), (20, 20)), mode='reflect')
    image_filtered = convolve2d(mirrored, core_filter, mode='same')
    image_filtered = np.abs(image_filtered[20 : 20 + grad_field.shape[0], 20 : 20 + grad_field.shape[1]])

    new_rows = int(block_size * np.ceil((float(num_rows)) / (float(block_size))))
    new_cols = int(block_size * np.ceil((float(num_cols)) / (float(block_size))))

    padded_img = np.zeros((new_rows, new_cols), dtype=complex)
    variance_matrix = np.zeros((new_rows, new_cols), dtype=complex)
    padded_img[0:num_rows][:, 0:num_cols] = param_img
    
    for i in range(0, num_rows, block_size):
        for j in range(0, num_cols, block_size):
            block = padded_img[i : i + block_size][:, j : j + block_size]
            variance_matrix[i : i + block_size][:, j : j + block_size] = np.var(block) * np.ones(block.shape)
            
    variance_matrix = variance_matrix[0:num_rows][:, 0:num_cols]    
    mask_variance = (variance_matrix > variance_thresh).astype(np.uint8)
    
    mask_variance = cv.morphologyEx(mask_variance, cv.MORPH_CLOSE, np.ones((k_size_close, k_size_close), np.uint8))
    mask_variance = cv.erode(mask_variance, np.ones((k_size_erode, k_size_erode), np.uint8))
    
    mask = image_filtered * mask_variance

    max_line = np.max(mask, axis=0)
    x_center = np.argmax(max_line)
    y_center = np.argmax(mask, axis=0)[x_center]

    return x_center, y_center


def crop_roi(param_h_roi, param_x_center, param_y_center, param_img):
    img_height, img_width  = param_img.shape

    if (param_y_center - param_h_roi//2 < 0) or (param_y_center + param_h_roi//2 > img_height - 1) or (param_x_center - param_h_roi//2 < 0) or (param_x_center + param_h_roi//2 > img_width - 1):
        padded_img = np.ones((img_height + 2*param_h_roi, img_width + 2*param_h_roi), dtype=np.uint8) * 255
        padded_img[param_h_roi:param_h_roi+img_height, param_h_roi:param_h_roi+img_width] = param_img
        cropped_roi = padded_img[param_y_center - param_h_roi//2 + param_h_roi : param_y_center + param_h_roi//2 + param_h_roi,
                       param_x_center - param_h_roi//2 + param_h_roi : param_x_center + param_h_roi//2 + param_h_roi]
    else:
        cropped_roi = param_img[param_y_center - param_h_roi//2 : param_y_center + param_h_roi//2 + 1,
                       param_x_center - param_h_roi//2 : param_x_center + param_h_roi//2 + 1]

    return cropped_roi


def display_center_point(img, x_center, y_center):
    plt.imshow(img, cmap='gray')
    plt.scatter(x_center, y_center, c='yellow', marker='x', label='Center Point')
    
    plt.title('Fingerprint with Center Point')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()


def divide_into_sectors(param_h_roi, param_nr_sectors, param_band_width, param_nr_sectors_band):
    vect_T = []
    vect_angles = []

    x_center, y_center = param_h_roi // 2, param_h_roi // 2

    for i in range(param_nr_sectors + 1):
        vect_T.append(i // param_nr_sectors_band)
        vect_angles.append((i % param_nr_sectors_band) * (2 * 180.0 / param_nr_sectors_band))

    sectors = []

    for i in range(param_nr_sectors):
        sector = []
        
        for x in range(param_h_roi):
            for y in range(param_h_roi):
                x0 = x - x_center
                y0 = y - y_center
                r = math.sqrt(x0**2 + y0**2)                  

                if param_band_width * (vect_T[i] + 1) <= r < param_band_width * (vect_T[i] + 2):
                    theta = math.degrees(math.atan2(y0, x0))
                    if theta < 0:
                        theta += 360.0  
                    
                    if i % param_nr_sectors_band == param_nr_sectors_band - 1 and i > 0:
                        if vect_angles[i] <= theta:
                            sector.append([x, y])
                    else:
                        if vect_angles[i] <= theta < vect_angles[i+1]:
                            sector.append([x, y])
        
        sectors.append(sector)
    
    return sectors

def plot_sectors(param_points_each_sector, param_h_roi, param_nr_bands, param_nr_sectors_band):
    images = []
    
    colors_rgb = [
        np.array([255, 0, 0]),     # Red
        np.array([0, 255, 0]),     # Green
        np.array([0, 0, 255]),     # Blue
        np.array([255, 255, 0]),   # Yellow
        np.array([255, 0, 255]),   # Magenta
        np.array([0, 255, 255]),   # Cyan
        np.array([128, 0, 0]),     # Maroon
        np.array([0, 128, 0]),     # Dark Green
        np.array([0, 0, 128]),     # Navy
        np.array([128, 128, 0]),   # Olive
        np.array([128, 0, 128]),   # Purple
        np.array([0, 128, 128]),   # Teal
        np.array([255, 165, 0]),   # Orange
        np.array([255, 192, 203]), # Pink
        np.array([128, 128, 128]), # Gray
        np.array([255, 255, 255])  # White
    ]

    cnt = 0

    for i in range(param_nr_bands):
        image = np.ones((param_h_roi, param_h_roi), dtype=np.uint8)   
        image = cv.cvtColor(image, cv.COLOR_GRAY2RGB) 

        for j in range(param_nr_sectors_band):
            for idx in param_points_each_sector[cnt]:
                image[idx[0], idx[1]] = colors_rgb[j]
        
            cnt += 1
            
        images.append(image)
    

    fig, axs = plt.subplots(1, 5, figsize=(15, 15))
    '''
    for i in range(len(param_points_each_sector)):
        image = np.ones((param_h_roi, param_h_roi), dtype=np.uint8)   
        image = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
        for idx in param_points_each_sector[i]:
           image[idx[0], idx[1]] = np.array([255, 255, 255])  # White
            
        images.append(image)
    

    fig, axs = plt.subplots(10, 8, figsize=(15, 15))
    '''

    axs = axs.flatten()

    for i, ax in enumerate(axs):
        ax.imshow(images[i])
        ax.axis('off')  # Turn off axis labels
        ax.set_title(f'Image {i+1}', fontsize=5)  # Set a title for each image

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()


def plot_circles_and_lines(param_h_roi, param_nr_sectors, param_band_width, param_nr_sectors_band):
    fig, ax = plt.subplots(figsize=(6, 6))

    fingerprint = np.ones((param_h_roi, param_h_roi))
    ax.imshow(fingerprint, cmap='gray', origin='upper')

    x_center, y_center = param_h_roi // 2, param_h_roi // 2

    ax.text(x_center, y_center, 'X', color='red', ha='center', va='center', fontsize=12)

    for i in range(6):
        circle = Circle((x_center, y_center), (i + 1) * param_band_width, fill=False, edgecolor='red', lw=1.2)
        ax.add_patch(circle)

    vect_angles = [(i % param_nr_sectors_band) * (2 * 180.0 / param_nr_sectors_band) for i in range(param_nr_sectors)]

    for angle in vect_angles:
        rad_angle = math.radians(angle)
        x_end = x_center + ((param_h_roi - 10) / 2) * np.cos(rad_angle)
        y_end = y_center + ((param_h_roi - 10) / 2) * np.sin(rad_angle)
        x_c = x_center + param_band_width * np.cos(rad_angle)
        y_c = y_center + param_band_width * np.sin(rad_angle)

        ax.plot([x_c, x_end], [y_c, y_end], 'red')

    ax.set_xlim(0, param_h_roi)
    ax.set_ylim(0, param_h_roi)
    ax.set_aspect('equal', adjustable='box')
    plt.show()

def normalize_sectors(param_points_each_sector, param_cropped_roi, target_mean=100.0, target_variance=100.0):
    mean_each_sector = np.zeros(len(param_points_each_sector))
    variance_each_sector = np.zeros(len(param_points_each_sector))
    norm_sectors = np.zeros_like(param_cropped_roi)

    for idx in range(len(param_points_each_sector)):
        for point in param_points_each_sector[idx]:
            mean_each_sector[idx] += param_cropped_roi[point[0], point[1]]

        mean_each_sector[idx] /= len(param_points_each_sector[idx])

    for idx in range(len(param_points_each_sector)):
        for point in param_points_each_sector[idx]:
            variance_each_sector[idx] += math.pow((param_cropped_roi[point[0], point[1]] - mean_each_sector[idx]), 2)

        variance_each_sector[idx] /= len(param_points_each_sector[idx])

    for idx in range(len(param_points_each_sector)):
        for point in param_points_each_sector[idx]:
            if variance_each_sector[idx] == 0:
                norm_sectors[point[0], point[1]] = target_mean
            else:
                pixel_value = param_cropped_roi[point[0], point[1]]
                invariant_formula = math.sqrt((target_variance + math.pow((pixel_value - mean_each_sector[idx]), 2)) / variance_each_sector[idx])
                
                if pixel_value > mean_each_sector[idx]:
                    norm_sectors[point[0], point[1]] = target_mean + invariant_formula
                else:
                    norm_sectors[point[0], point[1]] = target_mean - invariant_formula

    return norm_sectors

def get_gabor_filter(filter_idx, param_nr_filters):    
    sigma = 4
    gamma = 1.0
    psi = (-90) * np.pi / 180.0
    l = 10
    kernel_size = 33
    theta = filter_idx * np.pi / param_nr_filters
    
    cos = np.cos(filter_idx * np.pi / param_nr_filters)
    sin = np.sin(filter_idx * np.pi / param_nr_filters)
    
    sin_vect = [i * sin for i in range(-16, 17)]
    cos_vect = [i * cos for i in range(-16, 17)] 
    
    gabor_filter = np.zeros((kernel_size, kernel_size))
    
    for i in range(kernel_size):
        for j in range(kernel_size):
            xx = sin_vect[j] + cos_vect[i]
            yy = -sin_vect[i] + cos_vect[j]
            gabor_filter[j, i] = np.exp(-((xx**2) + (gamma * yy**2)) / (2 * sigma**2)) * np.cos((2 * np.pi * xx) / l + psi)
    
    #gabor_filter = cv.getGaborKernel((kernel_size, kernel_size), sigma, theta, l, gamma, psi)

    return gabor_filter 

def conv2fft(param_img, param_filter):
    '''
    img_height, img_width = param_img.shape
    filter_height, filter_width = param_filter.shape
    
    filtered_image = np.fft.ifft2(np.fft.fft2(param_img, [img_height+filter_height-1, img_width+filter_width-1]) * np.fft.fft2(param_filter, [img_height+filter_height-1, img_width+filter_width-1]))
    if not np.any(np.imag(param_img)) and not np.any(np.imag(param_filter)):
        filtered_image = np.real(filtered_image)

    px = ((filter_height - 1) + ((filter_height - 1) % 2)) // 2
    py = ((filter_width - 1) + ((filter_width - 1) % 2)) // 2
        
    return filtered_image[px:px+img_height, py:py+img_width]
    '''
    
    return fftconvolve(param_img, param_filter, mode='same')
     
    #return cv.filter2D(param_img, -1, param_filter) 


def determine_fingercode(param_img, param_points_each_sector):   
    mean_each_sector = np.zeros(len(param_points_each_sector))
    fingercode_vector = np.zeros(len(param_points_each_sector))

    for idx in range(len(param_points_each_sector)):
        for point in param_points_each_sector[idx]:
            mean_each_sector[idx] += param_img[point[0], point[1]]

        mean_each_sector[idx] /= len(param_points_each_sector[idx])

    for idx in range(len(param_points_each_sector)):
        for point in param_points_each_sector[idx]:
            fingercode_vector[idx] += np.abs(param_img[point[0], point[1]] - mean_each_sector[idx])

        fingercode_vector[idx] /= len(param_points_each_sector[idx])

    return fingercode_vector


def create_fingercode_image(param_img, param_fingercode, param_points_each_sector):
    fingercode_image = np.zeros_like(param_img)
    
    for idx in range(len(param_points_each_sector)):
        for point in param_points_each_sector[idx]:
            fingercode_image[point[0], point[1]] = param_fingercode[idx]

    return fingercode_image

def display_images(img_vector, num_img_per_row=2, num_images_per_col=4):
    fig, axs = plt.subplots(num_images_per_col, num_img_per_row, figsize=(15, 15))

    axs = axs.flatten()

    for i, ax in enumerate(axs):
        ax.imshow(img_vector[i], cmap='gray')
        ax.axis('off')  # Turn off axis labels
        ax.set_title(f'Image {i+1}', fontsize=5)  # Set a title for each image

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()
    

director_path = "D:/Licenta/74034_3_En_4_MOESM1_ESM/FVC2004/Dbs/DB1_A"
image_extension = '*.tif'

images_path = os.path.join(director_path, image_extension)
files = glob.glob(images_path)

for i in range(len(files)):
    print('Procesare imagine nr. %d...' % i)
    img = cv.imread(files[i], cv.IMREAD_GRAYSCALE)

    x_center, y_center = find_reference_point(img)
    #display_center_point(img, x_center, y_center)
    
    cropped_roi = crop_roi(h_roi, x_center, y_center, img)

    points_each_sector = divide_into_sectors(h_roi, nr_sectors, band_width, nr_sectors_band)
    #plot_sectors(points_each_sector, h_roi, nr_bands, nr_sectors_band)
    #plot_circles_and_lines(h_roi, nr_sectors, band_width, nr_sectors_band)

    norm_cropped_roi = normalize_sectors(points_each_sector, cropped_roi)

    fingercodes = []
    filtered_images = []
    fingercode_images = []

    for idx in range(nr_filters):
        gabor_filter = get_gabor_filter(idx, nr_filters)
        
        filtered_roi = conv2fft(norm_cropped_roi.copy(), gabor_filter)
        filtered_images.append(filtered_roi)
        
        fingercode = determine_fingercode(filtered_roi, points_each_sector)
        fingercodes.append(fingercode)
        
        fingercode_image = create_fingercode_image(filtered_roi, fingercode, points_each_sector)
        fingercode_images.append(fingercode_image)
       
    #filtered_images = np.array(filtered_images, dtype=np.uint8)   
    display_images(fingercode_images)   