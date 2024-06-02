import encrypt

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
band_width = 10 #b
nr_sectors = nr_bands * nr_sectors_band
h_roi = 10 + 2 * ((nr_bands + 1) * band_width) - 1 #ca sa fie impar => mijlocul e centrul


def find_reference_point(param_img):
    block_size = 8    
    variance_thresh = 20
    num_rows, num_cols = param_img.shape
    
    blurred_img = cv.GaussianBlur(param_img.copy(), (3, 3), 0)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    blurred_img = clahe.apply(blurred_img)
    
    gradient_x, gradient_y = np.gradient(blurred_img)    
    nominator = (gradient_x + 1j * gradient_y) ** 2
    denominator = np.abs((gradient_x + 1j * gradient_y) ** 2)

    grad_field = np.ones_like(param_img, dtype=complex)
    for i in range(param_img.shape[0]):
        for j in range(param_img.shape[1]):
            if denominator[i][j] != 0:
                grad_field[i][j] = nominator[i][j] / denominator[i][j]

    grid_x, grid_y = np.meshgrid(np.arange(-16, 17), np.arange(-16, 17))
    exponent = np.exp(-(grid_x ** 2 + grid_y ** 2) / (2 * np.sqrt(60) ** 2))
    core_filter = exponent * (grid_x + 1j * grid_y)

    padded_grad_field = np.pad(grad_field, ((20, 20), (20, 20)), mode='reflect')
    image_filtered = fftconvolve(padded_grad_field, core_filter, mode='same')
    image_filtered = np.abs(image_filtered[20 : 20 + grad_field.shape[0], 20 : 20 + grad_field.shape[1]])

    new_rows = int(block_size * np.ceil((float(num_rows)) / (float(block_size))))
    new_cols = int(block_size * np.ceil((float(num_cols)) / (float(block_size))))

    padded_img = np.zeros((new_rows, new_cols), dtype=complex)
    variance_matrix = np.zeros((new_rows, new_cols), dtype=complex)
    padded_img[0:num_rows][:, 0:num_cols] = param_img
    
    for i in range(0, num_rows, block_size):
        for j in range(0, num_cols, block_size):
            block = padded_img[i : i + block_size][:, j : j + block_size]
            variance_matrix[i : i + block_size][:, j : j + block_size] = np.var(block) 
            
    variance_matrix = variance_matrix[0:num_rows][:, 0:num_cols]    
    mask_variance = (variance_matrix > variance_thresh).astype(np.uint8)
    
    mask_variance = cv.morphologyEx(mask_variance, cv.MORPH_CLOSE, np.ones((8, 8), np.uint8), iterations=2)
    mask_variance = cv.morphologyEx(mask_variance, cv.MORPH_ERODE, np.ones((38, 38), np.uint8))
    mask_variance = cv.morphologyEx(mask_variance, cv.MORPH_ERODE, np.ones((8, 8), np.uint8), iterations=2)
    
    mask = image_filtered * mask_variance

    y_center, x_center = np.unravel_index(np.argmax(mask), mask.shape)
    return x_center, y_center


def crop_roi(param_h_roi, param_x_center, param_y_center, param_img):
    img_height, img_width  = param_img.shape

    if (param_y_center - param_h_roi//2 < 0) or (param_y_center + param_h_roi//2 > img_height - 1) or (param_x_center - param_h_roi//2 < 0) or (param_x_center + param_h_roi//2 > img_width - 1):
        return np.array([]) 
    else:
        return param_img[param_y_center - param_h_roi//2 : param_y_center + param_h_roi//2 + 1,
                       param_x_center - param_h_roi//2 : param_x_center + param_h_roi//2 + 1]


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


def normalize_sectors(param_points_each_sector, param_cropped_roi, target_mean=100.0, target_variance=100.0):
    mean_each_sector = np.zeros(len(param_points_each_sector))
    variance_each_sector = np.zeros(len(param_points_each_sector))
    norm_sectors = np.zeros_like(param_cropped_roi)
    #norm_sectors = param_cropped_roi.copy()

    for idx in range(len(param_points_each_sector)):
        for point in param_points_each_sector[idx]:
            mean_each_sector[idx] += param_cropped_roi[point[0], point[1]]

        mean_each_sector[idx] /= len(param_points_each_sector[idx])

    for idx in range(len(param_points_each_sector)):
        for point in param_points_each_sector[idx]:
            pixel = param_cropped_roi[point[0], point[1]]
            variance_each_sector[idx] += math.pow((pixel - mean_each_sector[idx]), 2)

        variance_each_sector[idx] /= (len(param_points_each_sector[idx]) - 1)

    for idx in range(len(param_points_each_sector)):
        for point in param_points_each_sector[idx]:
            if variance_each_sector[idx] == 0:
                norm_sectors[point[0], point[1]] = target_mean
                #variance_each_sector[idx] = 1
            else:
                pixel_value = param_cropped_roi[point[0], point[1]]
                invariant_formula = math.sqrt((target_variance * math.pow((pixel_value - mean_each_sector[idx]), 2)) / variance_each_sector[idx])
                
                if pixel_value > mean_each_sector[idx]:
                    norm_sectors[point[0], point[1]] = target_mean + invariant_formula
                else:
                    norm_sectors[point[0], point[1]] = target_mean - invariant_formula

    return norm_sectors


def add_mask(param_points_each_sector, param_cropped_roi):
    #norm_sectors = np.zeros_like(param_cropped_roi)
    norm_sectors = np.ones_like(param_cropped_roi) * 255.0

    for idx in range(len(param_points_each_sector)):
        for point in param_points_each_sector[idx]:
            norm_sectors[point[0], point[1]] = param_cropped_roi[point[0], point[1]]
    
    return norm_sectors


def get_even_symmetric_gabor_filter(filter_idx, param_nr_filters):
    sigma = 4.0
    lambda_ = 10.0  # Wavelength of the sinusoidal factor
    psi = 0 #(90-180) * np.pi / 180.0 
    gamma = 1.0
    kernel_size = 33  # Ensure kernel size is odd

    theta = filter_idx * np.pi / param_nr_filters
    sin_vect = [i * np.sin(theta) for i in range(-16, 17)]
    cos_vect = [i * np.cos(theta) for i in range(-16, 17)]

    gabor_filter = np.zeros((kernel_size, kernel_size), dtype=np.float64)
    
    for i in range(kernel_size):
        for j in range(kernel_size):
            xx = sin_vect[i] + cos_vect[j]
            yy = -sin_vect[j] + cos_vect[i]

            gabor_filter[i, j] = np.exp(-((xx**2) + (yy**2)) / (2 * sigma**2)) * np.cos(2 * np.pi * xx / lambda_) 

    #gabor_filter = cv.getGaborKernel((kernel_size, kernel_size), sigma, theta, lambda_, gamma, psi, ktype=cv.CV_64F)
    return gabor_filter


def apply_filter(param_img, param_filter):    
    #return fftconvolve(param_img, param_filter, mode='same')
    #return convolve2d(param_img, param_filter, mode='same')
    #return convolve(param_img, param_filter)
    return cv.filter2D(param_img, cv.CV_64F, param_filter)


def determine_fingercode(param_img, param_points_each_sector):   
    #mean_each_sector = np.zeros(len(param_points_each_sector))
    #fingercode_vector = np.zeros(len(param_points_each_sector))
    mean_each_sector = np.zeros(len(param_points_each_sector), dtype=np.float64)
    fingercode_vector = np.zeros(len(param_points_each_sector), dtype=np.float64)

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
    #fingercode_image = np.zeros_like(param_img)
    fingercode_image = np.ones_like(param_img) * 255.0
    
    for idx in range(len(param_points_each_sector)):
        for point in param_points_each_sector[idx]:
            fingercode_image[point[0], point[1]] = param_fingercode[idx]

    return fingercode_image


def process_image(img):
    x_center, y_center = find_reference_point(img)
    #display_center_point(img, x_center, y_center)
    
    cropped_roi = crop_roi(h_roi, x_center, y_center, img)
    
    if cropped_roi.shape[0] != 0:
        points_each_sector = divide_into_sectors(h_roi, nr_sectors, band_width, nr_sectors_band)
        #plot_sectors(points_each_sector, h_roi, nr_bands, nr_sectors_band)
        #plot_circles_and_lines(h_roi, nr_sectors, band_width, nr_sectors_band, x_center, y_center, img)
        
        #cropped_roi = add_mask(points_each_sector, cropped_roi)
        
        norm_cropped_roi = normalize_sectors(points_each_sector, cropped_roi)
        
        # pt criptare
        fingercodes = []
        filtered_images = []
        # ca sa pot vedea cum arata imaginea
        fingercode_images = []

        for idx in range(nr_filters):
            gabor_filter = get_even_symmetric_gabor_filter(idx, nr_filters)
            
            filtered_roi = apply_filter(norm_cropped_roi.copy(), gabor_filter)
            filtered_roi = add_mask(points_each_sector, filtered_roi)
            filtered_images.append(filtered_roi)
            
            fingercode = determine_fingercode(filtered_roi, points_each_sector)
            fingercodes.append(fingercode)
            
            fingercode_image = create_fingercode_image(filtered_roi, fingercode, points_each_sector)
            #filtered_roi = add_mask(points_each_sector, filtered_roi)
            fingercode_images.append(fingercode_image)
        
        filtered_images = np.array(filtered_images, dtype=np.uint8)   
        #display_images_filtre(filtered_images)  

        fingercode_images = np.array(fingercode_images, dtype=np.uint8)   
        #display_images_coduri(fingercode_images)
        
        fingercodes_encrypted = [encrypt.ecrypt_fingercode(fingercode) for fingercode in fingercodes]
        
        return fingercodes_encrypted, fingercodes
    
    return [], []

director_path = "D:/Licenta/74034_3_En_4_MOESM1_ESM/FVC2004/Dbs/DB1_A/"
director_path_save = "C:/Users/stefa/Desktop/Amprente_centru1/"
image_extension = '*.tif'

images_path = os.path.join(director_path, image_extension)
files = glob.glob(images_path)

for i in range(len(files)):
    print('Procesare imagine nr. %d...' % i)
    img = cv.imread(files[i], cv.IMREAD_GRAYSCALE)

    fing, fing2 = process_image(img)