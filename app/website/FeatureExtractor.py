import numpy as np
import cv2 as cv
import math

class FeatureExtractor:
    def __init__(self):
        self.nr_filters = 8
        self.nr_bands = 5
        self.nr_sectors_band = 16 #k
        self.band_width = 20 #b
        self.nr_sectors = self.nr_bands * self.nr_sectors_band
        self.h_roi = 10 + 2 * ((self.nr_bands + 1) * self.band_width) - 1 #ca sa fie impar => mijlocul e centrul

        self.center_point_image = None
        self.x_center = None
        self.y_center = None

        self.cropped_roi = None
        self.sectors_img = None

        self.sectors = None
        self.fingercodes = None
        self.fingercodes_images = None
        self.filtered_images = None

    def find_reference_point(self, param_img):
        block_size = 8    
        variance_thresh = 20
        num_rows, num_cols = param_img.shape
        
        blurred_img = cv.GaussianBlur(param_img.copy(), (3, 3), 0)
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        blurred_img = clahe.apply(blurred_img)
        
        gradient_x, gradient_y = np.gradient(blurred_img)    
        nominator = (gradient_x + 1j * gradient_y)** 2   
        denominator = np.abs(nominator)

        grad_field = np.ones_like(param_img, dtype=complex)
        for i in range(param_img.shape[0]):
            for j in range(param_img.shape[1]):
                if denominator[i][j] != 0:
                    grad_field[i][j] = nominator[i][j] / denominator[i][j]

        grid_x, grid_y = np.meshgrid(np.arange(-16, 17), np.arange(-16, 17))
        exponent = np.exp(-(grid_x ** 2 + grid_y ** 2) / (2 * np.sqrt(60) ** 2))
        core_filter = exponent * (grid_x + 1j * grid_y)

        img_height, img_width = grad_field.shape
        filter_height, filter_width = core_filter.shape

        image_filtered = np.fft.ifft2(np.fft.fft2(grad_field, [img_height+filter_height-1, img_width+filter_width-1]) * np.fft.fft2(core_filter, [img_height+filter_height-1, img_width+filter_width-1]))
        
        px = ((filter_height - 1) + (filter_height - 1) % 2) // 2
        py = ((filter_width - 1) + (filter_width - 1) % 2) // 2
        image_filtered = np.abs(image_filtered[px:px+img_height, py:py+img_width])
        
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
        
        mask_variance = cv.morphologyEx(mask_variance, cv.MORPH_CLOSE, np.ones((10, 10), np.uint8))#, iterations=2)
        mask_variance = cv.morphologyEx(mask_variance, cv.MORPH_ERODE, np.ones((44, 44), np.uint8))
        mask_variance = cv.morphologyEx(mask_variance, cv.MORPH_ERODE, np.ones((40, 40), np.uint8), iterations=3)
        
        mask = image_filtered * mask_variance

        self.y_center, self.x_center = np.unravel_index(np.argmax(mask), mask.shape)
        
        self.center_point_image = cv.cvtColor(param_img, cv.COLOR_GRAY2BGR)  
        cv.drawMarker(self.center_point_image, (self.x_center, self.y_center), color=(255, 255, 0), markerType=cv.MARKER_TILTED_CROSS, markerSize=25, thickness=2)

    def crop_roi(self, param_img):
        img_height, img_width  = param_img.shape

        if (self.y_center - self.h_roi//2 < 0) or (self.y_center + self.h_roi//2 > img_height - 1) or (self.x_center - self.h_roi//2 < 0) or (self.x_center + self.h_roi//2 > img_width - 1):
            return np.array([]) 
        else:
            return param_img[self.y_center - self.h_roi//2 : self.y_center + self.h_roi//2 + 1,
                        self.x_center - self.h_roi//2 : self.x_center + self.h_roi//2 + 1]
        
    def plot_circles_and_lines(self, param_image):
        self.sectors_img = cv.cvtColor(param_image, cv.COLOR_GRAY2RGB)  
        cv.drawMarker(self.sectors_img, (self.x_center, self.y_center), color=(255, 255, 0), markerType=cv.MARKER_TILTED_CROSS, markerSize=20, thickness=2)

        for i in range(self.nr_bands + 1):
            radius = (i + 1) * self.band_width
            cv.circle(self.sectors_img, (self.x_center, self.y_center), radius, (255, 0, 0), 1, cv.LINE_AA)

        vect_angles = [(i % self.nr_sectors_band) * (2 * 180.0 / self.nr_sectors_band) for i in range(self.nr_sectors)]

        for angle in vect_angles:
            rad_angle = math.radians(angle)
            x_end = int(self.x_center + ((self.h_roi - 10) / 2) * np.cos(rad_angle))
            y_end = int(self.y_center + ((self.h_roi - 10) / 2) * np.sin(rad_angle))
            x_c = int(self.x_center + self.band_width * np.cos(rad_angle))
            y_c = int(self.y_center + self.band_width * np.sin(rad_angle))

            cv.line(self.sectors_img, (x_c, y_c), (x_end, y_end), (255, 0, 0), 1, cv.LINE_AA)

    def divide_into_sectors(self):
        vect_T = []
        vect_angles = []

        x_center, y_center = self.h_roi // 2, self.h_roi // 2

        for i in range(self.nr_sectors + 1):
            vect_T.append(i // self.nr_sectors_band)
            vect_angles.append((i % self.nr_sectors_band) * (2 * 180.0 / self.nr_sectors_band))

        self.sectors = []

        for i in range(self.nr_sectors):
            sector = []
            
            for x in range(self.h_roi):
                for y in range(self.h_roi):
                    x0 = x - x_center
                    y0 = y - y_center
                    r = math.sqrt(x0**2 + y0**2)                  

                    if self.band_width * (vect_T[i] + 1) <= r < self.band_width * (vect_T[i] + 2):
                        theta = math.degrees(math.atan2(y0, x0))
                        if theta < 0:
                            theta += 360.0  
                        
                        if i % self.nr_sectors_band == self.nr_sectors_band - 1 and i > 0:
                            if vect_angles[i] <= theta:
                                sector.append([x, y])
                        else:
                            if vect_angles[i] <= theta < vect_angles[i+1]:
                                sector.append([x, y])
            
            self.sectors.append(sector)
        
        return self.sectors

    def normalize_sectors(self, param_cropped_roi, target_mean=100.0, target_variance=100.0):
        mean_each_sector = np.zeros(len(self.sectors))
        variance_each_sector = np.zeros(len(self.sectors))
        norm_sectors = np.zeros_like(param_cropped_roi)
        #norm_sectors = param_cropped_roi.copy()

        for idx in range(len(self.sectors)):
            for point in self.sectors[idx]:
                mean_each_sector[idx] += param_cropped_roi[point[0], point[1]]

            mean_each_sector[idx] /= len(self.sectors[idx])

        for idx in range(len(self.sectors)):
            for point in self.sectors[idx]:
                pixel = param_cropped_roi[point[0], point[1]]
                variance_each_sector[idx] += math.pow((pixel - mean_each_sector[idx]), 2)

            variance_each_sector[idx] /= (len(self.sectors[idx]) - 1)

        for idx in range(len(self.sectors)):
            for point in self.sectors[idx]:
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

    def add_mask(self, param_cropped_roi):
        #norm_sectors = np.zeros_like(param_cropped_roi)
        norm_sectors = np.ones_like(param_cropped_roi) * 255.0

        for idx in range(len(self.sectors)):
            for point in self.sectors[idx]:
                norm_sectors[point[0], point[1]] = param_cropped_roi[point[0], point[1]]
        
        return norm_sectors

    def get_even_symmetric_gabor_filter(self, filter_idx):
        sigma = 4.0
        lambda_ = 10.0 
        psi = (90-180) * np.pi / 180.0 
        gamma = 1.0
        kernel_size = 33 

        theta = filter_idx * np.pi / self.nr_filters
        sin_vect = [i * np.sin(theta) for i in range(-16, 17)]
        cos_vect = [i * np.cos(theta) for i in range(-16, 17)]

        gabor_filter = np.zeros((kernel_size, kernel_size), dtype=np.float64)
        
        for i in range(kernel_size):
            for j in range(kernel_size):
                xx = sin_vect[i] + cos_vect[j]
                yy = -sin_vect[j] + cos_vect[i]

                gabor_filter[i, j] = np.exp(-((xx**2) + (yy**2)) / (2 * sigma**2)) * np.cos(2 * np.pi * xx / lambda_) 

        return gabor_filter  

    @staticmethod
    def apply_filter(param_img, param_filter):  
        img_height, img_width = param_img.shape
        filter_height, filter_width = param_filter.shape

        if np.any(np.any(np.imag(param_img))) or np.any(np.any(np.imag(param_filter))):
                filtered_image = np.fft.ifft2(np.fft.fft2(param_img, [img_height+filter_height-1, img_width+filter_width-1]) * np.fft.fft2(param_filter, [img_height+filter_height-1, img_width+filter_width-1]))
        else:
            filtered_image = np.real(np.fft.ifft2(np.fft.fft2(param_img, [img_height+filter_height-1, img_width+filter_width-1]) * np.fft.fft2(param_filter, [img_height+filter_height-1, img_width+filter_width-1])))

        px = ((filter_height - 1) + (filter_height - 1) % 2) // 2
        py = ((filter_width - 1) + (filter_width - 1) % 2) // 2
        
        filtered_image = filtered_image[px:px+img_height, py:py+img_width]
        
        return filtered_image

    def determine_fingercode(self, param_img):   
        #mean_each_sector = np.zeros(len(self.sectors))
        #fingercode_vector = np.zeros(len(self.sectors))
        mean_each_sector = np.zeros(len(self.sectors), dtype=np.float64)
        fingercode_vector = np.zeros(len(self.sectors), dtype=np.float64)
        #fingercode_vector = np.zeros(len(self.sectors), dtype=np.float64)

        for idx in range(len(self.sectors)):
            for point in self.sectors[idx]:
                mean_each_sector[idx] += param_img[point[0], point[1]]

            mean_each_sector[idx] /= len(self.sectors[idx])

        for idx in range(len(self.sectors)):
            for point in self.sectors[idx]:
                fingercode_vector[idx] += np.abs(param_img[point[0], point[1]] - mean_each_sector[idx])
                #fingercode_vector[idx] += (param_img[point[0], point[1]] - mean_each_sector[idx])**2

            fingercode_vector[idx] /= len(self.sectors[idx])
            #fingercode_vector[idx] /= len(self.sectors[idx])
            #fingercode_vector[idx] = np.sqrt(self.sectors[idx])

        return fingercode_vector

    def create_fingercode_image(self, param_img, param_fingercode):
        #fingercode_image = np.zeros_like(param_img)
        fingercode_image = np.ones((self.h_roi, self.h_roi), dtype=np.uint8) * 255.0
        
        for idx in range(len(self.sectors)):
            for point in self.sectors[idx]:
                fingercode_image[point[0], point[1]] = param_fingercode[idx]

        return fingercode_image
    
    def get_cropped_roi(self, img):
        self.cropped_roi = self.crop_roi(img)
        
        if self.cropped_roi.shape[0] != 0:
            self.divide_into_sectors()
            self.plot_circles_and_lines(img)
        
        return self.cropped_roi

    def continue_process(self):
        norm_cropped_roi = self.normalize_sectors(self.cropped_roi)
            
        self.fingercodes = []
        self.fingercodes_images = []
        self.filtered_images = []

        for idx in range(self.nr_filters):
            gabor_filter = self.get_even_symmetric_gabor_filter(idx)

            '''gabor_filter_print = (gabor_filter - np.min(gabor_filter, axis=(0, 1))) /  (np.max(gabor_filter, axis=(0, 1)) -  np.min(gabor_filter, axis=(0, 1)))
            cv.imshow("gabor 2D", (gabor_filter_print * 255.0).astype(np.uint8))
            cv.waitKey()
            cv.destroyAllWindows()'''

            filtered_roi = self.apply_filter(norm_cropped_roi.copy(), gabor_filter)
            #filtered_roi = self.add_mask(filtered_roi)
            self.filtered_images.append(filtered_roi)
            
            fingercode = self.determine_fingercode(filtered_roi)
            #fingercode = self.add_mask(fingercode)
            self.fingercodes.append(fingercode)
            self.fingercodes_images.append(self.create_fingercode_image(filtered_roi, fingercode))

        clear_fingercode = np.array([code for fingercode in self.fingercodes for code in fingercode])
        return clear_fingercode
        
    def process_image(self, param_image):
        self.find_reference_point(param_image)
        self.cropped_roi = self.crop_roi(param_image)
        #self.cropped_roi = add_mask(points_each_sector, self.cropped_roi)

        if self.cropped_roi.shape[0] != 0:
            self.divide_into_sectors()
            norm_cropped_roi = self.normalize_sectors(self.cropped_roi)
            
            self.fingercodes = []
            self.filtered_images = []

            for idx in range(self.nr_filters):
                gabor_filter = self.get_even_symmetric_gabor_filter(idx)

                '''gabor_filter_print = (gabor_filter - np.min(gabor_filter, axis=(0, 1))) /  (np.max(gabor_filter, axis=(0, 1)) -  np.min(gabor_filter, axis=(0, 1)))
                cv.imshow("gabor 2D", (gabor_filter_print * 255.0).astype(np.uint8))
                cv.waitKey()
                cv.destroyAllWindows()'''

                filtered_roi = self.apply_filter(norm_cropped_roi.copy(), gabor_filter)
                #filtered_roi = self.add_mask(filtered_roi)
                self.filtered_images.append(filtered_roi)
                
                fingercode = self.determine_fingercode(filtered_roi)
                #fingercode = self.add_mask(fingercode)
                self.fingercodes.append(fingercode)

            clear_fingercode = np.array([code for fingercode in self.fingercodes for code in fingercode])
            return clear_fingercode
        
        return []