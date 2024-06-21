import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2 as cv
import threading
from RecognitionSystem import FingRecognitionSystem

class FingerprintRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistem de recunoaștere pe baza amprentei")
        self.root.state("zoom")  # Adjust window size to be wider than tall
        
        self.image_frame = tk.Frame(self.root)
        self.image_frame.place(relx=0.5, rely=0.4, anchor="center")

        self.distances_frame = tk.Frame(self.root)
        self.distances_frame.place_forget()
        
        self.select_btn = tk.Button(self.root, text="Selectați amprenta", font=("Helvetica", 10), command=self.open_image)
        self.select_btn.place(relx=0.5, rely=0.3, anchor="center")
        
        self.fing_rec_syst = FingRecognitionSystem()
        self.selected_image = None 

    def open_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.tif;*.png;*.jpg;*.jpeg;*.bmp")])
        
        if file_path:
            fingerprint_img = cv.imread(file_path, cv.IMREAD_GRAYSCALE)
            self.selected_image = fingerprint_img
            
            self.fing_rec_syst.find_center_point(fingerprint_img)
            self.select_btn.place_forget()
            self.display_image(self.fing_rec_syst.feature_extractor.center_point_image, self.image_frame, "Centrul evidențiat în amprenta selectată")
            self.root.update()
            
            self.root.after(0, self.get_roi, self.selected_image)

    def get_roi(self, image):
        self.fing_rec_syst.determine_cropped_roi(image)

        if self.fing_rec_syst.cropped_roi.shape[0] != 0:
            self.display_images(self.fing_rec_syst.feature_extractor.center_point_image, self.fing_rec_syst.feature_extractor.sectors_img, self.image_frame, "Centrul evidențiat în amprenta selectată", "Sectoare evidențiate în amprenta selectată")
            self.root.update()

            self.root.after(0, self.extract_and_match_fingercode)
        else:
            self.root.after(0, messagebox.showerror, "Eroare", "Amprentă plasată incorect! Selectați altă imagine.")
            self.root.after(3000, self.reset_interface)

    def extract_and_match_fingercode(self):
        self.fing_rec_syst.extract_fingercode_app()
        self.root.after(0, self.see_match_enc)
    
    def see_match_enc(self):
        result, clear_dist, enc_dist = self.fing_rec_syst.match()     

        if result != '':
            original_img = self.selected_image  # Imaginea originală selectată
            match_img = cv.imread(result, cv.IMREAD_GRAYSCALE)  # Imaginea găsită

            self.display_images(original_img, match_img, self.image_frame, "Amprenta selectată", "Amprenta găsită în baza de date")
            self.display_distances(clear_dist, enc_dist)

            self.reset_btn = tk.Button(self.root, text="Resetați", font=("Helvetica", 10), command=self.reset_interface)
            self.reset_btn.place(relx=0.5, rely=0.9, anchor="center")
        else:
            messagebox.showerror("Eroare", "Utilizatorul nu este înregistrat în baza de date!")
            self.reset_interface()
    
    def display_image(self, img, frame, descr):
        image = Image.fromarray(img)
        image.thumbnail((350, 350)) 
        photo = ImageTk.PhotoImage(image)
        
        for widget in frame.winfo_children():
            widget.destroy()

        image_description = tk.Label(frame, text=descr, font=("Helvetica", 12))
        image_description.pack(pady=(10, 5))

        img_label = tk.Label(frame, image=photo)
        img_label.image = photo  # Menținem o referință pentru a preveni garbage collection
        img_label.pack()

    def display_images(self, img_1, img_2, frame, descr_1, descr_2):
        resized_img_1 = ImageTk.PhotoImage(image=Image.fromarray(cv.resize(img_1, (350, 350))))
        resized_img_2 = ImageTk.PhotoImage(image=Image.fromarray(cv.resize(img_2, (350, 350))))

        for widget in frame.winfo_children():
            widget.destroy()

        # Descrierea pentru imaginea originală
        description_1 = tk.Label(frame, text=descr_1, font=("Helvetica", 12))
        description_1.grid(row=0, column=0, padx=(10, 100), pady=(10, 5))  

        # Afișăm imaginea originală sub descriere
        label_1 = tk.Label(frame, image=resized_img_1)
        label_1.image = resized_img_1  
        label_1.grid(row=1, column=0, padx=(10, 100), pady=10)  

        # Descrierea pentru imaginea de fingercode găsită
        description_2 = tk.Label(frame, text=descr_2, font=("Helvetica", 12))
        description_2.grid(row=0, column=1, padx=(100, 10), pady=(10, 5)) 

        # Afișăm imaginea de fingercode găsită sub descriere
        label_2 = tk.Label(frame, image=resized_img_2)
        label_2.image = resized_img_2 
        label_2.grid(row=1, column=1, padx=(100, 10), pady=10) 

    def display_fingercode_images(self, fingercode_images, frame):
        for widget in frame.winfo_children():
            widget.destroy()

        photo_images = []

        for fingercode_img in fingercode_images:
            image = Image.fromarray(fingercode_img)
            image.thumbnail((125, 125))
            photo = ImageTk.PhotoImage(image)
            photo_images.append(photo)

            img_label = tk.Label(frame, image=photo)
            img_label.image = photo  
            img_label.grid(row=0, column=len(photo_images)-1, padx=5, pady=5)

        frame.grid_columnconfigure(len(photo_images), weight=1)
    
    def display_distances(self, clear_distance, encrypted_distance):
        self.distances_frame.place(relx=0.5, rely=0.8, anchor="center")
        
        clear_distance_label = tk.Label(self.distances_frame, text=f"Distanța în domeniul clar: {clear_distance}", font=("Helvetica", 12))
        clear_distance_label.pack()
        
        encrypted_distance_label = tk.Label(self.distances_frame, text=f"Distanța în domeniul criptat: {encrypted_distance}", font=("Helvetica", 12))
        encrypted_distance_label.pack()
    
    def reset_interface(self):
        for widget in self.image_frame.winfo_children():
            widget.destroy()
        
        for widget in self.distances_frame.winfo_children():
            widget.destroy()
        
        self.image_frame.configure(borderwidth=0, relief="flat")
        self.select_btn.place(relx=0.5, rely=0.3, anchor="center")

        self.fing_rec_syst = FingRecognitionSystem()
        self.selected_image = None 
        
        if hasattr(self, 'reset_btn'):
            self.reset_btn.destroy()
        
if __name__ == "__main__":
    root = tk.Tk()
    app = FingerprintRecognitionApp(root)
    root.mainloop()