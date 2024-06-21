import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2 as cv
from RecognitionSystem import FingRecognitionSystem

class FingerprintRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistem de recunoaștere pe baza amprentei")
        self.root.state("zoomed")
        
        self.image_frame = tk.Frame(self.root)
        self.image_frame.place(relx=0.5, rely=0.5, anchor="center")

        self.distances_frame = tk.Frame(self.root)
        self.distances_frame.place_forget()
        
        self.select_btn = tk.Button(self.root, text="Selectați amprenta", font=("Helvetica", 11), command=self.open_image)
        self.select_btn.place(relx=0.5, rely=0.3, anchor="center")
        
        self.fing_rec_syst = FingRecognitionSystem()
        self.selected_image = None 

        self.result_label = tk.Label(self.root, text="", font=("Helvetica", 14), fg="black")
        self.result_label.place(relx=0.5, rely=0.05, anchor="center")

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
            self.show_error_message("Eroare", "Amprentă plasată incorect! Selectați altă imagine.")

    def extract_and_match_fingercode(self):
        self.fing_rec_syst.extract_fingercode_app()
        self.root.after(0, self.see_match_enc)
    
    def see_match_enc(self):
        match_path, match_fingercodes_image, selected_fingercodes_image, clear_dist, enc_dist = self.fing_rec_syst.match()     

        if match_path != '':
            self.result_label.config(text="Utilizator identificat", fg="green")
            selected_img = self.selected_image  # Imaginea selectată
            match_img = cv.imread(match_path, cv.IMREAD_GRAYSCALE)  # Imaginea găsită

            #self.display_images( "Amprenta selectată", "Amprenta găsită în baza de date")
            self.display_final_result(selected_img, match_img, selected_fingercodes_image, match_fingercodes_image, self.image_frame, "Amprenta selectată", "Amprenta găsită în baza de date")
            self.display_distances(clear_dist, enc_dist)

            self.reset_btn = tk.Button(self.root, text="Resetați", font=("Helvetica", 11), command=self.reset_interface)
            self.reset_btn.place(relx=0.5, rely=0.9, anchor="center")
        else:
            self.show_error_message("Eroare", "Utilizatorul nu este înregistrat în baza de date!")

    def show_error_message(self, title, message):
        messagebox.showerror(title, message)
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
        img_label.image = photo  # Mențin o referință pentru a preveni garbage collection
        img_label.pack()

    def display_images(self, img_1, img_2, frame, descr_1, descr_2):
        resized_img_1 = ImageTk.PhotoImage(image=Image.fromarray(cv.resize(img_1, (350, 350))))
        resized_img_2 = ImageTk.PhotoImage(image=Image.fromarray(cv.resize(img_2, (350, 350))))

        for widget in frame.winfo_children():
            widget.destroy()

        description_1 = tk.Label(frame, text=descr_1, font=("Helvetica", 12))
        description_1.grid(row=0, column=0, padx=(10, 100), pady=(10, 5))  

        label_1 = tk.Label(frame, image=resized_img_1)
        label_1.image = resized_img_1  
        label_1.grid(row=1, column=0, padx=(10, 100), pady=10)  

        description_2 = tk.Label(frame, text=descr_2, font=("Helvetica", 12))
        description_2.grid(row=0, column=1, padx=(100, 10), pady=(10, 5)) 

        label_2 = tk.Label(frame, image=resized_img_2)
        label_2.image = resized_img_2 
        label_2.grid(row=1, column=1, padx=(100, 10), pady=10)

    def display_final_result(self, img_1, img_2, fingercode_1, fingercode_2, frame, descr_1, descr_2):
        resized_img_1 = ImageTk.PhotoImage(image=Image.fromarray(cv.resize(img_1, (350, 300))))
        resized_img_2 = ImageTk.PhotoImage(image=Image.fromarray(cv.resize(img_2, (350, 300))))
        resized_fingercode_1 = ImageTk.PhotoImage(image=Image.fromarray(cv.resize(fingercode_1, (500, 250))))
        resized_fingercode_2 = ImageTk.PhotoImage(image=Image.fromarray(cv.resize(fingercode_2, (500, 250))))

        for widget in frame.winfo_children():
            widget.destroy()

        description_1 = tk.Label(frame, text=descr_1, font=("Helvetica", 12))
        description_1.grid(row=0, column=0, padx=(10, 150), pady=(5, 5))

        label_1 = tk.Label(frame, image=resized_img_1)
        label_1.image = resized_img_1
        label_1.grid(row=1, column=0, padx=(10, 150), pady=5)

        label_fingercode_1 = tk.Label(frame, image=resized_fingercode_1)
        label_fingercode_1.image = resized_fingercode_1
        label_fingercode_1.grid(row=2, column=0, padx=(10, 150), pady=35)

        description_2 = tk.Label(frame, text=descr_2, font=("Helvetica", 12))
        description_2.grid(row=0, column=1, padx=(150, 10), pady=(5, 5))

        label_2 = tk.Label(frame, image=resized_img_2)
        label_2.image = resized_img_2
        label_2.grid(row=1, column=1, padx=(150, 10), pady=5)

        label_fingercode_2 = tk.Label(frame, image=resized_fingercode_2)
        label_fingercode_2.image = resized_fingercode_2
        label_fingercode_2.grid(row=2, column=1, padx=(150, 10), pady=35)

    def display_distances(self, clear_distance, encrypted_distance):
        self.distances_frame.place(relx=0.5, rely=0.35, anchor="center")
        
        clear_distance_label = tk.Label(self.distances_frame, text=f"Distanța în domeniul clar: {clear_distance}", font=("Helvetica", 12), fg='black')
        clear_distance_label.pack()
        
        encrypted_distance_label = tk.Label(self.distances_frame, text=f"Distanța în domeniul criptat: {encrypted_distance}", font=("Helvetica", 12), fg='black')
        encrypted_distance_label.pack() 

    def reset_interface(self):
        for widget in self.image_frame.winfo_children():
            widget.destroy()
        
        for widget in self.distances_frame.winfo_children():
            widget.destroy()
        
        self.image_frame.configure(borderwidth=0, relief="flat")
        self.select_btn.place(relx=0.5, rely=0.3, anchor="center")

        self.result_label.config(text="")
        self.fing_rec_syst = FingRecognitionSystem()
        self.selected_image = None 
        
        if hasattr(self, 'reset_btn'):
            self.reset_btn.destroy()
        
if __name__ == "__main__":
    root = tk.Tk()
    app = FingerprintRecognitionApp(root)
    root.mainloop()