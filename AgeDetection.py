import cv2
import tkinter as tk
from tkinter import ttk
from keras.models import load_model
import numpy as np
from PIL import Image, ImageTk
from keras.applications.resnet50 import preprocess_input
from sklearn.preprocessing import LabelEncoder
from tkinter import messagebox
import time

CAMERA_INDEX = 0
FACE_CASCADE_PATH = 'haarcascade_frontalface_default.xml'
MODEL_PATH = 'BestModelResNet50.h5'
PRICE = {"Kid": "100 Baht", "Adult": "200 Baht", "OldPeople": 'Free'}

model = load_model(MODEL_PATH)
label_encoder = LabelEncoder()
label_encoder.fit(['Kid', 'Adult', 'OldPeople'])

def preprocess_image(image):
    image = cv2.resize(image, (224, 224))
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    return image

def print_ticket():
    print(f"Printing Ticket for detected {age_category}")
    messagebox.showinfo("Print", f"Printing Ticket for detected {age_category}")

def print_custom_ticket():
    selected_age_category = age_category_combobox.get()
    print(f"Printing Custom Ticket for {selected_age_category}.")
    messagebox.showinfo("Print", f"Printing Custom Ticket for {selected_age_category}")

def update_date_label(label):
    current_date = time.strftime('%Y-%m-%d')  
    label.config(text=f"{current_date}")
    label.after(60000, update_date_label, label) 

def update_time_label(label):
    current_time = time.strftime('%H:%M:%S')  
    label.config(text=f"{current_time}")
    label.after(1000, update_time_label, label)

def detect_and_estimate_age(frame, detected_age_category, ticket_price):
    try:
        face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            detected_age_category.set("No face detected")
            ticket_price.set("Ticket Price: N/A")
        else:
            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]
                face = preprocess_image(face)
                predictions = model.predict(face)
                age_category_idx = np.argmax(predictions, axis=1)
                global age_category 
                age_category = label_encoder.classes_[age_category_idx[0]]

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, age_category, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                detected_age_category.set(f"{age_category}")
                ticket_price.set(f"{'Free' if age_category == 'OldPeople' else PRICE[age_category]}")
    
        return frame
    except Exception as e:
        detected_age_category.set(f"Error: {str(e)}")
        ticket_price.set("Ticket Price: N/A")

def main():

    global age_category_combobox
    root = tk.Tk()
    root.title("Face Age Detection Ticketing System")
    root.geometry("1220x650")

    style = ttk.Style(root)
    style.configure('TButton', font=('JetBrains Mono', 10), background='light blue', padding=10)
    style.configure('TLabel', font=('JetBrains Mono', 12), padding=10, background='light grey')
    style.configure('TCombobox', font=('JetBrains Mono', 10), padding=10)

    main_frame = ttk.Frame(root)
    main_frame.grid(column=0, row=0, padx=10, pady=10, sticky="nsew")
    
    video_label = ttk.Label(main_frame)
    video_label.grid(column=0, row=0, rowspan=8, padx=10)
    
    right_frame = ttk.Frame(main_frame)
    right_frame.grid(column=1, row=0, rowspan=8)
    right_frame.grid_propagate(False) 
    right_frame.config(width=350, height=600)  
    
    for i in range(8):  
        right_frame.grid_rowconfigure(i, weight=1)
    right_frame.grid_columnconfigure(0, weight=1)
    
    ticket_details_label = ttk.Label(right_frame, text="รายละเอียดตั๋ว (Ticket Details)",background='#f0f0f0')
    ticket_details_label.grid(column=0, row=0,columnspan=2) 
    
    date_details_label = ttk.Label(right_frame, text="วันที่ (Data):",background='#f0f0f0')
    date_details_label.grid(column=0, row=1, sticky='e') 
    date_label = ttk.Label(right_frame, width=20,background="#ffffff")
    date_label.grid(column=1, row=1,sticky='w')
    update_date_label(date_label)
    
    time_details_label = ttk.Label(right_frame, text="เวลา (Time)",background='#f0f0f0')
    time_details_label.grid(column=0, row=2, sticky='e') 
    time_label = ttk.Label(right_frame, width=20,background="#ffffff")
    time_label.grid(column=1, row=2,sticky='w')
    update_time_label(time_label)

    detected_age_details_label = ttk.Label(right_frame, text="ประเภทของตั๋ว (Ticket Type):",background='#f0f0f0')
    detected_age_details_label.grid(column=0, row=3,columnspan=2) 
    detected_age_category = tk.StringVar()
    detected_age_label = ttk.Label(right_frame, textvariable=detected_age_category,background="#ffffff")
    detected_age_label.grid(column=0, row=4,columnspan=2) 
    
    ticket_price_details_label = ttk.Label(right_frame, text="ราคาตั๋ว (Ticket Price):",background='#f0f0f0')
    ticket_price_details_label.grid(column=0, row=5,columnspan=2) 
    ticket_price = tk.StringVar()
    ticket_price_label = ttk.Label(right_frame, textvariable=ticket_price,background="#ffffff")
    ticket_price_label.grid(column=0, row=6,columnspan=2)
    
    print_button = ttk.Button(right_frame, text="Print Ticket", command=print_ticket)
    print_button.grid(column=0, row=7,columnspan=2)
    
    age_category_combobox = ttk.Combobox(right_frame, values=("Kid", "Adult", "OldPeople"))
    age_category_combobox.grid(column=0, row=8, pady= 10,columnspan=2)
    age_category_combobox.configure(font=('JetBrains Mono', 10))
    age_category_combobox.set("เลือกช่วงอายุของที่ต้องการ")
    
    print_custom_button = ttk.Button(right_frame, text="Print Custom Ticket", command=print_custom_ticket)
    print_custom_button.grid(column=0, row=9,columnspan=2)
    
    cap = cv2.VideoCapture(CAMERA_INDEX)
    
    if not cap.isOpened():
        detected_age_category.set("Error: Camera not found")
        ticket_price.set("Ticket Price: N/A")
    else:
        def update_gui():
            ret, frame = cap.read()
            if ret:
                frame = detect_and_estimate_age(frame, detected_age_category, ticket_price)
                
   
                height, width = frame.shape[:2]
                new_width = 800
                new_height = int((new_width / width) * height)
                frame = cv2.resize(frame, (new_width, new_height))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
                video_label.config(image=photo)
                video_label.image = photo

                root.after(10, update_gui)
            else:
                detected_age_category.set("Error: Can't receive frame")
                ticket_price.set("Ticket Price: N/A")

    def on_closing():
        if cap.isOpened():
            cap.release()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    update_gui()
    root.mainloop()

if __name__ == "__main__":
    main()