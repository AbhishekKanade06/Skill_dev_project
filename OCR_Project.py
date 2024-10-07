import tkinter as tk
from tkinter import filedialog, messagebox
from paddleocr import PaddleOCR
import pandas as pd
import cv2
import os
from paddleocr import PaddleOCR
import matplotlib.pyplot as plt
import numpy as np
from keras._tf_keras.keras.preprocessing import image
from keras._tf_keras.keras.models import load_model
# Initialize PaddleOCR model
ocr = PaddleOCR(use_angle_cls=True, lang='en')
model=load_model('P&A_model.h5')
data=[]
ocr_data=[]
model_data=[]
#File Path Input
def select_file(): 
    global img_path
    img_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if img_path:
        path_label.config(text=f"Selected Image: {img_path}")
def preprocess_image(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply binary thresholding (invert colors)
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    return image, binary

def detect_lines_and_draw(image, binary_image):
    # Define the kernel size based on image width
    kernel_length = np.array(binary_image).shape[1] // 150

    # Detect vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
    vertical_lines = cv2.erode(binary_image, vertical_kernel, iterations=3)
    vertical_lines = cv2.dilate(vertical_lines, vertical_kernel, iterations=4)

    # Detect horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
    horizontal_lines = cv2.erode(binary_image, horizontal_kernel, iterations=3)
    horizontal_lines = cv2.dilate(horizontal_lines, horizontal_kernel, iterations=4)

    # Combine vertical and horizontal lines to get a grid
    grid = cv2.addWeighted(vertical_lines, 1, horizontal_lines, 1, 0.0)

    # Find contours in the grid image
    contours, _ = cv2.findContours(grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours row-wise and column-wise
    sorted_contours = sorted(contours, key=lambda ctr: (cv2.boundingRect(ctr)[1], cv2.boundingRect(ctr)[0]))

    # Draw bounding rectangles around detected cells
    for contour in sorted_contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Draw a red rectangle around the detected cell
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return image, sorted_contours

def process_cells(image, contours, output_dir):
    f=True
    cell_images = []
    row_index = 0
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        print(x,y,w,h)
        # Extract the cell from the original image
        cell = image[y:y + h, x:x + w]

        test_exract(cell,f)
        f=False
        
        # Update row index based on position
        if i < len(contours) - 1:
            _, next_y, _, _ = cv2.boundingRect(contours[i + 1])
            if next_y > y:  # Check if the next contour is in the next row
                row_index += 1

    return cell_images
def test_exract(cell,f):
    h,w,c=cell.shape
    if(h <=40 or w<=40 or h >=1800 or w >=1800):
        return
    if(f):
        return
    result=ocr.ocr(cell,cls=True)
    try:
        data.append(result[0][0][1][0])
        ocr_data.append(result[0][0][1][0])
    except :
        y=PA_model(cell)
        print(y)
        if y=='P' :
            data.append('P') 
            model_data.append('P')        
def PA_model(cell):
    x=cell
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    cell = cv2.filter2D(cell, -1, kernel)
    cell=cv2.resize(cell,(225,225))
    cell=np.expand_dims(cell, axis=0)
    predictions = model.predict(cell)
    predicted_class = np.argmax(predictions[0])

    class_labels ={0: 'Other1', 1: 'P'}
    return class_labels[predicted_class]

def main():
    # Preprocess the image
    image, binary_image = preprocess_image(img_path)

    # Detect lines and draw red rectangles around the cells
    image_with_lines, contours = detect_lines_and_draw(image, binary_image)

    # Save the output image with red bounding boxes
    output_image_path = 'image_with_lines.png'
    cv2.imwrite(output_image_path, image_with_lines)

    # Create output directory for individual cells
    output_cells_dir = 'output_cells'
    img=cv2.imread(img_path)
    # Save individual cells in the output directory
    process_cells(img, contours, output_cells_dir)
    print(data)
    print()
    print("OCR")
    print(ocr_data)
    print()
    print("Model")
    print(model_data)
    print()
    print(P_first(data))


    data_xl_1(data)

def P_first(data):
    name=False
    p=0
    for i in data:
        if not name and i.lower()=='name':
            name=True
        if name and '080' in i and p<2:
            return False
        if name and i=='P':
            p+=1
        if name and '080' in i and p>2:
            return True

def no_days(data):
        days=0
        for i in data:
            if i.isdigit():
                if(int(i)>31):
                    days+=2
                else:
                    days+=1
                
            elif "0801" in i:
                break
        return days


def data_xl_1(data):
    days=no_days(data)
    print(days)
    enroll=[]
    name=[]
    attendance=[]
    flag=False
    flag1=False
    counter=0
    if P_first(data):
        for i in range(0,len(data)):
            if data[i].lower()=='name':
                flag=True    
            if flag and len(data[i]) > 8:
                if '080' in data[i] :
                    enroll.append(data[i])
                else:
                    name.append(data[i])  
                    
                    attendance.append(counter)
                    
                    counter=0
            if ('P' in data[i] or data[i]=='d'or data[i]=='a') and flag:
                counter+=1 
    else:
        for i in range(0,len(data)):
            if data[i].lower()=='name':
                flag=True    
            if flag and len(data[i]) > 8:
                if '080' in data[i] :
                    enroll.append(data[i])
                else:
                    name.append(data[i]) 
                    if flag1:
                        attendance.append(counter)
                        counter=0
                    flag1=True    
            if ('P' in data[i] or data[i]=='d'or data[i]=='a') and flag1:
                counter+=1 

    print(enroll,len(enroll))
    print(name,len(name))
    print(attendance,len(attendance),sum(attendance))
    if(len(name)>len(enroll)):
        enroll =enroll+[enroll[1][0:6]+"______"]*(len(name)-len(enroll))
    if(len(name)<len(enroll)):
        name =name+[np.nan]*(-len(name)+len(enroll))    
    if(len(attendance)< len(name)):
        attendance=attendance+[0]*(len(name)-len(attendance))
    datafram=pd.DataFrame({
        "Enroll_no":enroll,
        "Name":name,
        "attendance":attendance
    })
    datafram["Percent"]=(datafram["attendance"]/days)*100 
    print(datafram)
    #dialogbox for savefile name
    excel_path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel files", "*.xlsx")])

    if excel_path:
        datafram.to_excel(excel_path, index=False)
        messagebox.showinfo("Success", "File created successfully!")


    dataframe_text.delete(1.0, tk.END)  
    dataframe_text.insert(tk.END, datafram.to_string(index=False))
    

#exit function
def exit_program():
    root.destroy()

#GUI Part
root = tk.Tk()
root.title("Attendance")
root.geometry("600x600")
root.configure(bg="#D0F0C0")  


font_style = ("Helvetica", 12)
button_style = {
    "bg": "black",   
    "fg": "blue",   
    "activebackground": "#333333",  
    "font": ("Helvetica", 12, "bold"),
    "width": 20,
    "bd": 0,
    "relief": "flat",
    
}
button_style1 = {
    "bg": "black",   
    "fg": "black",   
    "activebackground": "#333333",  
    "font": ("Helvetica", 12, "bold"),
    "width": 10,
    "bd": 0,
    "relief": "flat",
   
}
label_style = {
    "bg": "#D0F0C0",
    "fg": "#333333",
    "font": font_style
}


title_label = tk.Label(root, text="Welcome", font=("Helvetica", 16, "bold"), bg="#D0F0C0", fg="#2F4F4F")
title_label.pack(pady=20)

select_button = tk.Button(root, text="Select Image", command=select_file, **button_style)
select_button.pack(pady=10)
submit_button = tk.Button(root, text="Submit", command=main, **button_style)
submit_button.pack(pady=10)

path_label = tk.Label(root, text="No Image Selected", **label_style)
path_label.pack(pady=5)


dataframe_text = tk.Text(root, height=20, width=70, wrap='none', font=("Arial", 12))
dataframe_text.pack(pady=20)

footer_label = tk.Label(root, text="Developed by Abhishek Kanade", font=("Helvetica", 10), bg="#D0F0C0", fg="#2F4F4F")
footer_label.pack(side="bottom", pady=10)


exit_button = tk.Button(root, text="Close", command=exit_program, **button_style1)
exit_button.pack(side="bottom", pady=10)


root.mainloop()

