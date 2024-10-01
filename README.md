**Attendance Calculator** 
**Project Overview**
This project automates the extraction of attendance details from an image of an attendance sheet and converts it into a structured Excel sheet. The application leverages Optical Character Recognition (OCR) and image processing techniques to extract enroll numbers, names, attendance counts, and percentage attendance for each student.
**Key Features:**
Converts attendance sheet images to Excel sheets.
Extracts student details (enroll number, name) and computes attendance and percentage.
Uses PaddleOCR for text recognition.
Includes a pre-trained model for detecting attendance statuses (Present/Absent).
Provides a simple GUI for user interaction.

**2.Tools & Technologies**
**Python**: Core programming language.
**PaddleOCR**: OCR engine for text recognition from images.
Link:https://github.com/PaddlePaddle/PaddleOCR/tree/main
**OpenCV:** For image processing tasks such as thresholding, line detection, and contour analysis.
**Keras:** For loading the pre-trained model to classify attendance status (Present/Absent).
**Tkinter:** For building the graphical user interface (GUI).
**Pandas**: For creating and managing Excel files.

**3. System Architecture**
**Input Image:** The user provides an image of an attendance sheet via the GUI.
**Preprocessing:**
Converts the image to grayscale.
Applies binary thresholding for better contrast.
**Line Detection:** Horizontal and vertical lines in the table are detected to segment cells.
**OCR and Character Classification:**
Each cell is processed using PaddleOCR for text recognition.
For certain ambiguous cells (marked by "P" or "A"), the pre-trained model predicts if the student was Present (P) or Absent (A).
**Data Aggregation**: Enroll numbers, names, and attendance statuses are aggregated.
**Excel Generation:** The extracted data is saved in an Excel file along with the calculated percentage attendance.
**Output:** The final Excel file is created and saved to the user-specified location.

**Input Format:**
The attendnace sheet should be given in the form of an image( .jpg, .png,jpeg) containing attendance table columns enrollment number,name,attendance columns(containing handwritten A/P)  20 to 25.

**Naming Convention:** The name of each image should be in the format.
<Subject Name>_<Subject Code>_<Semester>_<Section>.png/jpg

**Output Format:**
The output will be a csv file or pdf with name ‘subject_section.csv’ that contains enrollment number,name, and total attendance of each student.



