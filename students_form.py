import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image

# Directory to store student images
student_image_dir = "students"

# Ensure the directory exists
if not os.path.exists(student_image_dir):
    os.makedirs(student_image_dir)

# Function to load student images
def load_student_images(image_dir):
    student_images = {}
    for file_name in os.listdir(image_dir):
        if file_name.endswith(".jpg") or file_name.endswith(".png"):
            name = file_name.split('.')[0]
            image_path = os.path.join(image_dir, file_name)
            img = cv2.imread(image_path)
            student_images[name] = img
    return student_images

# Function to upload student images
def upload_student_images():
    uploaded_files = st.file_uploader("Upload student images", type=["jpg", "png"], accept_multiple_files=True)
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        img = np.array(image)
        file_path = os.path.join(student_image_dir, uploaded_file.name)
        cv2.imwrite(file_path, img)
        st.success(f"Uploaded {uploaded_file.name}")

# Face recognition function
def recognize_face(face_img, known_faces):
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    detected_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in detected_faces:
        face = gray[y:y+h, x:x+w]
        for name, img in known_faces.items():
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            face_recognizer = cv2.face.LBPHFaceRecognizer_create()
            face_recognizer.train([img_gray], np.array([0]))
            label, confidence = face_recognizer.predict(face)
            if confidence < 100:
                return name
    return None

# Streamlit app
def main():
    st.title("Student Attendance System")

    # Upload student images
    st.header("Upload Student Images")
    upload_student_images()

    st.header("Mark Attendance")
    student_images = load_student_images(student_image_dir)

    # Show a dropdown with student names
    names = list(student_images.keys())
    selected_name = st.selectbox("Select your name", names)

    # Capture video from webcam
    st.write("Press 'Start Camera' to start capturing your face.")
    if st.button("Start Camera"):
        stframe = st.empty()
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            stframe.image(frame, channels="BGR", use_column_width=True)
            
            if st.button("Capture Face"):
                face_name = recognize_face(frame, student_images)
                if face_name == selected_name:
                    st.success(f"Attendance marked for {selected_name}.")
                else:
                    st.error("Face not recognized. Try again.")
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
