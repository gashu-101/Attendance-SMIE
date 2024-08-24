import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
import json
from datetime import date

# Directory where photos and JSON files will be stored
DATA_DIR = "students_data"
PHOTO_DIR = os.path.join(DATA_DIR, "photos")
STUDENTS_JSON_FILE = os.path.join(DATA_DIR, "students.json")
ATTENDANCE_JSON_FILE = os.path.join(DATA_DIR, "attendance_summary.json")

if not os.path.exists(PHOTO_DIR):
    os.makedirs(PHOTO_DIR)

# Initialize session state for student database
if 'students_db' not in st.session_state:
    if os.path.exists(STUDENTS_JSON_FILE):
        with open(STUDENTS_JSON_FILE, 'r') as f:
            students_data = json.load(f)
            st.session_state.students_db = pd.DataFrame(students_data)
    else:
        st.session_state.students_db = pd.DataFrame(columns=['Full Name', 'ID', 'Photo Paths', 'Section', 'Academic Year', 'Semester'])

# Function to collect student data
def collect_student_data():
    st.title("Student Data Collection")

    # Form to collect student data
    with st.form(key='student_data_form'):
        full_name = st.text_input("Full Name")
        student_id = st.text_input("Student ID")
        section = st.text_input("Section")
        ac_year = st.text_input("Academic Year")
        semester = st.text_input("Semester")

        # Camera feed placeholder
        camera_placeholder = st.empty()

        capture_button = st.form_submit_button(label="Capture Photo")
        submit_button = st.form_submit_button(label="Submit")

    # Initialize the camera and display the feed
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access the camera.")
            break

        # Display the camera feed
        camera_placeholder.image(frame, channels="BGR")

        if capture_button:
            # Construct the full path to the directory where the photo will be saved
            student_photo_dir = os.path.join(PHOTO_DIR, student_id)
            os.makedirs(student_photo_dir, exist_ok=True)

            # Now save the captured photo
            photo_filename = f"{student_id}_photo.jpg"
            photo_path = os.path.join(student_photo_dir, photo_filename)
            cv2.imwrite(photo_path, frame)
            st.success(f"Photo captured and saved as {photo_path}.")
            break

    cap.release()

    # Validate input fields and provide feedback
    if submit_button:
        if not full_name:
            st.error("Full Name is required.")
        elif not student_id:
            st.error("Student ID is required.")
        elif not section:
            st.error("Section is required.")
        elif not ac_year:
            st.error("Academic Year is required.")
        elif not semester:
            st.error("Semester is required.")
        else:
            # Add student data to the session state DataFrame
            st.session_state.students_db.loc[len(st.session_state.students_db)] = [full_name, student_id, [photo_path], section, ac_year, semester]

            # Store the student data in the JSON file
            student_data = st.session_state.students_db.to_dict(orient='records')
            with open(STUDENTS_JSON_FILE, 'w') as f:
                json.dump(student_data, f, indent=4)

            st.success("Student data collected successfully!")
            st.balloons()  # Optional: Adds a visual effect

# Function to perform face recognition and return face coordinates
def recognize_face(live_frame, stored_photo_path):
    stored_image = cv2.imread(stored_photo_path)
    gray_live_frame = cv2.cvtColor(live_frame, cv2.COLOR_BGR2GRAY)
    gray_stored_image = cv2.cvtColor(stored_image, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces_live = face_cascade.detectMultiScale(gray_live_frame, 1.1, 4)
    faces_stored = face_cascade.detectMultiScale(gray_stored_image, 1.1, 4)

    if len(faces_live) == 0 or len(faces_stored) == 0:
        return False, None

    (x_live, y_live, w_live, h_live) = faces_live[0]
    live_face = gray_live_frame[y_live:y_live+h_live, x_live:x_live+w_live]
    (x_stored, y_stored, w_stored, h_stored) = faces_stored[0]
    stored_face = gray_stored_image[y_stored:y_stored+h_stored, x_stored:x_stored+w_stored]

    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train([stored_face], np.array([0]))

    label, confidence = face_recognizer.predict(live_face)

    if confidence < 40:  # Stricter threshold
        return True, (x_live, y_live, w_live, h_live)
    else:
        return False, None

# Function to take attendance
def take_attendance():
    st.title("Take Attendance")

    # Ask for the date
    attendance_date = st.date_input("Select Attendance Date", date.today())

    if st.session_state.students_db.empty:
        st.warning("No student data available. Please collect student data first.")
        return

    section = st.selectbox("Select Section", st.session_state.students_db['Section'].unique())
    filtered_students = st.session_state.students_db[st.session_state.students_db['Section'] == section]

    if filtered_students.empty:
        st.warning("No students found for the selected section.")
        return

    st.text("Show the student's face to the camera")
    frame_placeholder = st.empty()

    cap = cv2.VideoCapture(0)

    attendance_recorded = set()  # Keep track of students for whom attendance has been recorded
    done = False  # Control variable for manual stopping
    
    attendance_summary = []  # List to store attendance data
    
    if st.button("Done", key="done_button"):
        done = True
            
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture video")
            break

        # Loop through each student in the filtered list and try to recognize
        for _, student_record in filtered_students.iterrows():
            if student_record['Full Name'] not in attendance_recorded:
                for photo_path in student_record['Photo Paths']:
                    recognized, face_coords = recognize_face(frame, photo_path)
                    if recognized:
                        # Draw a rectangle around the face
                        (x, y, w, h) = face_coords
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        # Put the student ID above the rectangle
                        cv2.putText(frame, student_record['ID'], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                        st.success(f"Attendance recorded for {student_record['Full Name']} on {attendance_date}")
                        attendance_recorded.add(student_record['Full Name'])
                        
                        # Store attendance data in summary list
                        attendance_summary.append({
                            "Full Name": student_record['Full Name'],
                            "ID": student_record['ID'],
                            "Date": str(attendance_date),
                            "Section": student_record['Section']
                        })
                        # Break the loop once a face is recognized to avoid multiple recognitions in one loop iteration
                        break

        # Display the live video frame with rectangles and text in the Streamlit app
        frame_placeholder.image(frame, channels="BGR")

        # Exit the loop after recognizing all students in the section
        if len(attendance_recorded) == len(filtered_students):
            st.success("Attendance recorded for all students in this section.")
            break

    cap.release()
    cv2.destroyAllWindows()

    if done:
        st.warning("Attendance process manually stopped.")
    
    # Save the attendance summary to the JSON file
    if os.path.exists(ATTENDANCE_JSON_FILE):
        with open(ATTENDANCE_JSON_FILE, 'r') as f:
            attendance_data = json.load(f)
    else:
        attendance_data = []
    
    attendance_data.extend(attendance_summary)
    with open(ATTENDANCE_JSON_FILE, 'w') as f:
        json.dump(attendance_data, f, indent=4)

# Function to analyze attendance
def analyze_attendance():
    st.title("Analyze Attendance")
    
    # Load the attendance summary JSON file
    if os.path.exists(ATTENDANCE_JSON_FILE):
        with open(ATTENDANCE_JSON_FILE, 'r') as f:
            attendance_data = json.load(f)
    else:
        # Create sample data if the file does not exist
        sample_data = [
            {"Full Name": "John Doe", "ID": "JD001", "Date": "2024-08-24", "Section": "A"},
            {"Full Name": "Jane Smith", "ID": "JS002", "Date": "2024-08-24", "Section": "A"},
            {"Full Name": "Alice Johnson", "ID": "AJ003", "Date": "2024-08-24", "Section": "B"},
        ]

        # Save the sample data to the JSON file
        with open(ATTENDANCE_JSON_FILE, 'w') as f:
            json.dump(sample_data, f, indent=4)

        st.info("Sample attendance data has been generated for analysis.")
        attendance_data = sample_data
    
    # Convert attendance data to DataFrame for analysis
    df = pd.DataFrame(attendance_data)
    st.write("### Attendance Data", df)

    # Display a summary of attendance per section
    attendance_per_section = df['Section'].value_counts().reset_index()
    attendance_per_section.columns = ['Section', 'Count']

    st.write("### Attendance Summary by Section")
    st.dataframe(attendance_per_section)

    # Plot a bar chart for attendance per section
    st.write("### Attendance Distribution by Section")
    st.bar_chart(attendance_per_section.set_index('Section'))

    # Display a summary of attendance per date
    attendance_per_date = df['Date'].value_counts().reset_index()
    attendance_per_date.columns = ['Date', 'Count']

    st.write("### Attendance Summary by Date")
    st.dataframe(attendance_per_date)

    # Plot a line chart for attendance per date
    st.write("### Attendance Over Time")
    st.line_chart(attendance_per_date.set_index('Date'))

    # Option to filter attendance data by section
    st.write("### Filter Attendance Data")
    selected_section = st.selectbox("Select Section to Filter", options=df['Section'].unique())
    filtered_df = df[df['Section'] == selected_section]
    st.write(f"Attendance Data for Section {selected_section}", filtered_df)

    # Additional analysis like percentage attendance
    total_students = df['Full Name'].nunique()
    attendance_percentage = (df['ID'].nunique() / total_students) * 100
    st.write(f"### Overall Attendance Percentage: {attendance_percentage:.2f}%")

# Streamlit interface
st.sidebar.title("SMIE Attendance")
option = st.sidebar.selectbox("Choose Option", ["Collect Student Data", "Take Attendance", "Analyze Attendance"])

if option == "Collect Student Data":
    collect_student_data()
elif option == "Take Attendance":
    take_attendance()
elif option == "Analyze Attendance":
    analyze_attendance()
