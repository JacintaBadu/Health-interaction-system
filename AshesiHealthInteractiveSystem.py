'''
1. Importing Libraries:
First, bring in all the libraries needed for our code.webbrowser is meant
to be use to search for reason from the web after the person given reason 
about their visit.sqlite3 is an inbuilt python library.Request is a HTTP library
 built in python for human being.

2. The rest of the code is same as the LiveVideoRecognition. Thus 
 capturing the video and reading the frame etc.
3. Speech Interaction - Welcome Message:
If a person is recognized, welcome them with a speech message using 
text-to-speech (pyttsx3) and print a greeting.

4. Database Check and Interaction:
Check the SQLite database to see if the recognized person has interacted before. 
If yes, ask for the reason for the visit and update the database.

5. Voice Recognition for Reason:
Use speech recognition (speech_recognition) to capture the person's reason for 
the visit. If not understood, prompt the person to enter the reason manually.

6.Web Browsing for Additional Information:
Use the captured reason to browse the web for related information to enhance the
 interaction. (Example: Measures to solve a health problem related to the reason.)

7. Database Insertion and Display:
Insert information about the person (name, student/faculty status, course/class,
department/position, reason) into the SQLite database.

8.Display Recognition Results:
Draw rectangles around recognized faces, display names, and continuously update 
the video feed.

9.Exiting the Program:
Provide an option to exit the program, display information about recognized 
individuals, and close the camera, OpenCV windows, and database connection 
when the user decides to quit.
'''

import webbrowser
import cv2
import face_recognition
import pickle
import numpy as np
import os
import time
import pyttsx3
import speech_recognition as sr
import sqlite3
import requests
from bs4 import BeautifulSoup #it helps to search, navigate and modify a parse tree.

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Initialize speech recognition
recognizer = sr.Recognizer()

# Load known face encodings
Encodings = []
Names = []
start_time = time.time()
image_dir = "Known_Images"

for root, dirs, files in os.walk(image_dir):
    for file in files:
        path = os.path.join(root, file)
        name = os.path.splitext(file)[0]
        person = face_recognition.load_image_file(path)
        encoding = face_recognition.face_encodings(person)[0]  # Assuming there is only one face in each image
        Encodings.append(encoding)
        Names.append(name)

# Database setup
conn = sqlite3.connect('face_recognition_database.db')
cursor = conn.cursor()

# Recreate the 'people' table with the 'reason' column
cursor.execute('''
    DROP TABLE IF EXISTS people
''')

cursor.execute('''
    CREATE TABLE people (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE,
        is_student INTEGER,
        course_name TEXT,
        class_name TEXT,
        department TEXT,
        position TEXT,
        first_visit_timestamp TEXT,
        last_visit_timestamp TEXT,
        reason TEXT
    )
''')
conn.commit()

# Load known face encodings from the pickle file
picklePath = 'Saving_train.pkl'
with open(picklePath, 'wb') as pickle_model:
    pickle.dump(Names, pickle_model)
    pickle.dump(Encodings, pickle_model)

# Real-time face recognition using the camera
camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()

    # Error handling for reading frames
    if not ret or frame is None:
        print("Error when trying to read from the frame")
        break

    # Resize the frame to reduce computational load
    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_positions = face_recognition.face_locations(frameRGB, model='hog')
    all_encodings = face_recognition.face_encodings(frameRGB, face_positions, model='small')

    # Loop over face positions and encodings
    for (top, right, bottom, left), face_encoding in zip(face_positions, all_encodings):
        name = 'Unknown Person'

        # Comparing unknown faces to known faces
        matches = face_recognition.compare_faces(Encodings, face_encoding, tolerance=0.6)
        if np.any(matches):
            face_index = np.where(matches)[0]
            distance = [np.linalg.norm(face_encoding - Encodings[i]) for i in face_index]
            closest_index = face_index[np.argmin(distance)]
            name = Names[closest_index]

            # Speak the person's name
            text = f"Hello {name}!"
            print(text)
            engine.say(text)
            engine.runAndWait()

            # Check if the person has interacted before
            cursor.execute('SELECT * FROM people WHERE name=?', (name,))
            person_data = cursor.fetchone()

            # Define variables outside the condition blocks
            class_name = None
            position = None
            department = None  # Added this line

            if person_data:
                # If the person has interacted before, ask for reason and duration
                engine.say(f"Welcome back, {name}! Please state your reason for the visit.")
                engine.runAndWait()
                with sr.Microphone() as source:
                    recognizer.adjust_for_ambient_noise(source)
                    audio = recognizer.listen(source)

                try:
                    reason = recognizer.recognize_google(audio)
                    print(reason)

                    # Browsing to get measures about the reason
                    print(f"Browsing information about: {reason}")
                    # Fetch information from the web
                    search_url = f"https://www.example.com/search?q= measure to solve {reason} health problem."
                    response = requests.get(search_url)
                    soup = BeautifulSoup(response.text, 'html.parser')

                     # Extract relevant information using BeautifulSoup
                     # Modify this part based on the structure of the webpage
                    result_elements = soup.find_all('div', class_='result')
                    for result in result_elements:
                            print(result.text)
                    measures = '\n'.join(result.text for result in result_elements)       

                except sr.UnknownValueError:
                    print("Could not understand the reason.")
                    reason = input("Enter your reason for the visit: ")
                    print(reason)
                    # Browsing to get measures about the reason
                    print(f"Browsing information about: {reason}")
                    # Fetch information from the web
                    search_url = f"https://www.example.com/search?q= measure to solve {reason} health problem."
                    response = requests.get(search_url)
                    soup = BeautifulSoup(response.text, 'html.parser')

                     # Extract relevant information using BeautifulSoup
                     # Modify this part based on the structure of the webpage
                    result_elements = soup.find_all('div', class_='result')
                    for result in result_elements:
                            print(result.text)
                    measures = '\n'.join(result.text for result in result_elements)       

                # Update the last visit timestamp and reason
                cursor.execute('''
                    UPDATE people
                    SET last_visit_timestamp=?, reason=?
                    WHERE id=?
                ''', (time.strftime('%Y-%m-%d %H:%M:%S'), reason, person_data[0]))
                conn.commit()

            else:
                # If the person is recognized for the first time, ask for additional information
                print("Unknown person. Please provide additional information.")

                # Ask if the person wants to provide additional information
                print("Do you want to provide additional information? (y/n)")
                response = input().lower()
                if response == 'y':
                    # Ask if the person is a student or faculty
                    engine.say(f"Hello {name}! Are you a student or faculty? (Speak or Enter 's' for student, 'f' for faculty)")
                    engine.runAndWait()
                    with sr.Microphone() as source:
                        recognizer.adjust_for_ambient_noise(source)
                        audio = recognizer.listen(source)

                    try:
                        response = recognizer.recognize_google(audio).lower()
                        print(response)
                    except sr.UnknownValueError:
                        print("Could not understand the response. Please enter 's' for student or 'f' for faculty.")
                        response = input("Enter 's' for student, 'f' for faculty: ").lower()

                    if response == 's':
                        is_student = True
                        print(response)
                        # Ask for course name
                        engine.say("Please state your course name.")
                        engine.runAndWait()
                        with sr.Microphone() as source:
                            recognizer.adjust_for_ambient_noise(source)
                            audio = recognizer.listen(source)
                        try:
                            course_name = recognizer.recognize_google(audio)
                            print(course_name)
                        except sr.UnknownValueError:
                            print("Could not understand the course name.")
                            course_name = input("Enter your course name: ")

                        # Ask for class name
                        engine.say("Please state your class name.")
                        engine.runAndWait()
                        with sr.Microphone() as source:
                            recognizer.adjust_for_ambient_noise(source)
                            audio = recognizer.listen(source)
                        try:
                            class_name = recognizer.recognize_google(audio)
                            print(class_name)
                        except sr.UnknownValueError:
                            print("Could not understand the class name.")
                            class_name = input("Enter your class name: ")

                        # Ask for reason
                        engine.say("Please state your reason for the visit.")
                        engine.runAndWait()
                        with sr.Microphone() as source:
                            recognizer.adjust_for_ambient_noise(source)
                            audio = recognizer.listen(source)
                        try:
                            reason = recognizer.recognize_google(audio)
                            print(reason)
                            # Browsing to get measures about the reason
                            print(f"Browsing information about: {reason}")
                            # Fetch information from the web
                            search_url = f"https://www.example.com/search?q= measure to solve {reason} health problem."
                            response = requests.get(search_url)
                            soup = BeautifulSoup(response.text, 'html.parser')

                             # Extracting important information using BeautifulSoup
                            result_elements = soup.find_all('div', class_='result')
                            for result in result_elements:
                                print(result.text)
                            measures = '\n'.join(result.text for result in result_elements)

                        except sr.UnknownValueError:
                            print("Could not understand response.")
                            reason = input("Enter your reason: ")   
                            # Browsing to get measures about the reason
                            print(f"Browsing information about: {reason}")
                            # Fetch information from the web
                            search_url = f"https://www.example.com/search?q= measure to solve {reason} health problem."
                            response = requests.get(search_url)
                            soup = BeautifulSoup(response.text, 'html.parser')

                             # Extract relevant information using BeautifulSoup
                            # Modify this part based on the structure of the webpage
                            result_elements = soup.find_all('div', class_='result')
                            for result in result_elements:
                                print(result.text)
                            measures = '\n'.join(result.text for result in result_elements)

                    elif response == 'f':
                        is_student = False
                        # Ask for department
                        engine.say("Please state your department.")
                        engine.runAndWait()
                        with sr.Microphone() as source:
                            recognizer.adjust_for_ambient_noise(source)
                            audio = recognizer.listen(source)
                        try:
                            department = recognizer.recognize_google(audio)
                            print(department)
                        except sr.UnknownValueError:
                            print("Could not understand the department.")
                            department = input("Enter your department: ")

                        # Ask for position
                        engine.say("Please state your position.")
                        engine.runAndWait()
                        with sr.Microphone() as source:
                            recognizer.adjust_for_ambient_noise(source)
                            audio = recognizer.listen(source)
                        try:
                            position = recognizer.recognize_google(audio)
                            print(position)
                        except sr.UnknownValueError:
                            print("Could not understand the position.")
                            position = input("Enter your position: ")

                        # Ask for reason for the visit
                        engine.say("Please state your reason for the visit.")
                        engine.runAndWait()
                        with sr.Microphone() as source:
                            recognizer.adjust_for_ambient_noise(source)
                            audio = recognizer.listen(source)
                        try:
                            reason= recognizer.recognize_google(audio)
                            print(reason)

                            # Browsing to get measures about the reason
                            print(f"Browsing information about: {reason}")
                            # Fetch information from the web
                            search_url = f"https://www.example.com/search?q= measure to solve {reason} health problem."
                            response = requests.get(search_url)
                            soup = BeautifulSoup(response.text, 'html.parser')

                             # Extract relevant information using BeautifulSoup
                            result_elements = soup.find_all('div', class_='result')
                            for result in result_elements:
                                print(result.text)
                            measures = '\n'.join(result.text for result in result_elements)

                        except sr.UnknownValueError:
                            print("Could not understand your response.")
                            reason = input("Enter your reason: ")

                            # Browsing to get measures about the reason
                            print(f"Browsing information about: {reason}")
                            # Fetch information from the web
                            search_url = f"https://www.example.com/search?q= measure to solve {reason} health problem."
                            response = requests.get(search_url)
                            soup = BeautifulSoup(response.text, 'html.parser')

                            # Extract relevant information using BeautifulSoup
                            result_elements = soup.find_all('div', class_='result')
                            for result in result_elements:
                                print(result.text)
                            measures = '\n'.join(result.text for result in result_elements)
                    else:
                        print("Invalid response. Please enter 's' for student or 'f' for faculty.")
                        continue


                    # Browsing to get measures about the reason
                    print(f"Browsing information about: {reason}")
                    # Fetch information from the web
                    search_url = f"https://www.example.com/search?q=measure to solve {reason} health problem."
                    response = requests.get(search_url)
                    soup = BeautifulSoup(response.text, 'html.parser')

                    # Extract relevant information using BeautifulSoup
                    result_elements = soup.find_all('div', class_='result')
                    for result in result_elements:
                        print(result.text)
                    measures = '\n'.join(result.text for result in result_elements)
                    # Insert information into the database
                    cursor.execute('''
                        INSERT INTO people (name, is_student, course_name, class_name, department, position, first_visit_timestamp,reason)
                        VALUES (?, ?, ?, ?, ?, ?, ?,?)
                    ''', (name, int(is_student), course_name, class_name, department, position, time.strftime('%Y-%m-%d %H:%M:%S'),reason))
                    conn.commit()

                    # Display the information for the person to see
                    print(f"Name: {name}")
                    print(f"Is Student: {'Yes' if is_student else 'No'}")
                    if is_student:
                        print(f"Course Name: {course_name}")
                        print(f"Class Name: {class_name}")
                        print(f"Reason: {reason}")
                        print(f"Measures: {measures}")
                    else:
                        print(f"Department: {department}")
                        print(f"Position: {position}")
                        print(f"Position: {reason}")
                    print(f"First Visit Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
                    print("-" * 30)

        cv2.rectangle(frameRGB, (left, top), (right, bottom), (255, 255, 0), 1)
        cv2.putText(frameRGB, name, (left, top - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 1)

    frameRGB = cv2.cvtColor(frameRGB, cv2.COLOR_RGB2BGR)
    cv2.imshow("Face Recognition", frameRGB)
    cv2.moveWindow("Face Recognition", 0, 0)

if cv2.waitKey(1) == ord('q'):
        # Ask the user if they want to quit
        print("Do you want to quit? (yes/no)")
        response = input().lower()
        if response == 'yes':
            # Display information about recognized individuals
            cursor.execute('SELECT * FROM people')
            recognized_people = cursor.fetchall()

            for person in recognized_people:
                print(f"Name: {person[1]}")
                print(f"Is Student: {'Yes' if person[2] else 'No'}")
                if person[2]:  # If the person is a student
                    print(f"Course Name: {person[3]}")
                    print(f"Class Name: {person[4]}")
                    print(f"reason: {person[5]}")
                else:  # If the person is faculty
                    print(f"Department: {person[6]}")
                    print(f"Position: {person[7]}")
                    print(f"Position: {person[8]}")
                    print(f"First Visit Timestamp: {person[9]}")
                    print(f"Last Visit Timestamp: {person[10]}")
                    print("-" * 30)

            # Close the camera and database connection
            camera.release()
            cv2.destroyAllWindows()
            conn.close()
           # break
        else:
            print("Continue with the code")