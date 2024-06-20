'''
1. Importing Libraries:
First, bring in all the libraries needed for our code. 

From importlibraies to the  displaying of the capture face is the "LiveVideoRecognition part of the code.
2. But a library in the pyttsx3 was imported for text-to-speech including speech recognition initialization.

3. In the while loop, a for loop is created for a continuous looping for left, right, bottom 
and top dimension of the face. An input called preference is asking from the user using pyttsxt3.

4. If the user mentioned or inputted any one of the preference, an if's statement start based on the input.

5. If it is music, the audio ask the question to input the song title, artiste. otherwise if documentaries, it ask for
title, and else if it is a movie, it ask for the title of movie, year of release. If any is not found,
 it asks the user to input the value again. And the loop can be terminated using "CTRL+ C"

'''
import cv2
import face_recognition
import pickle
import numpy as np
import os
import time
import pyttsx3
import speech_recognition as sr
import pygame
import pywhatkit

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

picklePath = 'Saving_train.pkl'
with open(picklePath, 'wb') as pickle_model:
    pickle.dump(Names, pickle_model)
    pickle.dump(Encodings, pickle_model)

# Load known face encodings from the pickle file
if os.path.exists(picklePath):
    with open(picklePath, 'rb') as pickle_model:
        Names = pickle.load(pickle_model)
        Encodings = pickle.load(pickle_model)

font = cv2.FONT_HERSHEY_SIMPLEX
Unknown_Dir = "/Users/traceyagbevem/Desktop/FaceRecognition/UnknownImages"

# Real-time face recognition using the camera
camera = cv2.VideoCapture(0)

pygame.init()

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

    for (top, right, bottom, left), face_encoding in zip(face_positions, all_encodings):
        name = 'Unknown Person'

        # Comparing unknown faces to known faces
        matches = face_recognition.compare_faces(Encodings, face_encoding, tolerance=0.6)
        if any(matches):
            face_index = np.where(matches)[0]
            distance = [np.linalg.norm(face_encoding - Encodings[i]) for i in face_index]
            closest_index = face_index[np.argmin(distance)]
            name = Names[closest_index]

            # Speak the person's name
            text = f"Hello {name}, welcome to entertainment option selection.! "
            engine.say(text)
            engine.runAndWait()

            # Ask user about preferences
            engine.say("Do you prefer music, documentaries, or movies? Speak or type your choice.")
            engine.runAndWait()

            # Get user preference through speech or text input
            with sr.Microphone() as source:
                try:
                    audio = recognizer.listen(source, timeout=0.3)
                    preference = recognizer.recognize_google(audio).lower()
                    print(f"Preference: {preference}")
                except sr.UnknownValueError:
                    print("Could not understand audio.")
                    preference = input("Please type your preference (music, documentaries, or movies): ").lower()

            if "music" in preference:
                # Ask for artist and song name
                engine.say("Please provide the artist name. Speak or type.")
                engine.runAndWait()

                with sr.Microphone() as source:
                    try:
                       audio = recognizer.listen(source, timeout=0.2)
                       artist_name = recognizer.recognize_google(audio).lower()
                       print(f"Artiste Name: {artist_name}")

                    except sr.UnknownValueError:
                        print("Could not understand audio.")
                        artist_name = input("Artist name: ").lower()

                engine.say("Please provide the title of the song. Speak or type.")
                engine.runAndWait()
                with sr.Microphone() as source:
                    try:
                       audio = recognizer.listen(source, timeout=0.2)
                       song_title = recognizer.recognize_google(audio).lower()
                       print(f"Song title: {song_title}")

                    except sr.UnknownValueError:
                        print("Could not understand audio.")
                        song_title = input("Song title: ").lower()

                # Search and play the song on YouTube
                pywhatkit.playonyt(f"{artist_name} {song_title}")

            elif "documentaries" in preference:
                # Ask for documentary title
                engine.say("Please provide the title of the documentary. Speak or type.")
                engine.runAndWait()
                with sr.Microphone() as source:
                    try:
                       audio = recognizer.listen(source, timeout=0.2)
                       documentary_title = recognizer.recognize_google(audio).lower()
                       print(f"Documentary title: {documentary_title}")

                    except sr.UnknownValueError:
                        print("Could not understand audio.")
                        documentary_title = input("Documentary title: ").lower()

                # Search and play the documentary on YouTube
                pywhatkit.playonyt(f"{documentary_title} documentary")

            elif "movies" in preference:
                # Ask for movie title and year
                engine.say("Please provide the title of the movie. Speak or type.")
                engine.runAndWait()
                with sr.Microphone() as source:
                    try:
                       audio = recognizer.listen(source, timeout=0.2)
                       movie_title = recognizer.recognize_google(audio).lower()
                       print(f"Movie title: {movie_title}")

                    except sr.UnknownValueError:
                        print("Could not understand audio.")
                        movie_title = input("Movie title: ").lower()

                engine.say("Please provide the release year of the movie. Speak or type.")
                engine.runAndWait()
                with sr.Microphone() as source:
                    try:
                       audio = recognizer.listen(source, timeout=0.2)
                       movie_year = recognizer.recognize_google(audio).lower()
                       print(f"Movie year: {movie_year}")

                    except sr.UnknownValueError:
                        print("Could not understand audio.")
                        movie_year = input("Release year: ").lower()

                # Search and play the movie on YouTube
                pywhatkit.playonyt(f"{movie_title} {movie_year} movie")

            else:
                engine.say("I'm sorry, I didn't understand your preference.")

        cv2.rectangle(frameRGB, (left, top), (right, bottom), (255, 255, 0), 1)
        cv2.putText(frameRGB, name, (left, top - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 1)

    frameRGB = cv2.cvtColor(frameRGB, cv2.COLOR_RGB2BGR)
    cv2.imshow("EpocCam", frameRGB)
    cv2.moveWindow("EpocCam", 0, 0)

    if cv2.waitKey(1) == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
