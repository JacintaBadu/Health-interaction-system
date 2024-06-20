'''
1.Importing Libraries:
First, bring in all the libraries needed for our code. We're using Pickle to store our trained model.

2.Understanding OS and Time:
Know that "os" stands for the operating system. We're also using "time" to figure out how long it takes to recognize faces.

3.Camera Setup and Video Capture:
Once our model is trained and saved, set up the camera to capture video from the laptop.

4.Grabbing Live Video Frames:
Use a loop to get frames from the live video. If a frame isn't captured, we handle it as an error. 
Otherwise, move on to the next part.

5. Adjusting Frame Size and Color:
Resize the captured frame and change its color to RGB. OpenCV prefers working with RGB.

6.Face Detection and Encoding:
Use the face recognition library to find face locations in the RGB frame, using the "hog" method 
for face detection.Get face encodings based on the identified face positions.

7.Comparing and Finding Names:
Check each encoding against our trained model. If there's a match within a set limit, get the 
corresponding name; otherwise, call it "unknown Person."

8. Drawing on the Video:
Draw a rectangle around the recognized face and display the person's name while the video is still playing.

9.Color Adjustment for Clarity:
Switch the frame back to BGR for better clarity.

10.Waiting for Key Press and Finish:
Wait for a key press and check if a key is pressed with a short delay.
Release the camera and close all the OpenCV windows.
'''
#1
import cv2
import face_recognition
import pickle
import numpy as np
import os
import time

# Load known face encodings
Encodings = []
Names = []
start_time = time.time()
image_dir = "Known_Images"
 # 2
for root, dirs, files in os.walk(image_dir):
    for file in files:
        path = os.path.join(root, file)
        name = os.path.splitext(file)[0]
        person = face_recognition.load_image_file(path)
        encoding = face_recognition.face_encodings(person)[0]  # Assuming there is only one face in each image
        Encodings.append(encoding)
        Names.append(name)
#1 
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

# 3: Real-time face recognition using the camera
camera = cv2.VideoCapture(0)

# 4
while True:
    ret, frame = camera.read()

    # Error handling for reading frames
    if not ret or frame is None:
        print("Error when trying to read from the frame")
        break

    # 5: Resize the frame to reduce computational load 
    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
       
    #6
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_positions = face_recognition.face_locations(frameRGB, model='hog')
    all_encodings = face_recognition.face_encodings(frameRGB, face_positions, model='small')

    for (top, right, bottom, left), face_encoding in zip(face_positions, all_encodings):
        name = 'Unknown Person'

        # 7  Comparing unknown faces to known faces
        matches = face_recognition.compare_faces(Encodings, face_encoding, tolerance=0.5)
        if np.any(matches):
            face_indices = np.where(matches)[0]
            names = [Names[i] for i in face_indices]
    # If you want to get the first name in the list, you can use:
    # name = names[0]
    # If you want to handle multiple matches, you might want to loop through names.
            for name in names:
                print(f"Detected: {name}")
        else:
            name = 'Unknown Person'
            print(f"No matches found. {name}")

        # if np.any(matches):
        #     face_index = np.where(matches)[0]
        #     # distance = [np.linalg.norm(face_encoding - Encodings[i]) for i in face_index]
        #     # closest_index = face_index[np.argmin(distance)]
        #     name = Names[face_index]

         #8
        cv2.rectangle(frameRGB, (left, top), (right, bottom), (255, 255, 0), 1)
        cv2.putText(frameRGB, name, (left, top - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0,255), 1)
        
        #9
    frameRGB = cv2.cvtColor(frameRGB, cv2.COLOR_RGB2BGR)
    cv2.imshow("EpocCam", frameRGB)
    cv2.moveWindow("EpocCam", 0, 0)
             # 10
    if cv2.waitKey(1) == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()