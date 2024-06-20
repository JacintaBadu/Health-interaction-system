'''
1.Importing Libraries:
First, bring in all the libraries needed for our code. We're using Pickle to store our trained model.

2.Understanding OS and Time:
Know that "os" stands for the operating system. We're also using "time" to 
figure out how long it takes to recognize faces.

3. After saving the models in the pickle file, unknown images are then trained 
to recognize the images and they are then added to the known images.

4. Color Adjustment for Clarity:
Switch the frame back to BGR for better clarity.

5. The duration was calculated to know the time it took for the model to be 
trained and saved.

6.Waiting for Key Press and Finish:
Wait for a key press and check if a key is pressed with a short delay.
Release the camera and close all the OpenCV windows.
'''
import face_recognition
import cv2
import os
import pickle
import numpy as np
import time

# Comment: Loading known images and their encodings
Encodings = []
Names = []
start_time = time.time()
image_dir = "Known_Images"

# looping through the known image directory to get the Names and Encodings
for root, dirs, files in os.walk(image_dir):
    for file in files:
        path = os.path.join(root, file)
        print(path)
        name = os.path.splitext(file)[0]
        print(name)
        person = face_recognition.load_image_file(path)
        encoding = face_recognition.face_encodings(person)[0]  # Assuming only one face per image
        Encodings.append(encoding)
        Names.append(name)
    print(Names)

# Comment: Saving known face encodings and names to a pickle file
picklePath = 'Saving_train.pkl'
with open(picklePath, 'wb') as pickle_model:
    pickle.dump(Names, pickle_model)
    pickle.dump(Encodings, pickle_model)

# Comment: Loading known face encodings and names from the pickle file
with open(picklePath, 'rb') as pickle_model:
    Names = pickle.load(pickle_model)
    Encodings = pickle.load(pickle_model)

font = cv2.FONT_HERSHEY_SIMPLEX # setting the font size
Unknown_Dir = "/Users/traceyagbevem/Desktop/FaceRecognition/UnknownImages"

# looping through the unknown images and deriving only the path
for root, dirs, files in os.walk(Unknown_Dir):
    print(files)
    for file in files:
        Unknown_path = os.path.join(root, file)
        print(Unknown_path)
        try:
            test_Unknown = face_recognition.load_image_file(Unknown_path) # loading the path for the unknown
            face_positions = face_recognition.face_locations(test_Unknown) 
            all_encodings = face_recognition.face_encodings(test_Unknown, face_positions)
            
            Recognition_Time = time.time() - start_time # duration for starting the recognizition
            print(f"Time to recognize unknown: {Recognition_Time} seconds")

            test_Unknown = cv2.cvtColor(test_Unknown, cv2.COLOR_RGB2BGR)
    
            # looping through the encodings for the unknown based on its face features and comparing it to the features of the known images 
            for (top, right, bottom, left), face_encoding in zip(face_positions, all_encodings):
                name = 'Unknown Person'

                # Comment: Comparing unknown faces to known faces
                matches = face_recognition.compare_faces(Encodings, face_encoding, tolerance=0.6)
                
                # comparing the match face to its corresponding Names in the array
                if np.any(matches):
                    face_index = np.where(matches)[0]
                    distance = [np.linalg.norm(face_encoding - Encodings[i]) for i in face_index]
                    closest_index = face_index[np.argmin(distance)]
                    name = Names[closest_index] 
                else:
                    # Comment: Handle the case when there are no matches
                    name = 'Unknown Person'
                 
                # as the images are loading , rectangle will be draw and their names will be added to it
                cv2.rectangle(test_Unknown, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.putText(test_Unknown, name, (left, top - 6), font, 1, (0, 0, 0), 1)

            cv2.imshow("Pictures", test_Unknown)
            cv2.moveWindow("Pictures", 0, 0)

        except Exception as e:
            print(f"Error trying to process unknown images: {e}")

# Comment: Moved display-related lines outside the loop
cv2.waitKey(0)
cv2.destroyAllWindows()
