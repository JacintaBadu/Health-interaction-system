'''
1.Initialization:
Import necessary libraries (cv2, face_recognition, time).
Record the start time for performance measurement.

2.Load Known Images and Encodings:
Load images of known individuals (e.g., Joe Biden) and encode their faces.

3.Create Training Data Arrays:
Create arrays with known face encodings and corresponding names.

4.print Dataset Loading Time:
Measure and print the time taken to load the known images and encodings.

5.Load and Process Test Image:
Load a test image with one or more unknown faces.
Find face locations and encodings in the test image.

6.Face Matching and Recognition:
Compare test image face encodings with known encodings.
If a match is found, recognize and label the person.

7.Display Result:
Show the test image with rectangles around recognized faces and their names using OpenCV.
Wait for a key event and then close the window.
'''


import cv2
import face_recognition
import time

# Record the start time
start_time = time.time()

# Import known images and their encodings
Biden_image = face_recognition.load_image_file("Known_Images/biden.jpeg")
Biden_face_encode = face_recognition.face_encodings(Biden_image)[0]
Biden_face_locations = face_recognition.face_locations(Biden_image)
Biden_image = cv2.cvtColor(Biden_image, cv2.COLOR_RGB2BGR)

Beverlyn_image = face_recognition.load_image_file("Known_Images/Beverlyn Amanfu.jpeg")
Beverlyn_face_encode = face_recognition.face_encodings(Beverlyn_image)[0]

Joel_image = face_recognition.load_image_file("Known_Images/Joel Asamoah.jpeg")
Joel_face_encode = face_recognition.face_encodings(Joel_image)[0]

Jacinta_image = face_recognition.load_image_file("Known_Images/Jacinta Badu .jpeg")
Jacinta_face_encode = face_recognition.face_encodings(Jacinta_image)[0]

Tracey_image = face_recognition.load_image_file("Known_Images/Tracey Agbevem.jpeg")
Tracey_face_encode = face_recognition.face_encodings(Tracey_image)[0]

# Create an array for all the images and their corresponding names (trained data)
encodings = [Biden_face_encode, Beverlyn_face_encode, Joel_face_encode, Jacinta_face_encode, Tracey_face_encode]
names = ['Joe Biden', 'Beverlyn Amanfu', 'Joel Asamoah', 'Jacinta Esi Amoawah Badu', 'Tracey Agbevem']

# Print the loading time for the dataset
dataset_time = time.time() - start_time
print(f"The loading time for the dataset: {dataset_time} seconds")

# Load the test image for unknown images
font = cv2.FONT_HERSHEY_SIMPLEX
testPerson = face_recognition.load_image_file("UnknownImages/Berv_Nic.jpeg")

# Finding face locations and encodings for the test image
face_positions = face_recognition.face_locations(testPerson)
all_encodings = face_recognition.face_encodings(testPerson, face_positions)

# Convert the image
testPerson = cv2.cvtColor(testPerson, cv2.COLOR_RGB2BGR)

# Finding the frame & matching
for (top, right, bottom, left), face_encoding in zip(face_positions, all_encodings):
    name = 'Unknown Person'

    # Comparing training set to unknown
    matches = face_recognition.compare_faces(encodings, face_encoding, tolerance=0.6)

    # Adjusting tolerance for multiple face positions
    if len(face_positions) > 1:
        matches = face_recognition.compare_faces(encodings, face_encoding, tolerance=0.5)

    if True in matches:
        first_match_index = matches.index(True)
        name = names[first_match_index]

    cv2.rectangle(testPerson, (left, top), (right, bottom), (255, 0, 0), 2)
    cv2.putText(testPerson, name, (left, top - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

# Display the image
cv2.imshow('Test Window', testPerson)
cv2.moveWindow('Test Window', 0, 0)

# Wait for a key event and then close the window
cv2.waitKey(0)
cv2.destroyAllWindows()
import cv2
import face_recognition
import time

# Record the start time
start_time = time.time()

# Import known images and their encodings
Biden_image = face_recognition.load_image_file("Known_Images/biden.jpeg")
Biden_face_encode = face_recognition.face_encodings(Biden_image)[0]
Biden_face_locations = face_recognition.face_locations(Biden_image)
Biden_image = cv2.cvtColor(Biden_image, cv2.COLOR_RGB2BGR)

Beverlyn_image = face_recognition.load_image_file("Known_Images/Beverlyn Amanfu.jpeg")
Beverlyn_face_encode = face_recognition.face_encodings(Beverlyn_image)[0]

Joel_image = face_recognition.load_image_file("Known_Images/Joel Asamoah.jpeg")
Joel_face_encode = face_recognition.face_encodings(Joel_image)[0]

Jacinta_image = face_recognition.load_image_file("Known_Images/Jacinta Badu .jpeg")
Jacinta_face_encode = face_recognition.face_encodings(Jacinta_image)[0]

Tracey_image = face_recognition.load_image_file("Known_Images/Tracey Agbevem.jpeg")
Tracey_face_encode = face_recognition.face_encodings(Tracey_image)[0]

# Create an array for all the images and their corresponding names (trained data)
encodings = [Biden_face_encode, Beverlyn_face_encode, Joel_face_encode, Jacinta_face_encode, Tracey_face_encode]
names = ['Joe Biden', 'Beverlyn Amanfu', 'Joel Asamoah', 'Jacinta Esi Amoawah Badu', 'Tracey Agbevem']

# Print the loading time for the dataset
dataset_time = time.time() - start_time
print(f"The loading time for the dataset: {dataset_time} seconds")

# Load the test image for unknown images
font = cv2.FONT_HERSHEY_SIMPLEX
#testPerson = face_recognition.load_image_file("UnknownImages/Jacinta & Kofi .jpeg")
testPerson = face_recognition.load_image_file("UnknownImages/Jacinta & Tracey.jpeg")
# testPerson = face_recognition.load_image_file("UnknownImages/crowd.jpg")
# testPerson = face_recognition.load_image_file("UnknownImages/Berv_Nic.jpeg")

# Finding face locations and encodings for the test image
face_positions = face_recognition.face_locations(testPerson)
all_encodings = face_recognition.face_encodings(testPerson, face_positions)

# Convert the image
testPerson = cv2.cvtColor(testPerson, cv2.COLOR_RGB2BGR)

# Finding the frame & matching
for (top, right, bottom, left), face_encoding in zip(face_positions, all_encodings):
    name = 'Unknown Person'

    # Comparing training set to unknown
    matches = face_recognition.compare_faces(encodings, face_encoding, tolerance=0.6)

    # Adjusting tolerance for multiple face positions
    if len(face_positions) > 1:
        matches = face_recognition.compare_faces(encodings, face_encoding, tolerance=0.5)

    if True in matches:
        first_match_index = matches.index(True)
        name = names[first_match_index]

    cv2.rectangle(testPerson, (left, top), (right, bottom), (255, 255, 0), 2)
    cv2.putText(testPerson, name, (left, top - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

# Display the image
cv2.imshow('Test Window', testPerson)
cv2.moveWindow('Test Window', 0, 0)

# Wait for a key event and then close the window
cv2.waitKey(0)
cv2.destroyAllWindows()
