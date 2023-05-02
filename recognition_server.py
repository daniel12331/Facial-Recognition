import cv2
import face_recognition
from flask import Flask, Response, render_template
import numpy
import math
import os
from waitress import serve


app = Flask(__name__)

def face_confidence(face_distance, face_match_threshold=0.6):
    range = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + '%'
    else:
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + '%'

def run_recognition():
    # Load known faces from images
    known_face_encodings = []
    known_face_names = []
    for image in os.listdir('faces'):
            face_image = face_recognition.load_image_file("faces/{image}")
            face_encoding = face_recognition.face_encodings(face_image)[0]
            known_face_encodings.append(face_encoding)
            known_face_names.append(image)

    # Start capturing video
    cap = cv2.VideoCapture(0)

    while True:
        # Read a frame from the video
        success, frame = cap.read()
        if not success:
            break

        # Resize frame of video to a smaller size
        frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        
        # Convert the frame from BGR to RGB
        rgb_frame = frame[:, :, ::-1]

        # Find all the faces and face encodings in the frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Loop through each face in the frame
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # See if the face is a match for the known faces
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # Use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = numpy.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                confidence = face_confidence(face_distances[best_match_index])

            # Draw a rectangle around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            # Draw the name of the face below the rectangle
            cv2.rectangle(frame, (left, bottom - 20), (right, bottom), (0, 255, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, f"{name} ({confidence})", (left + 6, bottom - 6), font, 0.5, (0, 0, 0), 1)

        # Encode the frame as JPEG and yield it
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(run_recognition(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=8080)
