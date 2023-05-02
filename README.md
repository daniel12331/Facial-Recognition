# Facial Recognition - Python
Distributed System Project

## ![Facial Recogntion](https://user-images.githubusercontent.com/62666691/235709950-fb454df4-7c8d-47ce-b7c4-76640d19f639.png)

## About
This project is a Flask application that uses the face recognition library to detect and recognize faces in real-time from a video stream. The application loads known faces from images stored in a directory, and compares them to faces detected in the video stream to determine if they match. If a match is found, the name of the known person is displayed on the video stream along with a confidence score.

The video stream is generated using the OpenCV library, and is served to the client using the Flask web framework. The application uses the Waitress web server to handle incoming requests, and is capable of handling multiple requests concurrently without the use of threads.

The project is useful for a variety of applications, such as security systems, attendance tracking, and personalized user experiences. The face recognition algorithm used in the project is accurate and efficient, and can be easily extended to recognize new faces by adding images to the known faces directory.



### Used the following tools: 

* [Facial Recognition Libary](https://pypi.org/project/face-recognition/)
* [OpenCV](https://opencv.org/)
* [Waitress](https://flask.palletsprojects.com/en/2.2.x/deploying/waitress/)


### Install and Launch

```
pip install requirements.txt
```
```
python server.py
```
