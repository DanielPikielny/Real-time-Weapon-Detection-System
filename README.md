# Real-time-Weapon-Detection-System
This project is a distributed real-time weapon detection system. It consists of a central server and multiple client-side jetson nano devices (or an alternative such as a raspberry.
The server uses a YOLO model to perform object detection and sends back the results to the clients, which then trigger alarms based on the detections.

Technologies and Libraries
Python

Flask: Web framework for the server API.

Flask-SocketIO: Enables real-time, bidirectional communication between the server and clients.

Ultralytics YOLO: A library for object detection using YOLO models.

OpenCV (cv2): Used on the client-side for video capture, processing, and display.

requests: Used by the clients to send video frames to the server.

pygame: Used on the client-side to play alarm sounds.

Features
Distributed Architecture: The system is designed to run on a central server and multiple client devices (e.g., Jetson boards).

Real-time Inference: Video frames are sent from the clients to the server for real-time weapon detection using a YOLO model.

Visual Feedback: The client applications display the video feed with bounding boxes and confidence scores around detected objects.

Audio Alarms: Different alarm sounds are triggered on the clients based on the detection source (e.g., local detection, detection from another client, or both).

Client-Server Communication: The system uses both HTTP requests for sending frames and WebSocket connections for real-time status updates and command messages.

Authentication: Clients are authenticated using API keys to ensure only authorized devices can connect to the server.

How to Run
Server
Install Dependencies:

Bash

pip install Flask flask-socketio ultralytics opencv-python numpy
Configuration: The server uses final_distance.pt as its model. You will need to place this file in the same directory.

Run the Server:

Bash

python Server.py
The server will start listening for client connections and inference requests.

Client (Jetson Devices)
Install Dependencies:

Bash

pip install opencv-python requests numpy pygame socketio
Configuration:

Set the VM_IP and VM_PORT in Jet1.py and Jet2.py to match the server's address.

Make sure the alarm sound files (Alarm_ME.mp3, Alarm_OTHER.mp3, etc.) are located at the specified paths or update the paths.

Run the Clients:

Bash

# For Jetson 1
python Jet1.py
# For Jetson 2
python Jet2.py
The clients will connect to the server, start capturing video, and send frames for inference.

Code Structure
Server.py: Contains the Flask application and the WeaponDetectionServer class, which handles API routes, WebSocket communication, and YOLO inference.

Jet1.py & Jet2.py: These files contain the WeaponDetectionClient class. They are responsible for video capture, sending frames to the server, receiving results, and managing visual and audio feedback.
