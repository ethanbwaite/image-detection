import cv2
import numpy as np
import socket

HOST = '136.27.22.160' # Desktop IP
PORT = 8888

class VideoClient:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.server_socket = None
        self.client_socket = None

    def start(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)
        print(f"Waiting for connection on {self.host}:{self.port}...")

        self.client_socket, address = self.server_socket.accept()
        print(f"Connected to {address}")

    def send_frames(self):
        cap = cv2.VideoCapture(1)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Do some processing on the frame if needed

            # Convert the frame to a byte string
            retval, buffer = cv2.imencode('.jpg', frame)
            data = buffer.tobytes()

            # Send the frame over the network
            self.client_socket.send(data)

        cap.release()
        self.client_socket.close()

    def stop(self):
        self.server_socket.close()

if __name__ == '__main__':
    server = VideoClient(HOST, PORT)
    server.start()
    server.send_frames()
    server.stop()