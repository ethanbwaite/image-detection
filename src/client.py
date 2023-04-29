import cv2
import numpy as np
import socket

HOST = '136.27.22.160' # Desktop IP
PORT = 8888

class VideoClient:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.socket = None

    def connect(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.host, self.port))
        print(f"Connected to {self.host}:{self.port}")

    def receive_frames(self):
        while True:
            # Receive the data from the socket
            data = self.socket.recv(4096)

            # If no data received, break the loop
            if not data:
                break

            # Convert the data to a cv2 frame
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Display the frame or do some processing on it
            cv2.imshow('Received Frame', frame)

            # Wait for a key press to exit the program
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
        self.socket.close()

if __name__ == '__main__':
    client = VideoClient(HOST, PORT)
    client.connect()
    client.receive_frames()