import cv2
import numpy as np
import socket
import pickle
import struct
import image_detection

HOST = '192.168.1.123' # Desktop IP
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

        data = b'' ### CHANGED
        payload_size = struct.calcsize("L") ### CHANGED

        while True:
            # Retrieve message size
            while len(data) < payload_size:
                data += self.socket.recv(4096)

            packed_msg_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = struct.unpack("L", packed_msg_size)[0] ### CHANGED

            # Retrieve all data based on message size
            while len(data) < msg_size:
                data += self.socket.recv(4096)

            frame_data = data[:msg_size]
            data = data[msg_size:] 

            # Extract frame
            frame = pickle.loads(frame_data)
            
            processed_frame = image_detection.detect_objects(frame)

            cv2.imshow('frame', processed_frame)

 
            # Wait for a key press to exit the program
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
        self.socket.close()

if __name__ == '__main__':
    client = VideoClient(HOST, PORT)
    client.connect()
    client.receive_frames()