import cv2
import numpy as np
import socket
import sys
import pickle
import struct
import time
import util
import base64


HOST = '192.168.1.123' # Desktop IP
PORT = 8888

log = util.get_logger()

class VideoServer:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.server_socket = None
        self.client_socket = None

    def start(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)
        log.info(f"Waiting for connection on {self.host}:{self.port}...")

        self.client_socket, address = self.server_socket.accept()
        log.info(f"Connected to {address}")

    def send_frames(self):
        cap = cv2.VideoCapture(1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H', '2', '6', '4'))
        
        period = 10 
        tick = 0
        total_time = 0
        while True:
            
            t0 = time.perf_counter()
            ret, frame = cap.read()

            if not ret:
                break
            # Serialize frame
            # data = pickle.dumps(frame)

            # Testing new encoder
            encoded, buffer = cv2.imencode('.jpg', frame)
            data = base64.b64encode(buffer)

            # Send message length first
            message_size = struct.pack("L", len(data)) ### CHANGED

            # Then data
            self.client_socket.sendall(message_size + data)

            # Calculate latency
            t1 = time.perf_counter()
            total_time += t1 - t0
            tick += 1
            
            if tick > period:
                tick = 0
                log.info(f"METRIC: Average latency per frame over {period} frames: [{total_time / period}s]")
                total_time = 0



        cap.release()
        self.client_socket.close()

    def stop(self):
        self.server_socket.close()

if __name__ == '__main__':
    server = VideoServer(HOST, PORT)
    server.start()
    while True:
        try:
            server.send_frames()
        except:
            log.error("Restarting socket...")
            server.start()
    server.stop()