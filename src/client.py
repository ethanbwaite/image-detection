import cv2
import numpy as np
import socket
import pickle
import struct
import image_detection
import time
import util, image_util
import traceback

HOST = '192.168.1.123' # Desktop IP
PORT = 8888

log = util.get_logger()

"""
Client that receives videos streamed from the server and applies processing to the frames
"""
class VideoClient:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.socket = None

    def connect(self):
        log.info(f"Connecting to {self.host}:{self.port}")
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.host, self.port))
        log.info(f"Connected to {self.host}:{self.port}")


    def receive_frames(self):
        period = 10 
        tick = 0
        total_time = 0
        data = b'' ### CHANGED
        payload_size = struct.calcsize("L") ### CHANGED

        # Image detection config settings
        should_process_frame = True
        should_log_detections = False

        # Keep track of past frames
        frame_history = []
        max_frames = 10

        # Keep track of unique objects
        known_objects = {}

        # Keep track of metadata associated with a known object
        # Such as age, time since last detection
        known_object_metadata = {}

        while True:
            try:
                t0 = time.perf_counter()

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

                # Perform object detection
                processed_frame, frame_metadata = image_detection.detect_objects(frame, should_process_frame, should_log_detections)

                frame_history.append(frame_metadata)
                if len(frame_history) > max_frames:
                    frame_history.pop(0)

                # Draw historical objects
                if should_process_frame:

                    # Track objects
                    known_objects = image_util.track_object(known_objects=known_objects, 
                                                            known_object_metadata=known_object_metadata,
                                                            candidate_objects=frame_metadata["boxes"])

                    # Draw historical tracking data
                    for past_frame_metadata in reversed(frame_history[:-1]):
                        processed_frame = image_detection.draw_detected_objects(processed_frame, past_frame_metadata, greyscale=True)

                    # Draw this frame's detections (on top of history)
                    # processed_frame = image_detection.draw_detected_objects(processed_frame, frame_metadata, greyscale=False)
                    processed_frame = image_detection.draw_known_objects(frame=processed_frame, known_objects=known_objects)

                # Calculate latency
                t1 = time.perf_counter()
                total_time += t1 - t0
                tick += 1
                
                if tick > period:
                    tick = 0
                    log.info(f"Average latency per frame over {period} frames: [{total_time / period}s]")
                    total_time = 0

                cv2.imshow('frame', processed_frame)

    
                # Wait for a key press to exit the program
                key = cv2.waitKey(1)
                if key == ord('d'):
                    should_process_frame = not should_process_frame
                if key == ord('l'):
                    should_log_detections = not should_log_detections
                if key == ord('q'):
                    break
            except Exception as e:
                log.error(f"Unable to process frame: {e}")
                traceback.print_exc()

        cv2.destroyAllWindows()
        self.socket.close()

if __name__ == '__main__':
    client = VideoClient(HOST, PORT)
    client.connect()
    client.receive_frames()