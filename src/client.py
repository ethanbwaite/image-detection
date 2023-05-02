import cv2
import numpy as np
import socket
import pickle
import struct
import image_detection
import time
import util, image_util
import traceback
import base64
import shapely
from pathlib import Path

HOST = '192.168.1.123' # Desktop IP
PORT = 8888
WIDTH = 1280
HEIGHT = 720
SAVE_FILE_PERIOD = 10
base_path = Path(__file__).parent
SAVE_FILE_NAME = "records.csv"
BOX_DETECTION_SAVE_FILE_NAME = "records-box.csv"

SAVE_FILE_PATH = (base_path / f"../data/{SAVE_FILE_NAME}").resolve()

BOX_DETECTION_SAVE_FILE_PATH = (base_path / f"../data/{BOX_DETECTION_SAVE_FILE_NAME}").resolve()

log = util.get_logger()
mouse_coords = (0, 0)
global_p0 = None
global_p1 = None

global_detection_paths = []

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

    # Create a mouse callback function
    def draw_mouse_coordinates(self, event, x, y, flags, param):
        global mouse_coords, global_detection_paths
        if event == cv2.EVENT_MOUSEMOVE:
            # Convert the coordinates to a string and display it on the frame
            mouse_coords = (x, y)
        if event == cv2.EVENT_LBUTTONUP:
            global_detection_paths.append(mouse_coords)
            log.info(f"Detection Line: {global_detection_paths}")
        if event == cv2.EVENT_RBUTTONUP:
            global_detection_paths = []

    def draw_annotations(self, frame, object_detected=False):

        # Draw current mouse coordinates
        cv2.putText(frame, "({}, {})".format(mouse_coords[0], mouse_coords[1]), mouse_coords, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(frame, "({}, {})".format(mouse_coords[0], mouse_coords[1]), mouse_coords, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Draw annotation lines
        if len(global_detection_paths) > 1:
            for i in range(len(global_detection_paths)-1):
                p0 = global_detection_paths[i]
                p1 = global_detection_paths[i+1]
                cv2.line(frame, p0, p1, (0,255,0), 2)

                cv2.putText(frame, "({}, {})".format(p0[0], p0[1]), p0, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                cv2.putText(frame, "({}, {})".format(p0[0], p0[1]), p0, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                cv2.putText(frame, "({}, {})".format(p1[0], p1[1]), p1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                cv2.putText(frame, "({}, {})".format(p1[0], p1[1]), p1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
        return frame


    def receive_frames(self):
        period = 100
        tick = 0
        total_time = 0
        total_inference_time = 0
        total_track_time = 0
        data = b''
        payload_size = struct.calcsize("L")

        # Image detection config settings
        should_process_frame = True
        should_log_detections = False
        should_draw_paths = False
        should_draw_history = False


        # Keep track of past frames
        frame_history = []
        max_frames = 10

        """
        Keep track of unique objects.
        known_objects is a dictionary of unique object IDs to bounding box coordinates
        in float form (each vector coordinate has range [0.0, 1.0])
        """
        known_objects = {}

        # Keep track of metadata associated with a known object
        # Such as age, time since last detection
        known_object_metadata = {}

        # Track objects that crossed the mouse defined line
        tallied_objects = {'person': 0, 'car': 0, 'motorcycle': 0, 'airplane': 0, 'bus': 0, 'truck': 0}
        
        # Record unique objects so we don't double count them
        detected_object_id_set = set()
        
        save_file_t0 = time.time()

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
                # frame = pickle.loads(frame_data)

                # Testing new encoder
                img = base64.b64decode(frame_data)
                npimg = np.fromstring(img, dtype=np.uint8)
                frame = cv2.imdecode(npimg, 1)

                # Perform object detection
                t2 = time.perf_counter()
                processed_frame, frame_metadata = image_detection.detect_objects(frame, 
                                                                                width=WIDTH,
                                                                                height=HEIGHT,
                                                                                should_detect_objects=should_process_frame, 
                                                                                should_log_detections=should_log_detections)
                t3 = time.perf_counter()

                frame_history = util.add_history(frame_history, frame_metadata, max_history=10)

                # Track if an object was detected this frame (crossed the annotation line)
                object_detected = False

                # Detect objects
                if should_process_frame:
                    t_track_0 = time.perf_counter()
                    # Track objects
                    known_objects, known_object_metadata = image_util.track_object(
                        known_objects=known_objects, 
                        known_object_metadata=known_object_metadata,
                        candidate_objects=zip(frame_metadata["labels"], frame_metadata["boxes"])
                    )
                    t_track_1 = time.perf_counter()

                    if len(global_detection_paths) > 1:
                        tallied_objects,
                        detected_object_id_set, 
                        object_detected = image_util.detect_object_crossed_line(global_detection_paths,             
                                                                                known_objects=known_objects,
                                                                                known_object_metadata=known_object_metadata,
                                                                                tally_dict=tallied_objects, 
                                                                                detected_object_id_set=detected_object_id_set,
                                                                                width=WIDTH,
                                                                                height=HEIGHT)
                    # Draw historical tracking data
                    if should_draw_history:
                        for past_frame_metadata in reversed(frame_history[:-1]):
                            processed_frame = image_detection.draw_detected_objects(processed_frame, past_frame_metadata, greyscale=True)

                    # Draw this frame's known objects (on top of history)
                    # processed_frame = image_detection.draw_detected_objects(processed_frame, frame_metadata, greyscale=False)
                    processed_frame = image_detection.draw_known_objects(frame=processed_frame, 
                                                                        known_objects=known_objects,
                                                                        known_object_metadata=known_object_metadata,
                                                                        draw_path=should_draw_paths,
                                                                        draw_history=100)

                # Calculate latency
                t1 = time.perf_counter()
                total_time += t1 - t0
                total_inference_time += t3 - t2
                total_track_time += t_track_1 - t_track_0
                tick += 1

                # Write records dict to CSV file every period
                if (time.time() - save_file_t0) > SAVE_FILE_PERIOD:
                    save_file_t0 = time.time()
                    util.write_dict_to_csv_with_timestamp(tallied_objects, SAVE_FILE_PATH)
                    log.info(f"Records {tallied_objects} saved to {SAVE_FILE_PATH}")
                    util.zero_out_dict_values(tallied_objects)


                
                if tick > period:
                    tick = 0
                    log.info(f"Average total latency over {period} frames: [{total_time / period}s]")
                    log.info(f"Average inference latency over {period} frames: [{total_inference_time / period}s] ({round(total_inference_time/total_time, 3) * 100}% of total)")
                    log.info(f"Average tracking latency over {period} frames: [{total_track_time / period}s] ({round(total_track_time/total_time, 3) * 100}% of total)")
                    log.info(f"Average FPS over {period} frames: [{period/total_time}]")

                    total_time = 0
                    total_inference_time = 0
                    total_track_time = 0

                # Setup frame
                frame_name = 'Traffic Detection'
                cv2.namedWindow(frame_name)
                cv2.setMouseCallback(frame_name, self.draw_mouse_coordinates)

                processed_frame = self.draw_annotations(processed_frame, object_detected)

                # Show frame
                cv2.imshow(frame_name, processed_frame)

    
                # Take user key input from the video frame
                key = cv2.waitKey(1)
                if key == ord('d'):
                    should_process_frame = not should_process_frame
                if key == ord('l'):
                    should_log_detections = not should_log_detections
                if key == ord('p'):
                    should_draw_paths = not should_draw_paths
                if key == ord('h'):
                    should_draw_history = not should_draw_history
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