import cv2, sys

def select_camera():
    num_cameras = int(sys.argv[1])  # Change this to the number of cameras you want to try
    for i in range(num_cameras):
        cap = cv2.VideoCapture(i)
        ret, frame = cap.read()
        if ret:
            print("Selected camera: ", i)
            return cap
        else:
            cap.release()
    print("Could not find any camera. Exiting.")
    exit()

select_camera()