import cv2
import numpy as np
import socket
import sys

# Set up the video capture object
cap = cv2.VideoCapture(0)

# Set up the socket
HOST = sys.argv[1] # The IP address of the receiving computer
PORT = 65432
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST, PORT))

# Set up the video codec and writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('send.avi', fourcc, 20.0, (640, 480))

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()

    # Write the frame to the video file
    out.write(frame)

    # Convert the frame to a string and send it over the network
    data = np.array(frame)
    stringData = data.tostring()
    s.send(str(len(stringData)).ljust(16).encode())
    s.send(stringData)

    # Display the frame
    cv2.imshow('Sending...', frame)

    # Exit if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and video writer
cap.release()
out.release()

# Close the socket
s.close()

# Close the display window
cv2.destroyAllWindows()