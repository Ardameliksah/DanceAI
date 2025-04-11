import cv2
import os
import sys
import numpy as np
import csv
# Add OpenPose paths
os.environ["PATH"] += os.pathsep + "C:/Users/ameli/openpose/build/x64/Release"
os.environ["PATH"] += os.pathsep + "C:/Users/ameli/openpose/build/bin"
os.environ["PATH"] += os.pathsep + "C:/Users/ameli/openpose/build/python/openpose/Release"
sys.path.append("C:/Users/ameli/openpose/build/python/openpose/Release")

# Import OpenPose
import pyopenpose as op

# Set up OpenPose parameters
params = {
    "model_folder": "C:/Users/ameli/openpose/models/",
    "model_pose": "BODY_25",
    "net_resolution": "-1x368",
    "disable_blending": False,
    "number_people_max":2,
}

# Initialize OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# Load video
video_path = "videos/Salsa Right Turn - Beginner Salsa Turns On2 _ TheDanceDojo.com.mp4"
cap = cv2.VideoCapture(video_path)

# Optional: Save output video
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter("output_salsa.avi", cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

frame_num = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    datum = op.Datum()
    datum.cvInputData = frame
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))

    print(f"Frame {frame_num} keypoints:")
    print(datum.poseKeypoints)

    # Show result
    cv2.imshow("OpenPose Salsa", datum.cvOutputData)
    out.write(datum.cvOutputData)
    frame_num += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()