import cv2
import os
import sys
import csv
import numpy as np

# === OpenPose DLL Setup ===
os.environ["PATH"] += os.pathsep + "C:/Users/ameli/openpose/build/x64/Release"
os.environ["PATH"] += os.pathsep + "C:/Users/ameli/openpose/build/bin"
sys.path.append("C:/Users/ameli/openpose/build/python/openpose/Release")

import pyopenpose as op

# === Load OpenCV Gender Classifier ===
MODEL_DIR = "C:/Users/ameli/Desktop/DanceAI/openCVmodels"
protoPath = os.path.join(MODEL_DIR, "deploy_gender.prototxt")
modelPath = os.path.join(MODEL_DIR, "gender_net.caffemodel")
genderNet = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
GENDER_LIST = ["male", "female"]

# === OpenPose Config ===
params = {
    "model_folder": "C:/Users/ameli/openpose/models/",
    "model_pose": "BODY_25",
    "face": True,
    "net_resolution": "-1x368",
    "disable_blending": False
}

opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# === Load Video ===
video_path = "videos/Salsa Right Turn - Beginner Salsa Turns On2 _ TheDanceDojo.com.mp4"
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

# === Prepare CSV Output ===
csv_path = "output_videos/rightTurnVideo/output_labeled_feet.csv"

# === Helper to crop face region from keypoints ===
def crop_face_from_keypoints(frame, face_keypoints):
    valid = face_keypoints[:, 2] > 0.3
    if not np.any(valid):
        return None
    points = face_keypoints[valid][:, :2]
    x, y, w, h = cv2.boundingRect(points.astype(np.int32))
    margin = 10
    x = max(0, x - margin)
    y = max(0, y - margin)
    w += 2 * margin
    h += 2 * margin
    return frame[y:y + h, x:x + w]

# === Gender detection from face crop ===
def classify_gender(face_img):
    blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
    genderNet.setInput(blob)
    preds = genderNet.forward()
    return GENDER_LIST[preds[0].argmax()]

# === MAIN LOOP ===
person_labels = {}  # person_idx -> gender
frame_num = 0

with open(csv_path, mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([
        "frame", "timestamp", "label",  # label = 'male' or 'female'
        "left_toe_x", "left_toe_y",
        "left_heel_x", "left_heel_y",
        "right_toe_x", "right_toe_y",
        "right_heel_x", "right_heel_y"
    ])

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        datum = op.Datum()
        datum.cvInputData = frame
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))

        pose_kps = datum.poseKeypoints
        face_kps = datum.faceKeypoints

        if pose_kps is not None and face_kps is not None:
            num_people = pose_kps.shape[0]
            for i in range(num_people):
                # Try to classify if not already labeled
                if i not in person_labels:
                    face_crop = crop_face_from_keypoints(frame, face_kps[i])
                    if face_crop is not None:
                        gender = classify_gender(face_crop)
                        person_labels[i] = gender

                        # Assign other person opposite gender if only two
                        if len(person_labels) == 1 and num_people == 2:
                            other = 1 - i
                            person_labels[other] = "female" if gender == "male" else "male"

                # Skip if still not labeled
                if i not in person_labels:
                    continue

                gender = person_labels[i]
                person = pose_kps[i]

                # Extract toe and heel Y positions with confidence check
                l_toe_x = person[19][0] if person[19][2] > 0.1 else None
                l_toe_y = person[19][1] if person[19][2] > 0.1 else None

                l_heel_x = person[21][0] if person[21][2] > 0.1 else None
                l_heel_y = person[21][1] if person[21][2] > 0.1 else None

                r_toe_x = person[22][0] if person[22][2] > 0.1 else None
                r_toe_y = person[22][1] if person[22][2] > 0.1 else None

                r_heel_x = person[24][0] if person[24][2] > 0.1 else None
                r_heel_y = person[24][1] if person[24][2] > 0.1 else None

                timestamp = round(frame_num / fps, 3)
                writer.writerow([
                    frame_num, round(timestamp, 3), gender,
                    l_toe_x, l_toe_y,
                    l_heel_x, l_heel_y,
                    r_toe_x, r_toe_y,
                    r_heel_x, r_heel_y
                ])


        frame_num += 1

cap.release()
print(" CSV with labeled foot positions saved: {csv_path}")
