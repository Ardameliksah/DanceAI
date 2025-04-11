import cv2
import pandas as pd
import numpy as np

# === Paths ===
video_path = "C:/Users/ameli/Desktop/DanceAI/videos/Salsa Right Turn - Beginner Salsa Turns On2 _ TheDanceDojo.com.mp4"
csv_path = "C:/Users/ameli/Desktop/DanceAI/output_videos/rightTurnVideo/output_labeled_feet.csv"

# === Load Video and CSV ===
cap = cv2.VideoCapture(video_path)
df = pd.read_csv(csv_path)

# === Video Properties ===
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# === Define Colors ===
colors = {
    "male": (255, 0, 0),      # Blue
    "female": (255, 0, 255),  # Purple
    "unknown": (0, 255, 255)  # Yellow
}

# === Main Loop ===
frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame_idx >= len(df):
        break

    row = df.iloc[frame_idx]
    label = row['label']
    color = colors.get(label, (0, 255, 255))

    # Draw Left Heel
    if pd.notna(row['left_heel_x']) and pd.notna(row['left_heel_y']):
        x = int(row['left_heel_x'])
        y = int(row['left_heel_y'])
        cv2.circle(frame, (x, y), 8, color, -1)

    # Draw Right Heel
    if pd.notna(row['right_heel_x']) and pd.notna(row['right_heel_y']):
        x = int(row['right_heel_x'])
        y = int(row['right_heel_y'])
        cv2.circle(frame, (x, y), 8, color, -1)

    # Add Label
    cv2.putText(frame, f"Frame {frame_idx} - {label}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

    cv2.imshow("Heel Overlay", frame)
    if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
        break

    frame_idx += 1

cap.release()
cv2.destroyAllWindows()
