import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import os
from datetime import datetime
import time

from db import save_to_db
from ocr import extract_number_plate

# Load model
model = YOLO(r"C:\Users\ananya_sharma\Downloads\object_detection\runs\detect\train\weights\best.pt")

cap = cv2.VideoCapture(0)

# save folder
save_path = "violations"
os.makedirs(save_path, exist_ok=True)

LINE_Y = 300

signal_buffer = deque(maxlen=10)

# spam control
last_saved_time = 0

# 🔴 RED LIGHT DETECTION
def detect_red_hsv(frame):
    roi = frame[50:150, 200:450]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 120, 120])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([170, 120, 120])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    red_mask = mask1 + mask2

    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

    red_pixels = cv2.countNonZero(red_mask)

    print("Red Pixels:", red_pixels)

    return red_pixels > 100


# 🚗 tracking
track_history = {}

while True:
    detected_plate = "UNKNOWN"

    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, imgsz=320, conf=0.5)

    # RED LIGHT
    red_light_now = detect_red_hsv(frame)
    signal_buffer.append(red_light_now)
    red_light = sum(signal_buffer) > len(signal_buffer)//2

    cv2.line(frame, (0, LINE_Y), (frame.shape[1], LINE_Y), (0, 0, 255), 2)

    class_names = ["with helmet", "without helmet", "rider", "number plate"]

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])

            if cls >= len(class_names):
                continue

            label_name = class_names[cls]

            color = (0, 255, 0)

            if label_name == "without helmet":
                color = (0, 0, 255)
            elif label_name == "number plate":
                color = (255, 255, 0)

            # 🟡 NUMBER PLATE OCR
            if label_name == "number plate":
                plate_crop = frame[y1:y2, x1:x2]

                if plate_crop.size > 0:
                    detected_plate = extract_number_plate(plate_crop)

                    cv2.putText(frame, f"Plate: {detected_plate}",
                                (x1, y1 - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (255, 255, 0), 2)

                    print("🔢 Plate:", detected_plate)

            # 🚨 HELMET VIOLATION
            if label_name == "without helmet":
                current_time = time.time()

                if current_time - last_saved_time > 3:
                    last_saved_time = current_time

                    cv2.putText(frame, "HELMET VIOLATION!", (x1, y2+20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{save_path}/helmet_{timestamp}.jpg"

                    cv2.imwrite(filename, frame)

                    print("🚨 HELMET SAVED:", filename)

                    save_to_db(detected_plate, "No Helmet", filename)

            # 🚗 RIDER LOGIC
            if label_name == "rider":
                center_x = (x1 + x2) // 2
                obj_id = f"{center_x}"

                if obj_id not in track_history:
                    track_history[obj_id] = deque(maxlen=5)

                track_history[obj_id].append(center_x)

                # 🚫 WRONG SIDE VIOLATION
                if len(track_history[obj_id]) >= 2:
                    direction = track_history[obj_id][-1] - track_history[obj_id][0]

                    if direction < -20:
                        current_time = time.time()

                        if current_time - last_saved_time > 3:
                            last_saved_time = current_time

                            cv2.putText(frame, "WRONG SIDE VIOLATION!",
                                        (x1, y2 + 40),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                        (0, 0, 255), 2)

                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"{save_path}/wrongside_{timestamp}.jpg"

                            cv2.imwrite(filename, frame)

                            print("🚫 WRONG SIDE SAVED:", filename)

                            save_to_db(detected_plate, "Wrong Side", filename)

                # 🚨 RED LIGHT VIOLATION
                if y2 > LINE_Y and red_light:
                    current_time = time.time()

                    if current_time - last_saved_time > 3:
                        last_saved_time = current_time

                        cv2.putText(frame, "RED LIGHT VIOLATION!",
                                    (x1, y2 + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    (0, 0, 255), 2)

                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"{save_path}/redlight_{timestamp}.jpg"

                        cv2.imwrite(filename, frame)

                        print("🚨 RED LIGHT SAVED:", filename)

                        save_to_db(detected_plate, "Red Light", filename)

            # DRAW BOX
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            cv2.putText(frame, label_name,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        color, 2)

    # SIGNAL DISPLAY
    status = "RED" if red_light else "GREEN"
    color = (0, 0, 255) if red_light else (0, 255, 0)

    cv2.putText(frame, f"Signal: {status}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    cv2.imshow("Traffic Violation System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()