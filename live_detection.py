import cv2
from ultralytics import YOLO

# Load your trained YOLO model
model = YOLO(r"C:\Users\ananya_sharma\Downloads\object_detection\runs\detect\train\weights\best.pt")   # <-- Replace with your path

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO on the frame
    results = model(frame)

    # Visualize detections on frame
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # XYXY coordinates
            x1, y1, x2, y2 = box.xyxy[0]

            # Confidence & Class
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            # Class names (from your data.yaml)
            class_names = ["with helmet", "without helmet", "rider", "number plate"]
            label = f"{class_names[cls]} {conf:.2f}"

            # Draw rectangle
            cv2.rectangle(
                frame, 
                (int(x1), int(y1)), 
                (int(x2), int(y2)), 
                (0, 255, 0), 
                2
            )

            # Put class label
            cv2.putText(
                frame,
                label,
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

    # Show the frame
    cv2.imshow("Helmet & Number Plate Detection - Live", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
