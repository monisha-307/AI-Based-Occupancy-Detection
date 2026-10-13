import cv2
from ultralytics import YOLO

# YOLO model
model = YOLO("yolo11n.pt")

# Camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:

    ret, frame = cap.read()

    if not ret:
        print("Camera frame varala")
        break

    # YOLO person detection
    results = model(frame, conf=0.15)

    person_count = 0

    for result in results:
        for box in result.boxes:

            class_id = int(box.cls[0])

            # Person class = 0
            if class_id == 0:

                person_count += 1

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    "Person",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

    # Automatic PIR simulation
    if person_count > 0:
        pir_motion = True
    else:
        pir_motion = False

    # Fusion
    if person_count > 0 or pir_motion:
        status = "OCCUPIED"
    else:
        status = "NOT OCCUPIED"

    # Display people count
    cv2.putText(
        frame,
        f"People: {person_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Display PIR status
    cv2.putText(
        frame,
        f"PIR: {pir_motion}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Display occupancy status
    cv2.putText(
        frame,
        status,
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Show camera
    cv2.imshow("AI Occupancy Detection", frame)

    # Press Q to quit
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

# Close
cap.release()
cv2.destroyAllWindows()