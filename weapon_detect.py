
from ultralytics import YOLO
import cv2
import requests

MODEL_PATH = "weapon_detector.pt"
BACKEND_URL = "http://localhost:4000/detection"

# Lower threshold for initial debugging
CONF = 0.70

CAM_ID = 0
IMG_SIZE = 640


def main():
    model = YOLO(MODEL_PATH)
    names = model.names

    print("Model loaded successfully!")
    print("Available classes:", names)

    cap = cv2.VideoCapture(CAM_ID)

    if not cap.isOpened():
        print("Could not open webcam")
        return

    print("Weapon Detection Started. Press Q to quit.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Could not read webcam frame")
            break

        # Run YOLO inference
        results = model.predict(
            source=frame,
            conf=CONF,
            imgsz=IMG_SIZE,
            verbose=False
        )

        r = results[0]
        boxes = r.boxes

        weapon_count = 0

        if boxes is not None and len(boxes) > 0:

            for b in boxes:

                x1, y1, x2, y2 = map(
                    int, b.xyxy[0].tolist()
                )

                cls_id = int(b.cls[0].item())
                confidence = float(b.conf[0].item())

                # Get actual predicted class
                label = names.get(cls_id, str(cls_id))

                weapon_count += 1

                # Notify backend (fire-and-forget, ignore errors)
                try:
                    requests.post(BACKEND_URL, json={"label": label, "confidence": confidence}, timeout=0.5)
                except Exception:
                    pass

                # Red bounding box
                color = (0, 0, 255)

                # Display class name and confidence
                text = f"{label.upper()} {confidence:.2f}"

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    3
                )

                # Label background
                (tw, th), _ = cv2.getTextSize(
                    text,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    2
                )

                label_y = max(y1 - th - 12, 0)

                cv2.rectangle(
                    frame,
                    (x1, label_y),
                    (x1 + tw + 10, label_y + th + 10),
                    color,
                    -1
                )

                cv2.putText(
                    frame,
                    text,
                    (x1 + 5, label_y + th + 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

        # Display detection count
        if weapon_count > 0:
            alert_text = f"ALERT: {weapon_count} Detection(s)"

            cv2.putText(
                frame,
                alert_text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2
            )

        else:
            cv2.putText(
                frame,
                "No objects detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        cv2.imshow("Weapon Detection - Live", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
