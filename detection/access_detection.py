
import cv2
import numpy as np
import keras
import sys
import time
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from face_recognition_module import (
    identify_person,
    load_registered_embeddings,
)

IMAGE_SIZE = 128
MASK_THRESHOLD = 0.5
MASK_EVERY_N_FRAMES = 3
RECOGNITION_INTERVAL = 2.0
IDENTITY_TIMEOUT = 4.0

# Load mask model
model = keras.models.load_model(
    str(ROOT / "face_mask_model.keras"),
    compile=False
)
print("Mask model loaded successfully!")

# Load registered reference faces once
load_registered_embeddings()

# Original single face detector
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades
    + "haarcascade_frontalface_alt2.xml"
)

if face_detector.empty():
    raise RuntimeError("Could not load Haar face detector.")

# Camera
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not camera.isOpened():
    raise RuntimeError("Camera could not be opened.")

# Shared recognition state
state_lock = threading.Lock()
person_name = "Unknown"
person_time = 0.0
recognition_running = False
last_recognition_start = 0.0


def recognition_worker(face_image):
    global person_name, person_time, recognition_running

    try:
        result = identify_person(face_image)
    except Exception as error:
        print("\nRecognition error:", error)
        result = "Unknown"

    with state_lock:
        person_name = result
        person_time = time.time()
        recognition_running = False


print("Camera started. Press Q to quit.")

frame_count = 0
last_face_seen = 0.0
cached_mask = None

try:
    while True:
        success, frame = camera.read()

        if not success:
            print("\nCould not read camera frame.")
            break

        frame = cv2.flip(frame, 1)
        now = time.time()

        # Original detector preprocessing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=5,
            minSize=(50, 50)
        )

        if len(faces) == 0:
            if now - last_face_seen > 1.0:
                cached_mask = None

                with state_lock:
                    person_name = "Unknown"
                    person_time = 0.0

            cv2.putText(
                frame,
                "No face detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 165, 255),
                2
            )

        else:
            # Process the largest face
            x, y, w, h = max(
                faces,
                key=lambda item: item[2] * item[3]
            )

            # Original crop margin
            margin = 20
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(frame.shape[1], x + w + margin)
            y2 = min(frame.shape[0], y + h + margin)

            face = frame[y1:y2, x1:x2]

            if face.size > 0:
                last_face_seen = now

                # Predict mask every few frames
                if (
                    cached_mask is None
                    or frame_count % MASK_EVERY_N_FRAMES == 0
                ):
                    resized = cv2.resize(
                        face,
                        (IMAGE_SIZE, IMAGE_SIZE)
                    )

                    face_rgb = cv2.cvtColor(
                        resized,
                        cv2.COLOR_BGR2RGB
                    )

                    batch = np.expand_dims(
                        face_rgb.astype("float32") / 255.0,
                        axis=0
                    )

                    prediction = float(
                        model.predict(batch, verbose=0)[0][0]
                    )

                    cached_mask = prediction < MASK_THRESHOLD

                # Run recognition in background
                with state_lock:
                    should_recognize = (
                        not recognition_running
                        and now - last_recognition_start
                        >= RECOGNITION_INTERVAL
                    )

                    if should_recognize:
                        recognition_running = True
                        last_recognition_start = now

                if should_recognize:
                    threading.Thread(
                        target=recognition_worker,
                        args=(face.copy(),),
                        daemon=True
                    ).start()

                with state_lock:
                    current_person = person_name
                    current_person_time = person_time

                if now - current_person_time > IDENTITY_TIMEOUT:
                    current_person = "Unknown"

                if cached_mask:
                    mask_label = "WITH MASK"
                    box_color = (0, 200, 0)

                    if current_person != "Unknown":
                        status = "REGISTERED - MASK OK"
                        status_color = (0, 200, 0)
                    else:
                        status = "VISITOR - MASK OK"
                        status_color = (0, 200, 255)

                else:
                    mask_label = "WITHOUT MASK"
                    box_color = (0, 0, 255)
                    status = "ACCESS DENIED - WEAR MASK"
                    status_color = (0, 0, 255)

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    box_color,
                    2
                )

                cv2.putText(
                    frame,
                    current_person,
                    (x1, max(25, y1 - 48)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    box_color,
                    2
                )

                cv2.putText(
                    frame,
                    mask_label,
                    (x1, max(45, y1 - 22)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    box_color,
                    2
                )

                cv2.putText(
                    frame,
                    status,
                    (x1, min(frame.shape[0] - 10, y2 + 25)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    status_color,
                    2
                )

        cv2.imshow("Smart Face Mask Access System", frame)
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()
    print("\nCamera closed.")
