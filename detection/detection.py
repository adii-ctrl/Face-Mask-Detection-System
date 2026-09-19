import cv2
import numpy as np
import tensorflow as tf

IMAGE_SIZE = 128

model = tf.keras.models.load_model("face_mask_model.keras")

print("Model loaded successfully!")

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_alt2.xml"
)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Camera could not be opened.")
    exit()

print("Camera started.")
print("Press Q to quit.")

# Previous detection
previous_faces = []

# Number of frames to keep detection alive
HOLD_FRAMES = 8

while True:

    success, frame = camera.read()

    if not success:
        print("Error: Could not read camera frame.")
        break

    frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=5,
        minSize=(50, 50)
    )

    current_faces = []

    # If face detected
    for x, y, w, h in faces:

        margin = 20

        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(frame.shape[1], x + w + margin)
        y2 = min(frame.shape[0], y + h + margin)

        face = frame[y1:y2, x1:x2]

        if face.size == 0:
            continue

        face = cv2.resize(
            face,
            (IMAGE_SIZE, IMAGE_SIZE)
        )

        face = cv2.cvtColor(
            face,
            cv2.COLOR_BGR2RGB
        )

        face = face.astype("float32") / 255.0

        face = np.expand_dims(face, axis=0)

        prediction = model.predict(
            face,
            verbose=0
        )[0][0]

        if prediction < 0.5:
            label = "With Mask"
            box_color = (0, 255, 0)
        else:
            label = "Without Mask"
            box_color = (0, 0, 255)

        current_faces.append(
            (x1, y1, x2, y2, label, box_color)
        )

    # If current frame detected a face
    if current_faces:

        previous_faces = current_faces
        hold_counter = HOLD_FRAMES

    else:

        # Keep previous detection for a few frames
        if previous_faces and hold_counter > 0:
            hold_counter -= 1
        else:
            previous_faces = []

    # Draw detection
    for x1, y1, x2, y2, label, box_color in previous_faces:

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            box_color,
            2
        )

        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            box_color,
            2
        )

    cv2.imshow(
        "Face Mask Detection System",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

print("Camera closed.")