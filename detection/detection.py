import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("face_mask_model.keras")

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

camera = cv2.VideoCapture(0)

while True:
    success, frame = camera.read()

    if not success:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50)
    )

    for x, y, w, h in faces:
        face = frame[y:y + h, x:x + w]

        face = cv2.resize(face, (128, 128))
        face = face.astype("float32") / 255.0
        face = np.expand_dims(face, axis=0)

        prediction = model.predict(face, verbose=0)[0][0]

        if prediction >= 0.5:
            label = "With Mask"
        else:
            label = "Without Mask"

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    cv2.imshow("Face Mask Detection System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()