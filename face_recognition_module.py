
from pathlib import Path

import cv2
import numpy as np
from deepface import DeepFace

BASE_DIR = Path(__file__).resolve().parent
REGISTERED_DIR = BASE_DIR / "registered_faces"

MODEL_NAME = "Facenet512"
THRESHOLD = 0.30

CASCADE_FILES = [
    "haarcascade_frontalface_alt2.xml",
    "haarcascade_frontalface_default.xml",
]

face_detectors = [
    cv2.CascadeClassifier(cv2.data.haarcascades + name)
    for name in CASCADE_FILES
]


def detect_faces(image):
    """Try multiple face detectors on the supplied image."""
    if image is None or image.size == 0:
        return []

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    for detector in face_detectors:
        if detector.empty():
            continue

        faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=4,
            minSize=(30, 30)
        )

        if len(faces) > 0:
            return list(faces)

    return []


def crop_largest_face(image):
    """Crop the largest detected face."""
    faces = detect_faces(image)

    if not faces:
        return None

    x, y, w, h = max(
        faces,
        key=lambda box: box[2] * box[3]
    )

    mx = int(w * 0.15)
    my = int(h * 0.15)

    x1 = max(0, x - mx)
    y1 = max(0, y - my)
    x2 = min(image.shape[1], x + w + mx)
    y2 = min(image.shape[0], y + h + my)

    face = image[y1:y2, x1:x2]

    return face if face.size > 0 else None


def crop_reference_face(image_path):
    image = cv2.imread(str(image_path))

    if image is None:
        return None

    return crop_largest_face(image)


def get_embedding(face_image):
    if face_image is None or face_image.size == 0:
        return None

    result = DeepFace.represent(
        img_path=face_image,
        model_name=MODEL_NAME,
        detector_backend="skip",
        enforce_detection=False
    )

    if not result:
        return None

    embedding = np.asarray(
        result[0]["embedding"],
        dtype=np.float32
    )

    norm = np.linalg.norm(embedding)

    if norm == 0:
        return None

    return embedding / norm


known_faces = []


def load_registered_embeddings():
    """Load registered photo embeddings when the program starts."""
    global known_faces
    known_faces = []

    extensions = {".jpg", ".jpeg", ".png"}

    if not REGISTERED_DIR.exists():
        print("ERROR: Registered folder not found:", REGISTERED_DIR)
        return

    for person_dir in sorted(REGISTERED_DIR.iterdir()):
        if not person_dir.is_dir():
            continue

        for photo in sorted(person_dir.iterdir()):
            if photo.suffix.lower() not in extensions:
                continue

            try:
                face = crop_reference_face(photo)

                if face is None:
                    print("SKIPPED - no face detected:", photo.name)
                    continue

                embedding = get_embedding(face)

                if embedding is not None:
                    known_faces.append(
                        (person_dir.name, embedding)
                    )
                    print("Loaded:", person_dir.name, "/", photo.name)

            except Exception as error:
                print("LOAD ERROR:", photo.name, error)

    print("Total reference faces loaded:", len(known_faces))


def identify_person(face_image):
    """Compare a live face crop against registered embeddings."""
    if not known_faces:
        return "Unknown"

    try:
        # The caller already supplies a detected face crop.
        # Avoid cropping it again, which could remove useful features.
        live = get_embedding(face_image)

        if live is None:
            return "Unknown"

        best_person = "Unknown"
        best_distance = float("inf")

        for person, reference in known_faces:
            if reference.shape != live.shape:
                continue

            distance = 1.0 - float(np.dot(live, reference))

            if distance < best_distance:
                best_distance = distance
                best_person = person

        print(
            f"\n[RESULT] Best match: {best_person} "
            f"| distance: {best_distance:.4f}"
        )

        if best_distance <= THRESHOLD:
            return best_person

        return "Unknown"

    except Exception as error:
        print("\n[RECOGNITION ERROR]", error)
        return "Unknown"
