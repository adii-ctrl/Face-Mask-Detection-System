import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

IMAGE_SIZE = 128
BATCH_SIZE = 32

validation_data = tf.keras.utils.image_dataset_from_directory(
    "dataset/validation",
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

validation_data = validation_data.map(
    lambda x, y: (x / 255.0, y)
)

model = tf.keras.models.load_model("face_mask_model.keras")

true_labels = []
predicted_labels = []

for images, labels in validation_data:
    predictions = model.predict(images, verbose=0)
    true_labels.extend(labels.numpy().flatten())
    predicted_labels.extend(
        (predictions.flatten() >= 0.5).astype(int)
    )

true_labels = np.array(true_labels)
predicted_labels = np.array(predicted_labels)

print("\nClassification Report:")
print(
    classification_report(
        true_labels,
        predicted_labels,
        target_names=["without_mask", "with_mask"]
    )
)

cm = confusion_matrix(true_labels, predicted_labels)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(6, 5))
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.xticks([0, 1], ["Without Mask", "With Mask"])
plt.yticks([0, 1], ["Without Mask", "With Mask"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.savefig("confusion_matrix.png")
plt.show()