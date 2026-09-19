import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

# =========================
# SETTINGS
# =========================
IMAGE_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 20

# =========================
# LOAD DATASET
# =========================
train_data = tf.keras.utils.image_dataset_from_directory(
    "dataset/train",
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=True
)
print("CLASS NAMES:", train_data.class_names)

validation_data = tf.keras.utils.image_dataset_from_directory(
    "dataset/validation",
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

# Check class names
print("Class names:", train_data.class_names)

# =========================
# DATA AUGMENTATION
# =========================
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.10),
    layers.RandomContrast(0.10)
])

# =========================
# NORMALIZATION
# =========================
normalization_layer = layers.Rescaling(1.0 / 255)

train_data = train_data.map(
    lambda x, y: (data_augmentation(normalization_layer(x)), y),
    num_parallel_calls=tf.data.AUTOTUNE
)

validation_data = validation_data.map(
    lambda x, y: (normalization_layer(x), y),
    num_parallel_calls=tf.data.AUTOTUNE
)

train_data = train_data.prefetch(tf.data.AUTOTUNE)
validation_data = validation_data.prefetch(tf.data.AUTOTUNE)

# =========================
# CALCULATE CLASS WEIGHTS
# =========================
class_counts = {}

for _, labels in train_data.unbatch():
    label = int(labels.numpy()[0])
    class_counts[label] = class_counts.get(label, 0) + 1

print("Class counts:", class_counts)

total = sum(class_counts.values())
num_classes = len(class_counts)

class_weight = {}

for class_id, count in class_counts.items():
    class_weight[class_id] = total / (num_classes * count)

print("Class weights:", class_weight)

# =========================
# CNN MODEL
# =========================
model = models.Sequential([

    layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(256, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),

    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),

    layers.Dense(1, activation="sigmoid")
])

# =========================
# COMPILE
# =========================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =========================
# TRAIN
# =========================
history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=EPOCHS,
    class_weight=class_weight
)

# =========================
# SAVE MODEL
# =========================
model.save("face_mask_model.keras")

print("Model training complete!")
print("Model saved as face_mask_model.keras")

# =========================
# ACCURACY GRAPH
# =========================
plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.savefig("accuracy_graph.png")
plt.show()

# =========================
# LOSS GRAPH
# =========================
plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.savefig("loss_graph.png")
plt.show()