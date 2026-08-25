import tensorflow as tf
from tensorflow.keras import layers, models

# Image settings
IMAGE_SIZE = 128
BATCH_SIZE = 32

# Load training dataset
train_data = tf.keras.utils.image_dataset_from_directory(
    "dataset/train",
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="binary"
)

# Load validation dataset
validation_data = tf.keras.utils.image_dataset_from_directory(
    "dataset/validation",
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="binary"
)

# Normalize pixel values
normalization_layer = layers.Rescaling(1./255)

train_data = train_data.map(
    lambda x, y: (normalization_layer(x), y)
)

validation_data = validation_data.map(
    lambda x, y: (normalization_layer(x), y)
)

# Create CNN model
model = models.Sequential([

    layers.Conv2D(32, (3, 3), activation="relu",
                  input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),

    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu"),

    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation="relu"),

    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),

    layers.Dense(128, activation="relu"),

    layers.Dropout(0.5),

    layers.Dense(1, activation="sigmoid")
])

# Compile model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Display model structure
model.summary()

# Train model
history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=10
)

# Save trained model
model.save("face_mask_model.keras")

print("Model training complete!")