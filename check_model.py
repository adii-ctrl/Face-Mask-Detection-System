import tensorflow as tf
import numpy as np
import glob
import os

IMAGE_SIZE = 128

model = tf.keras.models.load_model("face_mask_model.keras")

print("Model loaded successfully!")
print()

# Check one WITH MASK image
with_mask_images = glob.glob("dataset/validation/with_mask/*")

# Check one WITHOUT MASK image
without_mask_images = glob.glob("dataset/validation/without_mask/*")

print("With Mask images:", len(with_mask_images))
print("Without Mask images:", len(without_mask_images))
print()


def predict_image(image_path):
    image = tf.keras.utils.load_img(
        image_path,
        target_size=(IMAGE_SIZE, IMAGE_SIZE)
    )

    image = tf.keras.utils.img_to_array(image)

    image = image / 255.0

    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image, verbose=0)[0][0]

    print("Image:", os.path.basename(image_path))
    print("Prediction value:", prediction)

    if prediction < 0.5:
        print("Prediction: WITH MASK")
    else:
        print("Prediction: WITHOUT MASK")

    print("-" * 40)


print("Checking WITH MASK image:")
predict_image(with_mask_images[0])

print("Checking WITHOUT MASK image:")
predict_image(without_mask_images[0])