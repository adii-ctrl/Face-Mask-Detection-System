import os
import shutil
import random

source_folder = "dataset"
train_folder = "dataset/train"
validation_folder = "dataset/validation"

categories = ["with_mask", "without_mask"]

for category in categories:

    source_path = os.path.join(source_folder, category)
    train_path = os.path.join(train_folder, category)
    validation_path = os.path.join(validation_folder, category)

    os.makedirs(train_path, exist_ok=True)
    os.makedirs(validation_path, exist_ok=True)

    images = os.listdir(source_path)

    random.shuffle(images)

    split_point = int(len(images) * 0.8)

    train_images = images[:split_point]
    validation_images = images[split_point:]

    for image in train_images:
        source = os.path.join(source_path, image)
        destination = os.path.join(train_path, image)
        shutil.copy2(source, destination)

    for image in validation_images:
        source = os.path.join(source_path, image)
        destination = os.path.join(validation_path, image)
        shutil.copy2(source, destination)

    print(category)
    print("Training:", len(train_images))
    print("Validation:", len(validation_images))

print("Dataset splitting complete!")