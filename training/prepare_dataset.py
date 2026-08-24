import os
import cv2
import xml.etree.ElementTree as ET

images_path = "archive/images"
annotations_path = "archive/annotations"

os.makedirs("dataset/with_mask", exist_ok=True)
os.makedirs("dataset/without_mask", exist_ok=True)

for xml_file in os.listdir(annotations_path):

    if not xml_file.endswith(".xml"):
        continue

    xml_path = os.path.join(annotations_path, xml_file)

    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.find("filename").text
    image_path = os.path.join(images_path, filename)

    image = cv2.imread(image_path)

    if image is None:
        continue

    for i, obj in enumerate(root.findall("object")):

        label = obj.find("name").text

        box = obj.find("bndbox")

        xmin = int(box.find("xmin").text)
        ymin = int(box.find("ymin").text)
        xmax = int(box.find("xmax").text)
        ymax = int(box.find("ymax").text)

        face = image[ymin:ymax, xmin:xmax]

        if face.size == 0:
            continue

        if label == "with_mask":
            save_folder = "dataset/with_mask"

        elif label == "without_mask":
            save_folder = "dataset/without_mask"

        else:
            continue

        output_name = f"{filename}_{i}.jpg"
        output_path = os.path.join(save_folder, output_name)

        cv2.imwrite(output_path, face)

print("Dataset preparation complete!")