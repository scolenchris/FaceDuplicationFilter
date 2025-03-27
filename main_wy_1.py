import torch
import face_recognition
import os
import numpy as np
from tqdm import tqdm
from itertools import combinations
import csv
from datetime import datetime
import sys
from PIL import Image

sys.path.append("./")

from inference_sigle_function import inference

# 原始抵消版本


def load_images_from_folder(folder):
    images = []
    filenames = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = face_recognition.load_image_file(img_path)
        images.append(img)
        filenames.append(filename)
    return filenames, images


def encode_faces_and_average(images):
    encodings = []
    for img in images:
        # Assume each image has only one face for simplicity
        img = img.astype("uint8")
        encoding = face_recognition.face_encodings(img)

        if encoding:  # Check if any face is found
            encodings.append(encoding[0])

    if not encodings:
        raise ValueError("No faces found in any images.")

    # Calculate the average encoding to represent the single person's face ID
    average_encoding = np.mean(encodings, axis=0)
    return average_encoding


def calculate_similarity(known_encoding, test_encoding):
    # Calculate the Euclidean distance between the average known face and the test face
    # distance = np.linalg.norm(known_encoding - test_encoding)
    # Convert the distance into a similarity score (probability-like metric)
    # similarity = 1 / (1 + distance)  # Simple conversion to a probability-like score

    # face_distance方法
    similarity = face_recognition.face_distance([known_encoding], test_encoding)
    return similarity


def main():
    # Specify the folder containing the images
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    known_faces_folder = "./face/0"
    output_csv_path = (
        "./log_csv/similarity" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv"
    )

    # Load images and their filenames
    filenames, images = load_images_from_folder(known_faces_folder)

    # weights = {filename: 1 for filename in filenames} #average
    with torch.no_grad():
        weights = inference(dataset_loc=known_faces_folder, device=device)

    # Prepare CSV file
    with open(output_csv_path, mode="w", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write header
        csv_writer.writerow(["Combination", "Weighted Average Similarity"])
        min_similarity = 1.0
        min_similarity_combination = []
        # Generate all combinations of 5 images
        for combo_indices in tqdm(
            combinations(range(len(images)), 2),
            total=len(list(combinations(range(len(images)), 2))),
            desc="Calculating Similarity",
        ):
            # Get the images and filenames for this combination
            combo_images = [images[i] for i in combo_indices]
            combo_filenames = [filenames[i] for i in combo_indices]
            # print(combo_filenames)
            # Calculate the average encoding for this combination
            known_encoding = encode_faces_and_average(combo_images)

            # Validate with other images
            validation_indices = set(range(len(images))) - set(combo_indices)
            weighted_similarities = []
            total_weight = 0.0

            for idx in validation_indices:
                test_encoding = face_recognition.face_encodings(images[idx])
                if test_encoding:
                    similarity = calculate_similarity(known_encoding, test_encoding[0])
                    weight = weights[filenames[idx]]
                    weighted_similarities.append(similarity * weight)
                    total_weight += weight

            if total_weight > 0:
                weighted_average_similarity = sum(weighted_similarities) / total_weight
            else:
                weighted_average_similarity = 0.0

            if weighted_average_similarity < min_similarity:
                min_similarity = weighted_average_similarity
                min_similarity_combination = combo_filenames

            # Write to CSV
            combo_str = ", ".join(combo_filenames)
            csv_writer.writerow([combo_str, str(weighted_average_similarity)])
        print(f"Combination with minimum similarity: {min_similarity_combination}")
        print(f"Minimum similarity: {min_similarity}")

    print(f"Results written to {output_csv_path}")


if __name__ == "__main__":
    main()
