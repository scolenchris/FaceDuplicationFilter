import torch
import face_recognition
import os
import numpy as np
from tqdm import tqdm
from itertools import combinations
import csv
import sys

sys.path.append("./")
from inference_sigle_function import inference
from datetime import datetime
import shutil

# 增加了文件管理和保存最佳编码


# 加载图片返回图片名和图片的
def load_images_from_folder(folder):
    images = []
    filenames = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = face_recognition.load_image_file(img_path)
        images.append(img)
        filenames.append(filename)
    return filenames, images


# 对图片序列进行人脸特征编码，并取平均
def encode_faces(images):
    encodings = []
    for img in images:
        encoding = face_recognition.face_encodings(img)

        if encoding:
            encodings.append(encoding[0])
        else:
            encodings.append(None)  # 没检测到使用None填充

    if all(e is None for e in encodings):  # 改为全为none的时候抛出异常
        raise ValueError("No faces found in any images.")

    # average_encoding=np.mean(encodings,axis=0)  #不直接计算平均编码了
    return encodings


# 计算已知编码和测试编码的相似度，用距离，越小越近
def calculate_similarity(known_encoding, test_encoding):
    # face_distance方法
    similarity = face_recognition.face_distance([known_encoding], test_encoding)
    return similarity


def main():
    # 用到的路径
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    known_faces_folder = "./face/0"
    save_faces_folder = known_faces_folder.replace("face", "selected")
    output_csv_path = (
        "./log_csv/similarity" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv"
    )
    save_npy_path = os.path.join(
        save_faces_folder, "best_encoding.npy"
    )  # 将最佳图片组合编码保存到npy文件

    # 加载图片及文件名
    filenames, images = load_images_from_folder(known_faces_folder)  # 所有图片

    # 算图片质量分数
    with torch.no_grad():
        weights = inference(dataset_loc=known_faces_folder, device=device)

    # 算人脸编码
    encoding = encode_faces(images)

    # 准备写csv文件
    with open(output_csv_path, mode="w", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        # 文件头
        csv_writer.writerow(["Combination", "Weighted Average Similarity"])
        min_similarity = 1.0
        min_similarity_combination = []
        min_similarity_encoding = None
        # 挑选所有图片组合（2）
        for combo_indices in tqdm(
            combinations(range(len(images)), 2),
            total=len(list(combinations(range(len(images)), 2))),
            desc="Calculating Similarity",
        ):
            # 取组合的文件名
            # combo_images=[images[i] for i in combo_indices]  #这个已经用不到了
            combo_filenames = [filenames[i] for i in combo_indices]
            # 算组合平均编码
            combo_encodings = [encoding[i] for i in combo_indices]
            combo_encodings = [
                x for x in combo_encodings if x is not None
            ]  # 过滤一下，因为可能为none，就不能正常计算了
            known_encoding = np.mean(combo_encodings, axis=0)  # 图片组合（2）的平均编码

            # 计算组合平均编码与其他图片编码的距离
            validation_indices = set(range(len(images))) - set(combo_indices)
            weighted_similarities = []
            total_weight = 0.0

            for idx in validation_indices:
                test_encoding = encoding[
                    idx
                ]  # 分别计算出剩下图片的特征编码，并与图片组合（2）的平均编码进行相似度计算
                if test_encoding is not None:
                    similarity = calculate_similarity(known_encoding, test_encoding)
                    weight = weights[filenames[idx]]  # 使用图片质量分数调整相似度
                    weighted_similarities.append(similarity * weight)
                    total_weight += weight

            if total_weight > 0:
                weighted_average_similarity = (
                    sum(weighted_similarities) / total_weight
                )  # 加权平均相似度
            else:
                weighted_average_similarity = 0.0

            # 更新全局最小距离（最好相似度）并记录组合
            if weighted_average_similarity < min_similarity:
                min_similarity = weighted_average_similarity
                min_similarity_combination = combo_filenames
                min_similarity_encoding = known_encoding  # 保存最佳平均编码

            # 写入csv
            combo_str = ", ".join(combo_filenames)
            csv_writer.writerow([combo_str, str(weighted_average_similarity)])
        print(f"Combination with minimum similarity: {min_similarity_combination}")
        print(f"Minimum similarity: {min_similarity}")
        # print(f"Minimum similarity encoding: {min_similarity_encoding}")   #展示最佳编码

    print(f"Results written to {output_csv_path}")

    # 挑选最佳图片组合
    if min_similarity_combination:

        os.makedirs(save_faces_folder, exist_ok=True)

        # 遍历文件名并复制文件
        for filename in min_similarity_combination:
            # 构建原文件的完整路径
            original_file_path = os.path.join(known_faces_folder, filename)

            # 构建目标文件的完整路径
            new_file_path = os.path.join(save_faces_folder, filename)

            # 复制文件
            shutil.copy(original_file_path, new_file_path)
            print(f"Select file {filename} to {new_file_path}")

        # 保存最佳编码
        np.save(save_npy_path, min_similarity_encoding)  # 保存为.npy文件
        print("Encoding saved.")

    # 测试读取
    # encoding_from_file=np.load(save_npy_path)
    # print("Loaded encoding:",encoding_from_file)


if __name__ == "__main__":
    main()
