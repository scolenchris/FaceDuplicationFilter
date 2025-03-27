import os
import cv2
import face_recognition
from datetime import datetime
import numpy as np


def extract_faces_from_videos(
    video_folder, output_base_folder="./face", resize_threshold=800
):
    # Create a timestamped output folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = os.path.join(output_base_folder, f"extracted_faces_{timestamp}")
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over all video files in the input folder
    for video_filename in os.listdir(video_folder):
        video_path = os.path.join(video_folder, video_filename)

        # Skip if not a video file
        if not video_filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            continue

        # Create an output directory for each video
        video_output_folder = os.path.join(
            output_folder, os.path.splitext(video_filename)[0]
        )
        os.makedirs(video_output_folder, exist_ok=True)

        # Process the video
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_to_capture = np.linspace(
            0, frame_count - 1, 10, dtype=int
        )  # Select 10 frames

        captured_faces = 0
        frame_idx = 0

        while cap.isOpened() and captured_faces < 10:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx in frames_to_capture:
                # Convert the frame from BGR to RGB

                # frame_resized = cv2.resize(frame, (640, 360))  # 将帧大小调整为 640x360
                # rgb_frame = frame_resized[:, :, ::-1]
                # small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)   #将图片缩放到1/4倍
                # rgb_frame = small_frame[:, :, ::-1]

                # 自适应缩小
                height, width = frame.shape[:2]
                max_dim = max(height, width)

                # 最长边小于阈值才缩小
                if max_dim > resize_threshold:
                    scale_factor = 4
                    resized_frame = cv2.resize(
                        frame, (0, 0), fx=1.0 / scale_factor, fy=1.0 / scale_factor
                    )
                else:
                    resized_frame = frame
                rgb_frame = resized_frame[:, :, ::-1]

                # Detect faces in the frame
                face_locations = face_recognition.face_locations(rgb_frame)

                for face_index, face_location in enumerate(face_locations):
                    top, right, bottom, left = face_location

                    # 自适应缩放回来
                    if max_dim > resize_threshold:
                        top *= scale_factor
                        right *= scale_factor
                        bottom *= scale_factor
                        left *= scale_factor
                    else:
                        pass

                    face_image = frame[top:bottom, left:right]

                    # Save the face image
                    face_filename = f"face_{captured_faces}_{face_index}.jpg"
                    face_path = os.path.join(video_output_folder, face_filename)
                    cv2.imwrite(face_path, face_image)

                    captured_faces += 1
                    if captured_faces >= 10:
                        break

            frame_idx += 1

        cap.release()


if __name__ == "__main__":
    video_folder = "./video"
    output_folder = "./face"
    extract_faces_from_videos(video_folder, output_folder)
# # Example usage
# video_folder = "./video"
# output_folder = "./face"
# extract_faces_from_videos(video_folder, output_folder)
