import os

import cv2
from matplotlib import scale
import mediapipe as mp
import numpy as np


mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

curr_file_dir = os.path.dirname(os.path.abspath(__file__))
video_filename = 'sample.mp4'
scale_factor = 0.25

pose_estimator = mp_pose.Pose(model_complexity=2,
                              enable_segmentation=True)
cap = cv2.VideoCapture(os.path.join(curr_file_dir, video_filename))

while True:
    success, img = cap.read()

    if not success:
        print("Terminating because failed to read video")
        break

    # Scale the video and convert to rgb space
    img = cv2.resize(img, (0, 0), fx=scale_factor, fy=scale_factor)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_height, img_width, _ = img.shape

    results = pose_estimator.process(img_rgb)
    if not results.pose_landmarks:
        continue

    condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
    bg_image = np.zeros(img.shape, dtype=np.uint8)
    bg_image[:] = (192, 192, 192)  # gray
    annotated_image = np.where(condition, img, bg_image)

    mp_drawing.draw_landmarks(
        annotated_image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

    # Plot pose world landmarks.
    # mp_drawing.plot_landmarks(
    #     results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow("Pose Estimateion", annotated_image)
    cv2.waitKey(1)
