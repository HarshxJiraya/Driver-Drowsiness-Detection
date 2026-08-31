import numpy as np


# MediaPipe Face Mesh landmark indices
UPPER_LIP = 13
LOWER_LIP = 14
LEFT_MOUTH = 61
RIGHT_MOUTH = 291

def calculate_mar(face_landmark,frame_width,frame_height):
    def get_points(index):
        landmark = face_landmark[index]
        x = landmark.x * frame_width
        y = landmark.y * frame_height
        return np.array([x,y])

    upper_lip = get_points(UPPER_LIP)
    lower_lip = get_points(LOWER_LIP)
    left_mouth = get_points(LEFT_MOUTH)
    right_mouth = get_points(RIGHT_MOUTH)

    vertical_distance = np.linalg.norm(lower_lip - upper_lip)
    horizontal_distance = np.linalg.norm(left_mouth - right_mouth)

    if vertical_distance == 0:
        return 0.0

    mar = vertical_distance/horizontal_distance
    return mar

def is_yawning(mar,threshold = 0.6):
    return mar>threshold




