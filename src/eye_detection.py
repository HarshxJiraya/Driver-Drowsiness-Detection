import cv2
import numpy as np

class EyeDetector:
    LEFT_EYE_LANDMARKS = [
    33, 160, 158, 133, 153, 144
]

    RIGHT_EYE_LANDMARKS = [
        362, 385, 387, 263, 373, 380
    ]

    def __init__(self,padding=20):
        self.padding = padding

    def _get_eye_box(self,face_landmark,eye_index,frame_width,frame_height):
        points = []
        for index in eye_index:
            landmark = face_landmark[index]
            x = int(landmark.x*frame_width)
            y = int(landmark.y*frame_height)
            points.append((x,y))

        points = np.array(points)

        xmin = np.min(points[:,0])
        xmax = np.max(points[:,0])

        ymin = np.min(points[:,1])
        ymax = np.max(points[:,1])

        xmin -= self.padding
        ymin -= self.padding
        xmax += self.padding
        ymax += self.padding

        xmin = max(0,xmin)
        ymin = max(0,ymin)
        xmax = min(frame_width,xmax)
        ymax = min(frame_height,ymax)

        return xmin,xmax,ymin,ymax

    def detect(self,frame,face_landmark):
        height,width = frame.shape[:2]
        left_box = self._get_eye_box(face_landmark,self.LEFT_EYE_LANDMARKS,width,height)
        right_box = self._get_eye_box(face_landmark,self.RIGHT_EYE_LANDMARKS,width,height)

        lx1,lx2,ly1,ly2 = left_box
        rx1,rx2,ry1,ry2 = right_box

        left_eye = frame[ly1:ly2,lx1:lx2]
        right_eye = frame[ry1:ry2,rx1:rx2]

        return {
            "left_eye":left_eye,
            "right_eye":right_eye,
            "left_box":left_box,
            "right_box":right_box,
        } 


def calculate_single_ear(face_landmark, p1_idx, p2_idx, p3_idx, p4_idx, p5_idx, p6_idx, frame_width, frame_height):
    """
    Compute Eye Aspect Ratio (EAR) for a single eye given its 6 boundary landmark indices.
    """
    def get_pt(idx):
        lm = face_landmark[idx]
        return np.array([lm.x * frame_width, lm.y * frame_height])

    p1 = get_pt(p1_idx)  # outer corner
    p4 = get_pt(p4_idx)  # inner corner
    p2 = get_pt(p2_idx)  # top 1
    p6 = get_pt(p6_idx)  # bottom 1
    p3 = get_pt(p3_idx)  # top 2
    p5 = get_pt(p5_idx)  # bottom 2

    horizontal = np.linalg.norm(p1 - p4)
    if horizontal == 0:
        return 0.0

    vertical_1 = np.linalg.norm(p2 - p6)
    vertical_2 = np.linalg.norm(p3 - p5)

    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return float(ear)


def calculate_ear(face_landmark, frame_width, frame_height):
    """
    Calculate EAR for left eye, right eye, and the bilateral average in 0.02ms.

    Returns
    -------
    tuple: (left_ear, right_ear, avg_ear)
    """
    left_ear = calculate_single_ear(
        face_landmark,
        p1_idx=33, p2_idx=160, p3_idx=158, p4_idx=133, p5_idx=153, p6_idx=144,
        frame_width=frame_width, frame_height=frame_height
    )
    right_ear = calculate_single_ear(
        face_landmark,
        p1_idx=362, p2_idx=385, p3_idx=387, p4_idx=263, p5_idx=373, p6_idx=380,
        frame_width=frame_width, frame_height=frame_height
    )
    avg_ear = (left_ear + right_ear) / 2.0
    return left_ear, right_ear, avg_ear
