import cv2
import numpy as np

class EyeDetector:
    LEFT_EYE_LANDMARKS = [
    33, 160, 158, 133, 153, 144
]

    RIGHT_EYE_LANDMARKS = [
        362, 385, 387, 263, 373, 380
    ]

    def __init__(self,padding=10):
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