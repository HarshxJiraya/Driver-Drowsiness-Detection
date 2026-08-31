import time

class DrowsinessDetector:
    def __init__(self,eye_closed_threshold=2.0,yawning_threshold=3.0,drowsiness_duration=2.0):
        self.eye_closed_threshold = eye_closed_threshold
        self.yawning_threshold = yawning_threshold
        self.drowsiness_duration = drowsiness_duration

        self.eye_close_start = None
        self.yawning_start = None
        self.drowsy_start = None

    def update(self,left_eye_closed,right_eye_closed,yawning):
        current_time = time.time()

        both_eye_closed = left_eye_closed and right_eye_closed
        if both_eye_closed:
            if self.eye_close_start is None:
                self.eye_close_start = current_time
        else:
            self.eye_close_start = None

        if self.eye_close_start is not None:
            eye_closed_duration = current_time - self.eye_close_start
        else:
            eye_closed_duration = 0.0

        prolonged_eye_closed = eye_closed_duration > self.eye_closed_threshold


        if yawning:
            if self.yawning_start is None:
                self.yawning_start = current_time
        else:
            self.yawning_start = None

        if self.yawning_start is not None:
            yawn_duration = current_time - self.yawning_start
        else:
            yawn_duration = 0.0

        prolonged_yawning = yawn_duration > self.yawning_threshold


        drowsiness_signal = prolonged_yawning or prolonged_eye_closed

        if drowsiness_signal:
            if self.drowsy_start is None:
                self.drowsy_start = current_time
        else:
            self.drowsy_start = None

        if self.drowsy_start is not None:
            drowsiness_duration = current_time - self.drowsy_start
        else:
            drowsiness_duration = 0.0

        drowsy = drowsiness_duration > self.drowsiness_duration

        if drowsy:
            status = "Drowsy"
        else:
            status = "Awake"

        return {
            "status": status,
            "both_eyes_closed": both_eye_closed,
            "prolonged_eye_closure": prolonged_eye_closed,
            "confirmed_yawn": prolonged_yawning,
            "eye_closed_duration": eye_closed_duration,
            "yawn_duration": yawn_duration,
            "drowsy_duration": drowsiness_duration
        }




