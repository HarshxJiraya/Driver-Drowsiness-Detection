from pathlib import Path

import cv2
import mediapipe as mp


class FaceDetector:
    """
    MediaPipe Face Landmarker using VIDEO mode.

    Responsibilities:
    - Load the Face Landmarker model
    - Receive video/webcam frames
    - Detect facial landmarks synchronously
    - Return the result for the current frame
    """

    def __init__(
        self,
        model_path=None,
        num_faces=1,
        min_detection_confidence=0.5,
        min_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    ):

        if model_path is None:
            model_path = (
                Path(__file__).resolve().parent.parent
                / "model"
                / "face_landmarker.task"
            )

        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Face Landmarker model not found:\n{model_path}"
            )

        # MediaPipe Tasks Vision
        self.vision = mp.tasks.vision

        # Face Landmarker options
        options = self.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(model_path)
            ),

            # Synchronous video mode
            running_mode=self.vision.RunningMode.VIDEO,

            num_faces=num_faces,

            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        # Create Face Landmarker
        self.landmarker = (
            self.vision.FaceLandmarker.create_from_options(
                options
            )
        )

    def detect(self, frame, timestamp_ms):
        """
        Detect facial landmarks for the current frame.

        Parameters
        ----------
        frame : numpy.ndarray
            OpenCV BGR frame.

        timestamp_ms : int
            Increasing timestamp in milliseconds.

        Returns
        -------
        FaceLandmarkerResult
            MediaPipe face landmark result.
        """

        # OpenCV BGR → RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Convert to MediaPipe Image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Synchronous detection
        result = self.landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        return result

    def close(self):
        """
        Release MediaPipe resources.
        """

        self.landmarker.close()