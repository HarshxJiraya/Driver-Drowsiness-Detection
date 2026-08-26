from pathlib import Path

import cv2
import mediapipe as mp


class FaceDetector:
    """
    MediaPipe Face Landmarker using LIVE_STREAM mode.

    Responsibilities:
    - Load the Face Landmarker model
    - Receive live video frames
    - Detect facial landmarks asynchronously
    - Store the latest detection result
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

        # MediaPipe Tasks
        self.vision = mp.tasks.vision

        # Store the latest result
        self.latest_result = None

        # Face Landmarker options
        options = self.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(model_path)
            ),
            running_mode=self.vision.RunningMode.LIVE_STREAM,
            num_faces=num_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            result_callback=self._result_callback,
        )

        # Create Face Landmarker
        self.landmarker = (
            self.vision.FaceLandmarker.create_from_options(
                options
            )
        )

    def _result_callback(
        self,
        result,
        output_image,
        timestamp_ms
    ):
        """
        Callback called by MediaPipe when a result is available.
        """

        self.latest_result = result

        print(
        f"Callback received | "
        f"timestamp={timestamp_ms} | "
        f"faces={len(result.face_landmarks)}"
    )

    def detect(self, frame, timestamp_ms):
        """
        Submit a frame to MediaPipe for asynchronous processing.

        Parameters
        ----------
        frame : numpy.ndarray
            OpenCV BGR frame.

        timestamp_ms : int
            Increasing timestamp in milliseconds.
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

        # Submit frame asynchronously
        self.landmarker.detect_async(
            mp_image,
            timestamp_ms
        )

    def get_latest_result(self):
        """
        Return the most recent Face Landmarker result.
        """

        return self.latest_result

    def close(self):
        """
        Release MediaPipe resources.
        """

        self.landmarker.close()