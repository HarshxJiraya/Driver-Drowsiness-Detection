#!/usr/bin/env python3
"""
Driver Drowsiness Detection System - Main Application Entrypoint
================================================================
Real-time driver monitoring combining:
- Google MediaPipe Face Landmarker (478 3D landmarks)
- Geometric Eye Aspect Ratio (EAR) & Mouth Aspect Ratio (MAR)
- Fine-Tuned TensorFlow Lite CNN on MRL Eye Dataset (XNNPACK CPU Accelerated)
- Multi-Threaded Video Streaming (Zero-Latency Frame Reads)
- Universal Cross-Platform Audio Alarm (Windows, macOS, Linux, Raspberry Pi)

Usage:
    python main.py                     # Run live webcam detection
    python main.py --camera 0          # Select specific webcam index
    python main.py --source video.mp4  # Run detection on video file
    python main.py --no-alarm          # Silent mode (no audio alert)
"""

import argparse
import sys
import time
from pathlib import Path

import cv2

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.alarm import Alarm
from src.drowsiness_logic import DrowsinessDetector
from src.eye_detection import EyeDetector, calculate_ear
from src.eye_inference import EyeInference
from src.face_detection import FaceDetector
from src.video_stream import VideoStream
from src.yawn_detection import calculate_mar, is_yawning


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Real-Time Driver Drowsiness Detection System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=1,
        help="Primary camera device index (falls back to 0 if unavailable)",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Path to pre-recorded video file (overrides live camera if set)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Webcam stream capture width",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Webcam stream capture height",
    )
    parser.add_argument(
        "--ear-thresh",
        type=float,
        default=0.28,
        help="Eye Aspect Ratio (EAR) threshold for open eye bypass",
    )
    parser.add_argument(
        "--mar-thresh",
        type=float,
        default=0.60,
        help="Mouth Aspect Ratio (MAR) threshold for yawn detection",
    )
    parser.add_argument(
        "--eye-time",
        type=float,
        default=2.0,
        help="Consecutive eye closure threshold in seconds before drowsiness trigger",
    )
    parser.add_argument(
        "--yawn-time",
        type=float,
        default=3.0,
        help="Consecutive yawn duration in seconds before drowsiness trigger",
    )
    parser.add_argument(
        "--no-alarm",
        action="store_true",
        help="Disable audio alarm playback",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    print("=" * 60)
    print("      DRIVER DROWSINESS DETECTION SYSTEM (REAL-TIME)      ")
    print("=" * 60)

    # 1. Initialize Video Source
    is_video_file = args.source is not None
    if is_video_file:
        video_path = Path(args.source)
        if not video_path.exists():
            print(f"[ERROR] Video file not found: {video_path}")
            sys.exit(1)
        print(f"[*] Opening video file: {video_path}")
        cap = cv2.VideoCapture(str(video_path))
        stream = None
    else:
        print(f"[*] Starting Threaded VideoStream on camera index {args.camera}...")
        try:
            stream = VideoStream(src=args.camera, width=args.width, height=args.height).start()
            cap = None
            print("[✓] VideoStream initialized with zero-latency background capture.")
        except Exception as e:
            print(f"[ERROR] Failed to open camera: {e}")
            sys.exit(1)

    # 2. Initialize AI Models & Components
    print("[*] Loading AI models & detectors...")
    try:
        face_detector = FaceDetector()
        eye_detector = EyeDetector(padding=15)
        eye_inference = EyeInference()  # Auto-selects TFLite with XNNPACK
        drowsiness_detector = DrowsinessDetector(
            eye_closed_threshold=args.eye_time,
            yawning_threshold=args.yawn_time,
            drowsiness_duration=1.5,
        )
        alarm = Alarm(frequency=1000, duration_ms=400, interval_ms=200)
    except Exception as e:
        print(f"[ERROR] Initialization failed: {e}")
        if stream:
            stream.stop()
        sys.exit(1)

    print(f"[✓] Inference Engine : {'TFLite (XNNPACK CPU Accelerated)' if eye_inference.is_tflite else 'Keras'}")
    print(f"[✓] Audio Backend    : {alarm.backend} (Enabled: {not args.no_alarm})")
    print(f"[✓] Eye Time Limit   : {args.eye_time}s | Yawn Time Limit: {args.yawn_time}s")
    print("-" * 60)
    print("System active! Press 'q' or 'ESC' in the video window to quit.")
    print("-" * 60)

    # FPS Monitoring
    fps_start_time = time.time()
    fps_counter = 0
    current_fps = 0.0

    try:
        while True:
            # Capture frame
            if is_video_file:
                ret, frame = cap.read()
                if not ret:
                    print("[*] End of video stream.")
                    break
            else:
                grabbed, frame = stream.read()
                if not grabbed or frame is None:
                    time.sleep(0.005)
                    continue

            # FPS calculation
            # fps_counter += 1
            # fps_elapsed = time.time() - fps_start_time
            # if fps_elapsed >= 0.5:
            #     current_fps = fps_counter / fps_elapsed
            #     fps_counter = 0
            #     fps_start_time = time.time()

            # Monotonic timestamp in ms for MediaPipe
            current_timestamp_ms = int(time.time() * 1000)

            # 1. Detect Face Landmarks
            result = face_detector.detect(frame, current_timestamp_ms)

            if result is None or len(result.face_landmarks) == 0:
                cv2.putText(
                    frame,
                    "NO FACE DETECTED",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    f"FPS: {current_fps:.1f}",
                    (frame.shape[1] - 150, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )
                cv2.imshow("Driver Drowsiness Detection", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                continue

            face_landmark = result.face_landmarks[0]
            frame_height, frame_width = frame.shape[:2]

            # 2. Geometric EAR & Eye Crops
            left_ear, right_ear, avg_ear = calculate_ear(face_landmark, frame_width, frame_height)

            eye_data = eye_detector.detect(frame, face_landmark)
            left_eye = eye_data["left_eye"]
            right_eye = eye_data["right_eye"]
            left_box = eye_data["left_box"]
            right_box = eye_data["right_box"]

            # 3. Smart Hybrid Pre-Filtering
            # When eyes are clearly wide open (EAR > thresh), bypass CNN in 0.02ms
            if avg_ear > args.ear_thresh:
                left_label, left_confidence, left_prob = "Awake", 0.99, 0.01
                right_label, right_confidence, right_prob = "Awake", 0.99, 0.01
            else:
                # Run fine-tuned TFLite CNN in a single forward pass
                (left_label, left_confidence, left_prob), (right_label, right_confidence, right_prob) = (
                    eye_inference.predict_pair(left_eye, right_eye)
                )

            left_eye_closed = (left_label == "Sleepy")
            right_eye_closed = (right_label == "Sleepy")

            # 4. Yawn Detection (MAR)
            mar = calculate_mar(face_landmark, frame_width, frame_height)
            yawning = is_yawning(mar, threshold=args.mar_thresh)

            # 5. Temporal Drowsiness State Machine
            drowsiness_result = drowsiness_detector.update(left_eye_closed, right_eye_closed, yawning)
            status = drowsiness_result["status"]

            # 6. Render Visual Overlays
            lx1, lx2, ly1, ly2 = left_box
            rx1, rx2, ry1, ry2 = right_box
            eye_box_color = (0, 0, 255) if (left_eye_closed or right_eye_closed) else (0, 255, 0)
            cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), eye_box_color, 2)
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), eye_box_color, 2)

            # HUD Information Overlay
            # cv2.putText(frame, f"FPS: {current_fps:.1f}", (frame.shape[1] - 150, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Left Eye: {left_label} ({left_confidence * 100:.1f}%)", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(frame, f"Right Eye: {right_label} ({right_confidence * 100:.1f}%)", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(frame, f"MAR: {mar:.2f}", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(frame, f"Yawning: {'YES' if yawning else 'NO'}", (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255) if yawning else (255, 255, 255), 2)
            cv2.putText(frame, f"Eye Closed: {drowsiness_result['eye_closed_duration']:.1f}s", (30, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(frame, f"Yawn Duration: {drowsiness_result['yawn_duration']:.1f}s", (30, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

            # Driver Status Badge
            status_color = (0, 255, 0) if status == "Awake" else (0, 0, 255)
            cv2.putText(frame, f"STATUS: {status.upper()}", (30, 265), cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 3)

            # 7. Audio Alarm Control
            if not args.no_alarm:
                if status == "Drowsy":
                    alarm.start()
                elif status == "Awake":
                    alarm.stop()

            # Display Output Frame
            cv2.imshow("Driver Drowsiness Detection", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    except KeyboardInterrupt:
        print("\n[*] Interrupted by user.")
    finally:
        print("[*] Cleaning up resources...")
        alarm.stop()
        if stream is not None:
            stream.stop()
        if cap is not None:
            cap.release()
        face_detector.close()
        cv2.destroyAllWindows()
        print("[✓] Drowsiness Detection System stopped cleanly.")


if __name__ == "__main__":
    main()
