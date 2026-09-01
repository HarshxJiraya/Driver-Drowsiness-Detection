from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf


class EyeInference:
    """
    High-Performance Eye-State Classifier using TensorFlow Lite (with XNNPACK CPU acceleration)
    and fallback support for TensorFlow Keras.
    """

    def __init__(self, model_path=None, image_size=(224, 224), prefer_tflite=True):
        self.image_size = image_size
        self.class_names = {
            0: "Awake",
            1: "Sleepy"
        }

        model_dir = Path(__file__).resolve().parent.parent / "model"
        tflite_path = model_dir / "MRL_fineTuned.tflite"
        keras_path = model_dir / "MRL_fineTuned.keras"

        if model_path is not None:
            self.model_path = Path(model_path)
        elif prefer_tflite and tflite_path.exists():
            self.model_path = tflite_path
        elif keras_path.exists():
            self.model_path = keras_path
        elif tflite_path.exists():
            self.model_path = tflite_path
        else:
            raise FileNotFoundError("No trained model found (.tflite or .keras) in model/ directory.")

        self.is_tflite = (self.model_path.suffix.lower() == ".tflite")

        if self.is_tflite:
            print(f"Loading high-speed TFLite model from {self.model_path.name}...")
            self.interpreter = tf.lite.Interpreter(model_path=str(self.model_path))
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.input_index = self.input_details[0]["index"]
            self.output_index = self.output_details[0]["index"]

            # Pre-configure input tensor for batch size 2 (both eyes in parallel)
            self._allocated_batch_size = 2
            self.interpreter.resize_tensor_input(
                self.input_index, [2, self.image_size[0], self.image_size[1], 3]
            )
            self.interpreter.allocate_tensors()

            # Warm up interpreter
            dummy_batch = np.zeros((2, *self.image_size, 3), dtype=np.float32)
            self.interpreter.set_tensor(self.input_index, dummy_batch)
            self.interpreter.invoke()
            print("TFLite model loaded and warmed up successfully (XNNPACK accelerated).")
        else:
            print(f"Loading Keras model from {self.model_path.name}...")
            self.model = tf.keras.models.load_model(self.model_path)

            # Warm up model to eliminate first-frame execution latency
            dummy_batch = tf.zeros((2, *self.image_size, 3), dtype=tf.float32)
            _ = self.model(dummy_batch, training=False)
            print("Keras model loaded and warmup successful.")

    def preprocess(self, eye):
        """Preprocess a single eye image crop into a (224, 224, 3) float32 RGB array."""
        if eye is None or eye.size == 0:
            raise ValueError("Invalid image")

        eye = cv2.cvtColor(eye, cv2.COLOR_BGR2RGB)
        eye = cv2.resize(eye, self.image_size)
        eye = eye.astype(np.float32)
        return eye

    def preprocess_pair(self, left_eye, right_eye):
        """Preprocess and stack both eye crops into a single (2, 224, 224, 3) batch."""
        left_prep = self.preprocess(left_eye)
        right_prep = self.preprocess(right_eye)
        batch = np.stack([left_prep, right_prep], axis=0)
        return batch

    def _infer_batch(self, batch):
        """Run batch inference on preprocessed tensor."""
        if self.is_tflite:
            batch_size = batch.shape[0]
            if batch_size != self._allocated_batch_size:
                self.interpreter.resize_tensor_input(
                    self.input_index, [batch_size, self.image_size[0], self.image_size[1], 3]
                )
                self.interpreter.allocate_tensors()
                self._allocated_batch_size = batch_size

            self.interpreter.set_tensor(self.input_index, batch)
            self.interpreter.invoke()
            return self.interpreter.get_tensor(self.output_index)
        else:
            return self.model(batch, training=False).numpy()

    def _format_prediction(self, probability_sleepy):
        if probability_sleepy >= 0.5:
            class_index = 1
        else:
            class_index = 0

        label = self.class_names[class_index]
        if class_index == 1:
            confidence = probability_sleepy
        else:
            confidence = 1.0 - probability_sleepy

        return (label, confidence, probability_sleepy)

    def predict(self, eye):
        """Predict eye state for a single eye crop."""
        preprocessed_eye = self.preprocess(eye)
        tensor = np.expand_dims(preprocessed_eye, axis=0)
        prediction = self._infer_batch(tensor)
        probability_sleepy = float(prediction[0][0])
        return self._format_prediction(probability_sleepy)

    def predict_pair(self, left_eye, right_eye):
        """
        Run inference on both left and right eyes in a single batched forward pass.
        Tensor shape: (2, 224, 224, 3)

        Returns
        -------
        tuple: (left_result, right_result)
            Each result is (label, confidence, probability_sleepy)
        """
        batch = self.preprocess_pair(left_eye, right_eye)
        predictions = self._infer_batch(batch)

        left_prob = float(predictions[0][0])
        right_prob = float(predictions[1][0])

        left_result = self._format_prediction(left_prob)
        right_result = self._format_prediction(right_prob)

        return left_result, right_result

    def predict_batch(self, left_eye, right_eye):
        """Alias for predict_pair."""
        return self.predict_pair(left_eye, right_eye)
