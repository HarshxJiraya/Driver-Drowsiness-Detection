from pathlib import Path
import numpy as np
import tensorflow as tf
import cv2

class EyeInference:
    def __init__(self,model_path = None,image_size = (224,224)):
        if model_path is None:
            model_path = Path(__file__).resolve().parent.parent/"model"/"MRL_fineTuned.keras"

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError("No Model found")

        self.image_size = image_size

        self.class_names = {
            0: "Awake",
            1: "Sleepy"
        }

        print("Loading model...")

        self.model = tf.keras.models.load_model(model_path)

        # Warm up model to eliminate first-frame execution latency
        dummy_batch = tf.zeros((2, *self.image_size, 3), dtype=tf.float32)
        _ = self.model(dummy_batch, training=False)

        print("Model loading and warmup successful")



    def preprocess(self, eye):
        if eye is None or eye.size == 0:
            raise ValueError("Invalid image")

        eye = cv2.cvtColor(eye, cv2.COLOR_BGR2RGB)
        eye = cv2.resize(eye, self.image_size)
        eye = eye.astype(np.float32)
        return eye

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
        preprocessed_eye = self.preprocess(eye)
        tensor = np.expand_dims(preprocessed_eye, axis=0)

        # Direct tensor call is 5-10x faster than model.predict()
        prediction = self.model(tensor, training=False).numpy()
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
        left_prep = self.preprocess(left_eye)
        right_prep = self.preprocess(right_eye)

        batch = np.stack([left_prep, right_prep], axis=0)

        # Single forward pass for both eye crops simultaneously
        predictions = self.model(batch, training=False).numpy()

        left_prob = float(predictions[0][0])
        right_prob = float(predictions[1][0])

        left_result = self._format_prediction(left_prob)
        right_result = self._format_prediction(right_prob)

        return left_result, right_result

    def predict_batch(self, left_eye, right_eye):
        """Alias for predict_pair."""
        return self.predict_pair(left_eye, right_eye)
