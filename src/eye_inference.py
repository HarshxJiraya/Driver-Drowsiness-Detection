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

        print("Model loading in successfull")



    def preprocess(self,eye):
        if eye is None or eye.size==0:
            raise ValueError("Invalid image")

        eye = cv2.cvtColor(eye,cv2.COLOR_BGR2RGB)

        eye = cv2.resize(eye,self.image_size)

        eye = eye.astype(np.float32)

        eye = np.expand_dims(eye,axis=0)

        return eye

    def predict(self,eye):
        preproced_eye = self.preprocess(eye)

        prediction = self.model.predict(preproced_eye,verbose = 0)

        # print("RAW MODEL OUTPUT:", prediction)

        probability_sleepy = float(prediction[0][0])

        # print("PROBABILITY:", probability_sleepy)

        if probability_sleepy>=0.5:
            class_index = 1
        else:
            class_index = 0

        label = self.class_names[class_index]

        if class_index == 1:
            confidence = probability_sleepy
        else:
            confidence = 1.0 - probability_sleepy

        return (label,confidence,probability_sleepy)
    

