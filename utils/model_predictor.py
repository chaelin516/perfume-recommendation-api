# perfume_backend/utils/model_predictor.py

import numpy as np
import pickle
import tensorflow as tf

MODEL_PATH = "./models/final_model_perfume.keras"
ENCODER_PATH = "./models/encoder.pkl"

# 모델 & 인코더 로드 (최초 1회만)
model = tf.keras.models.load_model(MODEL_PATH)
encoder = pickle.load(open(ENCODER_PATH, "rb"))

def predict_emotion_cluster(user_input: list) -> int:
    """
    user_input: [gender, season, time, desired_impression, activity, weather]
    return: 예측된 감정 클러스터 (int: 0~5)
    """
    X = encoder.transform([user_input])
    pred = model.predict(X)
    return int(np.argmax(pred))
