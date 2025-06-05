# perfume_backend/utils/model_predictor.py

import numpy as np
import pickle
import tensorflow as tf
import os

MODEL_PATH = "./models/final_model.keras"
ENCODER_PATH = "./models/encoder.pkl"

# 전역 변수로 모델과 인코더를 저장 (lazy loading)
_model = None
_encoder = None

def load_model_and_encoder():
    """모델과 인코더를 lazy loading으로 로드"""
    global _model, _encoder
    
    if _model is None or _encoder is None:
        try:
            # compile=False로 모델 로드 (optimizer 문제 회피)
            print("📦 모델을 로딩 중...")
            _model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            print("✅ 모델 로딩 완료")
            
            print("📦 인코더를 로딩 중...")
            with open(ENCODER_PATH, "rb") as f:
                _encoder = pickle.load(f)
            print("✅ 인코더 로딩 완료")
            
        except FileNotFoundError as e:
            print(f"❌ 파일을 찾을 수 없습니다: {e}")
            raise e
        except Exception as e:
            print(f"❌ 모델/인코더 로딩 중 오류 발생: {e}")
            raise e
    
    return _model, _encoder

def predict_emotion_cluster(user_input: list) -> int:
    """
    user_input: [gender, season, time, desired_impression, activity, weather]
    return: 예측된 감정 클러스터 (int: 0~5)
    """
    try:
        model, encoder = load_model_and_encoder()
        
        # 입력 데이터 전처리
        X = encoder.transform([user_input])
        
        # 예측 수행
        pred = model.predict(X, verbose=0)  # verbose=0으로 로그 출력 제거
        cluster_id = int(np.argmax(pred))
        
        print(f"🔮 예측 결과: 클러스터 {cluster_id}")
        return cluster_id
        
    except Exception as e:
        print(f"❌ 예측 중 오류 발생: {e}")
        # 기본값 반환 (에러 발생 시)
        return 0

def check_model_files():
    """모델 파일들이 존재하는지 확인"""
    model_exists = os.path.exists(MODEL_PATH)
    encoder_exists = os.path.exists(ENCODER_PATH)
    
    print(f"📁 모델 파일 확인:")
    print(f"  - {MODEL_PATH}: {'✅ 존재' if model_exists else '❌ 없음'}")
    print(f"  - {ENCODER_PATH}: {'✅ 존재' if encoder_exists else '❌ 없음'}")
    
    return model_exists and encoder_exists

if __name__ == "__main__":
    # 테스트 실행
    check_model_files()
    test_input = ["women", "spring", "day", "elegant", "casual", "any"]
    result = predict_emotion_cluster(test_input)
    print(f"테스트 결과: {result}")