# utils/emotion_model_loader.py
# 🎭 감정 분석 모델 전용 로더 (vectorizer.pkl + 감정 모델)

import os
import pickle
import logging
import requests
from typing import Tuple, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)

# ─── 글로벌 변수 ────────────────────────────────────────────────────────────
_vectorizer = None
_emotion_model = None
_models_loaded = False

# ─── 모델 파일 경로 ────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "../models")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "vectorizer.pkl")
EMOTION_MODEL_PATH = os.path.join(MODELS_DIR, "emotion_model.pkl")

# ─── Google Drive 직접 다운로드 URL ────────────────────────────────────────
GOOGLE_DRIVE_URLS = {
    "vectorizer.pkl": "https://drive.google.com/uc?export=download&id=YOUR_VECTORIZER_FILE_ID",
    "emotion_model.pkl": "https://drive.google.com/uc?export=download&id=YOUR_EMOTION_MODEL_FILE_ID"
}


def download_file_from_google_drive(file_id: str, destination: str) -> bool:
    """Google Drive에서 파일 다운로드"""
    try:
        URL = f"https://drive.google.com/uc?export=download&id={file_id}"

        logger.info(f"📥 Google Drive에서 다운로드 시작: {os.path.basename(destination)}")

        session = requests.Session()
        response = session.get(URL, stream=True)

        # 큰 파일의 경우 확인 토큰 처리
        token = None
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                token = value
                break

        if token:
            params = {'id': file_id, 'confirm': token}
            response = session.get(URL, params=params, stream=True)

        # 디렉토리 생성
        os.makedirs(os.path.dirname(destination), exist_ok=True)

        # 파일 저장
        with open(destination, "wb") as f:
            for chunk in response.iter_content(chunk_size=32768):
                if chunk:
                    f.write(chunk)

        # 파일 크기 확인
        file_size = os.path.getsize(destination)
        logger.info(f"✅ 다운로드 완료: {os.path.basename(destination)} ({file_size:,} bytes)")

        return file_size > 1000  # 1KB 이상이면 성공

    except Exception as e:
        logger.error(f"❌ Google Drive 다운로드 실패: {e}")
        return False


def ensure_emotion_models() -> bool:
    """감정 분석 모델 파일들이 존재하는지 확인하고, 없으면 다운로드"""
    try:
        logger.info("🔍 감정 분석 모델 파일 확인 중...")

        models_to_check = [
            ("vectorizer.pkl", VECTORIZER_PATH),
            ("emotion_model.pkl", EMOTION_MODEL_PATH)
        ]

        all_files_exist = True

        for model_name, model_path in models_to_check:
            if os.path.exists(model_path):
                file_size = os.path.getsize(model_path)
                if file_size > 1000:  # 1KB 이상
                    logger.info(f"✅ {model_name}: 존재함 ({file_size:,} bytes)")
                else:
                    logger.warning(f"⚠️ {model_name}: 파일이 너무 작음 ({file_size} bytes)")
                    all_files_exist = False
            else:
                logger.warning(f"❌ {model_name}: 없음")
                all_files_exist = False

        # 파일이 없으면 다운로드 시도 (실제 구현에서는 올바른 file_id 필요)
        if not all_files_exist:
            logger.info("📥 누락된 감정 모델 파일 다운로드 시도...")

            # 현재는 파일 ID가 없으므로 로그만 출력
            logger.warning("⚠️ Google Drive 파일 ID가 설정되지 않음")
            logger.warning("⚠️ 감정 분석은 룰 기반으로 동작합니다")
            return False

            # 실제 구현 시 아래 코드 사용:
            # for model_name, model_path in models_to_check:
            #     if not os.path.exists(model_path):
            #         file_id = "YOUR_FILE_ID_HERE"  # 실제 Google Drive 파일 ID
            #         success = download_file_from_google_drive(file_id, model_path)
            #         if not success:
            #             logger.error(f"❌ {model_name} 다운로드 실패")
            #             return False

        return all_files_exist

    except Exception as e:
        logger.error(f"❌ 감정 모델 파일 확인 중 오류: {e}")
        return False


def load_vectorizer() -> Optional[Any]:
    """텍스트 벡터라이저 로드"""
    global _vectorizer

    if _vectorizer is not None:
        return _vectorizer

    try:
        if not os.path.exists(VECTORIZER_PATH):
            logger.warning(f"⚠️ 벡터라이저 파일이 없습니다: {VECTORIZER_PATH}")
            return None

        file_size = os.path.getsize(VECTORIZER_PATH)
        logger.info(f"📦 벡터라이저 로딩 시작: {file_size:,} bytes")

        with open(VECTORIZER_PATH, "rb") as f:
            _vectorizer = pickle.load(f)

        logger.info("✅ 벡터라이저 로딩 완료")

        # 벡터라이저 검증
        if hasattr(_vectorizer, 'transform'):
            test_text = ["테스트 텍스트"]
            test_result = _vectorizer.transform(test_text)
            logger.info(f"🧪 벡터라이저 테스트 성공: 입력 1개 → 출력 {test_result.shape}")
        else:
            logger.warning("⚠️ 벡터라이저에 transform 메서드가 없습니다")

        return _vectorizer

    except Exception as e:
        logger.error(f"❌ 벡터라이저 로딩 실패: {e}")
        _vectorizer = None
        return None


def load_emotion_model() -> Optional[Any]:
    """감정 분류 모델 로드"""
    global _emotion_model

    if _emotion_model is not None:
        return _emotion_model

    try:
        if not os.path.exists(EMOTION_MODEL_PATH):
            logger.warning(f"⚠️ 감정 모델 파일이 없습니다: {EMOTION_MODEL_PATH}")
            return None

        file_size = os.path.getsize(EMOTION_MODEL_PATH)
        logger.info(f"📦 감정 모델 로딩 시작: {file_size:,} bytes")

        with open(EMOTION_MODEL_PATH, "rb") as f:
            _emotion_model = pickle.load(f)

        logger.info("✅ 감정 모델 로딩 완료")

        # 모델 검증
        if hasattr(_emotion_model, 'predict'):
            logger.info("🧪 감정 모델 predict 메서드 확인 완료")
        else:
            logger.warning("⚠️ 감정 모델에 predict 메서드가 없습니다")

        return _emotion_model

    except Exception as e:
        logger.error(f"❌ 감정 모델 로딩 실패: {e}")
        _emotion_model = None
        return None


def initialize_emotion_models() -> bool:
    """감정 분석 모델들 초기화"""
    global _models_loaded

    try:
        logger.info("🎭 감정 분석 모델 초기화 시작...")

        # 1. 파일 존재 확인 및 다운로드
        files_ready = ensure_emotion_models()
        if not files_ready:
            logger.warning("⚠️ 감정 모델 파일 준비 실패 - 룰 기반으로 동작")
            _models_loaded = False
            return False

        # 2. 벡터라이저 로드
        vectorizer = load_vectorizer()
        if vectorizer is None:
            logger.error("❌ 벡터라이저 로딩 실패")
            _models_loaded = False
            return False

        # 3. 감정 모델 로드
        emotion_model = load_emotion_model()
        if emotion_model is None:
            logger.error("❌ 감정 모델 로딩 실패")
            _models_loaded = False
            return False

        # 4. 통합 테스트
        try:
            test_text = "이 향수 정말 좋아요! 기분이 좋아져요."
            vectorized = vectorizer.transform([test_text])
            prediction = emotion_model.predict(vectorized)

            logger.info(f"🧪 통합 테스트 성공: '{test_text}' → 예측 결과")
            logger.info(f"📊 벡터화 결과 shape: {vectorized.shape}")

        except Exception as test_error:
            logger.error(f"❌ 모델 통합 테스트 실패: {test_error}")
            _models_loaded = False
            return False

        _models_loaded = True
        logger.info("✅ 감정 분석 모델 초기화 완료")
        return True

    except Exception as e:
        logger.error(f"❌ 감정 모델 초기화 중 오류: {e}")
        _models_loaded = False
        return False


def get_emotion_models() -> Tuple[Optional[Any], Optional[Any]]:
    """로드된 감정 분석 모델들 반환"""
    return _vectorizer, _emotion_model


def get_emotion_models_status() -> dict:
    """감정 분석 모델 상태 정보 반환"""
    vectorizer_loaded = _vectorizer is not None
    emotion_model_loaded = _emotion_model is not None

    status = {
        "models_initialized": _models_loaded,
        "vectorizer_loaded": vectorizer_loaded,
        "emotion_model_loaded": emotion_model_loaded,
        "vectorizer_path": VECTORIZER_PATH,
        "emotion_model_path": EMOTION_MODEL_PATH,
        "vectorizer_exists": os.path.exists(VECTORIZER_PATH),
        "emotion_model_exists": os.path.exists(EMOTION_MODEL_PATH)
    }

    # 파일 크기 정보 추가
    if status["vectorizer_exists"]:
        status["vectorizer_size"] = os.path.getsize(VECTORIZER_PATH)

    if status["emotion_model_exists"]:
        status["emotion_model_size"] = os.path.getsize(EMOTION_MODEL_PATH)

    return status


def predict_emotion_with_models(text: str) -> Optional[dict]:
    """로드된 모델로 감정 예측"""
    if not _models_loaded or _vectorizer is None or _emotion_model is None:
        logger.warning("⚠️ 감정 모델이 로드되지 않음")
        return None

    try:
        # 텍스트 벡터화
        vectorized = _vectorizer.transform([text])

        # 감정 예측
        prediction = _emotion_model.predict(vectorized)

        # 확률 예측 (가능한 경우)
        if hasattr(_emotion_model, 'predict_proba'):
            probabilities = _emotion_model.predict_proba(vectorized)[0]
            confidence = float(max(probabilities))
        else:
            confidence = 0.7  # 기본 신뢰도

        result = {
            "prediction": prediction[0] if hasattr(prediction, '__getitem__') else str(prediction),
            "confidence": confidence,
            "method": "ML 모델 (vectorizer + 분류기)",
            "vectorized_shape": vectorized.shape
        }

        return result

    except Exception as e:
        logger.error(f"❌ 모델 기반 감정 예측 실패: {e}")
        return None


def reset_emotion_models():
    """감정 분석 모델 리셋"""
    global _vectorizer, _emotion_model, _models_loaded

    logger.info("🔄 감정 분석 모델 리셋...")

    _vectorizer = None
    _emotion_model = None
    _models_loaded = False

    logger.info("✅ 감정 분석 모델 리셋 완료")


# 모듈 로드 시 자동 초기화 (옵션)
if __name__ == "__main__":
    # 직접 실행 시 테스트
    logger.info("🧪 감정 모델 로더 테스트 시작...")

    success = initialize_emotion_models()
    status = get_emotion_models_status()

    print(f"초기화 성공: {success}")
    print(f"상태 정보: {status}")

    if success:
        test_result = predict_emotion_with_models("이 향수 정말 좋아요!")
        print(f"테스트 예측: {test_result}")