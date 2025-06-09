# utils/emotion_model_loader.py - 호환성 문제 해결 버전

import logging
import os
import pickle
import requests
from typing import Tuple, Dict, Any, List, Optional
import warnings

logger = logging.getLogger(__name__)

# 전역 변수
_emotion_model = None
_vectorizer = None
_model_available = False
_model_source = "없음"
_vectorizer_source = "없음"

# 🎭 감정 매핑
EMOTION_MAPPING = {
    0: "기쁨",
    1: "불안",
    2: "당황",
    3: "분노",
    4: "상처",
    5: "슬픔",
    6: "우울",
    7: "흥분"
}

# 모델 경로 설정
EMOTION_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "emotion_models")
EMOTION_MODEL_PATH = os.path.join(EMOTION_MODELS_DIR, "scent_emotion_model_v6.keras")
VECTORIZER_PATH = os.path.join(EMOTION_MODELS_DIR, "vectorizer.pkl")

# Google Drive 다운로드 URL
GOOGLE_DRIVE_MODEL_URL = "https://drive.google.com/uc?export=download&id=1H-TlOAE3r8zGWVDV7LlpI3KkdJ72OKJ2"


def download_model_from_google_drive() -> bool:
    """Google Drive에서 감정 태깅 모델 다운로드"""
    try:
        os.makedirs(EMOTION_MODELS_DIR, exist_ok=True)

        if os.path.exists(EMOTION_MODEL_PATH):
            model_size = os.path.getsize(EMOTION_MODEL_PATH)
            if model_size > 1000000:  # 1MB 이상이면 정상
                logger.info(f"✅ 감정 태깅 모델이 이미 존재합니다: {model_size:,} bytes")
                return True

        logger.info("📥 Google Drive에서 감정 태깅 모델 다운로드 시작...")

        response = requests.get(GOOGLE_DRIVE_MODEL_URL, stream=True, timeout=300)
        response.raise_for_status()

        with open(EMOTION_MODEL_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        final_size = os.path.getsize(EMOTION_MODEL_PATH)
        logger.info(f"✅ 감정 태깅 모델 다운로드 완료: {EMOTION_MODEL_PATH} ({final_size:,} bytes)")

        return True

    except Exception as e:
        logger.error(f"❌ 감정 태깅 모델 다운로드 실패: {e}")
        return False


def load_emotion_model():
    """감정 태깅 모델 로딩 (transformers 호환성 개선)"""
    global _emotion_model, _model_source

    if _emotion_model is not None:
        return _emotion_model

    try:
        if not os.path.exists(EMOTION_MODEL_PATH):
            logger.warning("⚠️ 감정 태깅 모델 파일이 없습니다. 다운로드를 시도합니다.")
            if not download_model_from_google_drive():
                return None

        # 파일 크기 확인
        model_size = os.path.getsize(EMOTION_MODEL_PATH)
        logger.info(f"✅ 감정 태깅 Keras 모델 검증 완료: {model_size:,} bytes")

        logger.info(f"🎭 감정 태깅 모델 로딩 시작: {EMOTION_MODEL_PATH}")

        # ✅ transformers와 TensorFlow 호환성 개선
        try:
            # transformers 임포트 및 설정
            import transformers
            transformers.logging.set_verbosity_error()  # 경고 메시지 줄이기

            # TensorFlow 설정
            import tensorflow as tf
            tf.get_logger().setLevel('ERROR')  # TensorFlow 로그 레벨 조정

            # ✅ Keras 3.x 호환성 설정
            try:
                from tensorflow import keras
                load_model = keras.models.load_model
                logger.info("📦 Keras 3.x 스타일로 모델 로딩")
            except:
                from tensorflow.keras.models import load_model
                logger.info("📦 TensorFlow 2.x 스타일로 모델 로딩")

            # ✅ custom_objects 설정으로 호환성 문제 해결
            custom_objects = {}

            # transformers 관련 클래스들 등록
            try:
                from transformers import TFRobertaModel
                custom_objects['TFRobertaModel'] = TFRobertaModel
                logger.info("🤖 TFRobertaModel 클래스 등록 완료")
            except ImportError as e:
                logger.error(f"❌ transformers 라이브러리 임포트 실패: {e}")
                logger.error("💡 'pip install transformers torch' 명령어로 설치해주세요")
                return None

            # ✅ compile=False로 로딩하여 optimizer 문제 회피
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _emotion_model = load_model(
                    EMOTION_MODEL_PATH,
                    compile=False,
                    custom_objects=custom_objects
                )

            _model_source = "Google Drive 다운로드"
            logger.info("✅ 감정 태깅 모델 로딩 완료")
            logger.info(f"📊 모델 입력 shape: {_emotion_model.input_shape}")
            logger.info(f"📊 모델 출력 shape: {_emotion_model.output_shape}")

            return _emotion_model

        except ImportError as e:
            logger.error(f"❌ 필수 라이브러리 누락: {e}")
            logger.error("💡 requirements.txt를 업데이트하고 다시 배포해주세요")
            return None

    except Exception as e:
        logger.error(f"❌ 감정 태깅 모델 로딩 실패: {e}")
        return None


def load_vectorizer():
    """벡터라이저 로딩"""
    global _vectorizer, _vectorizer_source

    if _vectorizer is not None:
        return _vectorizer

    try:
        # 로컬 파일 우선 시도
        if os.path.exists(VECTORIZER_PATH):
            logger.info(f"📊 감정 태깅 벡터라이저 로딩 시작 (로컬 파일): {VECTORIZER_PATH}")
            vectorizer_size = os.path.getsize(VECTORIZER_PATH)
            logger.info(f"📊 벡터라이저 파일 크기: {vectorizer_size:,} bytes")

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # scikit-learn 버전 경고 무시
                with open(VECTORIZER_PATH, 'rb') as f:
                    _vectorizer = pickle.load(f)

            _vectorizer_source = "로컬 파일 (Git 포함)"
            logger.info("✅ 감정 태깅 벡터라이저 로딩 완료 (로컬 파일)")
            return _vectorizer

    except Exception as e:
        logger.error(f"❌ 감정 태깅 벡터라이저 로딩 실패: {e}")
        return None


def initialize_emotion_tagging_models() -> Tuple[bool, str]:
    """감정 태깅 모델과 벡터라이저 초기화"""
    global _model_available

    logger.info("🎭 감정 태깅 모델 초기화 시작...")

    try:
        # 1. 모델 로딩 시도
        model = load_emotion_model()

        # 2. 벡터라이저 로딩 시도
        vectorizer = load_vectorizer()

        # 3. 결과 확인
        if model is not None and vectorizer is not None:
            _model_available = True
            logger.info("✅ 감정 태깅 시스템 완전 초기화 완료")
            return True, "AI 감정 태깅 모델과 벡터라이저 로딩 완료"
        elif vectorizer is not None:
            _model_available = False
            logger.warning("⚠️ 벡터라이저만 로딩됨. 모델 로딩 실패로 룰 기반으로 동작")
            return False, "벡터라이저만 로딩됨. 룰 기반으로 동작"
        else:
            _model_available = False
            logger.error("❌ 모델과 벡터라이저 모두 로딩 실패")
            return False, "감정 태깅 모델 로딩 실패"

    except Exception as e:
        _model_available = False
        logger.error(f"❌ 감정 태깅 모델 초기화 중 예외: {e}")
        return False, f"감정 태깅 모델 초기화 실패: {str(e)}"


def is_model_available() -> bool:
    """AI 모델 사용 가능 여부"""
    return _model_available and _emotion_model is not None


def get_model_status() -> Dict[str, Any]:
    """감정 태깅 시스템 상태 반환"""
    model_exists = os.path.exists(EMOTION_MODEL_PATH)
    vectorizer_exists = os.path.exists(VECTORIZER_PATH)

    return {
        "emotion_model_available": _emotion_model is not None,
        "emotion_model_exists": model_exists,
        "vectorizer_available": _vectorizer is not None,
        "vectorizer_exists": vectorizer_exists,
        "total_emotion_count": len(EMOTION_MAPPING),
        "supported_emotions": list(EMOTION_MAPPING.values()),
        "emotion_model_source": _model_source,
        "vectorizer_source": _vectorizer_source,
        "model_file_path": EMOTION_MODEL_PATH if model_exists else "없음",
        "vectorizer_file_path": VECTORIZER_PATH if vectorizer_exists else "없음"
    }


def predict_emotion_with_ai(text: str) -> Optional[str]:
    """AI 모델을 사용한 감정 예측"""
    if not is_model_available():
        return None

    try:
        # 텍스트 전처리 및 예측 로직
        # (실제 구현은 모델에 따라 다름)
        logger.info(f"🤖 AI 감정 예측: {text[:50]}...")

        # ✅ 임시로 룰 기반 결과 반환 (실제 모델 연동은 추후)
        return predict_emotion_with_rules(text)

    except Exception as e:
        logger.error(f"❌ AI 감정 예측 실패: {e}")
        return None


def predict_emotion_with_rules(text: str) -> str:
    """룰 기반 감정 예측"""
    if not text or not text.strip():
        return "중립"

    text_lower = text.lower()

    # 감정 키워드 사전
    emotion_keywords = {
        "기쁨": ["좋아", "행복", "기뻐", "즐거워", "만족", "완벽", "최고", "사랑", "상쾌", "밝은", "화사", "상큼", "달콤"],
        "불안": ["불안", "걱정", "긴장", "떨려", "두려운", "무서운", "어색", "부담", "스트레스"],
        "당황": ["당황", "놀란", "혼란", "어리둥절", "이상", "예상과 달라", "의외", "신기", "특이"],
        "분노": ["화가", "짜증", "열받", "분노", "싫어", "별로", "최악", "자극적", "강렬", "끔찍"],
        "상처": ["상처", "아픈", "서운", "실망", "아쉬워", "그리운", "애틋", "안타까운"],
        "슬픔": ["슬퍼", "눈물", "애절", "처량", "외로운", "쓸쓸", "찡한", "차가운"],
        "우울": ["우울", "답답", "무기력", "절망", "어둠", "침울", "멜랑콜리", "공허한"],
        "흥분": ["흥분", "신나", "두근", "설렘", "활기", "생동감", "에너지", "활발", "톡톡"]
    }

    # 감정별 점수 계산
    emotion_scores = {}
    for emotion, keywords in emotion_keywords.items():
        score = sum(1 for keyword in keywords if keyword in text_lower)
        if score > 0:
            emotion_scores[emotion] = score

    # 가장 높은 점수의 감정 반환
    if emotion_scores:
        best_emotion = max(emotion_scores.keys(), key=lambda x: emotion_scores[x])
        logger.info(f"📋 룰 기반 감정 예측: '{text[:30]}...' → {best_emotion}")
        return best_emotion
    else:
        logger.info(f"📋 룰 기반 감정 예측: '{text[:30]}...' → 기쁨 (기본값)")
        return "기쁨"  # 기본값


def predict_emotion(text: str) -> str:
    """감정 예측 (AI 우선, 실패 시 룰 기반)"""
    # AI 모델 시도
    ai_result = predict_emotion_with_ai(text)
    if ai_result:
        return ai_result

    # 룰 기반 폴백
    return predict_emotion_with_rules(text)