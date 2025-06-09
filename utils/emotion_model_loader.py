
# 🎭 감정 태깅 모델 로더 - 시향일기 자동 태깅용

import os
import pickle
import requests
import logging
from pathlib import Path
import gdown
from typing import Optional, Tuple, Any, List
import numpy as np

logger = logging.getLogger(__name__)

# 🔗 Google Drive 파일 ID 설정
EMOTION_MODEL_FILE_ID = "1JYUJvKVb44p63ctWe3c1G_qmdcDr-Xix"  # 감정 태깅 모델 파일 ID

# 로컬 모델 파일 경로
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "emotion_models"
EMOTION_MODEL_PATH = MODELS_DIR / "scent_emotion_model_v6.keras"
VECTORIZER_PATH = MODELS_DIR / "vectorizer.pkl"

# 🎭 감정 태그 매핑 (데이터셋 기준)
EMOTION_TAGS = {
    0: "기쁨",
    1: "불안",
    2: "당황",
    3: "분노",
    4: "상처",
    5: "슬픔",
    6: "우울",
    7: "흥분"
}

EMOTION_LABELS = {v: k for k, v in EMOTION_TAGS.items()}  # 역매핑

# 전역 변수로 모델과 벡터라이저 저장
_emotion_model = None
_vectorizer = None
_model_loaded = False


def create_models_directory():
    """모델 디렉토리 생성"""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"📁 감정 태깅 모델 디렉토리 확인: {MODELS_DIR}")


def download_from_google_drive_gdown(file_id: str, output_path: str) -> bool:
    """gdown 라이브러리를 사용한 Google Drive 다운로드"""
    try:
        logger.info(f"📥 Google Drive에서 감정 태깅 모델 다운로드 시작 (gdown): {file_id}")

        url = f"https://drive.google.com/uc?id={file_id}"
        output = gdown.download(url, output_path, quiet=False)

        if output and os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            logger.info(f"✅ 감정 태깅 모델 다운로드 완료: {output_path} ({file_size:,} bytes)")
            return True
        else:
            logger.error("❌ 감정 태깅 모델 gdown 다운로드 실패")
            return False

    except Exception as e:
        logger.error(f"❌ 감정 태깅 모델 gdown 다운로드 오류: {e}")
        return False


def download_from_google_drive_requests(file_id: str, destination: str) -> bool:
    """requests를 사용한 Google Drive 다운로드 (폴백)"""
    try:
        logger.info(f"📥 Google Drive에서 감정 태깅 모델 다운로드 시작 (requests): {file_id}")

        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        session = requests.Session()
        response = session.get(url, stream=True)

        if response.status_code == 200:
            if "virus scan warning" in response.text.lower() or "download_warning" in response.text:
                logger.info("🔍 대용량 파일 확인 토큰 처리 중...")
                confirm_url = f"https://drive.google.com/uc?export=download&confirm=t&id={file_id}"
                response = session.get(confirm_url, stream=True)

        if response.status_code == 200:
            os.makedirs(os.path.dirname(destination), exist_ok=True)

            total_size = 0
            with open(destination, 'wb') as f:
                for chunk in response.iter_content(chunk_size=32768):
                    if chunk:
                        f.write(chunk)
                        total_size += len(chunk)

                        if total_size % (10 * 1024 * 1024) == 0:
                            logger.info(f"📥 다운로드 진행: {total_size / 1024 / 1024:.1f}MB")

            file_size = os.path.getsize(destination)
            logger.info(f"✅ 감정 태깅 모델 requests 다운로드 완료: {destination} ({file_size:,} bytes)")
            return True
        else:
            logger.error(f"❌ 다운로드 실패: HTTP {response.status_code}")
            return False

    except Exception as e:
        logger.error(f"❌ 감정 태깅 모델 requests 다운로드 오류: {e}")
        return False


def download_model_file(file_id: str, output_path: str) -> bool:
    """감정 태깅 모델 파일 다운로드 (여러 방법 시도)"""
    if not file_id:
        logger.warning("⚠️ 감정 태깅 모델 파일 ID가 제공되지 않음")
        return False

    create_models_directory()

    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        if file_size > 1024:
            logger.info(f"✅ 감정 태깅 모델이 이미 존재: {output_path} ({file_size:,} bytes)")
            return True

    # 방법 1: gdown 시도
    try:
        import gdown
        if download_from_google_drive_gdown(file_id, output_path):
            return True
    except ImportError:
        logger.warning("⚠️ gdown 라이브러리가 없음. requests로 시도합니다.")
    except Exception as e:
        logger.warning(f"⚠️ gdown 방법 실패: {e}")

    # 방법 2: requests 시도
    if download_from_google_drive_requests(file_id, output_path):
        return True

    logger.error("❌ 모든 감정 태깅 모델 다운로드 방법 실패")
    return False


def verify_model_file(file_path: str) -> bool:
    """감정 태깅 모델 파일 유효성 검증"""
    try:
        if not os.path.exists(file_path):
            return False

        file_size = os.path.getsize(file_path)
        if file_size < 1024:
            logger.warning(f"⚠️ 감정 태깅 모델 파일 크기가 너무 작음: {file_size} bytes")
            return False

        if file_path.endswith('.keras'):
            try:
                import tensorflow as tf
                with open(file_path, 'rb') as f:
                    header = f.read(100)
                    if b'keras' in header.lower() or b'tensorflow' in header.lower():
                        logger.info(f"✅ 감정 태깅 Keras 모델 검증 완료: {file_size:,} bytes")
                        return True
            except Exception as e:
                logger.warning(f"⚠️ 감정 태깅 Keras 모델 검증 실패: {e}")

        elif file_path.endswith('.pkl'):
            try:
                with open(file_path, 'rb') as f:
                    header = f.read(10)
                    if header.startswith(b'\x80'):
                        logger.info(f"✅ 감정 태깅 Pickle 파일 검증 완료: {file_size:,} bytes")
                        return True
            except Exception as e:
                logger.warning(f"⚠️ 감정 태깅 Pickle 파일 검증 실패: {e}")

        logger.info(f"✅ 감정 태깅 모델 기본 검증 완료: {file_size:,} bytes")
        return True

    except Exception as e:
        logger.error(f"❌ 감정 태깅 모델 검증 중 오류: {e}")
        return False


def load_emotion_tagging_model():
    """감정 태깅 모델 로딩"""
    global _emotion_model

    if _emotion_model is not None:
        return _emotion_model

    try:
        # 모델 파일 다운로드 (필요한 경우)
        if not verify_model_file(str(EMOTION_MODEL_PATH)):
            logger.info("📥 감정 태깅 모델 파일 다운로드 중...")
            if not download_model_file(EMOTION_MODEL_FILE_ID, str(EMOTION_MODEL_PATH)):
                raise Exception("감정 태깅 모델 파일 다운로드 실패")

        # 모델 로딩
        logger.info(f"🎭 감정 태깅 모델 로딩 시작: {EMOTION_MODEL_PATH}")

        import tensorflow as tf

        # Keras 모델 로딩
        _emotion_model = tf.keras.models.load_model(str(EMOTION_MODEL_PATH), compile=False)

        logger.info(f"✅ 감정 태깅 모델 로딩 완료")
        logger.info(f"  - 입력 shape: {_emotion_model.input_shape}")
        logger.info(f"  - 출력 shape: {_emotion_model.output_shape}")
        logger.info(f"  - 예상 출력: 8개 감정 태그 확률")

        return _emotion_model

    except Exception as e:
        logger.error(f"❌ 감정 태깅 모델 로딩 실패: {e}")
        _emotion_model = None
        return None


def load_vectorizer():
    """벡터라이저 로딩 (로컬 파일 사용)"""
    global _vectorizer

    if _vectorizer is not None:
        return _vectorizer

    try:
        if os.path.exists(VECTORIZER_PATH):
            logger.info(f"📊 감정 태깅 벡터라이저 로딩 시작 (로컬 파일): {VECTORIZER_PATH}")

            file_size = os.path.getsize(VECTORIZER_PATH)
            logger.info(f"📊 벡터라이저 파일 크기: {file_size:,} bytes")

            with open(VECTORIZER_PATH, 'rb') as f:
                _vectorizer = pickle.load(f)

            logger.info("✅ 감정 태깅 벡터라이저 로딩 완료 (로컬 파일)")
            return _vectorizer
        else:
            logger.warning("⚠️ 감정 태깅 벡터라이저 파일이 없음 - 모델 내장 전처리 사용")
            logger.warning(f"  예상 경로: {VECTORIZER_PATH}")
            return None

    except Exception as e:
        logger.error(f"❌ 감정 태깅 벡터라이저 로딩 실패: {e}")
        _vectorizer = None
        return None


def preprocess_text(text: str) -> str:
    """텍스트 전처리 (간단한 정제)"""
    if not text:
        return ""

    # 기본 정제
    text = text.strip()
    # 추가적인 전처리가 필요하면 여기에 추가

    return text


def predict_emotion_tags(text: str) -> dict:
    """
    시향일기 텍스트에서 감정 태그 예측

    Args:
        text: 시향일기 텍스트

    Returns:
        {
            "success": bool,
            "predicted_emotion": str,
            "confidence": float,
            "all_probabilities": dict,
            "method": str
        }
    """
    try:
        logger.info(f"🎭 감정 태깅 예측 시작: '{text[:50]}...'")

        # 텍스트 전처리
        processed_text = preprocess_text(text)
        if not processed_text:
            return {
                "success": False,
                "error": "빈 텍스트",
                "method": "validation"
            }

        # 모델과 벡터라이저 로딩
        model = load_emotion_tagging_model()
        vectorizer = load_vectorizer()

        if model is None:
            # 모델이 없으면 룰 기반 폴백
            return _rule_based_emotion_tagging(processed_text)

        if vectorizer is None:
            logger.warning("⚠️ 벡터라이저가 없어 간단한 전처리 사용")
            # 간단한 전처리 로직 (실제 모델에 따라 조정 필요)
            return _simple_model_prediction(model, processed_text)

        # 벡터라이저를 사용한 텍스트 변환
        try:
            text_vector = vectorizer.transform([processed_text])

            # 모델 예측
            predictions = model.predict(text_vector, verbose=0)
            probabilities = predictions[0]  # 첫 번째 샘플의 확률들

            # 가장 높은 확률의 감정 태그
            predicted_label = int(np.argmax(probabilities))
            predicted_emotion = EMOTION_TAGS[predicted_label]
            confidence = float(probabilities[predicted_label])

            # 모든 감정별 확률
            all_probabilities = {
                EMOTION_TAGS[i]: float(prob)
                for i, prob in enumerate(probabilities)
            }

            logger.info(f"✅ 감정 태깅 예측 완료: {predicted_emotion} (신뢰도: {confidence:.3f})")

            return {
                "success": True,
                "predicted_emotion": predicted_emotion,
                "predicted_label": predicted_label,
                "confidence": confidence,
                "all_probabilities": all_probabilities,
                "method": "AI 모델",
                "processed_text": processed_text
            }

        except Exception as e:
            logger.error(f"❌ 모델 예측 중 오류: {e}")
            return _rule_based_emotion_tagging(processed_text)

    except Exception as e:
        logger.error(f"❌ 감정 태깅 예측 중 예외: {e}")
        return {
            "success": False,
            "error": str(e),
            "method": "exception"
        }


def _simple_model_prediction(model, text: str) -> dict:
    """벡터라이저 없이 간단한 모델 예측 (실험적)"""
    try:
        # 간단한 텍스트 인코딩 (실제 모델 입력에 맞게 조정 필요)
        # 여기서는 폴백으로 룰 기반 사용
        logger.warning("⚠️ 벡터라이저 없이는 모델 사용 불가, 룰 기반으로 전환")
        return _rule_based_emotion_tagging(text)

    except Exception as e:
        logger.error(f"❌ 간단한 모델 예측 실패: {e}")
        return _rule_based_emotion_tagging(text)


def _rule_based_emotion_tagging(text: str) -> dict:
    """룰 기반 감정 태깅 (폴백)"""
    try:
        logger.info(f"📋 룰 기반 감정 태깅 시작: '{text[:30]}...'")

        text_lower = text.lower()

        # 감정별 키워드 매칭
        emotion_scores = {}

        # 8개 감정별 키워드 (데이터셋 기반으로 작성)
        emotion_keywords = {
            "기쁨": ["좋", "행복", "기뻐", "즐거", "만족", "완벽", "사랑", "따뜻", "포근", "밝", "상쾌", "달콤"],
            "불안": ["불안", "걱정", "긴장", "떨", "두려", "무서", "조마조마", "어색", "부담", "스트레스"],
            "당황": ["당황", "놀", "혼란", "어리둥절", "멍", "모르겠", "헷갈", "이상", "의외", "신기"],
            "분노": ["화", "짜증", "열받", "분노", "싫", "별로", "최악", "자극적", "강렬", "과해"],
            "상처": ["상처", "아픈", "서운", "실망", "아쉬", "힘든", "섭섭", "그리운", "애틋"],
            "슬픔": ["슬", "눈물", "애절", "처량", "외로", "쓸쓸", "먹먹", "찡", "울컥", "진한"],
            "우울": ["우울", "답답", "무기력", "절망", "어둠", "침울", "멜랑콜리", "블루", "막막"],
            "흥분": ["흥분", "신나", "두근", "설렘", "활기", "생동감", "에너지", "활발", "톡톡", "생생"]
        }

        for emotion, keywords in emotion_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in text_lower:
                    score += text_lower.count(keyword)
            emotion_scores[emotion] = score

        # 최고 점수 감정 선택
        if any(score > 0 for score in emotion_scores.values()):
            predicted_emotion = max(emotion_scores.keys(), key=lambda x: emotion_scores[x])
            max_score = emotion_scores[predicted_emotion]
            confidence = min(max_score / len(text.split()) * 2, 1.0)  # 정규화
        else:
            # 기본값: 중립적 감정
            predicted_emotion = "기쁨"  # 기본값
            confidence = 0.3

        predicted_label = EMOTION_LABELS[predicted_emotion]

        # 모든 감정별 정규화된 확률
        total_score = sum(emotion_scores.values()) or 1
        all_probabilities = {
            emotion: score / total_score
            for emotion, score in emotion_scores.items()
        }

        logger.info(f"✅ 룰 기반 감정 태깅 완료: {predicted_emotion} (신뢰도: {confidence:.3f})")

        return {
            "success": True,
            "predicted_emotion": predicted_emotion,
            "predicted_label": predicted_label,
            "confidence": confidence,
            "all_probabilities": all_probabilities,
            "method": "룰 기반",
            "keyword_scores": emotion_scores
        }

    except Exception as e:
        logger.error(f"❌ 룰 기반 감정 태깅 실패: {e}")
        return {
            "success": False,
            "error": str(e),
            "method": "rule_based_error"
        }


def initialize_emotion_tagging_models() -> Tuple[bool, str]:
    """감정 태깅 모델 초기화"""
    global _model_loaded

    try:
        logger.info("🎭 감정 태깅 모델 초기화 시작...")

        # 모델 파일 확인 (감정 모델은 다운로드, 벡터라이저는 로컬)
        model_available = verify_model_file(str(EMOTION_MODEL_PATH))
        vectorizer_available = os.path.exists(VECTORIZER_PATH)

        logger.info(f"📋 감정 태깅 모델 파일 상태:")
        logger.info(f"  - 감정 태깅 모델: {'✅ 존재' if model_available else '❌ 다운로드 필요'}")
        logger.info(f"  - 벡터라이저: {'✅ 존재 (로컬)' if vectorizer_available else '❌ 없음 (로컬)'}")

        # 감정 모델 다운로드 (필요한 경우)
        if not model_available:
            logger.info("📥 감정 태깅 모델 파일 다운로드 시도...")
            if EMOTION_MODEL_FILE_ID:
                if download_model_file(EMOTION_MODEL_FILE_ID, str(EMOTION_MODEL_PATH)):
                    model_available = True
                    logger.info("✅ 감정 태깅 모델 다운로드 완료")
                else:
                    return False, "감정 태깅 모델 파일 다운로드 실패"
            else:
                return False, "감정 태깅 모델 파일 ID가 설정되지 않음"

        # 벡터라이저 확인 (로컬 파일)
        if not vectorizer_available:
            logger.warning(f"⚠️ 감정 태깅 벡터라이저 파일이 없습니다: {VECTORIZER_PATH}")
            logger.warning("  모델 내장 전처리를 사용하거나 룰 기반으로 동작합니다")

        # 모델 로딩 테스트
        if model_available:
            model = load_emotion_tagging_model()
            vectorizer = load_vectorizer()  # 실패해도 계속 진행

            if model is not None:
                logger.info("✅ 감정 태깅 모델 초기화 완료")
                logger.info(f"  - 감정 태깅 모델: ✅ 로드됨")
                logger.info(f"  - 벡터라이저: {'✅ 로드됨' if vectorizer is not None else '❌ 없음 (룰 기반 사용)'}")
                logger.info(f"  - 지원 감정: {list(EMOTION_TAGS.values())}")
                _model_loaded = True
                return True, "감정 태깅 모델 초기화 성공"
            else:
                return False, "감정 태깅 모델 로딩 실패"
        else:
            return False, "감정 태깅 모델 파일 준비 실패"

    except Exception as e:
        logger.error(f"❌ 감정 태깅 모델 초기화 실패: {e}")
        return False, f"초기화 중 오류: {str(e)}"


def is_model_available() -> bool:
    """감정 태깅 모델 사용 가능 여부 확인"""
    return _model_loaded and _emotion_model is not None


def get_model_status() -> dict:
    """감정 태깅 모델 상태 정보 반환"""
    return {
        "model_loaded": _model_loaded,
        "emotion_model_available": _emotion_model is not None,
        "vectorizer_available": _vectorizer is not None,
        "emotion_model_path": str(EMOTION_MODEL_PATH),
        "vectorizer_path": str(VECTORIZER_PATH),
        "emotion_model_exists": os.path.exists(EMOTION_MODEL_PATH),
        "vectorizer_exists": os.path.exists(VECTORIZER_PATH),
        "emotion_model_file_id": EMOTION_MODEL_FILE_ID,
        "vectorizer_source": "로컬 파일 (Git 포함)",
        "emotion_model_source": "Google Drive 다운로드",
        "supported_emotions": list(EMOTION_TAGS.values()),
        "total_emotion_count": len(EMOTION_TAGS),
        "emotion_label_mapping": EMOTION_LABELS
    }


def get_supported_emotions() -> List[str]:
    """지원하는 감정 태그 목록 반환"""
    return list(EMOTION_TAGS.values())


def get_emotion_label_mapping() -> dict:
    """감정-라벨 매핑 반환"""
    return EMOTION_LABELS.copy()


# 🧪 테스트 함수
def test_emotion_tagging(test_texts: List[str] = None):
    """감정 태깅 테스트"""
    if test_texts is None:
        test_texts = [
            "향기를 맡으니 내 안에 따뜻함이 번졌다.",
            "낯선 공간에서 이 향은 지나치게 도드라졌다.",
            "내가 뿌렸지만 내가 당황한 향이었다.",
            "나를 위한 공간이 향 하나로 낯설어졌다.",
            "그날의 말이 향처럼 다시 퍼져나갔다."
        ]

    logger.info("🧪 감정 태깅 테스트 시작...")

    for i, text in enumerate(test_texts, 1):
        logger.info(f"\n--- 테스트 {i} ---")
        logger.info(f"입력: {text}")

        result = predict_emotion_tags(text)

        if result.get("success"):
            logger.info(f"예측 감정: {result['predicted_emotion']}")
            logger.info(f"신뢰도: {result['confidence']:.3f}")
            logger.info(f"방법: {result['method']}")
        else:
            logger.error(f"예측 실패: {result.get('error', '알 수 없는 오류')}")

    logger.info("✅ 감정 태깅 테스트 완료!")


if __name__ == "__main__":
    # 직접 실행 시 테스트
    test_emotion_tagging()