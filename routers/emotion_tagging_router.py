# routers/emotion_tagging_router.py
# 🎯 감정 태깅 AI 모델 API 라우터 (scent_emotion_model_v6.keras 연동)

import os
import logging
import numpy as np
import pickle
from datetime import datetime
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

# TensorFlow 동적 임포트 (vectorizer.pkl 받기 전까지는 주석 처리)
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model

    TENSORFLOW_AVAILABLE = True
    logger = logging.getLogger("emotion_tagging")
    logger.info("✅ TensorFlow 사용 가능")
except ImportError as e:
    TENSORFLOW_AVAILABLE = False
    logger = logging.getLogger("emotion_tagging")
    logger.warning(f"⚠️ TensorFlow 없음: {e}")

# scikit-learn 동적 임포트
try:
    from sklearn.feature_extraction.text import TfidfVectorizer

    SKLEARN_AVAILABLE = True
    logger.info("✅ scikit-learn 사용 가능")
except ImportError as e:
    SKLEARN_AVAILABLE = False
    logger.warning(f"⚠️ scikit-learn 없음: {e}")

# 기존 감정 분석기 임포트 (룰 기반 폴백용)
from utils.emotion_analyzer import emotion_analyzer


# ─── 1. 스키마 정의 ─────────────────────────────────────────
class EmotionPredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000, description="감정을 분석할 텍스트")
    include_probabilities: bool = Field(False, description="전체 감정 확률 분포 포함 여부")

    @validator('text')
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError('텍스트가 비어있습니다.')
        return v.strip()


class EmotionPredictResponse(BaseModel):
    emotion: str = Field(..., description="예측된 감정")
    confidence: float = Field(..., ge=0.0, le=1.0, description="신뢰도 (0.0-1.0)")
    label: int = Field(..., ge=0, le=7, description="감정 라벨 (0-7)")
    method: str = Field(..., description="사용된 분석 방법")
    processing_time: float = Field(..., description="처리 시간 (초)")
    probabilities: Optional[Dict[str, float]] = Field(None, description="전체 감정별 확률 분포")


class EmotionHealthResponse(BaseModel):
    ai_model_available: bool = Field(..., description="AI 모델 사용 가능 여부")
    vectorizer_available: bool = Field(..., description="벡터라이저 사용 가능 여부")
    fallback_available: bool = Field(..., description="룰 기반 폴백 사용 가능 여부")
    supported_emotions: List[str] = Field(..., description="지원하는 감정 목록")
    model_info: Dict[str, Any] = Field(..., description="모델 정보")


# ─── 2. 감정 태깅 클래스 ─────────────────────────────────────
class WhiffEmotionTagger:
    """Whiff 전용 감정 태깅 클래스"""

    def __init__(self):
        # 8가지 감정 매핑 (데이터셋과 동일)
        self.emotion_labels = {
            0: "기쁨", 1: "불안", 2: "당황", 3: "분노",
            4: "상처", 5: "슬픔", 6: "우울", 7: "흥분"
        }

        # 파일 경로 설정 (기존 models와 분리)
        self.model_path = "emotion_models/scent_emotion_model_v6.keras"
        self.vectorizer_path = "emotion_models/vectorizer.pkl"

        # 모델 상태
        self.model = None
        self.vectorizer = None
        self.model_loaded = False
        self.vectorizer_loaded = False

        # 초기화 시도
        self._initialize_model()

        logger.info("🎭 Whiff 감정 태깅 시스템 초기화 완료")

    def _initialize_model(self):
        """모델 및 벡터라이저 초기화"""
        try:
            # 1. Keras 모델 로드 시도
            if TENSORFLOW_AVAILABLE and os.path.exists(self.model_path):
                logger.info(f"🤖 Keras 모델 로딩 시도: {self.model_path}")

                model_size = os.path.getsize(self.model_path)
                logger.info(f"📄 모델 파일 크기: {model_size:,}B ({model_size / 1024 / 1024:.1f}MB)")

                self.model = load_model(self.model_path, compile=False)
                self.model_loaded = True

                logger.info(f"✅ Keras 모델 로드 성공")
                logger.info(f"📊 모델 입력 shape: {self.model.input_shape}")
                logger.info(f"📊 모델 출력 shape: {self.model.output_shape}")
            else:
                logger.warning(f"⚠️ Keras 모델 파일 없음: {self.model_path}")

            # 2. 벡터라이저 로드 시도
            if SKLEARN_AVAILABLE and os.path.exists(self.vectorizer_path):
                logger.info(f"📦 벡터라이저 로딩 시도: {self.vectorizer_path}")

                with open(self.vectorizer_path, 'rb') as f:
                    self.vectorizer = pickle.load(f)
                self.vectorizer_loaded = True

                logger.info(f"✅ 벡터라이저 로드 성공")
                if hasattr(self.vectorizer, 'vocabulary_'):
                    logger.info(f"📊 어휘 크기: {len(self.vectorizer.vocabulary_)}")
            else:
                logger.warning(f"⚠️ 벡터라이저 파일 없음: {self.vectorizer_path}")

                # 🔧 임시 벡터라이저 생성 (개발용)
                if SKLEARN_AVAILABLE:
                    logger.info("🔧 개발용 임시 벡터라이저 생성 중...")
                    self._create_temporary_vectorizer()

        except Exception as e:
            logger.error(f"❌ 모델 초기화 실패: {e}")

    def _create_temporary_vectorizer(self):
        """개발용 임시 벡터라이저 생성 (vectorizer.pkl 받기 전까지)"""
        try:
            # 샘플 텍스트로 임시 벡터라이저 훈련
            sample_texts = [
                "향기를 맡으니 내 안에 따뜻함이 번졌다",
                "이 향은 불안한 마음을 달래주지 못했다",
                "갑자기 당황스러운 향이 코를 찔렀다",
                "화가 나는 냄새가 코를 자극했다",
                "마음이 아픈 향기였다",
                "슬픈 기억이 떠오르는 향수",
                "우울한 기분이 드는 향",
                "흥분되는 향기가 가슴을 뛰게 했다"
            ]

            # TF-IDF 벡터라이저 생성 (모델 학습 시와 유사한 설정)
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                lowercase=True,
                stop_words=None,
                token_pattern=r'[가-힣a-zA-Z0-9]+',  # 한글, 영문, 숫자
                min_df=1,
                max_df=0.95
            )

            # 샘플 텍스트로 피팅
            self.vectorizer.fit(sample_texts)
            self.vectorizer_loaded = True

            logger.info("✅ 임시 벡터라이저 생성 완료")
            logger.info(f"📊 임시 어휘 크기: {len(self.vectorizer.vocabulary_)}")
            logger.warning("⚠️ 이것은 개발용 임시 벡터라이저입니다. vectorizer.pkl 수령 후 교체하세요!")

        except Exception as e:
            logger.error(f"❌ 임시 벡터라이저 생성 실패: {e}")

    def preprocess_text(self, text: str) -> str:
        """텍스트 전처리"""
        if not text:
            return ""

        import re

        # 특수문자 제거 (한글, 영문, 숫자, 공백만 유지)
        text = re.sub(r'[^가-힣a-zA-Z0-9\s]', ' ', text)

        # 여러 공백을 하나로 변환
        text = re.sub(r'\s+', ' ', text)

        # 앞뒤 공백 제거
        text = text.strip()

        return text

    def predict_with_ai_model(self, text: str, include_probabilities: bool = False) -> Dict[str, Any]:
        """AI 모델을 사용한 감정 예측"""
        if not (self.model_loaded and self.vectorizer_loaded):
            raise Exception("AI 모델 또는 벡터라이저가 로드되지 않았습니다.")

        start_time = datetime.now()

        try:
            # 1. 텍스트 전처리
            processed_text = self.preprocess_text(text)

            if not processed_text:
                raise Exception("전처리 후 텍스트가 비어있습니다.")

            # 2. TF-IDF 벡터화
            text_vector = self.vectorizer.transform([processed_text])

            # 3. 모델 예측
            predictions = self.model.predict(text_vector.toarray(), verbose=0)
            predicted_label = int(np.argmax(predictions[0]))
            confidence = float(np.max(predictions[0]))

            # 4. 결과 구성
            result = {
                "emotion": self.emotion_labels[predicted_label],
                "confidence": confidence,
                "label": predicted_label,
                "method": "AI 모델 (Keras + TF-IDF)",
                "processing_time": (datetime.now() - start_time).total_seconds()
            }

            if include_probabilities:
                result["probabilities"] = {
                    self.emotion_labels[i]: float(prob)
                    for i, prob in enumerate(predictions[0])
                }

            logger.info(f"🤖 AI 예측 완료: '{text[:30]}...' → {result['emotion']} ({confidence:.3f})")

            return result

        except Exception as e:
            logger.error(f"❌ AI 모델 예측 실패: {e}")
            raise e

    def predict_with_rule_based(self, text: str, include_probabilities: bool = False) -> Dict[str, Any]:
        """룰 기반 감정 분석 (폴백)"""
        start_time = datetime.now()

        try:
            # 기존 emotion_analyzer 사용
            analysis_result = emotion_analyzer.analyze_emotion(text)

            if not analysis_result.get("success"):
                raise Exception("룰 기반 분석 실패")

            # Whiff 8가지 감정으로 매핑
            rule_emotion = analysis_result.get("primary_emotion", "중립")

            # 기존 감정을 8가지 감정으로 매핑
            emotion_mapping = {
                "기쁨": (0, "기쁨"),
                "불안": (1, "불안"),
                "당황": (2, "당황"),
                "분노": (3, "분노"),
                "상처": (4, "상처"),
                "슬픔": (5, "슬픔"),
                "우울": (6, "우울"),
                "흥분": (7, "흥분"),
                "중립": (0, "기쁨")  # 기본값
            }

            label, emotion = emotion_mapping.get(rule_emotion, (0, "기쁨"))
            confidence = analysis_result.get("confidence", 0.5)

            result = {
                "emotion": emotion,
                "confidence": confidence,
                "label": label,
                "method": "룰 기반 (폴백)",
                "processing_time": (datetime.now() - start_time).total_seconds()
            }

            if include_probabilities:
                # 기본 확률 분포 생성
                probs = [0.125] * 8  # 균등 분포
                probs[label] = confidence
                # 정규화
                total = sum(probs)
                probs = [p / total for p in probs]

                result["probabilities"] = {
                    self.emotion_labels[i]: prob
                    for i, prob in enumerate(probs)
                }

            logger.info(f"📋 룰 기반 예측 완료: '{text[:30]}...' → {emotion} ({confidence:.3f})")

            return result

        except Exception as e:
            logger.error(f"❌ 룰 기반 예측 실패: {e}")
            raise e

    def predict_emotion(self, text: str, include_probabilities: bool = False) -> Dict[str, Any]:
        """감정 예측 메인 함수 (AI 모델 우선, 실패 시 룰 기반)"""

        # 1. AI 모델 시도
        if self.model_loaded and self.vectorizer_loaded:
            try:
                return self.predict_with_ai_model(text, include_probabilities)
            except Exception as e:
                logger.warning(f"⚠️ AI 모델 실패, 룰 기반으로 폴백: {e}")

        # 2. 룰 기반 폴백
        try:
            return self.predict_with_rule_based(text, include_probabilities)
        except Exception as e:
            logger.error(f"❌ 모든 예측 방법 실패: {e}")

            # 3. 최종 안전장치
            return {
                "emotion": "기쁨",
                "confidence": 0.0,
                "label": 0,
                "method": "기본값 (에러 발생)",
                "processing_time": 0.0,
                "error": str(e)
            }

    def get_system_status(self) -> Dict[str, Any]:
        """시스템 상태 반환"""
        return {
            "ai_model_available": self.model_loaded,
            "vectorizer_available": self.vectorizer_loaded,
            "fallback_available": True,  # 룰 기반은 항상 사용 가능
            "supported_emotions": list(self.emotion_labels.values()),
            "model_info": {
                "tensorflow_available": TENSORFLOW_AVAILABLE,
                "sklearn_available": SKLEARN_AVAILABLE,
                "model_path": self.model_path,
                "vectorizer_path": self.vectorizer_path,
                "model_file_exists": os.path.exists(self.model_path),
                "vectorizer_file_exists": os.path.exists(self.vectorizer_path)
            }
        }


# ─── 3. 전역 인스턴스 생성 ─────────────────────────────────────
emotion_tagger = None


def get_emotion_tagger():
    """감정 태깅 인스턴스 반환 (의존성 주입용)"""
    global emotion_tagger
    if emotion_tagger is None:
        emotion_tagger = WhiffEmotionTagger()
    return emotion_tagger


# ─── 4. 라우터 설정 ─────────────────────────────────────────
router = APIRouter(prefix="/emotion", tags=["Emotion Tagging"])

# 시작 시 모델 초기화
logger.info("🚀 감정 태깅 라우터 초기화 시작...")
try:
    emotion_tagger = WhiffEmotionTagger()
    logger.info("✅ 감정 태깅 라우터 초기화 완료")
except Exception as e:
    logger.error(f"❌ 감정 태깅 라우터 초기화 실패: {e}")


# ─── 5. API 엔드포인트 ─────────────────────────────────────────

@router.post(
    "/predict",
    response_model=EmotionPredictResponse,
    summary="시향 일기 감정 예측",
    description=(
            "🎭 **시향 일기 텍스트의 감정을 예측합니다**\n\n"
            "**🤖 예측 방식:**\n"
            "1. **AI 모델 우선**: Keras + TF-IDF 기반 8가지 감정 분류\n"
            "2. **룰 기반 폴백**: AI 모델 실패 시 기존 룰 기반 분석 사용\n"
            "3. **안전장치**: 모든 방법 실패 시 기본값 반환\n\n"
            "**📊 지원 감정 (8가지):**\n"
            "- 기쁨 (0), 불안 (1), 당황 (2), 분노 (3)\n"
            "- 상처 (4), 슬픔 (5), 우울 (6), 흥분 (7)\n\n"
            "**💡 사용 예시:**\n"
            "```json\n"
            "{\n"
            "  \"text\": \"향기를 맡으니 내 안에 따뜻함이 번졌다\",\n"
            "  \"include_probabilities\": true\n"
            "}\n"
            "```"
    )
)
async def predict_emotion(
        request: EmotionPredictRequest,
        tagger: WhiffEmotionTagger = Depends(get_emotion_tagger)
):
    """감정 예측 API"""

    try:
        logger.info(f"🎭 감정 예측 요청: '{request.text[:50]}...'")

        # 감정 예측 수행
        result = tagger.predict_emotion(
            text=request.text,
            include_probabilities=request.include_probabilities
        )

        # 응답 형태로 변환
        response = EmotionPredictResponse(
            emotion=result["emotion"],
            confidence=result["confidence"],
            label=result["label"],
            method=result["method"],
            processing_time=result["processing_time"],
            probabilities=result.get("probabilities")
        )

        logger.info(f"✅ 감정 예측 완료: {response.emotion} ({response.confidence:.3f})")

        return response

    except Exception as e:
        logger.error(f"❌ 감정 예측 API 오류: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"감정 예측 중 오류가 발생했습니다: {str(e)}"
        )


@router.get(
    "/health",
    response_model=EmotionHealthResponse,
    summary="감정 태깅 시스템 상태 확인",
    description=(
            "🔍 **감정 태깅 시스템의 상태를 확인합니다**\n\n"
            "**📊 확인 항목:**\n"
            "- AI 모델 로딩 상태\n"
            "- 벡터라이저 로딩 상태\n"
            "- 룰 기반 폴백 사용 가능 여부\n"
            "- 지원하는 감정 목록\n"
            "- 시스템 환경 정보\n\n"
            "**💡 개발 단계에서 활용:**\n"
            "- 모델 파일 존재 여부 확인\n"
            "- 의존성 설치 상태 점검\n"
            "- 배포 후 정상 동작 검증"
    )
)
async def check_health(tagger: WhiffEmotionTagger = Depends(get_emotion_tagger)):
    """시스템 상태 확인 API"""

    try:
        status = tagger.get_system_status()

        response = EmotionHealthResponse(
            ai_model_available=status["ai_model_available"],
            vectorizer_available=status["vectorizer_available"],
            fallback_available=status["fallback_available"],
            supported_emotions=status["supported_emotions"],
            model_info=status["model_info"]
        )

        return response

    except Exception as e:
        logger.error(f"❌ 상태 확인 오류: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"상태 확인 중 오류가 발생했습니다: {str(e)}"
        )


@router.post(
    "/test",
    summary="감정 태깅 테스트 (개발용)",
    description="개발/디버깅용 감정 태깅 테스트 API"
)
async def test_emotion_prediction(
        texts: List[str],
        tagger: WhiffEmotionTagger = Depends(get_emotion_tagger)
):
    """개발용 일괄 테스트 API"""

    try:
        results = []

        for text in texts[:10]:  # 최대 10개까지
            result = tagger.predict_emotion(text, include_probabilities=True)
            results.append({
                "input": text,
                "output": result
            })

        return {
            "test_results": results,
            "system_status": tagger.get_system_status(),
            "total_tests": len(results)
        }

    except Exception as e:
        logger.error(f"❌ 테스트 API 오류: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"테스트 중 오류가 발생했습니다: {str(e)}"
        )