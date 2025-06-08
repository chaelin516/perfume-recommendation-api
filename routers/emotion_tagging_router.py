# routers/emotion_tagging_router.py
# 🎯 감정 태깅 AI 모델 API 라우터 (scent_emotion_model_v6.keras 연동)

import os
import pickle
import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import pandas as pd

# ─── 로거 설정 ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("emotion_tagging_router")

# ─── 경로 설정 (올바른 절대경로 사용) ─────────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
EMOTION_MODEL_PATH = os.path.join(BASE_DIR, "../emotion_models/scent_emotion_model_v6.keras")
VECTORIZER_PATH = os.path.join(BASE_DIR, "../emotion_models/vectorizer.pkl")

# ─── 글로벌 변수 ─────────────────────────────────────────────────────────────────
_emotion_model = None
_vectorizer = None
_model_loaded = False


# ─── 스키마 정의 ─────────────────────────────────────────────────────────────────
class EmotionTagRequest(BaseModel):
    text: str = Field(..., description="감정 태깅할 텍스트", example="이 향수는 정말 좋아요! 기분이 상쾌해집니다.")


class EmotionTagResponse(BaseModel):
    text: str = Field(..., description="입력 텍스트")
    emotion: str = Field(..., description="예측된 감정")
    confidence: float = Field(..., description="신뢰도 (0.0-1.0)")
    all_emotions: Dict[str, float] = Field(..., description="모든 감정별 확률")
    processing_time: float = Field(..., description="처리 시간 (초)")


# ─── 감정 클래스 매핑 ─────────────────────────────────────────────────────────────
EMOTION_LABELS = {
    0: "기쁨",
    1: "불안",
    2: "당황",
    3: "분노",
    4: "상처",
    5: "슬픔",
    6: "우울",
    7: "흥분"
}


class EmotionTagger:
    """감정 태깅 AI 모델 클래스"""

    def __init__(self):
        self.model_path = EMOTION_MODEL_PATH
        self.vectorizer_path = VECTORIZER_PATH
        self.model = None
        self.vectorizer = None
        self.model_loaded = False

        # 파일 존재 여부 확인
        self.check_files()

    def check_files(self):
        """모델 파일 존재 여부 확인"""
        model_exists = os.path.exists(self.model_path)
        vectorizer_exists = os.path.exists(self.vectorizer_path)

        logger.info(f"🔍 감정 모델 파일 확인:")
        logger.info(f"  - 모델 파일: {self.model_path} {'✅' if model_exists else '❌'}")
        logger.info(f"  - 벡터라이저: {self.vectorizer_path} {'✅' if vectorizer_exists else '❌'}")

        if model_exists:
            model_size = os.path.getsize(self.model_path)
            logger.info(f"  - 모델 크기: {model_size:,} bytes ({model_size / 1024 / 1024:.1f}MB)")

        return model_exists and vectorizer_exists

    def load_model(self):
        """모델과 벡터라이저 로딩"""
        if self.model_loaded:
            return True

        try:
            logger.info("🤖 감정 태깅 모델 로딩 시작...")

            # 1. TensorFlow 모델 로딩
            try:
                from tensorflow.keras.models import load_model
                self.model = load_model(self.model_path, compile=False)
                logger.info(f"✅ 감정 모델 로드 성공: {self.model.input_shape} → {self.model.output_shape}")
            except Exception as e:
                logger.error(f"❌ TensorFlow 모델 로딩 실패: {e}")
                return False

            # 2. 벡터라이저 로딩
            try:
                with open(self.vectorizer_path, "rb") as f:
                    self.vectorizer = pickle.load(f)
                logger.info(f"✅ 벡터라이저 로드 성공")
            except Exception as e:
                logger.error(f"❌ 벡터라이저 로딩 실패: {e}")
                return False

            self.model_loaded = True
            logger.info("🎉 감정 태깅 모델 로딩 완료!")
            return True

        except Exception as e:
            logger.error(f"❌ 감정 모델 로딩 중 오류: {e}")
            return False

    def predict_emotion(self, text: str) -> Dict:
        """텍스트의 감정 예측"""
        start_time = datetime.now()

        # 모델 로딩 확인
        if not self.model_loaded:
            if not self.load_model():
                raise Exception("모델 로딩에 실패했습니다.")

        try:
            # 1. 텍스트 전처리 및 벡터화
            text_vector = self.vectorizer.transform([text])

            # 2. 모델 예측
            predictions = self.model.predict(text_vector, verbose=0)
            emotion_probs = predictions[0]

            # 3. 결과 처리
            predicted_emotion_idx = int(np.argmax(emotion_probs))
            predicted_emotion = EMOTION_LABELS.get(predicted_emotion_idx, "알수없음")
            confidence = float(emotion_probs[predicted_emotion_idx])

            # 4. 모든 감정 확률
            all_emotions = {}
            for idx, prob in enumerate(emotion_probs):
                emotion_name = EMOTION_LABELS.get(idx, f"감정{idx}")
                all_emotions[emotion_name] = round(float(prob), 3)

            # 5. 처리 시간 계산
            processing_time = (datetime.now() - start_time).total_seconds()

            result = {
                "text": text,
                "emotion": predicted_emotion,
                "confidence": round(confidence, 3),
                "all_emotions": all_emotions,
                "processing_time": round(processing_time, 3)
            }

            logger.info(f"🎯 감정 예측 완료: '{text[:30]}...' → {predicted_emotion} ({confidence:.3f})")

            return result

        except Exception as e:
            logger.error(f"❌ 감정 예측 중 오류: {e}")
            raise e


# ─── 글로벌 감정 태거 인스턴스 ─────────────────────────────────────────────────────
emotion_tagger = EmotionTagger()

# ─── 라우터 설정 ─────────────────────────────────────────────────────────────────
router = APIRouter(prefix="/emotions", tags=["Emotion Tagging"])


@router.post(
    "/predict",
    response_model=EmotionTagResponse,
    summary="텍스트 감정 태깅",
    description=(
            "🎭 **AI 기반 텍스트 감정 분석**\n\n"
            "시향 일기나 리뷰 텍스트의 감정을 8가지 카테고리로 분류합니다.\n\n"
            "**📥 입력:**\n"
            "- 감정 분석할 텍스트 (한국어 권장)\n\n"
            "**📤 출력:**\n"
            "- 예측된 주요 감정 및 신뢰도\n"
            "- 8가지 감정별 상세 확률\n\n"
            "**🎯 지원 감정:**\n"
            "기쁨, 불안, 당황, 분노, 상처, 슬픔, 우울, 흥분"
    )
)
async def predict_emotion(request: EmotionTagRequest):
    """텍스트 감정 예측 API"""

    try:
        # 입력 검증
        if not request.text or len(request.text.strip()) == 0:
            raise HTTPException(status_code=400, detail="텍스트가 비어있습니다.")

        if len(request.text) > 1000:
            raise HTTPException(status_code=400, detail="텍스트가 너무 깁니다 (최대 1000자).")

        # 감정 예측
        result = emotion_tagger.predict_emotion(request.text.strip())

        return EmotionTagResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 감정 예측 API 오류: {e}")
        raise HTTPException(status_code=500, detail=f"감정 분석 중 오류가 발생했습니다: {str(e)}")


@router.get(
    "/status",
    summary="감정 태깅 시스템 상태",
    description="감정 태깅 AI 모델의 상태를 확인합니다."
)
def get_emotion_system_status():
    """감정 태깅 시스템 상태 확인"""

    try:
        # 파일 존재 여부
        model_exists = os.path.exists(EMOTION_MODEL_PATH)
        vectorizer_exists = os.path.exists(VECTORIZER_PATH)

        # 파일 크기
        model_size = os.path.getsize(EMOTION_MODEL_PATH) if model_exists else 0
        vectorizer_size = os.path.getsize(VECTORIZER_PATH) if vectorizer_exists else 0

        return {
            "system_status": "operational" if model_exists and vectorizer_exists else "files_missing",
            "model_info": {
                "model_file": EMOTION_MODEL_PATH,
                "model_exists": model_exists,
                "model_size_mb": round(model_size / 1024 / 1024, 2) if model_exists else 0,
                "vectorizer_file": VECTORIZER_PATH,
                "vectorizer_exists": vectorizer_exists,
                "vectorizer_size_kb": round(vectorizer_size / 1024, 2) if vectorizer_exists else 0
            },
            "model_loaded": emotion_tagger.model_loaded,
            "supported_emotions": list(EMOTION_LABELS.values()),
            "max_text_length": 1000,
            "last_checked": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"❌ 시스템 상태 확인 오류: {e}")
        return {
            "system_status": "error",
            "error": str(e),
            "last_checked": datetime.now().isoformat()
        }


@router.get(
    "/emotions",
    summary="지원 감정 목록",
    description="시스템에서 지원하는 모든 감정 카테고리를 반환합니다."
)
def get_supported_emotions():
    """지원하는 감정 목록 반환"""

    return {
        "emotions": EMOTION_LABELS,
        "total_count": len(EMOTION_LABELS),
        "categories": [
            {"id": k, "name": v, "description": f"{v} 관련 감정"}
            for k, v in EMOTION_LABELS.items()
        ]
    }