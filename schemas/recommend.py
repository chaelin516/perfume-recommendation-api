from pydantic import BaseModel, Field
from typing import Literal, List, Optional


# ✅ API와 모델 호환성을 보장하는 스키마 정의
class RecommendRequest(BaseModel):
    """
    향수 추천 요청 스키마

    모든 필드는 AI 모델의 encoder와 호환되도록 정의됨
    """

    gender: Literal['women', 'men', 'unisex'] = Field(
        ...,
        description="성별 선택",
        example="women"
    )

    season_tags: Literal['spring', 'summer', 'fall', 'winter'] = Field(
        ...,
        description="계절 선택",
        example="spring"
    )

    time_tags: Literal['day', 'night'] = Field(
        ...,
        description="시간대 선택",
        example="day"
    )

    desired_impression: Literal['confident', 'elegant', 'pure', 'friendly', 'mysterious', 'fresh'] = Field(
        ...,
        description="원하는 인상/감정",
        example="fresh"
    )

    activity: Literal['casual', 'work', 'date'] = Field(
        ...,
        description="활동 유형",
        example="casual"
    )

    weather: Literal['hot', 'cold', 'rainy', 'any'] = Field(
        ...,
        description="날씨 조건",
        example="any"
    )

    class Config:
        schema_extra = {
            "example": {
                "gender": "women",
                "season_tags": "spring",
                "time_tags": "day",
                "desired_impression": "fresh",
                "activity": "casual",
                "weather": "any"
            }
        }


# ✅ 추천 응답 향수 항목 (확장된 정보 포함)
class RecommendedPerfume(BaseModel):
    """단일 추천 향수 정보"""

    id: Optional[int] = Field(None, description="향수 고유 ID")
    name: str = Field(..., description="향수 이름")
    brand: str = Field(..., description="브랜드명")
    image_url: str = Field(..., description="이미지 URL")
    notes: Optional[str] = Field(None, description="향수 노트")
    emotions: Optional[str] = Field(None, description="감정/인상 정보")
    reason: Optional[str] = Field(None, description="추천 이유")
    score: Optional[float] = Field(None, description="추천 점수 (0.0-1.0)", ge=0.0, le=1.0)
    method: Optional[str] = Field(None, description="추천 방법 (AI/룰기반)")
    emotion_cluster: Optional[int] = Field(None, description="감정 클러스터 ID (0-5)")


# ✅ 추천 결과 응답 구조 (확장된 메타데이터 포함)
class RecommendResponse(BaseModel):
    """전체 추천 결과 응답"""

    success: bool = Field(True, description="추천 성공 여부")
    method_used: str = Field(..., description="사용된 추천 방법")
    processing_time_seconds: Optional[float] = Field(None, description="처리 시간 (초)")
    total_results: int = Field(..., description="총 추천 결과 수")

    recommended_perfumes: List[RecommendedPerfume] = Field(
        ...,
        description="추천된 향수 목록",
        min_items=0,
        max_items=20
    )

    # 메타데이터
    metadata: Optional[dict] = Field(
        None,
        description="추가 메타데이터 (디버깅/분석용)"
    )

    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "method_used": "AI 감정 클러스터 모델",
                "processing_time_seconds": 0.245,
                "total_results": 10,
                "recommended_perfumes": [
                    {
                        "id": 1,
                        "name": "Fresh Blossom",
                        "brand": "Spring Garden",
                        "image_url": "https://example.com/perfume1.jpg",
                        "notes": "bergamot, jasmine, white musk",
                        "emotions": "fresh, confident",
                        "reason": "🤖 AI가 95.2% 확률로 당신의 완벽한 향수라고 분석했습니다!",
                        "score": 0.952,
                        "method": "AI 감정 클러스터 모델",
                        "emotion_cluster": 1
                    }
                ],
                "metadata": {
                    "cluster_used": 1,
                    "cluster_confidence": 0.85,
                    "fallback_used": False
                }
            }
        }


# ✅ 에러 응답 스키마
class RecommendErrorResponse(BaseModel):
    """추천 실패 시 에러 응답"""

    success: bool = Field(False, description="추천 성공 여부")
    error_type: str = Field(..., description="에러 유형")
    error_message: str = Field(..., description="에러 메시지")
    fallback_available: bool = Field(..., description="대체 방법 사용 가능 여부")

    class Config:
        schema_extra = {
            "example": {
                "success": False,
                "error_type": "model_load_failed",
                "error_message": "AI 모델 로딩에 실패했습니다. 룰 기반 추천을 사용합니다.",
                "fallback_available": True
            }
        }


# ✅ 모델 상태 체크용 스키마
class ModelStatusResponse(BaseModel):
    """모델 상태 확인 응답"""

    model_available: bool = Field(..., description="AI 모델 사용 가능 여부")
    model_size_kb: float = Field(..., description="모델 파일 크기 (KB)")
    encoder_available: bool = Field(..., description="인코더 사용 가능 여부")
    fallback_encoder_ready: bool = Field(..., description="Fallback 인코더 준비 상태")
    supported_categories: dict = Field(..., description="지원되는 카테고리 목록")
    sklearn_version: str = Field(..., description="scikit-learn 버전")

    class Config:
        schema_extra = {
            "example": {
                "model_available": True,
                "model_size_kb": 31.2,
                "encoder_available": True,
                "fallback_encoder_ready": True,
                "supported_categories": {
                    "gender": ["women", "men", "unisex"],
                    "season_tags": ["spring", "summer", "fall", "winter"],
                    "time_tags": ["day", "night"],
                    "desired_impression": ["confident", "elegant", "pure", "friendly", "mysterious", "fresh"],
                    "activity": ["casual", "work", "date"],
                    "weather": ["hot", "cold", "rainy", "any"]
                },
                "sklearn_version": "1.5.2"
            }
        }


# ✅ 지원되는 모든 카테고리 정의 (검증용)
SUPPORTED_CATEGORIES = {
    "gender": ["women", "men", "unisex"],
    "season_tags": ["spring", "summer", "fall", "winter"],
    "time_tags": ["day", "night"],
    "desired_impression": ["confident", "elegant", "pure", "friendly", "mysterious", "fresh"],
    "activity": ["casual", "work", "date"],
    "weather": ["hot", "cold", "rainy", "any"]
}

# ✅ 감정 클러스터 매핑 (참조용)
EMOTION_CLUSTER_DESCRIPTIONS = {
    0: "차분한, 편안한",
    1: "자신감, 신선함",
    2: "우아함, 친근함",
    3: "순수함, 친근함",
    4: "신비로운, 매력적",
    5: "활기찬, 에너지"
}


# ✅ 유틸리티 함수들
def validate_request_categories(request: RecommendRequest) -> bool:
    """요청 데이터가 지원되는 카테고리에 포함되는지 검증"""

    checks = [
        request.gender in SUPPORTED_CATEGORIES["gender"],
        request.season_tags in SUPPORTED_CATEGORIES["season_tags"],
        request.time_tags in SUPPORTED_CATEGORIES["time_tags"],
        request.desired_impression in SUPPORTED_CATEGORIES["desired_impression"],
        request.activity in SUPPORTED_CATEGORIES["activity"],
        request.weather in SUPPORTED_CATEGORIES["weather"]
    ]

    return all(checks)


def get_category_mapping():
    """카테고리 매핑 정보 반환 (API 문서화용)"""
    return SUPPORTED_CATEGORIES


def get_emotion_cluster_info():
    """감정 클러스터 정보 반환"""
    return EMOTION_CLUSTER_DESCRIPTIONS