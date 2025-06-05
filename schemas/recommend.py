from pydantic import BaseModel, Field
from typing import Literal, List, Optional


# ✅ encoder.pkl과 호환되는 스키마 정의
class RecommendRequest(BaseModel):
    """
    향수 추천 요청 스키마

    encoder.pkl의 카테고리와 완전 호환되도록 정의됨
    """

    gender: Literal['men', 'unisex', 'women'] = Field(
        ...,
        description="성별 선택",
        example="women"
    )

    season_tags: Literal['fall', 'spring', 'summer', 'winter'] = Field(
        ...,
        description="계절 선택",
        example="spring"
    )

    time_tags: Literal['day', 'night'] = Field(
        ...,
        description="시간대 선택",
        example="day"
    )

    desired_impression: Literal[
        'confident, fresh',
        'confident, mysterious',
        'elegant, friendly',
        'pure, friendly'
    ] = Field(
        ...,
        description="원하는 인상/감정 조합",
        example="confident, fresh"
    )

    activity: Literal['casual', 'date', 'work'] = Field(
        ...,
        description="활동 유형",
        example="casual"
    )

    weather: Literal['any', 'cold', 'hot', 'rainy'] = Field(
        ...,
        description="날씨 조건",
        example="hot"
    )

    class Config:
        schema_extra = {
            "example": {
                "gender": "women",
                "season_tags": "spring",
                "time_tags": "day",
                "desired_impression": "confident, fresh",
                "activity": "casual",
                "weather": "hot"
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
                        "emotions": "confident, fresh",
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
                    "gender": ["men", "unisex", "women"],
                    "season_tags": ["fall", "spring", "summer", "winter"],
                    "time_tags": ["day", "night"],
                    "desired_impression": ["confident, fresh", "confident, mysterious", "elegant, friendly",
                                           "pure, friendly"],
                    "activity": ["casual", "date", "work"],
                    "weather": ["any", "cold", "hot", "rainy"]
                },
                "sklearn_version": "1.5.2"
            }
        }


# ✅ encoder.pkl과 완전 호환되는 카테고리 정의
SUPPORTED_CATEGORIES = {
    "gender": ["men", "unisex", "women"],
    "season_tags": ["fall", "spring", "summer", "winter"],
    "time_tags": ["day", "night"],
    "desired_impression": [
        "confident, fresh",
        "confident, mysterious",
        "elegant, friendly",
        "pure, friendly"
    ],
    "activity": ["casual", "date", "work"],
    "weather": ["any", "cold", "hot", "rainy"]
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


# ✅ 단일 감정 값을 조합 감정 값으로 매핑하는 헬퍼 함수 (하위 호환성용)
def map_single_to_combined_impression(single_impression: str) -> str:
    """
    단일 감정 값을 encoder.pkl 호환 조합 값으로 매핑
    하위 호환성 지원용 함수
    """
    mapping = {
        "confident": "confident, fresh",
        "fresh": "confident, fresh",
        "mysterious": "confident, mysterious",
        "elegant": "elegant, friendly",
        "friendly": "elegant, friendly",
        "pure": "pure, friendly"
    }

    return mapping.get(single_impression, "confident, fresh")  # 기본값


def get_available_impressions():
    """사용 가능한 모든 인상 조합 반환"""
    return SUPPORTED_CATEGORIES["desired_impression"]