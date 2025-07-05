# schemas/recommendation.py
from pydantic import BaseModel, Field
from typing import List
from datetime import datetime


class RecommendationItem(BaseModel):
    """개별 추천 향수 아이템"""
    perfume_name: str = Field(..., description="향수 이름")
    perfume_brand: str = Field(..., description="향수 브랜드")
    score: float = Field(..., description="추천 점수", ge=0.0, le=100.0)


class SaveRecommendationsRequest(BaseModel):
    """추천 결과 저장 요청"""
    user_id: str = Field(..., description="사용자 UID")
    recommend_round: int = Field(..., description="추천 라운드 (1 또는 2)", ge=1, le=2)
    recommendations: List[RecommendationItem] = Field(..., description="추천 향수 목록")

    class Config:
        schema_extra = {
            "example": {
                "user_id": "user123",
                "recommend_round": 1,
                "recommendations": [
                    {
                        "perfume_name": "블랙 오피움",
                        "perfume_brand": "YSL",
                        "score": 95.5
                    },
                    {
                        "perfume_name": "샤넬 No.5",
                        "perfume_brand": "Chanel",
                        "score": 89.2
                    }
                ]
            }
        }


class SaveRecommendationsResponse(BaseModel):
    """추천 결과 저장 응답"""
    message: str = Field(..., description="응답 메시지")
    saved_count: int = Field(..., description="저장된 항목 수")
    user_id: str = Field(..., description="사용자 UID")
    recommend_round: int = Field(..., description="추천 라운드")
    created_at: datetime = Field(..., description="생성 시간")


class RecommendationHistoryResponse(BaseModel):
    """추천 히스토리 응답"""
    id: int = Field(..., description="히스토리 ID")
    user_id: str = Field(..., description="사용자 UID")
    perfume_name: str = Field(..., description="향수 이름")
    perfume_brand: str = Field(..., description="향수 브랜드")
    score: float = Field(..., description="추천 점수")
    recommend_round: int = Field(..., description="추천 라운드")
    created_at: datetime = Field(..., description="생성 시간")


class GetRecommendationsResponse(BaseModel):
    """사용자 추천 조회 응답"""
    message: str = Field(..., description="응답 메시지")
    total_count: int = Field(..., description="전체 추천 수")
    recommendations: List[RecommendationHistoryResponse] = Field(..., description="추천 목록")
    user_id: str = Field(..., description="사용자 UID")