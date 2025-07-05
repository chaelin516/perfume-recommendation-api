# models/recommendation.py
from sqlmodel import SQLModel, Field
from datetime import datetime
from typing import Optional


class RecommendedPerfume(SQLModel, table=True):
    """추천 향수 저장 모델"""
    __tablename__ = "recommended_perfumes"

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: str = Field(index=True, description="사용자 UID")
    recommend_round: int = Field(description="추천 라운드 (1차, 2차)")
    perfume_name: str = Field(description="향수 이름")
    perfume_brand: str = Field(description="향수 브랜드")
    score: float = Field(description="추천 점수")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="생성 시간")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class RecommendationHistory(SQLModel, table=True):
    """추천 히스토리 모델"""
    __tablename__ = "recommendation_history"

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: str = Field(index=True, description="사용자 UID")
    request_data: str = Field(description="추천 요청 데이터 (JSON)")
    response_data: str = Field(description="추천 응답 데이터 (JSON)")
    recommendation_type: str = Field(description="추천 타입 (cluster, 2nd, etc)")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="생성 시간")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }