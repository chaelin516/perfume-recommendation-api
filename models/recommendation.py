from sqlmodel import SQLModel, Field
from typing import Optional
from datetime import datetime

class RecommendedPerfume(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: str
    recommend_round: int  # 1차 추천이면 1, 2차 추천이면 2
    perfume_name: str
    perfume_brand: str
    score: Optional[int] = None  # 사용자가 평가한 점수 (없으면 NULL)
    created_at: datetime = Field(default_factory=datetime.utcnow)
