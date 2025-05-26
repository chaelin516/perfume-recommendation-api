from pydantic import BaseModel
from typing import List, Optional

class PerfumeRecommendationItem(BaseModel):
    perfume_name: str
    perfume_brand: str
    score: Optional[int] = None

class SaveRecommendationsRequest(BaseModel):
    user_id: str
    recommend_round: int
    recommendations: List[PerfumeRecommendationItem]
