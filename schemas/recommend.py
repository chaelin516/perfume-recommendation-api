from pydantic import BaseModel
from typing import Literal, List

class RecommendRequest(BaseModel):
    gender: Literal['women', 'men', 'unisex']
    season: Literal['spring', 'summer', 'fall', 'winter']
    time: Literal['day', 'night']
    impression: Literal['confident', 'elegant', 'pure', 'friendly', 'mysterious', 'fresh']
    activity: Literal['casual', 'work', 'date']
    weather: Literal['hot', 'cold', 'rainy', 'any']


# 추천 응답 향수 항목
class RecommendedPerfume(BaseModel):
    id: int
    name: str
    brand: str
    image_url: str

# 추천 결과 응답 구조
class RecommendResponse(BaseModel):
    recommended_perfumes: List[RecommendedPerfume]
