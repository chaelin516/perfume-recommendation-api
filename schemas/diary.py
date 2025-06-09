# schemas/diary.py - Pydantic 경고 수정

from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Optional

# ✅ 클라이언트가 시향 일기를 작성할 때 사용하는 요청 모델
class DiaryCreateRequest(BaseModel):
    user_id: str
    perfume_name: str
    content: Optional[str] = None
    is_public: bool
    emotion_tags: Optional[List[str]] = []

    class Config:
        schema_extra = {
            "example": {
                "user_id": "john_doe",
                "perfume_name": "Chanel No.5",
                "content": "오늘은 봄바람이 느껴지는 향수와 산책했어요.",
                "is_public": False,
                "emotion_tags": ["calm", "spring"]
            }
        }


# ✅ 프론트엔드에 시향 일기 결과를 응답할 때 사용하는 모델
class DiaryResponse(BaseModel):
    id: str
    user_id: str = Field(..., alias="user_id")
    user_name: str = Field(..., alias="user_name")
    user_profile_image: str = Field(..., alias="user_profile_image")
    perfume_id: str = Field(..., alias="perfume_id")
    perfume_name: str = Field(..., alias="perfume_name")
    brand: str
    content: str
    tags: List[str] = Field(..., alias="tags")
    likes: int
    comments: int
    created_at: datetime = Field(..., alias="created_at")
    updated_at: datetime = Field(..., alias="updated_at")

    class Config:
        from_attributes = True  # orm_mode → from_attributes (Pydantic v2)
        populate_by_name = True  # allow_population_by_field_name → populate_by_name (Pydantic v2)
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

# schemas/course.py - Pydantic 경고 수정

from pydantic import BaseModel
from typing import Literal

class SimpleCourseRecommendRequest(BaseModel):
    gender: Literal["male", "female", "unisex"]
    emotion: str
    season: str
    time: str
    latitude: float
    longitude: float

class CourseItem(BaseModel):
    store: str
    address: str
    perfume_name: str
    brand: str
    image_url: str
    distance_km: float

# schemas/recommendation.py - Pydantic 경고 수정

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

