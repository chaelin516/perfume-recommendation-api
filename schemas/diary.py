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
        orm_mode = True
        allow_population_by_field_name = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
