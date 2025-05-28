# perfume_backend/schemas/diary.py

from pydantic import BaseModel, Field
from datetime import datetime
from typing import List

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
