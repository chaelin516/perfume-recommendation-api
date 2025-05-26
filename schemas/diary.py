from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List

class DiaryCreateRequest(BaseModel):
    user_id: str
    perfume_name: str
    content: Optional[str] = None  # ← 이 줄 반드시 있어야 함
    is_public: bool
    emotion_tags: Optional[List[str]] = []

class DiaryEntry(BaseModel):
    user_id: str
    perfume_name: str
    content: Optional[str] = None
    is_public: bool
    emotion_tags: Optional[List[str]] = []
    created_at: datetime
