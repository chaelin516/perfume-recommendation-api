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

# 추가로 routers/recommend_2nd_router.py에서 사용되는 스키마들도 수정

# routers/recommend_2nd_router.py 내부의 스키마 수정
class SecondRecommendRequest(BaseModel):
    """2차 추천 요청 스키마 - AI 모델 호출 포함"""

    user_preferences: UserPreferences = Field(
        ...,
        description="1차 추천을 위한 사용자 선호도 (AI 모델 입력)"
    )

    user_note_scores: Dict[str, int] = Field(
        ...,
        description="사용자의 노트별 선호도 점수 (0-5)",
        examples={
            "jasmine": 5,
            "rose": 4,
            "amber": 3,
            "musk": 0,
            "citrus": 2,
            "vanilla": 1
        }
    )

    emotion_proba: Optional[List[float]] = Field(
        None,
        description="6개 감정 클러스터별 확률 배열 (제공되지 않으면 AI 모델로 계산)",
        min_length=6,
        max_length=6,
        examples=[0.01, 0.03, 0.85, 0.02, 0.05, 0.04]
    )

    selected_idx: Optional[List[int]] = Field(
        None,
        description="1차 추천에서 선택된 향수 인덱스 목록 (제공되지 않으면 AI 모델로 계산)",
        min_length=1,
        max_length=20,
        examples=[23, 45, 102, 200, 233, 305, 399, 410, 487, 512]
    )

    @validator('user_note_scores')
    def validate_note_scores(cls, v):
        for note, score in v.items():
            if not isinstance(score, int) or score < 0 or score > 5:
                raise ValueError(f"노트 '{note}'의 점수는 0-5 사이의 정수여야 합니다.")
        return v

    # schema_extra 제거하고 examples 사용 (Pydantic v2 권장 방식)

    @validator('emotion_proba')
    def validate_emotion_proba(cls, v):
        if v is None:
            return v

        if len(v) != 6:
            raise ValueError("emotion_proba는 정확히 6개의 확률값을 가져야 합니다.")

        total = sum(v)
        if not (0.95 <= total <= 1.05):
            raise ValueError(f"emotion_proba의 합은 1.0에 가까워야 합니다. 현재: {total}")

        for prob in v:
            if not (0.0 <= prob <= 1.0):
                raise ValueError("각 확률값은 0.0-1.0 사이여야 합니다.")

        return v

    @validator('selected_idx')
    def validate_selected_idx(cls, v):
        if v is None:
            return v

        if len(set(v)) != len(v):
            raise ValueError("selected_idx에 중복된 인덱스가 있습니다.")

        for idx in v:
            if idx < 0:
                raise ValueError("인덱스는 0 이상이어야 합니다.")

        return v