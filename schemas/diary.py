# schemas/diary.py - 이미지 업로드 기능 추가된 버전

from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Optional


# ✅ 클라이언트가 시향 일기를 작성할 때 사용하는 요청 모델 (이미지 업로드 포함)
class DiaryCreateRequest(BaseModel):
    user_id: str
    perfume_name: str
    content: Optional[str] = None
    is_public: bool
    emotion_tags: Optional[List[str]] = []

    # 🆕 이미지 관련 필드들은 별도 API에서 처리 (multipart/form-data)

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


# 🆕 이미지 업로드 전용 요청 모델
class DiaryImageUploadRequest(BaseModel):
    user_id: str = Field(..., description="사용자 ID")
    diary_id: Optional[str] = Field(None, description="일기 ID (기존 일기에 이미지 추가시)")

    class Config:
        schema_extra = {
            "example": {
                "user_id": "john_doe",
                "diary_id": "diary_123456"
            }
        }


# ✅ 프론트엔드에 시향 일기 결과를 응답할 때 사용하는 모델 (이미지 필드 추가)
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

    # 🆕 이미지 관련 필드들
    image_url: Optional[str] = Field(None, description="원본 이미지 URL")
    thumbnail_url: Optional[str] = Field(None, description="썸네일 이미지 URL")
    image_filename: Optional[str] = Field(None, description="이미지 파일명")

    # 🆕 감정 분석 관련 필드들
    primary_emotion: Optional[str] = Field(None, description="주요 감정")
    emotion_confidence: Optional[float] = Field(None, description="감정 분석 신뢰도")

    class Config:
        from_attributes = True  # orm_mode → from_attributes (Pydantic v2)
        populate_by_name = True  # allow_population_by_field_name → populate_by_name (Pydantic v2)
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# 🆕 이미지 업로드 응답 모델
class ImageUploadResponse(BaseModel):
    success: bool = Field(..., description="업로드 성공 여부")
    message: str = Field(..., description="응답 메시지")
    image_url: Optional[str] = Field(None, description="업로드된 이미지 URL")
    thumbnail_url: Optional[str] = Field(None, description="썸네일 이미지 URL")
    filename: Optional[str] = Field(None, description="저장된 파일명")
    file_size: Optional[int] = Field(None, description="파일 크기 (bytes)")
    image_metadata: Optional[dict] = Field(None, description="이미지 메타데이터")

    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "이미지 업로드 성공",
                "image_url": "https://whiff-api-9nd8.onrender.com/uploads/diary_images/user123_abc123.jpg",
                "thumbnail_url": "https://whiff-api-9nd8.onrender.com/uploads/diary_images/thumbnails/thumb_user123_abc123.jpg",
                "filename": "user123_abc123.jpg",
                "file_size": 1048576,
                "image_metadata": {
                    "original_size": [1920, 1080],
                    "processed_size": [1920, 1080],
                    "thumbnail_size": [400, 225]
                }
            }
        }


# 🆕 시향 일기 + 이미지 통합 작성 요청 모델
class DiaryWithImageCreateRequest(BaseModel):
    user_id: str = Field(..., description="사용자 ID")
    perfume_name: str = Field(..., description="향수명")
    content: Optional[str] = Field(None, description="일기 내용")
    is_public: bool = Field(..., description="공개 여부")
    emotion_tags: Optional[List[str]] = Field([], description="감정 태그")

    # 이미지는 별도의 multipart field로 처리

    class Config:
        schema_extra = {
            "example": {
                "user_id": "john_doe",
                "perfume_name": "Chanel No.5",
                "content": "오늘은 봄바람이 느껴지는 향수와 산책했어요. 사진도 함께!",
                "is_public": True,
                "emotion_tags": ["happy", "spring", "photo"]
            }
        }


# 🆕 이미지 관리 관련 응답 모델들
class ImageStatsResponse(BaseModel):
    total_images: int = Field(..., description="총 이미지 수")
    total_size_mb: float = Field(..., description="총 용량 (MB)")
    upload_dir: str = Field(..., description="업로드 디렉토리")

    class Config:
        schema_extra = {
            "example": {
                "total_images": 156,
                "total_size_mb": 89.7,
                "upload_dir": "/app/uploads/diary_images"
            }
        }


class ImageDeleteResponse(BaseModel):
    success: bool = Field(..., description="삭제 성공 여부")
    message: str = Field(..., description="삭제 결과 메시지")
    deleted_files: List[str] = Field(..., description="삭제된 파일 목록")

    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "이미지 삭제 완료",
                "deleted_files": ["user123_abc123.jpg", "thumb_user123_abc123.jpg"]
            }
        }