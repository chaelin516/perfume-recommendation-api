# routers/diary_router.py - 감정 태그 + 이미지 업로드 + 강화된 삭제 기능 완전 통합 버전

import os
import json
import uuid
import logging
import asyncio
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Query, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# 의존성 모듈들
from utils.auth_utils import verify_firebase_token_optional, get_firebase_status
import asyncio
import re

# ─── 로깅 설정 ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("diary_router")

# ─── 라우터 생성 ─────────────────────────────────────────────────────────────────
router = APIRouter(
    prefix="/diaries",
    tags=["시향 일기"],
    responses={404: {"description": "Not found"}}
)

# ─── 데이터 경로 설정 ────────────────────────────────────────────────────────────
DATA_DIR = Path("data")
DIARY_DATA_PATH = DATA_DIR / "diary_data.json"
DIARY_IMAGES_DIR = DATA_DIR / "diary_images"

# 디렉토리 생성
DATA_DIR.mkdir(exist_ok=True)
DIARY_IMAGES_DIR.mkdir(exist_ok=True)


# ─── 데이터 모델 정의 ────────────────────────────────────────────────────────────
class DiaryEntry(BaseModel):
    """시향 일기 입력 모델"""
    user_id: Optional[str] = Field(None, description="사용자 ID")
    perfume_name: str = Field(..., min_length=1, max_length=100, description="향수 이름")
    content: Optional[str] = Field(None, max_length=2000, description="일기 내용")
    emotion_tags: Optional[List[str]] = Field(default=[], description="감정 태그")
    is_public: bool = Field(default=False, description="공개 여부")
    rating: Optional[float] = Field(None, ge=0, le=5, description="향수 평점")
    mood: Optional[str] = Field(None, max_length=50, description="기분")
    weather: Optional[str] = Field(None, max_length=50, description="날씨")
    location: Optional[str] = Field(None, max_length=100, description="장소")


class DiaryUpdateRequest(BaseModel):
    """일기 수정 요청 모델"""
    content: Optional[str] = Field(None, max_length=2000, description="수정할 내용")
    emotion_tags: Optional[List[str]] = Field(None, description="수정할 감정 태그")
    is_public: Optional[bool] = Field(None, description="공개 여부 변경")
    rating: Optional[float] = Field(None, ge=0, le=5, description="향수 평점 수정")
    mood: Optional[str] = Field(None, max_length=50, description="기분 수정")
    weather: Optional[str] = Field(None, max_length=50, description="날씨 수정")
    location: Optional[str] = Field(None, max_length=100, description="장소 수정")


# ─── 유틸리티 함수들 ────────────────────────────────────────────────────────────
async def get_default_user():
    """기본 사용자 정보 반환"""
    return {
        "uid": "anonymous_user",
        "email": "anonymous@example.com",
        "name": "익명 사용자",
        "picture": ""
    }


def sanitize_content(content: str) -> str:
    """내용 정리 및 검증"""
    if not content:
        return ""

    # HTML 태그 제거
    content = re.sub(r'<[^>]+>', '', content)

    # 특수 문자 정리
    content = content.strip()

    # 길이 제한
    if len(content) > 2000:
        content = content[:2000]

    return content


async def rule_based_emotion_analysis(content: str, perfume_name: str) -> Dict:
    """룰 기반 감정 분석 (간단한 구현)"""
    try:
        if not content or not content.strip():
            return {
                "success": False,
                "primary_emotion": "중립",
                "confidence": 0.0,
                "emotion_tags": ["#neutral"],
                "analysis_method": "no_content"
            }

        content_lower = content.lower()

        # 간단한 감정 키워드 매칭
        emotion_keywords = {
            "행복": ["좋다", "행복", "기쁘다", "즐겁다", "만족", "좋아요", "최고"],
            "사랑": ["사랑", "로맨틱", "달콤", "설레", "매력", "감동"],
            "평온": ["평온", "차분", "고요", "안정", "편안", "릴렉스"],
            "상쾌": ["상쾌", "시원", "깔끔", "산뜻", "청량", "프레시"],
            "우아": ["우아", "고급", "세련", "품격", "클래식", "엘레간트"],
            "중립": ["보통", "그냥", "무난", "평범"]
        }

        detected_emotions = []
        for emotion, keywords in emotion_keywords.items():
            for keyword in keywords:
                if keyword in content_lower:
                    detected_emotions.append(emotion)
                    break

        if not detected_emotions:
            detected_emotions = ["중립"]

        primary_emotion = detected_emotions[0]
        confidence = min(0.8, len(detected_emotions) * 0.3 + 0.2)

        # 태그 생성
        emotion_tags = [f"#{emotion}" for emotion in detected_emotions[:3]]

        return {
            "success": True,
            "primary_emotion": primary_emotion,
            "confidence": confidence,
            "emotion_tags": emotion_tags,
            "analysis_method": "rule_based",
            "context_detected": {
                "has_positive_words": any(word in content_lower for word in ["좋다", "행복", "만족"]),
                "has_negative_words": any(word in content_lower for word in ["싫다", "나쁘다", "실망"]),
                "content_length": len(content)
            },
            "perfume_type": "기타"
        }

    except Exception as e:
        logger.error(f"❌ 감정 분석 오류: {e}")
        return {
            "success": False,
            "primary_emotion": "중립",
            "confidence": 0.1,
            "emotion_tags": ["#error"],
            "analysis_method": "error"
        }


async def save_uploaded_image(file_content: bytes, diary_id: str, filename: str) -> str:
    """이미지 파일 저장 (간단한 구현)"""
    try:
        # 파일 확장자 추출
        file_ext = filename.split('.')[-1].lower() if '.' in filename else 'jpg'

        # 저장할 파일명 생성
        save_filename = f"{diary_id}.{file_ext}"
        save_path = DIARY_IMAGES_DIR / save_filename

        # 파일 저장
        with open(save_path, 'wb') as f:
            f.write(file_content)

        # 상대 경로 반환
        return f"diary_images/{save_filename}"

    except Exception as e:
        logger.error(f"❌ 이미지 저장 오류: {e}")
        return None


async def delete_uploaded_image(image_path: str) -> bool:
    """업로드된 이미지 삭제"""
    try:
        if image_path:
            full_path = DATA_DIR / image_path
            if full_path.exists():
                full_path.unlink()
                return True
        return False
    except Exception as e:
        logger.error(f"❌ 이미지 삭제 오류: {e}")
        return False


def load_diary_data() -> List[Dict]:
    """일기 데이터 로드"""
    try:
        if DIARY_DATA_PATH.exists():
            with open(DIARY_DATA_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    except Exception as e:
        logger.error(f"❌ 일기 데이터 로드 실패: {e}")
        return []


def save_diary_data(data: List[Dict]) -> bool:
    """일기 데이터 저장"""
    try:
        with open(DIARY_DATA_PATH, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        logger.error(f"❌ 일기 데이터 저장 실패: {e}")
        return False


def get_diary_by_id(diary_id: str) -> Optional[Dict]:
    """특정 일기 조회"""
    diary_data = load_diary_data()
    return next((diary for diary in diary_data if diary.get("id") == diary_id), None)


def update_diary_by_id(diary_id: str, updated_data: Dict) -> bool:
    """특정 일기 업데이트"""
    diary_data = load_diary_data()
    for i, diary in enumerate(diary_data):
        if diary.get("id") == diary_id:
            diary_data[i].update(updated_data)
            diary_data[i]["updated_at"] = datetime.now().isoformat()
            return save_diary_data(diary_data)
    return False


def delete_diary_by_id(diary_id: str) -> bool:
    """특정 일기 삭제"""
    diary_data = load_diary_data()
    original_count = len(diary_data)
    diary_data = [diary for diary in diary_data if diary.get("id") != diary_id]

    if len(diary_data) < original_count:
        return save_diary_data(diary_data)
    return False


# ─── 감정 분석 및 태그 처리 함수들 ────────────────────────────────────────────────
async def process_emotion_analysis(content: str, perfume_name: str) -> Dict:
    """감정 분석 처리"""
    if not content or not content.strip():
        return {
            "success": False,
            "primary_emotion": "중립",
            "confidence": 0.0,
            "emotion_tags": ["#neutral"],
            "analysis_method": "no_content"
        }

    try:
        # 룰 기반 감정 분석 (비동기)
        analysis_result = await asyncio.wait_for(
            rule_based_emotion_analysis(content, perfume_name),
            timeout=10.0
        )

        if analysis_result and analysis_result.get("success"):
            return analysis_result
        else:
            # 분석 실패 시 기본값 반환
            return {
                "success": False,
                "primary_emotion": "중립",
                "confidence": 0.3,
                "emotion_tags": ["#neutral"],
                "analysis_method": "fallback"
            }

    except asyncio.TimeoutError:
        logger.warning("⏰ 감정 분석 시간 초과")
        return {
            "success": False,
            "primary_emotion": "중립",
            "confidence": 0.2,
            "emotion_tags": ["#timeout"],
            "analysis_method": "timeout"
        }
    except Exception as e:
        logger.error(f"❌ 감정 분석 오류: {e}")
        return {
            "success": False,
            "primary_emotion": "중립",
            "confidence": 0.1,
            "emotion_tags": ["#error"],
            "analysis_method": "error"
        }


def merge_emotion_tags(manual_tags: List[str], auto_tags: List[str]) -> List[str]:
    """수동 태그와 자동 태그 병합"""
    # 중복 제거 및 정리
    all_tags = list(set(manual_tags + auto_tags))

    # 빈 태그 제거 및 정리
    cleaned_tags = []
    for tag in all_tags:
        if tag and tag.strip():
            cleaned_tag = tag.strip()
            if not cleaned_tag.startswith('#'):
                cleaned_tag = f'#{cleaned_tag}'
            cleaned_tags.append(cleaned_tag)

    return cleaned_tags[:10]  # 최대 10개 태그로 제한


# ─── 이미지 처리 함수들 ──────────────────────────────────────────────────────────
async def process_diary_image(file: UploadFile, diary_id: str) -> Optional[str]:
    """일기 이미지 처리"""
    if not file:
        return None

    try:
        # 이미지 검증
        if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
            raise HTTPException(status_code=400, detail="지원되지 않는 이미지 형식입니다.")

        # 파일 크기 검증 (5MB 제한)
        file_size = 0
        file_content = await file.read()
        file_size = len(file_content)

        if file_size > 5 * 1024 * 1024:  # 5MB
            raise HTTPException(status_code=400, detail="이미지 크기가 5MB를 초과합니다.")

        # 이미지 저장
        image_path = await save_uploaded_image(file_content, diary_id, file.filename)
        return image_path

    except Exception as e:
        logger.error(f"❌ 이미지 처리 오류: {e}")
        return None


# ─── API 엔드포인트들 ────────────────────────────────────────────────────────────

@router.post("/", summary="시향 일기 작성")
async def create_diary(
        entry: DiaryEntry,
        user=Depends(verify_firebase_token_optional)
):
    """
    시향 일기 작성

    - 감정 분석 자동 실행
    - 감정 태그 자동 생성 및 병합
    - 사용자 인증 선택적 적용
    """
    try:
        # 사용자 정보 처리
        if not user:
            user = get_default_user()

        user_id = entry.user_id or user.get("uid", "anonymous_user")

        # 일기 내용 검증 및 정리
        content = sanitize_content(entry.content) if entry.content else ""

        # 기본 일기 데이터 생성
        now = datetime.now().isoformat()
        diary_id = str(uuid.uuid4())

        logger.info(f"📝 새 일기 작성: {user_id} - {entry.perfume_name}")

        # 감정 분석 실행
        emotion_analysis = await process_emotion_analysis(content, entry.perfume_name)

        # 감정 태그 병합
        manual_tags = entry.emotion_tags or []
        auto_tags = emotion_analysis.get("emotion_tags", [])
        merged_tags = merge_emotion_tags(manual_tags, auto_tags)

        # 일기 데이터 구성
        diary = {
            "id": diary_id,
            "user_id": user_id,
            "user_name": user.get("name", user_id),
            "user_profile_image": user.get("picture", ""),
            "perfume_id": f"perfume_{entry.perfume_name.lower().replace(' ', '_')}",
            "perfume_name": entry.perfume_name,
            "brand": "Unknown Brand",
            "content": content,
            "emotion_tags": merged_tags,
            "rating": entry.rating,
            "mood": entry.mood,
            "weather": entry.weather,
            "location": entry.location,
            "likes": 0,
            "comments": 0,
            "is_public": entry.is_public,
            "created_at": now,
            "updated_at": now,
            "image_path": None,  # 이미지 업로드는 별도 엔드포인트에서 처리

            # 감정 분석 결과
            "emotion_analysis": emotion_analysis,
            "primary_emotion": emotion_analysis.get("primary_emotion", "중립"),
            "emotion_confidence": emotion_analysis.get("confidence", 0.0),
            "emotion_tags_auto": auto_tags,
            "emotion_analysis_status": "completed" if emotion_analysis.get("success") else "failed",
            "analysis_method": emotion_analysis.get("analysis_method", "unknown")
        }

        # 데이터 저장
        diary_data = load_diary_data()
        diary_data.append(diary)

        if save_diary_data(diary_data):
            return JSONResponse(
                status_code=200,
                content={
                    "message": "시향 일기가 성공적으로 저장되었습니다.",
                    "diary_id": diary_id,
                    "user_id": user_id,
                    "emotion_analysis": {
                        "status": diary["emotion_analysis_status"],
                        "method": diary["analysis_method"],
                        "primary_emotion": diary["primary_emotion"],
                        "confidence": diary["emotion_confidence"],
                        "merged_tags": merged_tags,
                        "auto_tags_count": len(auto_tags),
                        "manual_tags_count": len(manual_tags)
                    }
                }
            )
        else:
            raise HTTPException(status_code=500, detail="일기 저장에 실패했습니다.")

    except Exception as e:
        logger.error(f"❌ 일기 저장 오류: {e}")
        raise HTTPException(status_code=500, detail=f"일기 저장 중 오류: {str(e)}")


@router.get("/", summary="시향 일기 목록 조회")
async def get_diary_list(
        public: Optional[bool] = Query(None, description="공개 여부 필터"),
        page: Optional[int] = Query(1, description="페이지 번호"),
        size: Optional[int] = Query(10, description="페이지 크기"),
        user_id: Optional[str] = Query(None, description="사용자 ID 필터"),
        emotion: Optional[str] = Query(None, description="감정 필터"),
        sort: Optional[str] = Query("created_at", description="정렬 기준 (created_at, likes, rating)")
):
    """
    시향 일기 목록 조회

    - 필터링 및 페이징 지원
    - 감정별 필터링 지원
    - 다양한 정렬 옵션
    """
    try:
        diary_data = load_diary_data()

        # 필터링
        filtered_data = diary_data

        if public is not None:
            filtered_data = [d for d in filtered_data if d.get("is_public") == public]

        if user_id:
            filtered_data = [d for d in filtered_data if d.get("user_id") == user_id]

        if emotion:
            filtered_data = [d for d in filtered_data
                             if emotion.lower() in d.get("primary_emotion", "").lower() or
                             any(emotion.lower() in tag.lower() for tag in d.get("emotion_tags", []))]

        # 정렬
        if sort == "likes":
            filtered_data.sort(key=lambda x: x.get("likes", 0), reverse=True)
        elif sort == "rating":
            filtered_data.sort(key=lambda x: x.get("rating", 0) or 0, reverse=True)
        else:  # created_at (기본값)
            filtered_data.sort(key=lambda x: x.get("created_at", ""), reverse=True)

        # 페이징
        total_count = len(filtered_data)
        start_idx = (page - 1) * size
        end_idx = start_idx + size
        paginated_data = filtered_data[start_idx:end_idx]

        # 응답 데이터 구성
        response_data = []
        for diary in paginated_data:
            response_data.append({
                "id": diary.get("id"),
                "user_id": diary.get("user_id"),
                "user_name": diary.get("user_name"),
                "perfume_name": diary.get("perfume_name"),
                "content": diary.get("content", "")[:100] + "..." if len(diary.get("content", "")) > 100 else diary.get(
                    "content", ""),
                "emotion_tags": diary.get("emotion_tags", []),
                "primary_emotion": diary.get("primary_emotion"),
                "rating": diary.get("rating"),
                "mood": diary.get("mood"),
                "weather": diary.get("weather"),
                "location": diary.get("location"),
                "likes": diary.get("likes", 0),
                "comments": diary.get("comments", 0),
                "is_public": diary.get("is_public", False),
                "created_at": diary.get("created_at"),
                "image_path": diary.get("image_path")
            })

        return {
            "message": "일기 목록 조회 성공",
            "data": response_data,
            "pagination": {
                "page": page,
                "size": size,
                "total": total_count,
                "total_pages": (total_count + size - 1) // size,
                "has_next": end_idx < total_count,
                "has_prev": page > 1
            },
            "filters": {
                "public": public,
                "user_id": user_id,
                "emotion": emotion,
                "sort": sort
            }
        }

    except Exception as e:
        logger.error(f"❌ 일기 목록 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=f"일기 목록 조회 중 오류: {str(e)}")


@router.post("/{diary_id}/image", summary="일기 이미지 업로드")
async def upload_diary_image(
        diary_id: str,
        file: UploadFile = File(...),
        user=Depends(verify_firebase_token_optional)
):
    """
    일기 이미지 업로드

    - 이미지 검증 및 크기 제한
    - 기존 이미지 자동 삭제
    - 다양한 이미지 형식 지원
    """
    try:
        # 일기 존재 확인
        diary = get_diary_by_id(diary_id)
        if not diary:
            raise HTTPException(status_code=404, detail="일기를 찾을 수 없습니다.")

        # 권한 확인 (작성자만 수정 가능)
        if user and diary.get("user_id") != user.get("uid"):
            raise HTTPException(status_code=403, detail="이 일기를 수정할 권한이 없습니다.")

        # 기존 이미지 삭제
        if diary.get("image_path"):
            await delete_uploaded_image(diary.get("image_path"))

        # 새 이미지 저장
        image_path = await process_diary_image(file, diary_id)

        if image_path:
            # 일기 데이터 업데이트
            update_data = {"image_path": image_path}
            if update_diary_by_id(diary_id, update_data):
                return {
                    "message": "이미지가 성공적으로 업로드되었습니다.",
                    "diary_id": diary_id,
                    "image_path": image_path
                }
            else:
                raise HTTPException(status_code=500, detail="일기 업데이트에 실패했습니다.")
        else:
            raise HTTPException(status_code=500, detail="이미지 업로드에 실패했습니다.")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 이미지 업로드 오류: {e}")
        raise HTTPException(status_code=500, detail=f"이미지 업로드 중 오류: {str(e)}")


@router.put("/{diary_id}", summary="일기 수정")
async def update_diary(
        diary_id: str,
        update_data: DiaryUpdateRequest,
        user=Depends(verify_firebase_token_optional)
):
    """
    일기 수정

    - 부분 수정 지원
    - 감정 태그 재분석 옵션
    - 권한 확인
    """
    try:
        # 일기 존재 확인
        diary = get_diary_by_id(diary_id)
        if not diary:
            raise HTTPException(status_code=404, detail="일기를 찾을 수 없습니다.")

        # 권한 확인
        if user and diary.get("user_id") != user.get("uid"):
            raise HTTPException(status_code=403, detail="이 일기를 수정할 권한이 없습니다.")

        # 업데이트할 데이터 준비
        updated_fields = {}

        # 내용 업데이트 시 감정 재분석
        if update_data.content is not None:
            updated_fields["content"] = sanitize_content(update_data.content)

            # 감정 재분석 실행
            emotion_analysis = await process_emotion_analysis(
                updated_fields["content"],
                diary.get("perfume_name", "")
            )

            # 감정 분석 결과 업데이트
            updated_fields.update({
                "emotion_analysis": emotion_analysis,
                "primary_emotion": emotion_analysis.get("primary_emotion", "중립"),
                "emotion_confidence": emotion_analysis.get("confidence", 0.0),
                "emotion_tags_auto": emotion_analysis.get("emotion_tags", []),
                "emotion_analysis_status": "completed" if emotion_analysis.get("success") else "failed",
                "analysis_method": emotion_analysis.get("analysis_method", "unknown")
            })

        # 기타 필드 업데이트
        if update_data.emotion_tags is not None:
            # 기존 자동 태그와 새 수동 태그 병합
            auto_tags = diary.get("emotion_tags_auto", [])
            manual_tags = update_data.emotion_tags
            updated_fields["emotion_tags"] = merge_emotion_tags(manual_tags, auto_tags)

        if update_data.is_public is not None:
            updated_fields["is_public"] = update_data.is_public

        if update_data.rating is not None:
            updated_fields["rating"] = update_data.rating

        if update_data.mood is not None:
            updated_fields["mood"] = update_data.mood

        if update_data.weather is not None:
            updated_fields["weather"] = update_data.weather

        if update_data.location is not None:
            updated_fields["location"] = update_data.location

        # 업데이트 실행
        if update_diary_by_id(diary_id, updated_fields):
            return {
                "message": "일기가 성공적으로 수정되었습니다.",
                "diary_id": diary_id,
                "updated_fields": list(updated_fields.keys()),
                "emotion_reanalyzed": "content" in updated_fields
            }
        else:
            raise HTTPException(status_code=500, detail="일기 수정에 실패했습니다.")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 일기 수정 오류: {e}")
        raise HTTPException(status_code=500, detail=f"일기 수정 중 오류: {str(e)}")


@router.delete("/{diary_id}", summary="일기 삭제")
async def delete_diary(
        diary_id: str,
        user=Depends(verify_firebase_token_optional)
):
    """
    일기 삭제

    - 권한 확인
    - 관련 이미지 자동 삭제
    - 안전한 삭제 처리
    """
    try:
        # 일기 존재 확인
        diary = get_diary_by_id(diary_id)
        if not diary:
            raise HTTPException(status_code=404, detail="일기를 찾을 수 없습니다.")

        # 권한 확인
        if user and diary.get("user_id") != user.get("uid"):
            raise HTTPException(status_code=403, detail="이 일기를 삭제할 권한이 없습니다.")

        # 관련 이미지 삭제
        if diary.get("image_path"):
            try:
                await delete_uploaded_image(diary.get("image_path"))
                logger.info(f"✅ 일기 이미지 삭제 완료: {diary.get('image_path')}")
            except Exception as e:
                logger.warning(f"⚠️ 일기 이미지 삭제 실패: {e}")

        # 일기 데이터 삭제
        if delete_diary_by_id(diary_id):
            return {
                "message": "일기가 성공적으로 삭제되었습니다.",
                "diary_id": diary_id,
                "user_id": diary.get("user_id"),
                "deleted_items": {
                    "diary": True,
                    "image": diary.get("image_path") is not None
                }
            }
        else:
            raise HTTPException(status_code=500, detail="일기 삭제에 실패했습니다.")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 일기 삭제 오류: {e}")
        raise HTTPException(status_code=500, detail=f"일기 삭제 중 오류: {str(e)}")


@router.delete("/user/{user_id}", summary="사용자 모든 일기 삭제")
async def delete_user_diaries(
        user_id: str,
        user=Depends(verify_firebase_token_optional)
):
    """
    특정 사용자의 모든 일기 삭제

    - 관리자 권한 또는 본인만 가능
    - 모든 관련 이미지 자동 삭제
    - 일괄 삭제 처리
    """
    try:
        # 권한 확인 (본인 또는 관리자)
        if user:
            current_user_id = user.get("uid")
            is_admin = user.get("email", "").endswith("@admin.whiff.com")  # 관리자 권한 확인

            if current_user_id != user_id and not is_admin:
                raise HTTPException(status_code=403, detail="이 작업을 수행할 권한이 없습니다.")

        # 사용자 일기 목록 조회
        diary_data = load_diary_data()
        user_diaries = [diary for diary in diary_data if diary.get("user_id") == user_id]

        if not user_diaries:
            return {
                "message": "삭제할 일기가 없습니다.",
                "user_id": user_id,
                "deleted_count": 0
            }

        # 관련 이미지 삭제
        deleted_images = 0
        for diary in user_diaries:
            if diary.get("image_path"):
                try:
                    await delete_uploaded_image(diary.get("image_path"))
                    deleted_images += 1
                except Exception as e:
                    logger.warning(f"⚠️ 이미지 삭제 실패: {e}")

        # 일기 데이터 삭제
        original_count = len(diary_data)
        filtered_data = [diary for diary in diary_data if diary.get("user_id") != user_id]
        deleted_count = original_count - len(filtered_data)

        if save_diary_data(filtered_data):
            return {
                "message": f"사용자 {user_id}의 모든 일기가 삭제되었습니다.",
                "user_id": user_id,
                "deleted_count": deleted_count,
                "deleted_images": deleted_images
            }
        else:
            raise HTTPException(status_code=500, detail="일기 삭제에 실패했습니다.")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 사용자 일기 삭제 오류: {e}")
        raise HTTPException(status_code=500, detail=f"사용자 일기 삭제 중 오류: {str(e)}")


@router.get("/stats/summary", summary="일기 통계 요약")
async def get_diary_stats_summary(
        user_id: Optional[str] = Query(None, description="특정 사용자 통계"),
        user=Depends(verify_firebase_token_optional)
):
    """
    일기 통계 요약 정보

    - 전체 통계 또는 사용자별 통계
    - 감정 분포, 평점 평균, 작성 빈도 등
    - 월별/주별 통계 제공
    """
    try:
        diary_data = load_diary_data()

        # 사용자 필터링
        if user_id:
            diary_data = [d for d in diary_data if d.get("user_id") == user_id]

        if not diary_data:
            return {
                "message": "통계 데이터가 없습니다.",
                "user_id": user_id,
                "stats": {
                    "total_diaries": 0,
                    "emotion_distribution": {},
                    "average_rating": 0,
                    "public_ratio": 0
                }
            }

        # 기본 통계
        total_diaries = len(diary_data)
        public_diaries = len([d for d in diary_data if d.get("is_public", False)])

        # 감정 분포 계산
        emotion_count = {}
        for diary in diary_data:
            emotion = diary.get("primary_emotion", "중립")
            emotion_count[emotion] = emotion_count.get(emotion, 0) + 1

        # 평점 통계
        ratings = [d.get("rating") for d in diary_data if d.get("rating") is not None]
        average_rating = sum(ratings) / len(ratings) if ratings else 0

        # 태그 분포 (상위 10개)
        tag_count = {}
        for diary in diary_data:
            for tag in diary.get("emotion_tags", []):
                tag_count[tag] = tag_count.get(tag, 0) + 1

        popular_tags = sorted(tag_count.items(), key=lambda x: x[1], reverse=True)[:10]

        # 월별 작성 통계 (최근 12개월)
        from collections import defaultdict
        monthly_stats = defaultdict(int)

        for diary in diary_data:
            created_at = diary.get("created_at", "")
            if created_at:
                try:
                    month_key = created_at[:7]  # YYYY-MM 형식
                    monthly_stats[month_key] += 1
                except:
                    continue

        # 향수 브랜드별 통계
        brand_count = {}
        for diary in diary_data:
            brand = diary.get("brand", "Unknown")
            brand_count[brand] = brand_count.get(brand, 0) + 1

        popular_brands = sorted(brand_count.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            "message": "일기 통계 조회 성공",
            "user_id": user_id,
            "stats": {
                "total_diaries": total_diaries,
                "public_diaries": public_diaries,
                "public_ratio": round(public_diaries / total_diaries * 100, 1) if total_diaries > 0 else 0,
                "emotion_distribution": emotion_count,
                "average_rating": round(average_rating, 2),
                "popular_tags": popular_tags,
                "monthly_stats": dict(monthly_stats),
                "popular_brands": popular_brands,
                "has_images": len([d for d in diary_data if d.get("image_path")]),
                "analysis_success_rate": len([d for d in diary_data if d.get(
                    "emotion_analysis_status") == "completed"]) / total_diaries * 100 if total_diaries > 0 else 0
            }
        }

    except Exception as e:
        logger.error(f"❌ 통계 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=f"통계 조회 중 오류: {str(e)}")


@router.get("/emotions/analysis", summary="감정 분석 결과 조회")
async def get_emotion_analysis_results(
        user_id: Optional[str] = Query(None, description="특정 사용자"),
        emotion: Optional[str] = Query(None, description="특정 감정 필터"),
        limit: Optional[int] = Query(20, description="결과 개수 제한"),
        user=Depends(verify_firebase_token_optional)
):
    """
    감정 분석 결과 조회

    - 사용자별 감정 패턴 분석
    - 특정 감정의 일기 목록
    - 감정 변화 추이 분석
    """
    try:
        diary_data = load_diary_data()

        # 사용자 필터링
        if user_id:
            diary_data = [d for d in diary_data if d.get("user_id") == user_id]

        # 감정 필터링
        if emotion:
            diary_data = [d for d in diary_data
                          if emotion.lower() in d.get("primary_emotion", "").lower()]

        # 최근 순으로 정렬 및 개수 제한
        diary_data.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        diary_data = diary_data[:limit]

        # 감정 분석 결과 구성
        results = []
        for diary in diary_data:
            analysis = diary.get("emotion_analysis", {})

            results.append({
                "diary_id": diary.get("id"),
                "perfume_name": diary.get("perfume_name"),
                "created_at": diary.get("created_at"),
                "primary_emotion": diary.get("primary_emotion"),
                "confidence": diary.get("emotion_confidence", 0),
                "analysis_status": diary.get("emotion_analysis_status"),
                "analysis_method": diary.get("analysis_method"),
                "auto_tags": diary.get("emotion_tags_auto", []),
                "manual_tags": [tag for tag in diary.get("emotion_tags", [])
                                if tag not in diary.get("emotion_tags_auto", [])],
                "content_preview": diary.get("content", "")[:100] + "..." if len(
                    diary.get("content", "")) > 100 else diary.get("content", ""),
                "context_detected": analysis.get("context_detected", {}),
                "perfume_type": analysis.get("perfume_type", "기타")
            })

        # 전체 감정 분포 계산
        all_diary_data = load_diary_data()
        if user_id:
            all_diary_data = [d for d in all_diary_data if d.get("user_id") == user_id]

        emotion_distribution = {}
        for diary in all_diary_data:
            emotion = diary.get("primary_emotion", "중립")
            emotion_distribution[emotion] = emotion_distribution.get(emotion, 0) + 1

        return {
            "message": "감정 분석 결과 조회 성공",
            "user_id": user_id,
            "emotion_filter": emotion,
            "results": results,
            "total_analyzed": len(results),
            "emotion_distribution": emotion_distribution,
            "analysis_summary": {
                "successful_analysis": len([r for r in results if r["analysis_status"] == "completed"]),
                "failed_analysis": len([r for r in results if r["analysis_status"] == "failed"]),
                "average_confidence": sum([r["confidence"] for r in results]) / len(results) if results else 0
            }
        }

    except Exception as e:
        logger.error(f"❌ 감정 분석 결과 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=f"감정 분석 결과 조회 중 오류: {str(e)}")


@router.post("/{diary_id}/like", summary="일기 좋아요")
async def like_diary(
        diary_id: str,
        user=Depends(verify_firebase_token_optional)
):
    """
    일기 좋아요 토글

    - 좋아요 추가/제거
    - 중복 좋아요 방지
    - 실시간 카운트 업데이트
    """
    try:
        # 일기 존재 확인
        diary = get_diary_by_id(diary_id)
        if not diary:
            raise HTTPException(status_code=404, detail="일기를 찾을 수 없습니다.")

        # 사용자 정보
        user_id = user.get("uid") if user else "anonymous"

        # 좋아요 정보 처리 (간단한 구현)
        current_likes = diary.get("likes", 0)
        liked_users = diary.get("liked_users", [])

        if user_id in liked_users:
            # 좋아요 취소
            liked_users.remove(user_id)
            new_likes = max(0, current_likes - 1)
            action = "unliked"
        else:
            # 좋아요 추가
            liked_users.append(user_id)
            new_likes = current_likes + 1
            action = "liked"

        # 업데이트
        update_data = {
            "likes": new_likes,
            "liked_users": liked_users
        }

        if update_diary_by_id(diary_id, update_data):
            return {
                "message": f"일기 {action} 성공",
                "diary_id": diary_id,
                "action": action,
                "new_likes_count": new_likes,
                "user_id": user_id
            }
        else:
            raise HTTPException(status_code=500, detail="좋아요 처리에 실패했습니다.")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 좋아요 처리 오류: {e}")
        raise HTTPException(status_code=500, detail=f"좋아요 처리 중 오류: {str(e)}")


@router.get("/search", summary="일기 검색")
async def search_diaries(
        q: str = Query(..., description="검색어"),
        search_type: str = Query("all", description="검색 타입 (all, content, perfume, tags)"),
        page: int = Query(1, description="페이지 번호"),
        size: int = Query(10, description="페이지 크기"),
        user=Depends(verify_firebase_token_optional)
):
    """
    일기 검색

    - 내용, 향수명, 태그별 검색
    - 통합 검색 및 필터링
    - 검색 결과 하이라이팅
    """
    try:
        diary_data = load_diary_data()

        # 공개 일기만 검색 (비로그인 시) 또는 본인 일기 포함 (로그인 시)
        if user:
            user_id = user.get("uid")
            # 공개 일기 + 본인 일기
            searchable_data = [d for d in diary_data
                               if d.get("is_public", False) or d.get("user_id") == user_id]
        else:
            # 공개 일기만
            searchable_data = [d for d in diary_data if d.get("is_public", False)]

        # 검색 실행
        search_results = []
        search_term = q.lower()

        for diary in searchable_data:
            match_score = 0
            match_fields = []

            # 검색 타입별 매칭
            if search_type in ["all", "content"]:
                content = diary.get("content", "").lower()
                if search_term in content:
                    match_score += 3
                    match_fields.append("content")

            if search_type in ["all", "perfume"]:
                perfume_name = diary.get("perfume_name", "").lower()
                if search_term in perfume_name:
                    match_score += 5
                    match_fields.append("perfume")

            if search_type in ["all", "tags"]:
                tags = [tag.lower() for tag in diary.get("emotion_tags", [])]
                if any(search_term in tag for tag in tags):
                    match_score += 2
                    match_fields.append("tags")

            # 추가 검색 필드
            if search_type == "all":
                # 감정, 브랜드, 위치 등에서도 검색
                if search_term in diary.get("primary_emotion", "").lower():
                    match_score += 1
                    match_fields.append("emotion")

                if search_term in diary.get("brand", "").lower():
                    match_score += 2
                    match_fields.append("brand")

                if search_term in diary.get("location", "").lower():
                    match_score += 1
                    match_fields.append("location")

            # 매칭된 결과 추가
            if match_score > 0:
                search_results.append({
                    "diary": diary,
                    "match_score": match_score,
                    "match_fields": match_fields
                })

        # 점수 순 정렬
        search_results.sort(key=lambda x: x["match_score"], reverse=True)

        # 페이징
        total_count = len(search_results)
        start_idx = (page - 1) * size
        end_idx = start_idx + size
        paginated_results = search_results[start_idx:end_idx]

        # 응답 데이터 구성
        response_data = []
        for result in paginated_results:
            diary = result["diary"]
            response_data.append({
                "id": diary.get("id"),
                "user_id": diary.get("user_id"),
                "user_name": diary.get("user_name"),
                "perfume_name": diary.get("perfume_name"),
                "content": diary.get("content", "")[:200] + "..." if len(diary.get("content", "")) > 200 else diary.get(
                    "content", ""),
                "emotion_tags": diary.get("emotion_tags", []),
                "primary_emotion": diary.get("primary_emotion"),
                "rating": diary.get("rating"),
                "likes": diary.get("likes", 0),
                "created_at": diary.get("created_at"),
                "match_score": result["match_score"],
                "match_fields": result["match_fields"]
            })

        return {
            "message": "검색 완료",
            "query": q,
            "search_type": search_type,
            "results": response_data,
            "pagination": {
                "page": page,
                "size": size,
                "total": total_count,
                "total_pages": (total_count + size - 1) // size,
                "has_next": end_idx < total_count,
                "has_prev": page > 1
            },
            "search_summary": {
                "total_found": total_count,
                "avg_match_score": sum([r["match_score"] for r in search_results]) / len(
                    search_results) if search_results else 0,
                "search_in_public_only": user is None
            }
        }

    except Exception as e:
        logger.error(f"❌ 일기 검색 오류: {e}")
        raise HTTPException(status_code=500, detail=f"일기 검색 중 오류: {str(e)}")


# ─── 관리자 전용 API ─────────────────────────────────────────────────────────────

@router.get("/admin/all", summary="전체 일기 관리 (관리자)")
async def admin_get_all_diaries(
        include_private: bool = Query(False, description="비공개 일기 포함"),
        user=Depends(verify_firebase_token_optional)
):
    """
    관리자 전용: 전체 일기 조회

    - 모든 사용자 일기 조회
    - 비공개 일기 포함 옵션
    - 상세 통계 정보
    """
    try:
        # 관리자 권한 확인
        if not user or not user.get("email", "").endswith("@admin.whiff.com"):
            raise HTTPException(status_code=403, detail="관리자 권한이 필요합니다.")

        diary_data = load_diary_data()

        # 비공개 일기 필터링
        if not include_private:
            diary_data = [d for d in diary_data if d.get("is_public", False)]

        # 상세 통계 계산
        stats = {
            "total_diaries": len(diary_data),
            "public_diaries": len([d for d in diary_data if d.get("is_public", False)]),
            "users_count": len(set([d.get("user_id") for d in diary_data])),
            "with_images": len([d for d in diary_data if d.get("image_path")]),
            "analysis_completed": len([d for d in diary_data if d.get("emotion_analysis_status") == "completed"]),
            "average_rating": sum([d.get("rating", 0) for d in diary_data if d.get("rating")]) / len(
                [d for d in diary_data if d.get("rating")]) if [d for d in diary_data if d.get("rating")] else 0
        }

        return {
            "message": "관리자 일기 조회 성공",
            "admin_user": user.get("email"),
            "data": diary_data,
            "stats": stats,
            "include_private": include_private
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 관리자 일기 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=f"관리자 일기 조회 중 오류: {str(e)}")


@router.delete("/admin/cleanup", summary="데이터 정리 (관리자)")
async def admin_cleanup_data(
        remove_failed_analysis: bool = Query(False, description="분석 실패 일기 삭제"),
        remove_empty_content: bool = Query(False, description="내용 없는 일기 삭제"),
        remove_old_days: Optional[int] = Query(None, description="N일 이전 일기 삭제"),
        user=Depends(verify_firebase_token_optional)
):
    """
    관리자 전용: 데이터 정리

    - 분석 실패 일기 정리
    - 빈 내용 일기 정리
    - 오래된 일기 정리
    """
    try:
        # 관리자 권한 확인
        if not user or not user.get("email", "").endswith("@admin.whiff.com"):
            raise HTTPException(status_code=403, detail="관리자 권한이 필요합니다.")

        diary_data = load_diary_data()
        original_count = len(diary_data)
        cleanup_stats = {
            "failed_analysis_removed": 0,
            "empty_content_removed": 0,
            "old_diaries_removed": 0
        }

        # 정리 작업 실행
        cleaned_data = diary_data.copy()

        # 1. 분석 실패 일기 제거
        if remove_failed_analysis:
            before_count = len(cleaned_data)
            cleaned_data = [d for d in cleaned_data
                            if d.get("emotion_analysis_status") != "failed"]
            cleanup_stats["failed_analysis_removed"] = before_count - len(cleaned_data)

        # 2. 빈 내용 일기 제거
        if remove_empty_content:
            before_count = len(cleaned_data)
            cleaned_data = [d for d in cleaned_data
                            if d.get("content", "").strip()]
            cleanup_stats["empty_content_removed"] = before_count - len(cleaned_data)

        # 3. 오래된 일기 제거
        if remove_old_days:
            from datetime import datetime, timedelta
            cutoff_date = (datetime.now() - timedelta(days=remove_old_days)).isoformat()
            before_count = len(cleaned_data)
            cleaned_data = [d for d in cleaned_data
                            if d.get("created_at", "") > cutoff_date]
            cleanup_stats["old_diaries_removed"] = before_count - len(cleaned_data)

        # 정리 결과 저장
        total_removed = original_count - len(cleaned_data)

        if total_removed > 0:
            save_diary_data(cleaned_data)

        return {
            "message": f"데이터 정리 완료: {total_removed}개 일기 제거",
            "admin_user": user.get("email"),
            "original_count": original_count,
            "final_count": len(cleaned_data),
            "cleanup_stats": cleanup_stats
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 관리자 데이터 정리 오류: {e}")
        raise HTTPException(status_code=500, detail=f"관리자 데이터 정리 중 오류: {str(e)}")


# ─── 헬스체크 및 상태 확인 ───────────────────────────────────────────────────────

@router.get("/health", summary="시향 일기 모듈 상태 확인")
async def health_check():
    """
    시향 일기 모듈 헬스체크

    - 데이터 파일 상태
    - 감정 분석 모듈 상태
    - 이미지 업로드 디렉토리 상태
    """
    try:
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "modules": {}
        }

        # 데이터 파일 상태 확인
        try:
            diary_data = load_diary_data()
            health_status["modules"]["data_file"] = {
                "status": "ok",
                "diary_count": len(diary_data),
                "file_exists": DIARY_DATA_PATH.exists()
            }
        except Exception as e:
            health_status["modules"]["data_file"] = {
                "status": "error",
                "error": str(e)
            }
            health_status["status"] = "degraded"

        # 감정 분석 모듈 상태 확인
        try:
            test_analysis = await process_emotion_analysis("테스트 내용입니다", "테스트 향수")
            health_status["modules"]["emotion_analysis"] = {
                "status": "ok",
                "test_result": test_analysis.get("success", False)
            }
        except Exception as e:
            health_status["modules"]["emotion_analysis"] = {
                "status": "error",
                "error": str(e)
            }
            health_status["status"] = "degraded"

        # 이미지 디렉토리 상태 확인
        health_status["modules"]["image_storage"] = {
            "status": "ok" if DIARY_IMAGES_DIR.exists() else "warning",
            "directory_exists": DIARY_IMAGES_DIR.exists(),
            "writable": os.access(DIARY_IMAGES_DIR, os.W_OK) if DIARY_IMAGES_DIR.exists() else False
        }

        return health_status

    except Exception as e:
        logger.error(f"❌ 헬스체크 오류: {e}")
        return {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }