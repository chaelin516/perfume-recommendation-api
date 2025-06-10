# routers/diary_router.py - 이미지 업로드 기능 추가된 버전

from fastapi import APIRouter, Query, HTTPException, BackgroundTasks, Body, File, UploadFile, Form, Request
from fastapi.responses import JSONResponse
from schemas.diary import (
    DiaryCreateRequest, DiaryResponse, DiaryWithImageCreateRequest,
    ImageUploadResponse, ImageStatsResponse, ImageDeleteResponse
)
from schemas.common import BaseResponse
from utils.image_utils import (
    save_uploaded_image, get_image_url, get_thumbnail_url,
    delete_image_files, get_upload_stats, validate_image_file
)
from datetime import datetime, date
from typing import Optional, List, Dict, Any
import os
import json
import uuid
import logging
import asyncio
import time
from collections import Counter
from pydantic import BaseModel, Field
import re

# 🎭 로거 설정
logger = logging.getLogger("diary_router")

# ✅ 라우터 생성
router = APIRouter(prefix="/diaries", tags=["Diary"])

# 📂 데이터 파일 경로
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DIARY_PATH = os.path.join(BASE_DIR, "../data/diary_data.json")

# 🎯 룰 기반 감정 분석 사전 정의 (기존 코드와 동일)
EMOTION_RULES = {
    "기쁨": {
        "keywords": [
            "좋", "훌륭", "향긋", "달콤", "상큼", "깔끔", "사랑", "완벽", "최고", "멋진",
            "환상적", "놀라운", "아름다운", "우아한", "세련된", "고급스러운", "매력적",
            "기분좋", "행복", "즐거운", "만족", "감동", "황홀", "반함", "좋아해", "마음에 들",
            "포근", "따뜻", "편안", "부드러운", "은은한", "우아", "고혹적", "신비로운"
        ],
        "tags": ["#happy", "#joyful", "#pleasant", "#nice", "#satisfied", "#lovely"],
        "base_confidence": 0.7
    },
    "설렘": {
        "keywords": [
            "설레", "두근", "떨림", "기대", "호기심", "궁금", "신기", "새로운", "특별한",
            "독특한", "매혹적", "흥미로운", "재미있", "신선한", "생동감", "활기찬",
            "짜릿", "스릴", "흥분", "가슴이 뛰", "심장이", "떨려"
        ],
        "tags": ["#excited", "#curious", "#anticipation", "#thrilling", "#fascinating"],
        "base_confidence": 0.65
    },
    "평온": {
        "keywords": [
            "평온", "고요", "잔잔", "차분", "안정", "평화", "조용", "편안", "여유로운",
            "느긋", "릴렉스", "휴식", "치유", "힐링", "진정", "마음이 편해", "스트레스",
            "피로가 풀", "숨을 쉬기", "깊게 호흡", "명상", "사색", "생각에 잠기"
        ],
        "tags": ["#calm", "#peaceful", "#relaxed", "#healing", "#serene"],
        "base_confidence": 0.6
    },
    "자신감": {
        "keywords": [
            "자신감", "당당", "세련", "고급", "품격", "우아", "카리스마", "멋있", "섹시",
            "매력적", "강렬", "파워풀", "임팩트", "프로페셔널", "성숙한", "어른스러운"
        ],
        "tags": ["#confident", "#elegant", "#sophisticated", "#charismatic", "#powerful"],
        "base_confidence": 0.65
    },
    "활력": {
        "keywords": [
            "활력", "에너지", "생동감", "활기", "싱그러운", "상쾌", "시원", "청량감",
            "톡톡", "팝", "활발", "역동적", "젊은", "발랄", "명랑", "생기발랄"
        ],
        "tags": ["#energetic", "#fresh", "#vibrant", "#lively", "#dynamic"],
        "base_confidence": 0.6
    },
    "로맨틱": {
        "keywords": [
            "로맨틱", "낭만", "사랑", "달콤", "부드러운", "따뜻한", "포근", "감미로운",
            "달콤쌉쌀", "심쿵", "로맨스", "데이트", "연인", "커플", "달달한"
        ],
        "tags": ["#romantic", "#sweet", "#lovely", "#tender", "#affectionate"],
        "base_confidence": 0.7
    },
    "그리움": {
        "keywords": [
            "그리움", "향수", "추억", "그립", "옛날", "어릴적", "추상적", "몽환적",
            "아련", "쓸쓸", "서정적", "감성적", "애틋", "생각나", "기억"
        ],
        "tags": ["#nostalgic", "#memory", "#longing", "#sentimental", "#wistful"],
        "base_confidence": 0.6
    }
}

# 🌍 상황별 감정 부스터 (기존과 동일)
CONTEXT_BOOSTERS = {
    "계절": {
        "봄": {"기쁨": 0.2, "활력": 0.15, "로맨틱": 0.1},
        "여름": {"활력": 0.25, "자신감": 0.15, "기쁨": 0.1},
        "가을": {"그리움": 0.2, "평온": 0.15, "로맨틱": 0.1},
        "겨울": {"평온": 0.2, "그리움": 0.15, "로맨틱": 0.1}
    },
    "시간": {
        "아침": {"활력": 0.2, "자신감": 0.15},
        "낮": {"기쁨": 0.15, "활력": 0.1},
        "저녁": {"로맨틱": 0.2, "평온": 0.15},
        "밤": {"그리움": 0.2, "평온": 0.15, "로맨틱": 0.1}
    },
    "상황": {
        "데이트": {"로맨틱": 0.3, "설렘": 0.2},
        "업무": {"자신감": 0.2, "활력": 0.15},
        "휴식": {"평온": 0.25, "기쁨": 0.1},
        "외출": {"활력": 0.15, "자신감": 0.1}
    }
}

# 🎨 향수 타입별 감정 매핑 (기존과 동일)
PERFUME_TYPE_EMOTIONS = {
    "플로럴": ["로맨틱", "기쁨", "평온"],
    "시트러스": ["활력", "기쁨", "자신감"],
    "우디": ["자신감", "평온", "그리움"],
    "바닐라": ["평온", "로맨틱", "그리움"],
    "머스크": ["자신감", "로맨틱", "평온"],
    "프루티": ["기쁨", "활력", "설렘"]
}


# ✅ 기존 유틸리티 함수들 (동일)
def load_diary_data():
    """시향 일기 데이터 로딩"""
    if os.path.exists(DIARY_PATH):
        try:
            with open(DIARY_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.info(f"✅ 시향 일기 데이터 로딩: {len(data)}개")
            return data
        except Exception as e:
            logger.error(f"❌ 시향 일기 데이터 로딩 실패: {e}")
    return []


def save_diary_data(data):
    """시향 일기 데이터 저장"""
    try:
        os.makedirs(os.path.dirname(DIARY_PATH), exist_ok=True)
        with open(DIARY_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"✅ 시향 일기 데이터 저장: {len(data)}개")
    except Exception as e:
        logger.error(f"❌ 시향 일기 데이터 저장 실패: {e}")


def get_default_user():
    """기본 사용자 정보"""
    return {
        "uid": "anonymous",
        "name": "익명 사용자",
        "email": "anonymous@example.com",
        "picture": ""
    }


async def rule_based_emotion_analysis(content: str, perfume_name: str = "") -> dict:
    """룰 기반 감정 분석 (기존 코드와 동일)"""
    try:
        text = f"{content} {perfume_name}".lower()
        emotion_scores = {}

        for emotion, config in EMOTION_RULES.items():
            score = config["base_confidence"]
            keyword_matches = 0

            for keyword in config["keywords"]:
                if keyword in text:
                    keyword_matches += 1
                    score += 0.1

            if keyword_matches > 0:
                emotion_scores[emotion] = min(score, 1.0)

        if not emotion_scores:
            return {
                "success": False,
                "primary_emotion": "중립",
                "confidence": 0.3,
                "emotion_tags": ["#neutral"],
                "analysis_method": "rule_based"
            }

        primary_emotion = max(emotion_scores, key=emotion_scores.get)
        confidence = emotion_scores[primary_emotion]

        emotion_tags = EMOTION_RULES[primary_emotion]["tags"][:3]

        return {
            "success": True,
            "primary_emotion": primary_emotion,
            "confidence": confidence,
            "emotion_tags": emotion_tags,
            "analysis_method": "rule_based",
            "all_scores": emotion_scores
        }

    except Exception as e:
        logger.error(f"❌ 룰 기반 감정 분석 오류: {e}")
        return {
            "success": False,
            "primary_emotion": "중립",
            "confidence": 0.3,
            "emotion_tags": ["#neutral"],
            "analysis_method": "error"
        }


# 전역 데이터 로딩
diary_data = load_diary_data()


# ================================
# 🆕 이미지 업로드 관련 API들
# ================================

@router.post("/upload-image", summary="이미지 업로드 (단독)", response_model=ImageUploadResponse)
async def upload_diary_image(
        request: Request,
        user_id: str = Form(..., description="사용자 ID"),
        image: UploadFile = File(..., description="업로드할 이미지 파일")
):
    """
    시향 일기용 이미지만 업로드하는 API

    - 지원 형식: JPG, JPEG, PNG, WEBP
    - 최대 크기: 10MB
    - 자동 리사이징 및 썸네일 생성
    """
    try:
        logger.info(f"📸 이미지 업로드 요청: {user_id} - {image.filename}")

        # 1. 이미지 파일 검증
        is_valid, message = validate_image_file(image)
        if not is_valid:
            return ImageUploadResponse(
                success=False,
                message=message,
                image_url=None,
                thumbnail_url=None,
                filename=None
            )

        # 2. 이미지 저장 및 처리
        success, result, metadata = await save_uploaded_image(image, user_id)

        if not success:
            return ImageUploadResponse(
                success=False,
                message=result,
                image_url=None,
                thumbnail_url=None,
                filename=None
            )

        # 3. URL 생성
        base_url = str(request.base_url)
        image_url = get_image_url(result, base_url)
        thumbnail_url = get_thumbnail_url(result, base_url)

        logger.info(f"✅ 이미지 업로드 성공: {result}")

        return ImageUploadResponse(
            success=True,
            message="이미지 업로드 성공",
            image_url=image_url,
            thumbnail_url=thumbnail_url,
            filename=result,
            file_size=metadata.get("file_size") if metadata else None,
            image_metadata=metadata
        )

    except Exception as e:
        logger.error(f"❌ 이미지 업로드 오류: {e}")
        return ImageUploadResponse(
            success=False,
            message=f"이미지 업로드 중 오류가 발생했습니다: {str(e)}",
            image_url=None,
            thumbnail_url=None,
            filename=None
        )


@router.post("/with-image", summary="시향 일기 + 이미지 동시 작성")
async def create_diary_with_image(
        request: Request,
        user_id: str = Form(..., description="사용자 ID"),
        perfume_name: str = Form(..., description="향수명"),
        content: str = Form(None, description="일기 내용"),
        is_public: bool = Form(..., description="공개 여부"),
        emotion_tags: str = Form("[]", description="감정 태그 (JSON 배열 문자열)"),
        image: UploadFile = File(..., description="첨부할 이미지")
):
    """
    시향 일기 작성과 이미지 업로드를 동시에 처리하는 API

    - 일기 작성 + 이미지 업로드를 한 번에 처리
    - 자동 감정 분석 포함
    - 이미지 실패 시에도 일기는 저장됨
    """
    try:
        # 1. emotion_tags JSON 파싱
        try:
            import json as py_json
            parsed_tags = py_json.loads(emotion_tags) if emotion_tags else []
        except:
            parsed_tags = []

        logger.info(f"📝📸 일기+이미지 작성: {user_id} - {perfume_name}")

        # 2. 이미지 업로드 먼저 처리
        image_url = None
        thumbnail_url = None
        image_filename = None
        image_metadata = {}

        if image:
            is_valid, validation_message = validate_image_file(image)
            if is_valid:
                success, result, metadata = await save_uploaded_image(image, user_id)
                if success:
                    base_url = str(request.base_url)
                    image_url = get_image_url(result, base_url)
                    thumbnail_url = get_thumbnail_url(result, base_url)
                    image_filename = result
                    image_metadata = metadata or {}
                    logger.info(f"✅ 이미지 저장 성공: {result}")
                else:
                    logger.warning(f"⚠️ 이미지 저장 실패: {result}")
            else:
                logger.warning(f"⚠️ 이미지 검증 실패: {validation_message}")

        # 3. 시향 일기 작성
        user = get_default_user()
        now = datetime.now().isoformat()
        diary_id = str(uuid.uuid4())

        # 4. 룰 기반 감정 분석
        initial_analysis = None
        if content and content.strip():
            try:
                initial_analysis = await asyncio.wait_for(
                    rule_based_emotion_analysis(content, perfume_name),
                    timeout=5.0
                )
            except Exception as e:
                logger.error(f"❌ 감정 분석 오류: {e}")
                initial_analysis = {
                    "success": False,
                    "primary_emotion": "중립",
                    "confidence": 0.3,
                    "emotion_tags": ["#neutral"],
                    "analysis_method": "error"
                }

        # 5. 일기 데이터 생성 (이미지 정보 포함)
        diary = {
            "id": diary_id,
            "user_id": user_id,
            "user_name": user_id,
            "user_profile_image": user.get("picture", ""),
            "perfume_id": f"perfume_{perfume_name.lower().replace(' ', '_')}",
            "perfume_name": perfume_name,
            "brand": "Unknown Brand",
            "content": content or "",
            "tags": parsed_tags or [],
            "likes": 0,
            "comments": 0,
            "is_public": is_public,
            "created_at": now,
            "updated_at": now,

            # 🆕 이미지 관련 필드들
            "image_url": image_url,
            "thumbnail_url": thumbnail_url,
            "image_filename": image_filename,
            "image_metadata": image_metadata,

            # 감정 분석 정보
            "emotion_analysis": initial_analysis,
            "primary_emotion": initial_analysis.get("primary_emotion", "중립") if initial_analysis else "중립",
            "emotion_confidence": initial_analysis.get("confidence", 0.0) if initial_analysis else 0.0,
            "emotion_tags_auto": initial_analysis.get("emotion_tags", []) if initial_analysis else [],
            "emotion_analysis_status": "completed" if initial_analysis and initial_analysis.get(
                "success") else "failed",
            "analysis_method": "rule_based"
        }

        # 태그 병합 (사용자 입력 태그 + 자동 분석 태그)
        if initial_analysis and initial_analysis.get("emotion_tags"):
            auto_tags = initial_analysis.get("emotion_tags", [])
            manual_tags = parsed_tags or []
            diary["tags"] = list(set(manual_tags + auto_tags))

        # 6. 저장
        diary_data.append(diary)
        save_diary_data(diary_data)

        return JSONResponse(
            status_code=200,
            content={
                "message": "시향 일기 + 이미지 저장 성공",
                "diary_id": diary_id,
                "user_id": user_id,
                "image_uploaded": image_url is not None,
                "image_url": image_url,
                "thumbnail_url": thumbnail_url,
                "emotion_analysis": {
                    "status": diary["emotion_analysis_status"],
                    "method": "rule_based",
                    "primary_emotion": diary["primary_emotion"],
                    "confidence": diary["emotion_confidence"]
                }
            }
        )

    except Exception as e:
        logger.error(f"❌ 일기+이미지 저장 오류: {e}")
        return JSONResponse(
            status_code=500,
            content={"message": f"저장 중 오류: {str(e)}"}
        )


@router.put("/diaries/{diary_id}/add-image", summary="기존 일기에 이미지 추가")
async def add_image_to_diary(
        diary_id: str,
        request: Request,
        user_id: str = Form(..., description="사용자 ID"),
        image: UploadFile = File(..., description="추가할 이미지")
):
    """기존 시향 일기에 이미지를 추가하는 API"""
    try:
        logger.info(f"📸➕ 기존 일기에 이미지 추가: {diary_id}")

        # 1. 해당 일기 찾기
        diary_data = load_diary_data()
        diary_index = None
        target_diary = None

        for i, diary in enumerate(diary_data):
            if diary.get("id") == diary_id and diary.get("user_id") == user_id:
                diary_index = i
                target_diary = diary
                break

        if not target_diary:
            return JSONResponse(
                status_code=404,
                content={"message": "해당 일기를 찾을 수 없습니다."}
            )

        # 2. 이미지 업로드
        is_valid, message = validate_image_file(image)
        if not is_valid:
            return JSONResponse(
                status_code=400,
                content={"message": message}
            )

        success, result, metadata = await save_uploaded_image(image, user_id)
        if not success:
            return JSONResponse(
                status_code=500,
                content={"message": result}
            )

        # 3. 일기 데이터 업데이트
        base_url = str(request.base_url)
        image_url = get_image_url(result, base_url)
        thumbnail_url = get_thumbnail_url(result, base_url)

        # 기존 이미지 삭제 (선택사항)
        old_filename = target_diary.get("image_filename")
        if old_filename:
            delete_image_files(old_filename)
            logger.info(f"🗑️ 기존 이미지 삭제: {old_filename}")

        # 일기 데이터 업데이트
        diary_data[diary_index]["image_url"] = image_url
        diary_data[diary_index]["thumbnail_url"] = thumbnail_url
        diary_data[diary_index]["image_filename"] = result
        diary_data[diary_index]["image_metadata"] = metadata
        diary_data[diary_index]["updated_at"] = datetime.now().isoformat()

        save_diary_data(diary_data)

        return JSONResponse(
            status_code=200,
            content={
                "message": "일기에 이미지 추가 완료",
                "diary_id": diary_id,
                "image_url": image_url,
                "thumbnail_url": thumbnail_url,
                "filename": result
            }
        )

    except Exception as e:
        logger.error(f"❌ 일기 이미지 추가 오류: {e}")
        return JSONResponse(
            status_code=500,
            content={"message": f"이미지 추가 중 오류: {str(e)}"}
        )


@router.delete("/images/{filename}", summary="이미지 삭제", response_model=ImageDeleteResponse)
async def delete_diary_image(filename: str, user_id: str = Query(..., description="사용자 ID")):
    """업로드된 이미지 파일 삭제"""
    try:
        logger.info(f"🗑️ 이미지 삭제 요청: {filename} by {user_id}")

        # 1. 파일명에서 사용자 ID 확인 (보안)
        if not filename.startswith(user_id[:10]):  # 파일명 앞부분에 사용자ID 포함 확인
            return ImageDeleteResponse(
                success=False,
                message="삭제 권한이 없습니다.",
                deleted_files=[]
            )

        # 2. 일기 데이터에서 해당 이미지 사용하는 일기 찾아서 업데이트
        diary_data = load_diary_data()
        updated = False

        for diary in diary_data:
            if diary.get("image_filename") == filename and diary.get("user_id") == user_id:
                diary["image_url"] = None
                diary["thumbnail_url"] = None
                diary["image_filename"] = None
                diary["image_metadata"] = {}
                diary["updated_at"] = datetime.now().isoformat()
                updated = True
                break

        if updated:
            save_diary_data(diary_data)

        # 3. 실제 파일 삭제
        success = delete_image_files(filename)
        deleted_files = []

        if success:
            deleted_files = [filename, f"thumb_{filename}"]
            message = "이미지 삭제 완료"
        else:
            message = "이미지 파일 삭제 실패"

        return ImageDeleteResponse(
            success=success,
            message=message,
            deleted_files=deleted_files
        )

    except Exception as e:
        logger.error(f"❌ 이미지 삭제 오류: {e}")
        return ImageDeleteResponse(
            success=False,
            message=f"이미지 삭제 중 오류: {str(e)}",
            deleted_files=[]
        )


@router.get("/images/stats", summary="이미지 업로드 통계", response_model=ImageStatsResponse)
async def get_image_upload_stats():
    """업로드된 이미지들의 통계 정보 조회"""
    try:
        stats = get_upload_stats()
        return ImageStatsResponse(
            total_images=stats.get("total_files", 0),
            total_size_mb=stats.get("total_size_mb", 0.0),
            upload_dir=stats.get("upload_dir", "")
        )
    except Exception as e:
        logger.error(f"❌ 이미지 통계 조회 오류: {e}")
        return ImageStatsResponse(
            total_images=0,
            total_size_mb=0.0,
            upload_dir="error"
        )


# ================================
# ✅ 기존 API들 (이미지 필드 추가된 버전)
# ================================

@router.post("/", summary="시향 일기 작성 (텍스트만)")
async def create_diary_entry(
        entry: DiaryCreateRequest = Body(
            ...,
            example={
                "user_id": "john_doe",
                "perfume_name": "Chanel No.5",
                "content": "오늘은 봄바람이 느껴지는 향수와 산책했어요.",
                "is_public": False,
                "emotion_tags": ["calm", "spring"]
            }
        )
):
    """
    시향 일기 작성 (텍스트만, 이미지 없음)

    - 기존 API와 동일하지만 이미지 필드들이 null로 설정됨
    - 별도로 이미지를 추가하고 싶으면 /diaries/{diary_id}/add-image 사용
    """
    try:
        user = get_default_user()
        user_id = entry.user_id if entry.user_id else "anonymous_user"

        now = datetime.now().isoformat()
        diary_id = str(uuid.uuid4())

        logger.info(f"📝 새 일기 작성 (텍스트만): {user_id} - {entry.perfume_name}")

        # 룰 기반 감정 분석
        initial_analysis = None
        if entry.content and entry.content.strip():
            try:
                initial_analysis = await asyncio.wait_for(
                    rule_based_emotion_analysis(entry.content, entry.perfume_name),
                    timeout=5.0
                )
            except Exception as e:
                logger.error(f"❌ 감정 분석 오류: {e}")
                initial_analysis = {
                    "success": False,
                    "primary_emotion": "중립",
                    "confidence": 0.3,
                    "emotion_tags": ["#neutral"],
                    "analysis_method": "error"
                }

        # 일기 데이터 생성 (이미지 필드들은 null)
        diary = {
            "id": diary_id,
            "user_id": user_id,
            "user_name": user_id,
            "user_profile_image": user.get("picture", ""),
            "perfume_id": f"perfume_{entry.perfume_name.lower().replace(' ', '_')}",
            "perfume_name": entry.perfume_name,
            "brand": "Unknown Brand",
            "content": entry.content or "",
            "tags": entry.emotion_tags or [],
            "likes": 0,
            "comments": 0,
            "is_public": entry.is_public,
            "created_at": now,
            "updated_at": now,

            # 🆕 이미지 관련 필드들 (null로 초기화)
            "image_url": None,
            "thumbnail_url": None,
            "image_filename": None,
            "image_metadata": {},

            # 감정 분석 정보
            "emotion_analysis": initial_analysis,
            "primary_emotion": initial_analysis.get("primary_emotion", "중립") if initial_analysis else "중립",
            "emotion_confidence": initial_analysis.get("confidence", 0.0) if initial_analysis else 0.0,
            "emotion_tags_auto": initial_analysis.get("emotion_tags", []) if initial_analysis else [],
            "emotion_analysis_status": "completed" if initial_analysis and initial_analysis.get(
                "success") else "failed",
            "analysis_method": "rule_based"
        }

        # 태그 병합
        if initial_analysis and initial_analysis.get("emotion_tags"):
            auto_tags = initial_analysis.get("emotion_tags", [])
            manual_tags = entry.emotion_tags or []
            diary["tags"] = list(set(manual_tags + auto_tags))

        # 저장
        diary_data.append(diary)
        save_diary_data(diary_data)

        return JSONResponse(
            status_code=200,
            content={
                "message": "시향 일기가 성공적으로 저장되었습니다.",
                "diary_id": diary_id,
                "user_id": user_id,
                "has_image": False,
                "emotion_analysis": {
                    "status": diary["emotion_analysis_status"],
                    "method": "rule_based",
                    "primary_emotion": diary["primary_emotion"],
                    "confidence": diary["emotion_confidence"]
                }
            }
        )

    except Exception as e:
        logger.error(f"❌ 일기 저장 오류: {e}")
        return JSONResponse(
            status_code=500,
            content={"message": f"일기 저장 중 오류: {str(e)}"}
        )


@router.get("/", summary="시향 일기 목록 조회 (이미지 포함)")
async def get_diary_list(
        public: Optional[bool] = Query(None, description="공개 여부 필터"),
        page: Optional[int] = Query(1, description="페이지 번호"),
        size: Optional[int] = Query(10, description="페이지 크기"),
        keyword: Optional[str] = Query(None, description="검색 키워드"),
        has_image: Optional[bool] = Query(None, description="이미지 포함 여부 필터")
):
    try:
        filtered_data = diary_data.copy()

        # 필터링
        if public is not None:
            filtered_data = [d for d in filtered_data if d.get("is_public") == public]

        if keyword:
            filtered_data = [d for d in filtered_data
                             if keyword.lower() in d.get("content", "").lower()
                             or keyword.lower() in d.get("perfume_name", "").lower()]

        # 🆕 이미지 포함 여부 필터
        if has_image is not None:
            if has_image:
                filtered_data = [d for d in filtered_data if d.get("image_url")]
            else:
                filtered_data = [d for d in filtered_data if not d.get("image_url")]

        # 정렬 및 페이징
        filtered_data.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        start = (page - 1) * size
        end = start + size
        paginated_data = filtered_data[start:end]

        # 응답 데이터 변환 (이미지 정보 포함)
        response_data = []
        for item in paginated_data:
            response_data.append({
                "id": item.get("id", ""),
                "user_name": item.get("user_name", "익명"),
                "perfume_name": item.get("perfume_name", ""),
                "content": item.get("content", ""),
                "tags": item.get("tags", []),
                "primary_emotion": item.get("primary_emotion", "중립"),
                "emotion_confidence": item.get("emotion_confidence", 0.0),
                "analysis_method": item.get("analysis_method", "rule_based"),
                "likes": item.get("likes", 0),
                "created_at": item.get("created_at", ""),
                # 🆕 이미지 관련 정보
                "image_url": item.get("image_url"),
                "thumbnail_url": item.get("thumbnail_url"),
                "has_image": bool(item.get("image_url"))
            })

        return BaseResponse(
            message=f"시향 일기 목록 조회 성공 (총 {len(filtered_data)}개)",
            result={
                "diaries": response_data,
                "total_count": len(filtered_data),
                "page": page,
                "size": size,
                "has_next": end < len(filtered_data),
                "analysis_method": "rule_based",
                "image_support": True  # 🆕 이미지 지원 여부
            }
        )

    except Exception as e:
        logger.error(f"❌ 일기 목록 조회 오류: {e}")
        return JSONResponse(
            status_code=500,
            content={"message": f"서버 오류: {str(e)}"}
        )