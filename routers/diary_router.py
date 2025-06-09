# routers/diary_router.py - 긴급 수정: UserPreferences 완전 제거

from fastapi import APIRouter, Query, Depends, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from schemas.diary import DiaryCreateRequest, DiaryResponse
from schemas.common import BaseResponse
from datetime import datetime, date
from typing import Optional, List, Dict, Any
from utils.auth_utils import verify_firebase_token_optional, get_firebase_status
import os
import json
import uuid
import logging
import asyncio
import time
from collections import Counter
from pydantic import BaseModel, Field

# 🎭 로거 설정
logger = logging.getLogger("diary_router")

# 🎭 안전한 감정 분석기 임포트
EMOTION_ANALYZER_AVAILABLE = False
emotion_analyzer = None


def safe_import_emotion_analyzer():
    """감정 분석기 안전한 임포트"""
    global EMOTION_ANALYZER_AVAILABLE, emotion_analyzer

    try:
        from utils.emotion_analyzer import emotion_analyzer as ea
        test_emotions = ea.get_supported_emotions()
        if test_emotions and len(test_emotions) > 0:
            emotion_analyzer = ea
            EMOTION_ANALYZER_AVAILABLE = True
            logger.info(f"✅ 감정 분석기 로드 성공: {len(test_emotions)}개 감정 지원")
            return True
    except Exception as e:
        logger.warning(f"⚠️ 감정 분석기 로드 실패: {e}")

    EMOTION_ANALYZER_AVAILABLE = False
    logger.info("📋 기본 감정 분석 모드로 동작합니다")
    return False


# 초기화 시도
safe_import_emotion_analyzer()

# ✅ 라우터 생성
router = APIRouter(prefix="/diaries", tags=["Diary"])

# 📂 데이터 파일 경로
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DIARY_PATH = os.path.join(BASE_DIR, "../data/diary_data.json")

# 기본 감정 태그
DEFAULT_EMOTION_TAGS = {
    "positive": ["#happy", "#joyful", "#pleasant", "#nice"],
    "negative": ["#sad", "#disappointed", "#unpleasant", "#bad"],
    "neutral": ["#neutral", "#normal", "#okay"]
}


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
        return True
    except Exception as e:
        logger.error(f"❌ 시향 일기 데이터 저장 실패: {e}")
        return False


# 전역 데이터
diary_data = load_diary_data()


async def fallback_emotion_analysis(text: str, perfume_name: str = "") -> Dict[str, Any]:
    """폴백 감정 분석 (키워드 기반)"""
    text_lower = text.lower()
    positive_keywords = ["좋", "훌륭", "향긋", "달콤", "상큼", "깔끔", "사랑", "완벽", "최고"]
    negative_keywords = ["싫", "별로", "이상", "안좋", "실망", "아쉬", "후회"]

    positive_score = sum(1 for keyword in positive_keywords if keyword in text_lower)
    negative_score = sum(1 for keyword in negative_keywords if keyword in text_lower)

    if positive_score > negative_score:
        primary_emotion = "기쁨"
        emotion_tags = DEFAULT_EMOTION_TAGS["positive"]
        confidence = min(0.7, 0.4 + positive_score * 0.1)
    elif negative_score > positive_score:
        primary_emotion = "실망"
        emotion_tags = DEFAULT_EMOTION_TAGS["negative"]
        confidence = min(0.7, 0.4 + negative_score * 0.1)
    else:
        primary_emotion = "중립"
        emotion_tags = DEFAULT_EMOTION_TAGS["neutral"]
        confidence = 0.3

    return {
        "success": True,
        "primary_emotion": primary_emotion,
        "confidence": confidence,
        "emotion_tags": emotion_tags,
        "analysis_method": "fallback_keyword_based",
        "processing_time": 0.001
    }


async def safe_analyze_emotion(text: str, perfume_name: str = "") -> Dict[str, Any]:
    """안전한 감정 분석"""
    if EMOTION_ANALYZER_AVAILABLE and emotion_analyzer:
        try:
            result = await emotion_analyzer.analyze_emotion(text, use_model=False)
            if result.get("success"):
                return result
        except Exception as e:
            logger.warning(f"⚠️ AI 감정 분석 실패, 폴백 사용: {e}")

    return await fallback_emotion_analysis(text, perfume_name)


# ✅ API 엔드포인트들

@router.get("/emotion-status", summary="감정 분석 시스템 상태")
async def check_emotion_status():
    """감정 분석 시스템 상태 확인"""
    return JSONResponse(content={
        "emotion_analyzer_available": EMOTION_ANALYZER_AVAILABLE,
        "supported_emotions": emotion_analyzer.get_supported_emotions() if EMOTION_ANALYZER_AVAILABLE else ["기쁨", "실망",
                                                                                                            "중립"],
        "system_status": "ai_available" if EMOTION_ANALYZER_AVAILABLE else "fallback_only",
        "fallback_method": "keyword_based"
    })


@router.post("/", summary="시향 일기 작성")
async def write_diary(
        entry: DiaryCreateRequest,
        background_tasks: BackgroundTasks,
        user=Depends(verify_firebase_token_optional)
):
    try:
        user_id = user["uid"]
        now = datetime.now().isoformat()
        diary_id = str(uuid.uuid4())

        logger.info(f"📝 새 일기 작성: {user.get('name', '익명')} - {entry.perfume_name}")

        # 감정 분석
        initial_analysis = None
        if entry.content and entry.content.strip():
            try:
                initial_analysis = await asyncio.wait_for(
                    safe_analyze_emotion(entry.content, entry.perfume_name),
                    timeout=3.0
                )
            except:
                initial_analysis = await fallback_emotion_analysis(entry.content, entry.perfume_name)

        # 일기 데이터 생성
        diary = {
            "id": diary_id,
            "user_id": user_id,
            "user_name": user.get("name", "익명 사용자"),
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

            # 감정 분석 정보
            "emotion_analysis": initial_analysis,
            "primary_emotion": initial_analysis.get("primary_emotion", "중립") if initial_analysis else "중립",
            "emotion_confidence": initial_analysis.get("confidence", 0.0) if initial_analysis else 0.0,
            "emotion_tags_auto": initial_analysis.get("emotion_tags", []) if initial_analysis else [],
            "emotion_analysis_status": "completed" if initial_analysis else "no_analysis"
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
                "emotion_analysis": {
                    "status": diary["emotion_analysis_status"],
                    "primary_emotion": diary["primary_emotion"],
                    "confidence": diary["emotion_confidence"],
                    "analyzer_available": EMOTION_ANALYZER_AVAILABLE
                }
            }
        )

    except Exception as e:
        logger.error(f"❌ 일기 저장 오류: {e}")
        return JSONResponse(
            status_code=500,
            content={"message": f"일기 저장 중 오류: {str(e)}"}
        )


@router.get("/", summary="시향 일기 목록 조회")
async def get_diary_list(
        public: Optional[bool] = Query(None, description="공개 여부 필터"),
        page: Optional[int] = Query(1, description="페이지 번호"),
        size: Optional[int] = Query(10, description="페이지 크기"),
        keyword: Optional[str] = Query(None, description="검색 키워드")
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

        # 정렬 및 페이징
        filtered_data.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        start = (page - 1) * size
        end = start + size
        paginated_data = filtered_data[start:end]

        # 응답 데이터 변환
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
                "likes": item.get("likes", 0),
                "created_at": item.get("created_at", "")
            })

        return BaseResponse(
            message=f"시향 일기 목록 조회 성공 (총 {len(filtered_data)}개)",
            result={
                "diaries": response_data,
                "total_count": len(filtered_data),
                "page": page,
                "size": size,
                "has_next": end < len(filtered_data)
            }
        )

    except Exception as e:
        logger.error(f"❌ 일기 목록 조회 오류: {e}")
        return JSONResponse(
            status_code=500,
            content={"message": f"서버 오류: {str(e)}"}
        )


@router.post("/{diary_id}/like", summary="시향 일기 좋아요")
async def like_diary(diary_id: str):
    try:
        for diary in diary_data:
            if diary.get("id") == diary_id:
                diary["likes"] = diary.get("likes", 0) + 1
                diary["updated_at"] = datetime.now().isoformat()
                save_diary_data(diary_data)
                return JSONResponse(content={"message": "좋아요가 추가되었습니다."})

        return JSONResponse(status_code=404, content={"message": "일기를 찾을 수 없습니다."})

    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"오류: {str(e)}"})


@router.delete("/{diary_id}/unlike", summary="시향 일기 좋아요 취소")
async def unlike_diary(diary_id: str):
    try:
        for diary in diary_data:
            if diary.get("id") == diary_id:
                diary["likes"] = max(0, diary.get("likes", 0) - 1)
                diary["updated_at"] = datetime.now().isoformat()
                save_diary_data(diary_data)
                return JSONResponse(content={"message": "좋아요가 취소되었습니다."})

        return JSONResponse(status_code=404, content={"message": "일기를 찾을 수 없습니다."})

    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"오류: {str(e)}"})


# 🎯 초기화 로깅
logger.info("✅ 시향 일기 라우터 초기화 완료")
logger.info(f"  - 감정 분석기 사용 가능: {'✅' if EMOTION_ANALYZER_AVAILABLE else '❌'}")
logger.info(f"  - 기존 일기 데이터: {len(diary_data)}개")