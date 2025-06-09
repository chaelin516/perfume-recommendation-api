# routers/diary_router.py - 안전한 감정 태깅 연동 완전판

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

# 🎭 감정 분석기 임포트 (안전한 방식)
logger = logging.getLogger("diary_router")

# 감정 분석기 가용성 확인
EMOTION_ANALYZER_AVAILABLE = False
emotion_analyzer = None


def initialize_emotion_analyzer():
    """감정 분석기 안전한 초기화"""
    global EMOTION_ANALYZER_AVAILABLE, emotion_analyzer

    try:
        # 1단계: 모듈 임포트 시도
        from utils.emotion_analyzer import emotion_analyzer as ea
        emotion_analyzer = ea

        # 2단계: 간단한 테스트
        test_result = ea.get_supported_emotions()
        if test_result and len(test_result) > 0:
            EMOTION_ANALYZER_AVAILABLE = True
            logger.info(f"✅ 감정 분석기 로드 성공 - 지원 감정: {test_result}")
            return True
        else:
            raise Exception("지원 감정 목록 획득 실패")

    except ImportError as e:
        logger.warning(f"⚠️ emotion_analyzer 모듈 없음: {e}")
        EMOTION_ANALYZER_AVAILABLE = False
        return False
    except Exception as e:
        logger.warning(f"⚠️ 감정 분석기 초기화 실패: {e}")
        EMOTION_ANALYZER_AVAILABLE = False
        return False


# 초기화 시도
initialize_emotion_analyzer()

router = APIRouter(prefix="/diaries", tags=["Diary"])

# 📂 시향 일기 데이터 파일 경로
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DIARY_PATH = os.path.join(BASE_DIR, "../data/diary_data.json")

# 📊 감정 분석 통계 파일 경로
EMOTION_STATS_PATH = os.path.join(BASE_DIR, "../data/emotion_stats.json")

# 📦 기존 데이터 로딩
if os.path.exists(DIARY_PATH):
    try:
        with open(DIARY_PATH, "r", encoding="utf-8") as f:
            diary_data = json.load(f)
        logger.info(f"✅ 시향 일기 데이터 로딩 완료: {len(diary_data)}개 항목")
    except Exception as e:
        logger.error(f"❌ 시향 일기 데이터 로딩 실패: {e}")
        diary_data = []
else:
    diary_data = []
    logger.info("⚠️ 시향 일기 데이터 파일이 없습니다. 새로 생성됩니다.")


# 📊 감정 분석 통계 로딩
def load_emotion_stats() -> Dict[str, Any]:
    """감정 분석 통계 로드"""
    try:
        if os.path.exists(EMOTION_STATS_PATH):
            with open(EMOTION_STATS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"❌ 감정 통계 로딩 실패: {e}")

    return {
        "total_analyses": 0,
        "successful_analyses": 0,
        "failed_analyses": 0,
        "emotion_distribution": {},
        "average_confidence": 0.0,
        "last_updated": datetime.now().isoformat()
    }


def save_emotion_stats(stats: Dict[str, Any]) -> bool:
    """감정 분석 통계 저장"""
    try:
        stats["last_updated"] = datetime.now().isoformat()
        os.makedirs(os.path.dirname(EMOTION_STATS_PATH), exist_ok=True)
        with open(EMOTION_STATS_PATH, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        logger.error(f"❌ 감정 통계 저장 실패: {e}")
        return False


# 🔒 안전한 감정 분석 함수
async def safe_analyze_emotion(content: str, perfume_name: str = "") -> Dict[str, Any]:
    """
    안전한 감정 분석 함수

    Args:
        content: 분석할 텍스트
        perfume_name: 향수 이름 (컨텍스트용)

    Returns:
        감정 분석 결과 딕셔너리
    """
    analysis_start_time = time.time()

    # 기본 결과 구조
    default_result = {
        "success": False,
        "primary_emotion": "중립",
        "confidence": 0.0,
        "emotion_tags": ["#neutral"],
        "analysis_method": "fallback",
        "processing_time": 0.0,
        "error_message": None
    }

    try:
        # 입력 검증
        if not content or not content.strip():
            default_result.update({
                "success": True,
                "analysis_method": "validation",
                "error_message": "빈 텍스트"
            })
            return default_result

        # 텍스트 길이 제한 (안전장치)
        if len(content) > 2000:
            logger.warning(f"⚠️ 텍스트가 너무 깁니다: {len(content)}자")
            content = content[:2000]

        # 감정 분석기 사용 가능 여부 확인
        if not EMOTION_ANALYZER_AVAILABLE:
            logger.info("📋 감정 분석기 사용 불가, 룰 기반 폴백 사용")
            fallback_result = await fallback_emotion_analysis(content, perfume_name)
            fallback_result["processing_time"] = time.time() - analysis_start_time
            return fallback_result

        # 감정 분석 실행
        logger.info(f"🎭 감정 분석 시작: '{content[:50]}{'...' if len(content) > 50 else ''}'")

        # 향수 이름이 있으면 컨텍스트에 추가
        analysis_text = content
        if perfume_name:
            analysis_text = f"향수 '{perfume_name}'에 대한 후기: {content}"

        # 감정 분석 실행 (타임아웃 설정)
        try:
            analysis_result = await asyncio.wait_for(
                emotion_analyzer.analyze_emotion(analysis_text, use_model=True),
                timeout=10.0  # 10초 타임아웃
            )
        except asyncio.TimeoutError:
            logger.warning("⏰ 감정 분석 타임아웃 (10초)")
            raise Exception("감정 분석 타임아웃")

        # 결과 검증
        if not analysis_result or not analysis_result.get("success"):
            error_msg = analysis_result.get("message", "알 수 없는 오류") if analysis_result else "분석 결과 없음"
            raise Exception(f"감정 분석 실패: {error_msg}")

        # 신뢰도 검증
        confidence = analysis_result.get("confidence", 0.0)
        if confidence < 0.3:
            logger.warning(f"⚠️ 낮은 신뢰도: {confidence:.3f}")

        # 성공 결과 구성
        result = {
            "success": True,
            "primary_emotion": analysis_result.get("primary_emotion", "중립"),
            "confidence": round(confidence, 3),
            "emotion_tags": analysis_result.get("emotion_tags", ["#neutral"]),
            "analysis_method": analysis_result.get("method", "ai_model"),
            "processing_time": round(time.time() - analysis_start_time, 3),
            "analysis_details": analysis_result.get("analysis_details"),
            "analyzer_version": analysis_result.get("analyzer_version")
        }

        logger.info(f"✅ 감정 분석 성공: {result['primary_emotion']} (신뢰도: {result['confidence']:.3f})")
        return result

    except Exception as e:
        logger.error(f"❌ 감정 분석 중 오류: {e}")

        # 안전한 폴백 시도
        try:
            fallback_result = await fallback_emotion_analysis(content, perfume_name)
            fallback_result.update({
                "processing_time": round(time.time() - analysis_start_time, 3),
                "error_message": str(e)
            })
            return fallback_result
        except Exception as fallback_error:
            logger.error(f"❌ 폴백 감정 분석도 실패: {fallback_error}")

            # 최종 안전장치
            default_result.update({
                "processing_time": round(time.time() - analysis_start_time, 3),
                "error_message": f"분석 실패: {str(e)}"
            })
            return default_result


async def fallback_emotion_analysis(content: str, perfume_name: str = "") -> Dict[str, Any]:
    """
    폴백 감정 분석 (간단한 룰 기반)
    """
    logger.info("📋 폴백 감정 분석 시작")

    # 간단한 키워드 기반 감정 분석
    positive_keywords = [
        "좋아", "좋은", "좋네", "마음에 들어", "만족", "완벽", "최고", "사랑", "예뻐",
        "상쾌", "밝은", "화사", "싱그러운", "상큼", "달콤", "포근", "따뜻", "부드러운"
    ]

    negative_keywords = [
        "별로", "안 좋아", "싫어", "이상해", "어색해", "부담스러워", "과해", "독해",
        "자극적", "역겨운", "끔찍", "최악", "실망", "아쉬워"
    ]

    excitement_keywords = [
        "신나", "흥분", "두근", "설렘", "활기", "에너지", "생생한", "활력", "발랄한"
    ]

    calm_keywords = [
        "차분", "편안", "은은", "부드러운", "평온", "고요", "안정", "릴랙스"
    ]

    content_lower = content.lower()

    # 키워드 매칭
    positive_count = sum(1 for keyword in positive_keywords if keyword in content_lower)
    negative_count = sum(1 for keyword in negative_keywords if keyword in content_lower)
    excitement_count = sum(1 for keyword in excitement_keywords if keyword in content_lower)
    calm_count = sum(1 for keyword in calm_keywords if keyword in content_lower)

    # 감정 결정
    if positive_count > negative_count:
        if excitement_count > calm_count:
            primary_emotion = "기쁨"
            emotion_tags = ["#joyful", "#positive", "#happy"]
        else:
            primary_emotion = "기쁨"
            emotion_tags = ["#satisfied", "#positive", "#calm"]
    elif negative_count > positive_count:
        primary_emotion = "상처"
        emotion_tags = ["#disappointed", "#negative"]
    elif excitement_count > 0:
        primary_emotion = "흥분"
        emotion_tags = ["#excited", "#energetic"]
    elif calm_count > 0:
        primary_emotion = "기쁨"  # 차분함을 긍정적으로 해석
        emotion_tags = ["#calm", "#peaceful"]
    else:
        primary_emotion = "중립"
        emotion_tags = ["#neutral"]

    # 신뢰도 계산
    total_keywords = positive_count + negative_count + excitement_count + calm_count
    confidence = min(0.7, 0.3 + (total_keywords * 0.1))  # 최대 70%

    result = {
        "success": True,
        "primary_emotion": primary_emotion,
        "confidence": round(confidence, 3),
        "emotion_tags": emotion_tags,
        "analysis_method": "fallback_rule_based",
        "analysis_details": {
            "positive_count": positive_count,
            "negative_count": negative_count,
            "excitement_count": excitement_count,
            "calm_count": calm_count
        }
    }

    logger.info(f"📋 폴백 분석 완료: {primary_emotion} (신뢰도: {confidence:.3f})")
    return result


def update_emotion_statistics(analysis_result: Dict[str, Any]) -> None:
    """감정 분석 통계 업데이트"""
    try:
        stats = load_emotion_stats()

        stats["total_analyses"] += 1

        if analysis_result.get("success"):
            stats["successful_analyses"] += 1

            # 감정 분포 업데이트
            emotion = analysis_result.get("primary_emotion", "중립")
            stats["emotion_distribution"][emotion] = stats["emotion_distribution"].get(emotion, 0) + 1

            # 평균 신뢰도 업데이트
            confidence = analysis_result.get("confidence", 0.0)
            current_avg = stats.get("average_confidence", 0.0)
            total_successful = stats["successful_analyses"]
            stats["average_confidence"] = round(
                (current_avg * (total_successful - 1) + confidence) / total_successful, 3
            )
        else:
            stats["failed_analyses"] += 1

        # 통계 저장
        save_emotion_stats(stats)

    except Exception as e:
        logger.error(f"❌ 감정 통계 업데이트 실패: {e}")


async def analyze_emotion_in_background(diary_id: str, content: str, perfume_name: str = "") -> None:
    """백그라운드에서 감정 분석 및 업데이트"""
    try:
        logger.info(f"🔄 백그라운드 감정 분석 시작: {diary_id}")

        # 감정 분석 실행
        analysis_result = await safe_analyze_emotion(content, perfume_name)

        # 일기 데이터에서 해당 일기 찾기
        diary_found = False
        for diary in diary_data:
            if diary.get("id") == diary_id:
                # 감정 분석 결과 업데이트
                diary["emotion_analysis"] = analysis_result
                diary["emotion_tags"] = analysis_result.get("emotion_tags", ["#neutral"])
                diary["primary_emotion"] = analysis_result.get("primary_emotion", "중립")
                diary["emotion_confidence"] = analysis_result.get("confidence", 0.0)
                diary["emotion_updated_at"] = datetime.now().isoformat()
                diary_found = True
                break

        if diary_found:
            # 파일에 저장
            with open(DIARY_PATH, "w", encoding="utf-8") as f:
                json.dump(diary_data, f, ensure_ascii=False, indent=2)

            logger.info(f"✅ 백그라운드 감정 분석 완료: {diary_id} -> {analysis_result.get('primary_emotion')}")
        else:
            logger.warning(f"⚠️ 일기를 찾을 수 없음: {diary_id}")

        # 통계 업데이트
        update_emotion_statistics(analysis_result)

    except Exception as e:
        logger.error(f"❌ 백그라운드 감정 분석 실패: {diary_id}, {e}")


# ✅ Firebase 상태 확인 API
@router.get("/firebase-status", summary="Firebase 상태 확인", description="Firebase 인증 서비스 상태를 확인합니다.")
async def check_firebase_status():
    return get_firebase_status()


# ✅ 감정 분석 상태 확인 API
@router.get("/emotion-status", summary="감정 분석 시스템 상태", description="감정 분석 시스템의 상태와 통계를 확인합니다.")
async def check_emotion_status():
    """감정 분석 시스템 상태 확인"""
    try:
        # 기본 상태 정보
        status_info = {
            "emotion_analyzer_available": EMOTION_ANALYZER_AVAILABLE,
            "supported_emotions": [],
            "analysis_statistics": load_emotion_stats(),
            "system_status": "operational" if EMOTION_ANALYZER_AVAILABLE else "fallback_only"
        }

        # 감정 분석기가 사용 가능한 경우 추가 정보
        if EMOTION_ANALYZER_AVAILABLE:
            try:
                status_info["supported_emotions"] = emotion_analyzer.get_supported_emotions()
                status_info["analyzer_stats"] = emotion_analyzer.get_analysis_stats()
            except Exception as e:
                logger.error(f"❌ 감정 분석기 상태 확인 중 오류: {e}")
                status_info["analyzer_error"] = str(e)

        return JSONResponse(content=status_info)

    except Exception as e:
        logger.error(f"❌ 감정 상태 확인 실패: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"감정 상태 확인 중 오류: {str(e)}"}
        )


# ✅ 시향 일기 작성 API (감정 분석 포함)
@router.post("/", summary="시향 일기 작성 (감정 분석 포함)",
             description="사용자가 향수에 대해 작성한 시향 일기를 저장하고 자동으로 감정을 분석합니다.")
async def write_diary(entry: DiaryCreateRequest, background_tasks: BackgroundTasks,
                      user=Depends(verify_firebase_token_optional)):
    try:
        user_id = user["uid"]
        now = datetime.now().isoformat()
        diary_id = str(uuid.uuid4())

        logger.info(f"📝 새 일기 작성 시작: {user.get('name', '익명')} - {entry.perfume_name}")

        # 초기 감정 분석 (빠른 분석)
        initial_analysis = None
        if entry.content and entry.content.strip():
            try:
                # 빠른 초기 분석 (3초 타임아웃)
                initial_analysis = await asyncio.wait_for(
                    safe_analyze_emotion(entry.content, entry.perfume_name),
                    timeout=3.0
                )
                logger.info(f"✅ 초기 감정 분석 완료: {initial_analysis.get('primary_emotion')}")
            except asyncio.TimeoutError:
                logger.warning("⏰ 초기 감정 분석 타임아웃, 백그라운드에서 처리")
                initial_analysis = None
            except Exception as e:
                logger.warning(f"⚠️ 초기 감정 분석 실패: {e}, 백그라운드에서 재시도")
                initial_analysis = None

        # 새 일기 항목 생성
        diary = {
            "id": diary_id,
            "user_id": user_id,
            "user_name": user.get("name", "익명 사용자"),
            "user_profile_image": user.get("picture", ""),
            "perfume_id": f"perfume_{entry.perfume_name.lower().replace(' ', '_')}",
            "perfume_name": entry.perfume_name,
            "brand": "Dummy Brand",  # 실제 브랜드 연동 필요
            "content": entry.content or "",
            "tags": entry.emotion_tags or [],
            "likes": 0,
            "comments": 0,
            "is_public": entry.is_public,
            "created_at": now,
            "updated_at": now,

            # 감정 분석 관련 필드
            "emotion_analysis": initial_analysis,
            "primary_emotion": initial_analysis.get("primary_emotion", "분석중") if initial_analysis else "분석중",
            "emotion_confidence": initial_analysis.get("confidence", 0.0) if initial_analysis else 0.0,
            "emotion_tags_auto": initial_analysis.get("emotion_tags", []) if initial_analysis else [],
            "emotion_updated_at": now if initial_analysis else None,
            "emotion_analysis_status": "completed" if initial_analysis else "pending"
        }

        # 자동 생성된 감정 태그를 기존 태그와 병합
        if initial_analysis and initial_analysis.get("emotion_tags"):
            auto_emotion_tags = initial_analysis.get("emotion_tags", [])
            manual_tags = entry.emotion_tags or []
            # 중복 제거하면서 병합
            combined_tags = list(set(manual_tags + auto_emotion_tags))
            diary["tags"] = combined_tags

        diary_data.append(diary)

        # 파일에 저장
        os.makedirs(os.path.dirname(DIARY_PATH), exist_ok=True)
        with open(DIARY_PATH, "w", encoding="utf-8") as f:
            json.dump(diary_data, f, ensure_ascii=False, indent=2)

        # 백그라운드에서 정밀 감정 분석 (초기 분석이 실패했거나 신뢰도가 낮은 경우)
        if not initial_analysis or initial_analysis.get("confidence", 0) < 0.5:
            if entry.content and entry.content.strip():
                background_tasks.add_task(
                    analyze_emotion_in_background,
                    diary_id,
                    entry.content,
                    entry.perfume_name
                )
                logger.info(f"🔄 백그라운드 정밀 감정 분석 예약: {diary_id}")

        logger.info(f"[DIARY] 새 일기 저장됨: {user.get('name', '익명')} - {entry.perfume_name}")

        return JSONResponse(
            status_code=200,
            content={
                "message": "시향 일기가 성공적으로 저장되었습니다.",
                "diary_id": diary["id"],
                "emotion_analysis": {
                    "status": diary["emotion_analysis_status"],
                    "primary_emotion": diary["primary_emotion"],
                    "confidence": diary["emotion_confidence"],
                    "processing_note": "백그라운드에서 정밀 분석 중입니다." if diary[
                                                                    "emotion_analysis_status"] == "pending" else "분석 완료"
                }
            }
        )

    except Exception as e:
        logger.error(f"❌ 일기 저장 중 오류: {e}")
        return JSONResponse(
            status_code=500,
            content={"message": f"일기 저장 중 오류가 발생했습니다: {str(e)}"}
        )


# ✅ 감정 태그 수동 업데이트 API
@router.patch("/{diary_id}/emotion-tags", summary="감정 태그 수동 업데이트",
              description="일기의 감정 태그를 사용자가 수동으로 수정합니다.")
async def update_emotion_tags(
        diary_id: str,
        emotion_tags: List[str],
        user=Depends(verify_firebase_token_optional)
):
    """감정 태그 수동 업데이트"""
    try:
        user_id = user["uid"]

        # 일기 찾기 및 권한 확인
        diary_found = False
        for diary in diary_data:
            if diary.get("id") == diary_id:
                if diary.get("user_id") != user_id:
                    raise HTTPException(status_code=403, detail="수정 권한이 없습니다.")

                # 감정 태그 업데이트
                diary["tags"] = emotion_tags
                diary["emotion_tags_manual"] = emotion_tags
                diary["emotion_manual_updated_at"] = datetime.now().isoformat()
                diary["updated_at"] = datetime.now().isoformat()
                diary_found = True
                break

        if not diary_found:
            raise HTTPException(status_code=404, detail="해당 일기를 찾을 수 없습니다.")

        # 파일에 저장
        with open(DIARY_PATH, "w", encoding="utf-8") as f:
            json.dump(diary_data, f, ensure_ascii=False, indent=2)

        logger.info(f"✅ 감정 태그 수동 업데이트: {diary_id} -> {emotion_tags}")

        return JSONResponse(
            content={
                "message": "감정 태그가 성공적으로 업데이트되었습니다.",
                "diary_id": diary_id,
                "updated_tags": emotion_tags
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 감정 태그 업데이트 중 오류: {e}")
        raise HTTPException(status_code=500, detail=f"감정 태그 업데이트 중 오류: {str(e)}")


# ✅ 감정 재분석 API
@router.post("/{diary_id}/reanalyze-emotion", summary="감정 재분석",
             description="특정 일기의 감정을 다시 분석합니다.")
async def reanalyze_emotion(
        diary_id: str,
        background_tasks: BackgroundTasks,
        user=Depends(verify_firebase_token_optional)
):
    """감정 재분석"""
    try:
        user_id = user["uid"]

        # 일기 찾기 및 권한 확인
        target_diary = None
        for diary in diary_data:
            if diary.get("id") == diary_id:
                if diary.get("user_id") != user_id:
                    raise HTTPException(status_code=403, detail="재분석 권한이 없습니다.")
                target_diary = diary
                break

        if not target_diary:
            raise HTTPException(status_code=404, detail="해당 일기를 찾을 수 없습니다.")

        content = target_diary.get("content", "")
        if not content.strip():
            raise HTTPException(status_code=400, detail="분석할 내용이 없습니다.")

        # 재분석 상태 업데이트
        target_diary["emotion_analysis_status"] = "reanalyzing"
        target_diary["emotion_updated_at"] = datetime.now().isoformat()

        # 파일에 저장
        with open(DIARY_PATH, "w", encoding="utf-8") as f:
            json.dump(diary_data, f, ensure_ascii=False, indent=2)

        # 백그라운드에서 재분석
        background_tasks.add_task(
            analyze_emotion_in_background,
            diary_id,
            content,
            target_diary.get("perfume_name", "")
        )

        logger.info(f"🔄 감정 재분석 요청: {diary_id}")

        return JSONResponse(
            content={
                "message": "감정 재분석이 시작되었습니다.",
                "diary_id": diary_id,
                "status": "reanalyzing",
                "estimated_time": "30초 이내"
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 감정 재분석 요청 중 오류: {e}")
        raise HTTPException(status_code=500, detail=f"감정 재분석 요청 중 오류: {str(e)}")


# ✅ 시향 일기 목록 조회 API (감정 필터링 포함)
@router.get("/", summary="시향 일기 목록 조회 (감정 필터링 포함)",
            description="저장된 모든 시향 일기를 반환하며, 감정 기반 필터링을 지원합니다.",
            response_model=BaseResponse, response_model_by_alias=True)
async def get_diary_list(
        public: Optional[bool] = Query(None, description="공개 여부 필터 (true/false)"),
        date_filter: Optional[date] = Query(None, description="작성 날짜 필터 (YYYY-MM-DD)"),
        sort: Optional[str] = Query("desc", description="정렬 방식 (desc: 최신순, asc: 오래된순)"),
        page: Optional[int] = Query(1, description="페이지 번호 (1부터 시작)"),
        size: Optional[int] = Query(10, description="페이지 당 항목 수"),
        keyword: Optional[str] = Query(None, description="내용 또는 향수명 키워드 검색"),
        emotion: Optional[str] = Query(None, description="감정 태그 필터링"),
        primary_emotion: Optional[str] = Query(None, description="주요 감정 필터링"),
        min_confidence: Optional[float] = Query(None, description="최소 감정 신뢰도 (0.0-1.0)", ge=0.0, le=1.0)
):
    try:
        filtered_data = []

        for diary in diary_data:
            # 공개 여부 필터
            if public is not None and diary.get("is_public") != public:
                continue

            # 날짜 필터
            if date_filter is not None:
                created_str = diary.get("created_at", "")
                if "T" in created_str:
                    diary_date = created_str.split("T")[0]
                    if diary_date != date_filter.isoformat():
                        continue

            # 키워드 검색
            if keyword:
                content_match = keyword.lower() in diary.get("content", "").lower()
                perfume_match = keyword.lower() in diary.get("perfume_name", "").lower()
                if not (content_match or perfume_match):
                    continue

            # 감정 태그 필터
            if emotion:
                tags = diary.get("tags", [])
                if isinstance(tags, list):
                    if emotion.lower() not in [tag.lower() for tag in tags]:
                        continue
                else:
                    if emotion.lower() not in str(tags).lower():
                        continue

            # 주요 감정 필터
            if primary_emotion:
                diary_emotion = diary.get("primary_emotion", "")
                if primary_emotion.lower() not in diary_emotion.lower():
                    continue

            # 감정 신뢰도 필터
            if min_confidence is not None:
                confidence = diary.get("emotion_confidence", 0.0)
                if confidence < min_confidence:
                    continue

            filtered_data.append(diary)

        # 정렬
        reverse = sort != "asc"
        filtered_data.sort(
            key=lambda x: x.get("created_at") or "1970-01-01T00:00:00",
            reverse=reverse
        )

        # 페이지네이션
        start = (page - 1) * size
        end = start + size
        paginated_data = filtered_data[start:end]

        # 응답 데이터 변환 (감정 정보 포함)
        response_data = []
        for item in paginated_data:
            try:
                diary_item = {
                    "id": item.get("id", ""),
                    "user_id": item.get("user_id", ""),
                    "user_name": item.get("user_name", "익명"),
                    "user_profile_image": item.get("user_profile_image", ""),
                    "perfume_id": item.get("perfume_id", ""),
                    "perfume_name": item.get("perfume_name", ""),
                    "brand": item.get("brand", "Unknown"),
                    "content": item.get("content", ""),
                    "tags": item.get("tags", []),
                    "likes": item.get("likes", 0),
                    "comments": item.get("comments", 0),
                    "created_at": item.get("created_at", ""),
                    "updated_at": item.get("updated_at", ""),

                    # 감정 분석 정보
                    "primary_emotion": item.get("primary_emotion", "중립"),
                    "emotion_confidence": item.get("emotion_confidence", 0.0),
                    "emotion_tags_auto": item.get("emotion_tags_auto", []),
                    "emotion_analysis_status": item.get("emotion_analysis_status", "unknown"),
                    "emotion_updated_at": item.get("emotion_updated_at")
                }
                response_data.append(diary_item)
            except Exception as e:
                logger.warning(f"⚠️ 일기 항목 변환 오류: {e}")
                continue

        return BaseResponse(
            message=f"시향 일기 목록 조회 성공 (총 {len(filtered_data)}개, 페이지: {page})",
            result={
                "diaries": response_data,
                "total_count": len(filtered_data),
                "page": page,
                "size": size,
                "has_next": end < len(filtered_data),
                "emotion_filters_applied": {
                    "emotion_tag": emotion,
                    "primary_emotion": primary_emotion,
                    "min_confidence": min_confidence
                }
            }
        )

    except Exception as e:
        logger.error(f"❌ 일기 목록 조회 오류: {e}")
        return JSONResponse(
            status_code=500,
            content={"message": f"서버 오류: {str(e)}"}
        )


# ✅ 감정 분석 통계 API
@router.get("/emotion-statistics", summary="감정 분석 통계",
            description="전체 일기의 감정 분석 통계를 반환합니다.")
async def get_emotion_statistics():
    """감정 분석 통계"""
    try:
        stats = load_emotion_stats()

        # 현재 일기 데이터에서 실시간 통계 계산
        current_emotions = []
        successful_analyses = 0
        total_confidence = 0.0

        for diary in diary_data:
            if diary.get("emotion_analysis_status") == "completed":
                emotion = diary.get("primary_emotion")
                confidence = diary.get("emotion_confidence", 0.0)

                if emotion and emotion != "분석중":
                    current_emotions.append(emotion)
                    successful_analyses += 1
                    total_confidence += confidence

        # 현재 감정 분포
        current_distribution = dict(Counter(current_emotions))

        # 평균 신뢰도
        avg_confidence = (total_confidence / successful_analyses) if successful_analyses > 0 else 0.0

        return JSONResponse(
            content={
                "overview": {
                    "total_diaries": len(diary_data),
                    "analyzed_diaries": successful_analyses,
                    "analysis_success_rate": round((successful_analyses / len(diary_data) * 100),
                                                   2) if diary_data else 0,
                    "average_confidence": round(avg_confidence, 3)
                },
                "current_emotion_distribution": current_distribution,
                "historical_statistics": stats,
                "top_emotions": sorted(current_distribution.items(), key=lambda x: x[1], reverse=True)[:5],
                "emotion_analyzer_status": {
                    "available": EMOTION_ANALYZER_AVAILABLE,
                    "mode": "AI + Fallback" if EMOTION_ANALYZER_AVAILABLE else "Fallback Only"
                }
            }
        )

    except Exception as e:
        logger.error(f"❌ 감정 통계 조회 오류: {e}")
        return JSONResponse(
            status_code=500,
            content={"message": f"감정 통계 조회 중 오류: {str(e)}"}
        )


# ✅ 테스트 감정 분석 API
@router.post("/test-emotion-analysis", summary="감정 분석 테스트",
             description="텍스트에 대한 감정 분석을 테스트합니다.")
async def test_emotion_analysis(
        content: str,
        perfume_name: Optional[str] = None
):
    """감정 분석 테스트"""
    try:
        if not content.strip():
            raise HTTPException(status_code=400, detail="분석할 텍스트가 비어있습니다.")

        logger.info(f"🧪 감정 분석 테스트: '{content[:50]}{'...' if len(content) > 50 else ''}'")

        # 감정 분석 실행
        analysis_result = await safe_analyze_emotion(content, perfume_name or "")

        return JSONResponse(
            content={
                "input": {
                    "content": content,
                    "perfume_name": perfume_name,
                    "content_length": len(content)
                },
                "analysis_result": analysis_result,
                "system_info": {
                    "analyzer_available": EMOTION_ANALYZER_AVAILABLE,
                    "analysis_method": analysis_result.get("analysis_method"),
                    "processing_time": analysis_result.get("processing_time")
                }
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 감정 분석 테스트 오류: {e}")
        raise HTTPException(status_code=500, detail=f"감정 분석 테스트 중 오류: {str(e)}")


# ✅ 기존 API들 (좋아요, 사용자별 조회 등) - 그대로 유지
@router.post("/{diary_id}/like", summary="시향 일기 좋아요 추가")
async def like_diary(diary_id: str):
    try:
        found = False
        for diary in diary_data:
            if diary.get("id") == diary_id:
                diary["likes"] = diary.get("likes", 0) + 1
                diary["updated_at"] = datetime.now().isoformat()
                found = True
                break

        if not found:
            return JSONResponse(status_code=404, content={"message": "해당 일기를 찾을 수 없습니다."})

        with open(DIARY_PATH, "w", encoding="utf-8") as f:
            json.dump(diary_data, f, ensure_ascii=False, indent=2)

        return JSONResponse(status_code=200, content={"message": "좋아요가 추가되었습니다."})

    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"좋아요 처리 중 오류: {str(e)}"})


@router.delete("/{diary_id}/unlike", summary="시향 일기 좋아요 취소")
async def unlike_diary(diary_id: str):
    try:
        found = False
        for diary in diary_data:
            if diary.get("id") == diary_id:
                diary["likes"] = max(0, diary.get("likes", 0) - 1)
                diary["updated_at"] = datetime.now().isoformat()
                found = True
                break

        if not found:
            return JSONResponse(status_code=404, content={"message": "해당 일기를 찾을 수 없습니다."})

        with open(DIARY_PATH, "w", encoding="utf-8") as f:
            json.dump(diary_data, f, ensure_ascii=False, indent=2)

        return JSONResponse(status_code=200, content={"message": "좋아요가 취소되었습니다."})

    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"좋아요 취소 처리 중 오류: {str(e)}"})


@router.get("/user/{user_id}", summary="사용자별 일기 조회")
async def get_user_diaries(user_id: str, public_only: bool = Query(True, description="공개 일기만 조회할지 여부")):
    try:
        user_diaries = []

        for diary in diary_data:
            if diary.get("user_id") == user_id:
                if public_only and not diary.get("is_public", False):
                    continue
                user_diaries.append(diary)

        user_diaries.sort(key=lambda x: x.get("created_at", ""), reverse=True)

        return JSONResponse(
            status_code=200,
            content={
                "message": f"사용자 {user_id}의 일기 조회 완료",
                "data": user_diaries,
                "count": len(user_diaries)
            }
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"사용자 일기 조회 중 오류: {str(e)}"}
        )


@router.get("/status", summary="일기 시스템 상태", description="일기 시스템의 전체 상태를 확인합니다.")
async def get_diary_system_status():
    emotion_stats = load_emotion_stats()

    return {
        "diary_system": {
            "diary_count": len(diary_data),
            "diary_file_exists": os.path.exists(DIARY_PATH),
            "diary_file_path": DIARY_PATH
        },
        "emotion_system": {
            "analyzer_available": EMOTION_ANALYZER_AVAILABLE,
            "emotion_stats": emotion_stats,
            "stats_file_exists": os.path.exists(EMOTION_STATS_PATH)
        },
        "firebase_status": get_firebase_status(),
        "message": "일기 시스템 상태 정보입니다."
    }