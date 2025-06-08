# routers/diary_router.py - 감정 태깅 연동 버전

from fastapi import APIRouter, Query, Depends, HTTPException
from fastapi.responses import JSONResponse
from schemas.diary import DiaryCreateRequest, DiaryResponse
from schemas.common import BaseResponse
from datetime import datetime, date
from typing import Optional
from utils.auth_utils import verify_firebase_token_optional, get_firebase_status

import os, json, uuid, logging

# 🆕 감정 태깅 연동을 위한 임포트
try:
    from routers.emotion_tagging_router import get_emotion_tagger

    EMOTION_TAGGING_AVAILABLE = True
    logger = logging.getLogger("diary_router")
    logger.info("✅ 감정 태깅 모듈 사용 가능")
except ImportError as e:
    EMOTION_TAGGING_AVAILABLE = False
    logger = logging.getLogger("diary_router")
    logger.warning(f"⚠️ 감정 태깅 모듈 없음: {e}")

router = APIRouter(prefix="/diaries", tags=["Diary"])

# 📂 시향 일기 데이터 파일 경로
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DIARY_PATH = os.path.join(BASE_DIR, "../data/diary_data.json")

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


# 🆕 감정 태깅 함수
async def analyze_diary_emotion(content: str):
    """시향 일기 내용의 감정을 분석합니다."""
    if not EMOTION_TAGGING_AVAILABLE or not content or not content.strip():
        # 감정 태깅을 사용할 수 없거나 내용이 없으면 기본값 반환
        return {
            "emotion": "기쁨",
            "confidence": 0.0,
            "label": 0,
            "method": "기본값 (감정 태깅 불가)"
        }

    try:
        # 감정 태깅 인스턴스 가져오기
        emotion_tagger = get_emotion_tagger()

        # 감정 예측 수행
        result = emotion_tagger.predict_emotion(content, include_probabilities=False)

        logger.info(f"🎭 일기 감정 분석 완료: '{content[:30]}...' → {result['emotion']} ({result['confidence']:.3f})")

        return result

    except Exception as e:
        logger.error(f"❌ 일기 감정 분석 실패: {e}")
        # 실패 시 기본값 반환
        return {
            "emotion": "기쁨",
            "confidence": 0.0,
            "label": 0,
            "method": "기본값 (분석 실패)",
            "error": str(e)
        }


# ✅ Firebase 상태 확인 API
@router.get("/firebase-status", summary="Firebase 상태 확인", description="Firebase 인증 서비스 상태를 확인합니다.")
async def check_firebase_status():
    return get_firebase_status()


# ✅ 시향 일기 작성 API (감정 태깅 자동 연동)
@router.post(
    "/",
    summary="시향 일기 작성 (감정 태깅 자동 적용)",
    description=(
            "📝 **시향 일기를 작성하고 AI가 자동으로 감정을 분석합니다**\n\n"
            "**🎭 자동 감정 분석:**\n"
            "- 일기 내용을 AI 모델로 분석하여 8가지 감정 중 하나를 자동 태깅\n"
            "- 감정: 기쁨, 불안, 당황, 분노, 상처, 슬픔, 우울, 흥분\n"
            "- AI 모델 실패 시 룰 기반 분석으로 폴백\n\n"
            "**📊 저장되는 정보:**\n"
            "- 기존 일기 정보 (사용자, 향수명, 내용 등)\n"
            "- 🆕 AI가 분석한 감정 태그 및 신뢰도\n"
            "- 🆕 감정 분석 방법 정보\n\n"
            "**💡 사용법:**\n"
            "- 기존과 동일하게 일기 작성\n"
            "- AI가 자동으로 감정을 분석하여 태그 추가\n"
            "- 프론트엔드에서 emotion_tags 필드로 감정 정보 활용"
    )
)
async def write_diary(entry: DiaryCreateRequest, user=Depends(verify_firebase_token_optional)):
    try:
        user_id = user["uid"]
        user_name = user.get("name", "익명 사용자")
        now = datetime.now().isoformat()

        # 🎭 감정 분석 수행 (AI 모델 또는 룰 기반)
        emotion_analysis = await analyze_diary_emotion(entry.content or "")

        # 🆕 감정 태그 배열 생성 (기존 emotion_tags + AI 분석 결과)
        emotion_tags = entry.emotion_tags or []

        # AI가 분석한 감정을 태그로 추가 (중복 제거)
        ai_emotion = emotion_analysis.get("emotion", "기쁨")
        if ai_emotion not in emotion_tags:
            emotion_tags.append(ai_emotion)

        # 새 일기 항목 생성
        diary = {
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "user_name": user_name,
            "user_profile_image": user.get("picture", ""),
            "perfume_id": f"perfume_{entry.perfume_name.lower().replace(' ', '_')}",
            "perfume_name": entry.perfume_name,
            "brand": "Dummy Brand",  # 실제 브랜드 연동 필요
            "content": entry.content or "",
            "tags": emotion_tags,  # 기존 + AI 감정 태그
            "likes": 0,
            "comments": 0,
            "is_public": entry.is_public,
            "created_at": now,
            "updated_at": now,

            # 🆕 감정 분석 결과 추가
            "ai_emotion_analysis": {
                "predicted_emotion": emotion_analysis.get("emotion"),
                "confidence": emotion_analysis.get("confidence", 0.0),
                "emotion_label": emotion_analysis.get("label", 0),
                "analysis_method": emotion_analysis.get("method"),
                "analyzed_at": now,
                "error": emotion_analysis.get("error")  # 분석 실패 시 에러 정보
            }
        }

        diary_data.append(diary)

        # 파일에 저장
        os.makedirs(os.path.dirname(DIARY_PATH), exist_ok=True)
        with open(DIARY_PATH, "w", encoding="utf-8") as f:
            json.dump(diary_data, f, ensure_ascii=False, indent=2)

        logger.info(f"[DIARY] 새 일기 저장됨: {user_name} - {entry.perfume_name}")
        logger.info(f"[EMOTION] AI 분석 결과: {ai_emotion} (신뢰도: {emotion_analysis.get('confidence', 0):.3f})")

        return JSONResponse(
            status_code=200,
            content={
                "message": "시향 일기가 성공적으로 저장되었습니다.",
                "diary_id": diary["id"],
                "emotion_analysis": {  # 🆕 감정 분석 결과를 클라이언트에 반환
                    "predicted_emotion": ai_emotion,
                    "confidence": emotion_analysis.get("confidence", 0.0),
                    "method": emotion_analysis.get("method"),
                    "all_tags": emotion_tags
                }
            }
        )

    except Exception as e:
        logger.error(f"❌ 일기 저장 중 오류: {e}")
        return JSONResponse(
            status_code=500,
            content={"message": f"일기 저장 중 오류가 발생했습니다: {str(e)}"}
        )


# ✅ 시향 일기 목록 조회 API (감정 분석 정보 포함)
@router.get(
    "/",
    summary="시향 일기 목록 조회 (감정 정보 포함)",
    description="저장된 모든 시향 일기를 감정 분석 정보와 함께 반환합니다.",
    response_model=BaseResponse,
    response_model_by_alias=True
)
async def get_diary_list(
        public: Optional[bool] = Query(None, description="공개 여부 필터 (true/false)"),
        date_filter: Optional[date] = Query(None, description="작성 날짜 필터 (YYYY-MM-DD)"),
        sort: Optional[str] = Query("desc", description="정렬 방식 (desc: 최신순, asc: 오래된순)"),
        page: Optional[int] = Query(1, description="페이지 번호 (1부터 시작)"),
        size: Optional[int] = Query(10, description="페이지 당 항목 수"),
        keyword: Optional[str] = Query(None, description="내용 또는 향수명 키워드 검색"),
        emotion: Optional[str] = Query(None, description="감정 태그 필터링"),
        ai_emotion: Optional[str] = Query(None, description="🆕 AI 분석 감정 필터링")  # 🆕 추가
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

            # 기존 감정 태그 필터
            if emotion:
                tags = diary.get("tags", [])
                if isinstance(tags, list):
                    if emotion.lower() not in [tag.lower() for tag in tags]:
                        continue
                else:
                    if emotion.lower() not in str(tags).lower():
                        continue

            # 🆕 AI 분석 감정 필터
            if ai_emotion:
                ai_emotion_analysis = diary.get("ai_emotion_analysis", {})
                predicted_emotion = ai_emotion_analysis.get("predicted_emotion", "")
                if ai_emotion.lower() != predicted_emotion.lower():
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

                    # 🆕 AI 감정 분석 정보 추가
                    "ai_emotion_analysis": item.get("ai_emotion_analysis", {
                        "predicted_emotion": "기쁨",
                        "confidence": 0.0,
                        "emotion_label": 0,
                        "analysis_method": "정보 없음"
                    })
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

                # 🆕 감정 분석 통계 추가
                "emotion_statistics": _calculate_emotion_statistics(filtered_data)
            }
        )

    except Exception as e:
        logger.error(f"❌ 일기 목록 조회 오류: {e}")
        return JSONResponse(
            status_code=500,
            content={"message": f"서버 오류: {str(e)}"}
        )


# 🆕 감정 통계 계산 함수
def _calculate_emotion_statistics(diary_list):
    """일기 목록의 감정 분석 통계를 계산합니다."""
    emotion_counts = {}
    confidence_sum = 0.0
    analyzed_count = 0

    for diary in diary_list:
        ai_analysis = diary.get("ai_emotion_analysis", {})
        emotion = ai_analysis.get("predicted_emotion")
        confidence = ai_analysis.get("confidence", 0.0)

        if emotion:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            confidence_sum += confidence
            analyzed_count += 1

    avg_confidence = confidence_sum / analyzed_count if analyzed_count > 0 else 0.0

    return {
        "emotion_distribution": emotion_counts,
        "total_analyzed": analyzed_count,
        "average_confidence": round(avg_confidence, 3),
        "most_common_emotion": max(emotion_counts.items(), key=lambda x: x[1])[0] if emotion_counts else None
    }


# 🆕 감정 분석 재실행 API (관리자용)
@router.post(
    "/reanalyze-emotions",
    summary="감정 분석 재실행 (관리자용)",
    description="기존 일기들의 감정을 AI 모델로 재분석합니다."
)
async def reanalyze_emotions(
        diary_ids: Optional[list] = None,
        user=Depends(verify_firebase_token_optional)
):
    """기존 일기들의 감정을 재분석합니다."""

    try:
        reanalyzed_count = 0
        error_count = 0

        for i, diary in enumerate(diary_data):
            # 특정 일기 ID들만 처리하거나 전체 처리
            if diary_ids and diary.get("id") not in diary_ids:
                continue

            content = diary.get("content", "")
            if not content.strip():
                continue

            try:
                # 감정 재분석
                emotion_analysis = await analyze_diary_emotion(content)

                # 일기 데이터 업데이트
                diary_data[i]["ai_emotion_analysis"] = {
                    "predicted_emotion": emotion_analysis.get("emotion"),
                    "confidence": emotion_analysis.get("confidence", 0.0),
                    "emotion_label": emotion_analysis.get("label", 0),
                    "analysis_method": emotion_analysis.get("method"),
                    "analyzed_at": datetime.now().isoformat(),
                    "reanalyzed": True
                }

                # 태그에도 감정 추가 (중복 제거)
                tags = diary_data[i].get("tags", [])
                ai_emotion = emotion_analysis.get("emotion", "기쁨")
                if ai_emotion not in tags:
                    tags.append(ai_emotion)
                    diary_data[i]["tags"] = tags

                reanalyzed_count += 1

                logger.info(
                    f"[REANALYZE] {diary.get('id')}: {ai_emotion} ({emotion_analysis.get('confidence', 0):.3f})")

            except Exception as e:
                logger.error(f"❌ 일기 {diary.get('id')} 재분석 실패: {e}")
                error_count += 1

        # 파일에 저장
        with open(DIARY_PATH, "w", encoding="utf-8") as f:
            json.dump(diary_data, f, ensure_ascii=False, indent=2)

        return JSONResponse(
            status_code=200,
            content={
                "message": f"감정 재분석 완료: {reanalyzed_count}개 성공, {error_count}개 실패",
                "reanalyzed_count": reanalyzed_count,
                "error_count": error_count,
                "total_diaries": len(diary_data)
            }
        )

    except Exception as e:
        logger.error(f"❌ 감정 재분석 중 오류: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"감정 재분석 중 오류가 발생했습니다: {str(e)}"
        )


# ✅ 기존 API들 유지 (좋아요, 좋아요 취소, 사용자별 일기 조회 등)
@router.post("/{diary_id}/like", summary="시향 일기 좋아요 추가", description="해당 시향 일기의 좋아요 수를 1 증가시킵니다.")
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

        # 파일에 저장
        with open(DIARY_PATH, "w", encoding="utf-8") as f:
            json.dump(diary_data, f, ensure_ascii=False, indent=2)

        return JSONResponse(status_code=200, content={"message": "좋아요가 추가되었습니다."})

    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"좋아요 처리 중 오류: {str(e)}"})


@router.delete("/{diary_id}/unlike", summary="시향 일기 좋아요 취소", description="해당 시향 일기의 좋아요 수를 1 감소시킵니다.")
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

        # 파일에 저장
        with open(DIARY_PATH, "w", encoding="utf-8") as f:
            json.dump(diary_data, f, ensure_ascii=False, indent=2)

        return JSONResponse(status_code=200, content={"message": "좋아요가 취소되었습니다."})

    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"좋아요 취소 처리 중 오류: {str(e)}"})


@router.get("/user/{user_id}", summary="사용자별 일기 조회", description="특정 사용자가 작성한 일기 목록을 반환합니다.")
async def get_user_diaries(user_id: str, public_only: bool = Query(True, description="공개 일기만 조회할지 여부")):
    try:
        user_diaries = []

        for diary in diary_data:
            if diary.get("user_id") == user_id:
                if public_only and not diary.get("is_public", False):
                    continue
                user_diaries.append(diary)

        # 최신순으로 정렬
        user_diaries.sort(key=lambda x: x.get("created_at", ""), reverse=True)

        return JSONResponse(
            status_code=200,
            content={
                "message": f"사용자 {user_id}의 일기 조회 완료",
                "data": user_diaries,
                "count": len(user_diaries),
                "emotion_statistics": _calculate_emotion_statistics(user_diaries)  # 🆕 사용자별 감정 통계
            }
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"사용자 일기 조회 중 오류: {str(e)}"}
        )


# ✅ 시스템 상태 확인 (감정 태깅 정보 포함)
@router.get("/status", summary="일기 시스템 상태 (감정 태깅 포함)", description="일기 시스템의 상태와 감정 태깅 시스템 상태를 확인합니다.")
async def get_diary_system_status():
    emotion_tagging_status = None

    # 감정 태깅 시스템 상태 확인
    if EMOTION_TAGGING_AVAILABLE:
        try:
            emotion_tagger = get_emotion_tagger()
            emotion_tagging_status = emotion_tagger.get_system_status()
        except Exception as e:
            emotion_tagging_status = {"error": str(e)}

    return {
        "diary_count": len(diary_data),
        "diary_file_exists": os.path.exists(DIARY_PATH),
        "diary_file_path": DIARY_PATH,
        "firebase_status": get_firebase_status(),
        "emotion_tagging_available": EMOTION_TAGGING_AVAILABLE,
        "emotion_tagging_status": emotion_tagging_status,  # 🆕 감정 태깅 상태
        "emotion_statistics": _calculate_emotion_statistics(diary_data),  # 🆕 전체 감정 통계
        "message": "일기 시스템 상태 정보입니다."
    }