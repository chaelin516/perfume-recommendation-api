# routers/diary_router.py - 감정 태깅 기능 연동 버전

from fastapi import APIRouter, Query, Depends
from fastapi.responses import JSONResponse
from schemas.diary import DiaryCreateRequest, DiaryResponse
from schemas.common import BaseResponse
from datetime import datetime, date
from typing import Optional
from utils.auth_utils import verify_firebase_token_optional, get_firebase_status

import os, json, uuid

router = APIRouter(prefix="/diaries", tags=["Diary"])

# 📂 시향 일기 데이터 파일 경로
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DIARY_PATH = os.path.join(BASE_DIR, "../data/diary_data.json")

# 📦 기존 데이터 로딩
if os.path.exists(DIARY_PATH):
    try:
        with open(DIARY_PATH, "r", encoding="utf-8") as f:
            diary_data = json.load(f)
        print(f"✅ 시향 일기 데이터 로딩 완료: {len(diary_data)}개 항목")
    except Exception as e:
        print(f"❌ 시향 일기 데이터 로딩 실패: {e}")
        diary_data = []
else:
    diary_data = []
    print("⚠️ 시향 일기 데이터 파일이 없습니다. 새로 생성됩니다.")


# 🎭 감정 태깅 함수 (안전한 import)
def get_emotion_tags_for_text(text: str) -> dict:
    """텍스트에서 감정 태그 예측 (안전한 호출)"""
    try:
        from utils.emotion_tagging_model_loader import predict_emotion_tags, is_model_available

        if not text or not text.strip():
            return {
                "success": False,
                "predicted_emotion": "기쁨",  # 기본값
                "confidence": 0.0,
                "method": "빈 텍스트"
            }

        # 감정 태깅 모델 사용 가능 여부 확인
        if is_model_available():
            print(f"🎭 AI 감정 태깅 사용: '{text[:30]}...'")
            result = predict_emotion_tags(text)
        else:
            print(f"📋 룰 기반 감정 태깅 사용: '{text[:30]}...'")
            # 모델이 없으면 룰 기반 사용
            from utils.emotion_tagging_model_loader import _rule_based_emotion_tagging
            result = _rule_based_emotion_tagging(text)

        return result

    except ImportError as e:
        print(f"⚠️ 감정 태깅 모듈 import 실패: {e}")
        # 폴백: 간단한 룰 기반
        return _simple_rule_based_tagging(text)
    except Exception as e:
        print(f"❌ 감정 태깅 중 오류: {e}")
        # 폴백: 간단한 룰 기반
        return _simple_rule_based_tagging(text)


def _simple_rule_based_tagging(text: str) -> dict:
    """간단한 룰 기반 감정 태깅 (완전 폴백)"""
    try:
        if not text:
            return {
                "success": True,
                "predicted_emotion": "기쁨",
                "confidence": 0.3,
                "method": "기본값"
            }

        text_lower = text.lower()

        # 간단한 키워드 기반 매칭
        if any(word in text_lower for word in ["좋", "행복", "사랑", "완벽", "달콤", "따뜻"]):
            return {"success": True, "predicted_emotion": "기쁨", "confidence": 0.7, "method": "간단 룰"}
        elif any(word in text_lower for word in ["불안", "걱정", "떨", "두려"]):
            return {"success": True, "predicted_emotion": "불안", "confidence": 0.7, "method": "간단 룰"}
        elif any(word in text_lower for word in ["당황", "놀", "혼란", "이상"]):
            return {"success": True, "predicted_emotion": "당황", "confidence": 0.7, "method": "간단 룰"}
        elif any(word in text_lower for word in ["화", "짜증", "싫", "최악"]):
            return {"success": True, "predicted_emotion": "분노", "confidence": 0.7, "method": "간단 룰"}
        elif any(word in text_lower for word in ["상처", "아픈", "실망", "그리운"]):
            return {"success": True, "predicted_emotion": "상처", "confidence": 0.7, "method": "간단 룰"}
        elif any(word in text_lower for word in ["슬", "눈물", "외로", "쓸쓸"]):
            return {"success": True, "predicted_emotion": "슬픔", "confidence": 0.7, "method": "간단 룰"}
        elif any(word in text_lower for word in ["우울", "답답", "무기력", "어둠"]):
            return {"success": True, "predicted_emotion": "우울", "confidence": 0.7, "method": "간단 룰"}
        elif any(word in text_lower for word in ["흥분", "신나", "설렘", "활기"]):
            return {"success": True, "predicted_emotion": "흥분", "confidence": 0.7, "method": "간단 룰"}
        else:
            return {"success": True, "predicted_emotion": "기쁨", "confidence": 0.4, "method": "기본값"}

    except Exception as e:
        print(f"❌ 간단 룰 기반 태깅 실패: {e}")
        return {"success": True, "predicted_emotion": "기쁨", "confidence": 0.3, "method": "오류 폴백"}


# ✅ Firebase 상태 확인 API
@router.get("/firebase-status", summary="Firebase 상태 확인", description="Firebase 인증 서비스 상태를 확인합니다.")
async def check_firebase_status():
    return get_firebase_status()


# ✅ 시향 일기 작성 API (🆕 감정 태깅 자동 적용)
@router.post("/", summary="시향 일기 작성 (감정 태깅 자동 적용)", description="사용자가 향수에 대해 작성한 시향 일기를 저장하고 자동으로 감정 태그를 추가합니다.")
async def write_diary(entry: DiaryCreateRequest, user=Depends(verify_firebase_token_optional)):
    try:
        user_id = user["uid"]
        now = datetime.now().isoformat()

        # 🎭 자동 감정 태깅 수행
        emotion_result = {"predicted_emotion": "기쁨", "confidence": 0.0, "method": "기본값"}

        if entry.content and entry.content.strip():
            print(f"🎭 시향일기 감정 태깅 시작: 사용자 {user.get('name', '익명')}")
            emotion_result = get_emotion_tags_for_text(entry.content)
            print(
                f"🎭 감정 태깅 결과: {emotion_result.get('predicted_emotion', '알 수 없음')} (신뢰도: {emotion_result.get('confidence', 0):.3f})")

        # 기존 emotion_tags에 예측된 감정 추가
        auto_emotion_tags = [emotion_result.get("predicted_emotion", "기쁨")]
        user_emotion_tags = entry.emotion_tags or []

        # 중복 제거하면서 합치기
        final_emotion_tags = list(set(auto_emotion_tags + user_emotion_tags))

        # 새 일기 항목 생성
        diary = {
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "user_name": user.get("name", "익명 사용자"),
            "user_profile_image": user.get("picture", ""),
            "perfume_id": f"perfume_{entry.perfume_name.lower().replace(' ', '_')}",
            "perfume_name": entry.perfume_name,
            "brand": "Dummy Brand",  # 실제 브랜드 연동 필요
            "content": entry.content or "",
            "tags": final_emotion_tags,  # 🆕 자동 태깅된 감정 포함
            "likes": 0,
            "comments": 0,
            "is_public": entry.is_public,
            "created_at": now,
            "updated_at": now,
            # 🆕 감정 태깅 메타데이터 추가
            "emotion_tagging": {
                "auto_predicted": emotion_result.get("predicted_emotion", "기쁨"),
                "confidence": emotion_result.get("confidence", 0.0),
                "method": emotion_result.get("method", "기본값"),
                "user_provided": entry.emotion_tags or [],
                "final_tags": final_emotion_tags
            }
        }

        diary_data.append(diary)

        # 파일에 저장
        os.makedirs(os.path.dirname(DIARY_PATH), exist_ok=True)
        with open(DIARY_PATH, "w", encoding="utf-8") as f:
            json.dump(diary_data, f, ensure_ascii=False, indent=2)

        print(f"[DIARY] 새 일기 저장됨: {user.get('name', '익명')} - {entry.perfume_name}")
        print(
            f"[EMOTION] 자동 태깅: {emotion_result.get('predicted_emotion')} (신뢰도: {emotion_result.get('confidence', 0):.3f})")

        return JSONResponse(
            status_code=200,
            content={
                "message": "시향 일기가 성공적으로 저장되었습니다.",
                "diary_id": diary["id"],
                "emotion_tagging": {
                    "auto_predicted": emotion_result.get("predicted_emotion", "기쁨"),
                    "confidence": emotion_result.get("confidence", 0.0),
                    "method": emotion_result.get("method", "기본값"),
                    "final_tags": final_emotion_tags
                }
            }
        )

    except Exception as e:
        print(f"❌ 일기 저장 중 오류: {e}")
        return JSONResponse(
            status_code=500,
            content={"message": f"일기 저장 중 오류가 발생했습니다: {str(e)}"}
        )


# ✅ 시향 일기 목록 조회 API
@router.get("/", summary="시향 일기 목록 조회", description="저장된 모든 시향 일기를 반환합니다.", response_model=BaseResponse,
            response_model_by_alias=True)
async def get_diary_list(
        public: Optional[bool] = Query(None, description="공개 여부 필터 (true/false)"),
        date_filter: Optional[date] = Query(None, description="작성 날짜 필터 (YYYY-MM-DD)"),
        sort: Optional[str] = Query("desc", description="정렬 방식 (desc: 최신순, asc: 오래된순)"),
        page: Optional[int] = Query(1, description="페이지 번호 (1부터 시작)"),
        size: Optional[int] = Query(10, description="페이지 당 항목 수"),
        keyword: Optional[str] = Query(None, description="내용 또는 향수명 키워드 검색"),
        emotion: Optional[str] = Query(None, description="감정 태그 필터링")
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

            # 🆕 감정 태그 필터 (자동 태깅 결과도 포함)
            if emotion:
                tags = diary.get("tags", [])
                if isinstance(tags, list):
                    if emotion.lower() not in [tag.lower() for tag in tags]:
                        continue
                else:
                    if emotion.lower() not in str(tags).lower():
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

        # 응답 데이터 변환 (필요한 필드만 포함)
        response_data = []
        for item in paginated_data:
            try:
                # DiaryResponse 스키마에 맞게 데이터 변환
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
                    # 🆕 감정 태깅 정보 포함 (선택적)
                    "emotion_tagging": item.get("emotion_tagging", {})
                }
                response_data.append(diary_item)
            except Exception as e:
                print(f"⚠️ 일기 항목 변환 오류: {e}")
                continue

        return BaseResponse(
            message=f"시향 일기 목록 조회 성공 (총 {len(filtered_data)}개, 페이지: {page})",
            result={
                "diaries": response_data,
                "total_count": len(filtered_data),
                "page": page,
                "size": size,
                "has_next": end < len(filtered_data),
                # 🆕 감정 태깅 통계 추가
                "emotion_stats": _calculate_emotion_stats(filtered_data)
            }
        )

    except Exception as e:
        print(f"❌ 일기 목록 조회 오류: {e}")
        return JSONResponse(
            status_code=500,
            content={"message": f"서버 오류: {str(e)}"}
        )


def _calculate_emotion_stats(diaries: list) -> dict:
    """감정 태그 통계 계산"""
    try:
        emotion_counts = {}
        auto_tagging_stats = {"ai_model": 0, "rule_based": 0, "simple_rule": 0, "default": 0}

        for diary in diaries:
            # 감정 태그 개수
            tags = diary.get("tags", [])
            for tag in tags:
                emotion_counts[tag] = emotion_counts.get(tag, 0) + 1

            # 자동 태깅 방법 통계
            emotion_tagging = diary.get("emotion_tagging", {})
            method = emotion_tagging.get("method", "unknown")
            if "AI" in method:
                auto_tagging_stats["ai_model"] += 1
            elif "룰 기반" in method:
                auto_tagging_stats["rule_based"] += 1
            elif "간단 룰" in method:
                auto_tagging_stats["simple_rule"] += 1
            else:
                auto_tagging_stats["default"] += 1

        return {
            "emotion_distribution": emotion_counts,
            "auto_tagging_methods": auto_tagging_stats,
            "total_diaries": len(diaries)
        }

    except Exception as e:
        print(f"⚠️ 감정 통계 계산 오류: {e}")
        return {}


# 🆕 감정 태깅 테스트 API
@router.post("/test-emotion-tagging", summary="감정 태깅 테스트", description="텍스트에 대한 감정 태깅을 테스트합니다.")
async def test_emotion_tagging_api(text: str):
    """감정 태깅 테스트 API"""
    try:
        if not text or not text.strip():
            return JSONResponse(
                status_code=400,
                content={"message": "텍스트를 입력해주세요."}
            )

        print(f"🧪 감정 태깅 테스트 요청: '{text[:50]}...'")

        # 감정 태깅 수행
        result = get_emotion_tags_for_text(text)

        return JSONResponse(
            content={
                "message": "감정 태깅 테스트 완료",
                "input_text": text,
                "result": result,
                "timestamp": datetime.now().isoformat()
            }
        )

    except Exception as e:
        print(f"❌ 감정 태깅 테스트 중 오류: {e}")
        return JSONResponse(
            status_code=500,
            content={"message": f"감정 태깅 테스트 중 오류: {str(e)}"}
        )


# 🆕 감정 태깅 상태 확인 API
@router.get("/emotion-tagging-status", summary="감정 태깅 시스템 상태", description="감정 태깅 시스템의 상태를 확인합니다.")
async def get_emotion_tagging_status():
    """감정 태깅 시스템 상태 확인"""
    try:
        from utils.emotion_tagging_model_loader import get_model_status, is_model_available

        status = get_model_status()
        is_available = is_model_available()

        return JSONResponse(
            content={
                "emotion_tagging_available": is_available,
                "model_status": status,
                "supported_emotions": status.get("supported_emotions", []),
                "system_ready": True,
                "fallback_available": True  # 룰 기반은 항상 사용 가능
            }
        )

    except Exception as e:
        print(f"❌ 감정 태깅 상태 확인 중 오류: {e}")
        return JSONResponse(
            content={
                "emotion_tagging_available": False,
                "error": str(e),
                "fallback_available": True,
                "system_ready": False
            }
        )

# 기존 API들 (좋아요, 좋아요 취소, 사용자별 일기 조회, 시스템 상태 등)은 그대로 유지...