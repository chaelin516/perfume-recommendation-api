from fastapi import APIRouter, Query, Depends
from fastapi.responses import JSONResponse
from schemas.diary import DiaryCreateRequest, DiaryResponse
from schemas.common import BaseResponse
from datetime import datetime, date
from typing import Optional
from utils.auth_utils import verify_firebase_token  # 🔐 Firebase 인증 함수

import os, json, uuid

router = APIRouter(prefix="/diaries", tags=["Diary"])

# 파일 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DIARY_PATH = os.path.join(BASE_DIR, "../data/diary_data.json")

# 기존 데이터 로딩
if os.path.exists(DIARY_PATH):
    with open(DIARY_PATH, "r", encoding="utf-8") as f:
        diary_data = json.load(f)
else:
    diary_data = []

# ✅ 시향 일기 작성 API
@router.post("/", summary="시향 일기 작성", description="사용자가 향수에 대해 작성한 시향 일기를 저장합니다.")
async def write_diary(entry: DiaryCreateRequest, user=Depends(verify_firebase_token)):
    user_id = user["uid"]

    now = datetime.now().isoformat()

    diary = {
        "id": str(uuid.uuid4()),
        "user_id": user_id,
        "user_name": user.get("name", "익명 사용자"),
        "user_profile_image": user.get("picture", ""),
        "perfume_id": f"perfume_{entry.perfume_name.lower().replace(' ', '_')}",
        "perfume_name": entry.perfume_name,
        "brand": "Dummy Brand",  # 실제 프로젝트에서는 향수 브랜드 정보 연동 필요
        "content": entry.content or "",
        "tags": entry.emotion_tags or [],
        "likes": 0,
        "comments": 0,
        "created_at": now,
        "updated_at": now
    }

    diary_data.append(diary)

    with open(DIARY_PATH, "w", encoding="utf-8") as f:
        json.dump(diary_data, f, ensure_ascii=False, indent=2)

    return JSONResponse(
        status_code=200,
        content={"message": "시향 일기가 성공적으로 저장되었습니다."}
    )

# ✅ 시향 일기 목록 조회 API
@router.get("/", summary="시향 일기 목록 조회", description="저장된 모든 시향 일기를 반환합니다.", response_model=BaseResponse, response_model_by_alias=True)
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
            if public is not None and diary.get("is_public") != public:
                continue

            if date_filter is not None:
                created_str = diary.get("created_at", "")
                if "T" not in created_str or created_str.split("T")[0] != date_filter.isoformat():
                    continue

            if keyword:
                if keyword.lower() not in diary.get("content", "").lower() and keyword.lower() not in diary.get("perfume_name", "").lower():
                    continue

            if emotion:
                if emotion.lower() not in [tag.lower() for tag in diary.get("tags", [])]:
                    continue

            filtered_data.append(diary)

        reverse = sort != "asc"
        filtered_data.sort(
            key=lambda x: x.get("created_at") or "1970-01-01T00:00:00",
            reverse=reverse
        )

        start = (page - 1) * size
        end = start + size
        paginated_data = filtered_data[start:end]

        response_data = [DiaryResponse(**item).dict(by_alias=True) for item in paginated_data]

        return BaseResponse(message="시향 일기 목록 조회 성공", result=response_data)

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"서버 오류: {str(e)}"}
        )

# ✅ 시향 일기 좋아요 추가 API
@router.post("/{diary_id}/like", summary="시향 일기 좋아요 추가", description="해당 시향 일기의 좋아요 수를 1 증가시킵니다.")
async def like_diary(diary_id: str):
    found = False

    for diary in diary_data:
        if diary["id"] == diary_id:
            diary["likes"] = diary.get("likes", 0) + 1
            diary["updated_at"] = datetime.now().isoformat()
            found = True
            break

    if not found:
        return JSONResponse(status_code=404, content={"message": "해당 일기를 찾을 수 없습니다."})

    with open(DIARY_PATH, "w", encoding="utf-8") as f:
        json.dump(diary_data, f, ensure_ascii=False, indent=2)

    return JSONResponse(status_code=200, content={"message": "좋아요가 추가되었습니다."})

# ✅ 시향 일기 좋아요 취소 API
@router.delete("/{diary_id}/unlike", summary="시향 일기 좋아요 취소", description="해당 시향 일기의 좋아요 수를 1 감소시킵니다.")
async def unlike_diary(diary_id: str):
    found = False

    for diary in diary_data:
        if diary["id"] == diary_id:
            diary["likes"] = max(0, diary.get("likes", 0) - 1)
            diary["updated_at"] = datetime.now().isoformat()
            found = True
            break

    if not found:
        return JSONResponse(status_code=404, content={"message": "해당 일기를 찾을 수 없습니다."})

    with open(DIARY_PATH, "w", encoding="utf-8") as f:
        json.dump(diary_data, f, ensure_ascii=False, indent=2)

    return JSONResponse(status_code=200, content={"message": "좋아요가 취소되었습니다."})
