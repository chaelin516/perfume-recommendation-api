from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
from perfume_backend.schemas.diary import DiaryCreateRequest, DiaryEntry
from perfume_backend.schemas.common import BaseResponse
from datetime import datetime, date
from typing import Optional
import os, json

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

# 시향 일기 작성 API
@router.post("/", summary="시향 일기 작성", description="사용자가 향수에 대해 작성한 시향 일기를 저장합니다.")
async def write_diary(entry: DiaryCreateRequest):
    # created_at 자동 생성
    diary = DiaryEntry(
        user_id=entry.user_id,
        perfume_name=entry.perfume_name,
        content=entry.content,
        is_public=entry.is_public,
        emotion_tags=entry.emotion_tags,
        created_at=datetime.now()
    )

    # JSON 직렬화 가능한 dict로 변환 후 저장
    diary_data.append(diary.model_dump())

    with open(DIARY_PATH, "w", encoding="utf-8") as f:
        json.dump(diary_data, f, ensure_ascii=False, indent=2, default=str)

    return JSONResponse(
        status_code=200,
        content={"message": "시향 일기가 성공적으로 저장되었습니다."}
    )


# 시향 일기 조회 API (필터 + 정렬 + 페이징 포함)
@router.get("/", summary="시향 일기 목록 조회", description="저장된 모든 시향 일기를 반환합니다.", response_model=BaseResponse)
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
                if "T" not in created_str or created_str.split("T")[0] != date_filter.isoformat():
                    continue

            # 키워드 필터 (내용 or 향수명)
            if keyword:
                if keyword.lower() not in diary.get("content", "").lower() and keyword.lower() not in diary.get("perfume_name", "").lower():
                    continue

            # 감정 태그 필터
            if emotion:
                if emotion.lower() not in [tag.lower() for tag in diary.get("emotion_tags", [])]:
                    continue

            filtered_data.append(diary)

        # 정렬
        reverse = sort != "asc"
        filtered_data.sort(
            key=lambda x: x.get("created_at") or "1970-01-01T00:00:00",
            reverse=reverse
        )

        
        # 페이징
        start = (page - 1) * size
        end = start + size
        paginated_data = filtered_data[start:end]

        return BaseResponse(message="시향 일기 목록 조회 성공", result=paginated_data)

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"서버 오류: {str(e)}"}
        )
