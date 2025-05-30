from fastapi import APIRouter, Query, Depends
from fastapi.responses import JSONResponse
from schemas.diary import DiaryCreateRequest, DiaryResponse
from schemas.common import BaseResponse
from datetime import datetime, date
from typing import Optional
from utils.auth_utils import verify_firebase_token_optional, get_firebase_status  # 🔐 선택적 Firebase 인증

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

# ✅ Firebase 상태 확인 API
@router.get("/firebase-status", summary="Firebase 상태 확인", description="Firebase 인증 서비스 상태를 확인합니다.")
async def check_firebase_status():
    return get_firebase_status()

# ✅ 시향 일기 작성 API (Firebase 선택적 로그인)
@router.post("/", summary="시향 일기 작성", description="사용자가 향수에 대해 작성한 시향 일기를 저장합니다.")
async def write_diary(entry: DiaryCreateRequest, user=Depends(verify_firebase_token_optional)):
    try:
        user_id = user["uid"]
        now = datetime.now().isoformat()

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
            "tags": entry.emotion_tags or [],
            "likes": 0,
            "comments": 0,
            "is_public": entry.is_public,
            "created_at": now,
            "updated_at": now
        }

        diary_data.append(diary)

        # 파일에 저장
        os.makedirs(os.path.dirname(DIARY_PATH), exist_ok=True)
        with open(DIARY_PATH, "w", encoding="utf-8") as f:
            json.dump(diary_data, f, ensure_ascii=False, indent=2)

        print(f"[DIARY] 새 일기 저장됨: {user.get('name', '익명')} - {entry.perfume_name}")

        return JSONResponse(
            status_code=200,
            content={"message": "시향 일기가 성공적으로 저장되었습니다.", "diary_id": diary["id"]}
        )
    
    except Exception as e:
        print(f"❌ 일기 저장 중 오류: {e}")
        return JSONResponse(
            status_code=500,
            content={"message": f"일기 저장 중 오류가 발생했습니다: {str(e)}"}
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
                    "updated_at": item.get("updated_at", "")
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
                "has_next": end < len(filtered_data)
            }
        )

    except Exception as e:
        print(f"❌ 일기 목록 조회 오류: {e}")
        return JSONResponse(
            status_code=500,
            content={"message": f"서버 오류: {str(e)}"}
        )

# ✅ 좋아요 추가 API
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

# ✅ 좋아요 취소 API
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

# ✅ 특정 사용자의 일기 조회
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
                "count": len(user_diaries)
            }
        )
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"사용자 일기 조회 중 오류: {str(e)}"}
        )

# ✅ 시스템 상태 확인
@router.get("/status", summary="일기 시스템 상태", description="일기 시스템의 상태를 확인합니다.")
async def get_diary_system_status():
    return {
        "diary_count": len(diary_data),
        "diary_file_exists": os.path.exists(DIARY_PATH),
        "diary_file_path": DIARY_PATH,
        "firebase_status": get_firebase_status(),
        "message": "일기 시스템 상태 정보입니다."
    }