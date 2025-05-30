from fastapi import APIRouter, Depends
from utils.auth_utils import verify_firebase_token_optional, get_firebase_status

router = APIRouter(prefix="/users", tags=["User"])

# ✅ 로그인된 사용자 정보 조회
@router.get(
    "/me",
    summary="내 정보 조회",
    description="현재 로그인한 사용자의 정보를 반환합니다."
)
async def get_my_info(user=Depends(verify_firebase_token_optional)):
    uid = user["uid"]

    # 🔁 사용자 정보 조회 함수 정의 (간단한 목업)
    # 추후 실제 DB 연동이 필요하면 models.user_model 쪽으로 분리 가능
    user_info = {
        "uid": uid,
        "email": user.get("email", ""),
        "name": user.get("name", ""),
        "picture": user.get("picture", ""),
        "is_test_user": uid.startswith("test-")  # 테스트 사용자 여부
    }

    return {
        "message": "사용자 정보 조회 성공",
        "data": user_info,
        "firebase_status": get_firebase_status()
    }

# ✅ 사용자 설정 정보 조회 (더미 데이터)
@router.get(
    "/settings",
    summary="사용자 설정 조회",
    description="사용자의 설정 정보를 반환합니다."
)
async def get_user_settings(user=Depends(verify_firebase_token_optional)):
    # 더미 설정 데이터
    settings = {
        "notification_enabled": True,
        "public_profile": True,
        "preferred_language": "ko",
        "theme": "light",
        "marketing_consent": False
    }
    
    return {
        "message": "사용자 설정 조회 성공",
        "data": settings
    }

# ✅ 사용자 프로필 업데이트 (더미 기능)
@router.put(
    "/profile",
    summary="프로필 업데이트",
    description="사용자 프로필 정보를 업데이트합니다."
)
async def update_user_profile(
    name: str = None,
    bio: str = None,
    user=Depends(verify_firebase_token_optional)
):
    updated_fields = {}
    if name:
        updated_fields["name"] = name
    if bio:
        updated_fields["bio"] = bio
    
    return {
        "message": "프로필 업데이트 성공",
        "data": {
            "uid": user["uid"],
            "updated_fields": updated_fields,
            "updated_at": "2025-05-30T12:00:00Z"
        }
    }

# ✅ 사용자 통계 정보
@router.get(
    "/stats",
    summary="사용자 통계",
    description="사용자의 활동 통계를 반환합니다."
)
async def get_user_stats(user=Depends(verify_firebase_token_optional)):
    # 더미 통계 데이터
    stats = {
        "total_diaries": 5,
        "total_likes_received": 12,
        "total_comments": 3,
        "favorite_perfume_brands": ["Dior", "Chanel", "Tom Ford"],
        "most_used_emotions": ["elegant", "fresh", "romantic"],
        "joined_date": "2025-01-15",
        "days_active": 45
    }
    
    return {
        "message": "사용자 통계 조회 성공",
        "data": stats
    }