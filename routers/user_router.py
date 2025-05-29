from fastapi import APIRouter, Depends
from utils.auth_utils import verify_firebase_token

router = APIRouter(prefix="/users", tags=["User"])

# ✅ 로그인된 사용자 정보 조회
@router.get(
    "/me",
    summary="내 정보 조회",
    description="현재 로그인한 사용자의 정보를 반환합니다."
)
async def get_my_info(user=Depends(verify_firebase_token)):
    uid = user["uid"]

    # 🔁 사용자 정보 조회 함수 정의 (간단한 목업)
    # 추후 실제 DB 연동이 필요하면 models.user_model 쪽으로 분리 가능
    user_info = {
        "uid": uid,
        "email": user.get("email", ""),
        "name": user.get("name", ""),
        "picture": user.get("picture", "")
    }

    return user_info
