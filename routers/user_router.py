# routers/user_router.py

from fastapi import APIRouter, Depends
from utils.auth_utils import verify_firebase_token
from models.user_model import get_user

router = APIRouter(prefix="/users", tags=["User"])

# ✅ 로그인된 사용자 정보 조회
@router.get("/me", summary="내 정보 조회", description="현재 로그인한 사용자의 정보를 반환합니다.")
async def get_my_info(user=Depends(verify_firebase_token)):
    uid = user["uid"]
    user_info = get_user(uid)

    if not user_info:
        return {"message": "사용자 정보를 찾을 수 없습니다."}
    
    return user_info
