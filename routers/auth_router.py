# routers/auth_router.py

from fastapi import APIRouter, Depends
from utils.auth_utils import verify_firebase_token

router = APIRouter(prefix="/auth", tags=["Auth"])

# ✅ Swagger 테스트용 간단 인증 확인 API
@router.post("/test", summary="Firebase 토큰 유효성 테스트", description="Firebase 토큰이 유효한지 확인합니다.")
async def test_token(user=Depends(verify_firebase_token)):
    return {
        "message": f"{user.get('name', '알 수 없음')}님, 인증되었습니다.",
        "uid": user["uid"],
        "email": user.get("email")
    }
