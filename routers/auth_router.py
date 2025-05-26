# perfume_backend/routers/auth_router.py

from fastapi import APIRouter, HTTPException, Header
from fastapi.responses import JSONResponse
from firebase_admin import auth
schemas.common import BaseResponse
import firebase_admin

router = APIRouter(prefix="/auth", tags=["Auth"])

@router.post("/login", summary="Firebase 로그인", description="iOS에서 전달된 Firebase ID 토큰을 검증합니다.")
async def login(id_token: str = Header(..., description="Firebase에서 발급된 ID 토큰")):
    try:
        decoded_token = auth.verify_id_token(id_token)
        uid = decoded_token["uid"]
        email = decoded_token.get("email", "unknown")

        return BaseResponse(message="로그인 성공", result={"uid": uid, "email": email})
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"인증 실패: {str(e)}")
