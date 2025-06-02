# routers/auth_router.py - 확장 버전

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr
from utils.auth_utils import verify_firebase_token_optional, get_firebase_status
from models.user_model import save_user
from firebase_admin import auth
import logging

router = APIRouter(prefix="/auth", tags=["Auth"])


# 요청/응답 스키마
class EmailPasswordRegister(BaseModel):
    email: EmailStr
    password: str
    name: str


class EmailPasswordLogin(BaseModel):
    email: EmailStr
    password: str


class GoogleLoginRequest(BaseModel):
    id_token: str  # Google에서 받은 ID 토큰


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class ResetPasswordRequest(BaseModel):
    oob_code: str  # Firebase에서 받은 재설정 코드
    new_password: str


class VerifyEmailRequest(BaseModel):
    id_token: str


# ✅ 기존 토큰 테스트 API
@router.post("/test", summary="Firebase 토큰 유효성 테스트")
async def test_token(user=Depends(verify_firebase_token_optional)):
    return {
        "message": f"{user.get('name', '알 수 없음')}님, 인증되었습니다.",
        "uid": user["uid"],
        "email": user.get("email")
    }


# 🆕 이메일/비밀번호 회원가입
@router.post("/register", summary="이메일/비밀번호 회원가입")
async def register_with_email(request: EmailPasswordRegister):
    try:
        # Firebase에서 사용자 생성
        user_record = auth.create_user(
            email=request.email,
            password=request.password,
            display_name=request.name,
            email_verified=False
        )

        # 이메일 인증 링크 생성
        verification_link = auth.generate_email_verification_link(request.email)

        # 사용자 정보 저장 (DB)
        await save_user(
            uid=user_record.uid,
            email=request.email,
            name=request.name
        )

        return JSONResponse(
            status_code=201,
            content={
                "message": "회원가입이 완료되었습니다. 이메일 인증을 확인해주세요.",
                "uid": user_record.uid,
                "email_verification_link": verification_link
            }
        )

    except auth.EmailAlreadyExistsError:
        raise HTTPException(
            status_code=400,
            detail="이미 존재하는 이메일 주소입니다."
        )
    except Exception as e:
        logging.error(f"Registration error: {e}")
        raise HTTPException(
            status_code=500,
            detail="회원가입 중 오류가 발생했습니다."
        )


# 🆕 이메일/비밀번호 로그인
@router.post("/login", summary="이메일/비밀번호 로그인")
async def login_with_email(request: EmailPasswordLogin):
    try:
        # Firebase에서는 클라이언트 사이드에서 로그인을 처리하므로
        # 여기서는 사용자 존재 여부만 확인
        user_record = auth.get_user_by_email(request.email)

        # 실제 로그인은 iOS 앱에서 Firebase SDK로 처리
        return JSONResponse(
            content={
                "message": "사용자 확인 완료. iOS 앱에서 Firebase 로그인을 진행하세요.",
                "user_exists": True,
                "email_verified": user_record.email_verified
            }
        )

    except auth.UserNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="존재하지 않는 사용자입니다."
        )
    except Exception as e:
        logging.error(f"Login error: {e}")
        raise HTTPException(
            status_code=500,
            detail="로그인 확인 중 오류가 발생했습니다."
        )


# 🆕 구글 로그인 처리
@router.post("/google-login", summary="구글 로그인")
async def google_login(request: GoogleLoginRequest):
    try:
        # Google ID 토큰 검증
        decoded_token = auth.verify_id_token(request.id_token)

        uid = decoded_token["uid"]
        email = decoded_token.get("email")
        name = decoded_token.get("name")
        picture = decoded_token.get("picture")

        # 사용자 정보 저장/업데이트
        await save_user(
            uid=uid,
            email=email,
            name=name,
            picture=picture
        )

        return JSONResponse(
            content={
                "message": "구글 로그인이 완료되었습니다.",
                "user": {
                    "uid": uid,
                    "email": email,
                    "name": name,
                    "picture": picture
                }
            }
        )

    except auth.InvalidIdTokenError:
        raise HTTPException(
            status_code=401,
            detail="유효하지 않은 구글 토큰입니다."
        )
    except Exception as e:
        logging.error(f"Google login error: {e}")
        raise HTTPException(
            status_code=500,
            detail="구글 로그인 중 오류가 발생했습니다."
        )


# 🆕 비밀번호 재설정 요청
@router.post("/forgot-password", summary="비밀번호 재설정 이메일 발송")
async def forgot_password(request: ForgotPasswordRequest):
    try:
        # 비밀번호 재설정 링크 생성
        reset_link = auth.generate_password_reset_link(request.email)

        return JSONResponse(
            content={
                "message": "비밀번호 재설정 이메일이 발송되었습니다.",
                "reset_link": reset_link
            }
        )

    except auth.UserNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="존재하지 않는 이메일 주소입니다."
        )
    except Exception as e:
        logging.error(f"Password reset error: {e}")
        raise HTTPException(
            status_code=500,
            detail="비밀번호 재설정 요청 중 오류가 발생했습니다."
        )


# 🆕 이메일 인증 발송
@router.post("/send-verification", summary="이메일 인증 발송")
async def send_email_verification(request: VerifyEmailRequest):
    try:
        # ID 토큰에서 사용자 정보 추출
        decoded_token = auth.verify_id_token(request.id_token)
        email = decoded_token.get("email")

        if not email:
            raise HTTPException(
                status_code=400,
                detail="토큰에서 이메일 정보를 찾을 수 없습니다."
            )

        # 이메일 인증 링크 생성
        verification_link = auth.generate_email_verification_link(email)

        return JSONResponse(
            content={
                "message": "이메일 인증 링크가 발송되었습니다.",
                "verification_link": verification_link
            }
        )

    except auth.InvalidIdTokenError:
        raise HTTPException(
            status_code=401,
            detail="유효하지 않은 토큰입니다."
        )
    except Exception as e:
        logging.error(f"Email verification error: {e}")
        raise HTTPException(
            status_code=500,
            detail="이메일 인증 발송 중 오류가 발생했습니다."
        )


# 🆕 로그아웃 (토큰 무효화)
@router.post("/logout", summary="로그아웃")
async def logout(user=Depends(verify_firebase_token_optional)):
    try:
        # Firebase에서 사용자의 refresh token 무효화
        auth.revoke_refresh_tokens(user["uid"])

        return JSONResponse(
            content={
                "message": "로그아웃이 완료되었습니다.",
                "uid": user["uid"]
            }
        )

    except Exception as e:
        logging.error(f"Logout error: {e}")
        raise HTTPException(
            status_code=500,
            detail="로그아웃 중 오류가 발생했습니다."
        )


# ✅ Firebase 상태 확인
@router.get("/firebase-status", summary="Firebase 상태 확인")
async def check_firebase_status():
    return get_firebase_status()