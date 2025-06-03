# routers/auth_router.py - 422 에러 해결 버전

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr, Field, validator
from utils.auth_utils import verify_firebase_token_optional, get_firebase_status
from models.user_model import save_user
from firebase_admin import auth
import logging
import re

router = APIRouter(prefix="/auth", tags=["Auth"])


# ✅ 개선된 요청/응답 스키마
class EmailPasswordRegister(BaseModel):
    email: EmailStr = Field(..., description="사용자 이메일 주소", example="user@example.com")
    password: str = Field(..., min_length=6, max_length=50, description="비밀번호 (최소 6자)", example="password123")
    name: str = Field(..., min_length=1, max_length=50, description="사용자 이름", example="홍길동")

    @validator('password')
    def validate_password(cls, v):
        if len(v) < 6:
            raise ValueError('비밀번호는 최소 6자 이상이어야 합니다.')
        return v

    @validator('name')
    def validate_name(cls, v):
        if not v or not v.strip():
            raise ValueError('이름은 필수 항목입니다.')
        return v.strip()

    class Config:
        schema_extra = {
            "example": {
                "email": "user@example.com",
                "password": "password123",
                "name": "홍길동"
            }
        }


class EmailPasswordLogin(BaseModel):
    email: EmailStr = Field(..., description="로그인 이메일")
    password: str = Field(..., description="로그인 비밀번호")


class GoogleLoginRequest(BaseModel):
    id_token: str = Field(..., description="Google ID 토큰")


class ForgotPasswordRequest(BaseModel):
    email: EmailStr = Field(..., description="비밀번호 재설정할 이메일")


class VerifyEmailRequest(BaseModel):
    id_token: str = Field(..., description="Firebase ID 토큰")


# ✅ 응답 스키마
class RegisterResponse(BaseModel):
    message: str
    uid: str
    email: str
    email_sent: bool = False
    verification_link: str = None


# ✅ 기존 토큰 테스트 API
@router.post("/test", summary="Firebase 토큰 유효성 테스트")
async def test_token(user=Depends(verify_firebase_token_optional)):
    return {
        "message": f"{user.get('name', '알 수 없음')}님, 인증되었습니다.",
        "uid": user["uid"],
        "email": user.get("email")
    }


# 🆕 이메일/비밀번호 회원가입 (개선된 에러 처리)
@router.post(
    "/register",
    summary="이메일/비밀번호 회원가입",
    response_model=RegisterResponse,
    responses={
        201: {"description": "회원가입 성공"},
        400: {"description": "이미 존재하는 이메일 또는 잘못된 입력"},
        422: {"description": "입력 데이터 형식 오류"},
        500: {"description": "서버 내부 오류"}
    }
)
async def register_with_email(request: EmailPasswordRegister):
    try:
        # 입력 데이터 로깅 (디버깅용)
        logging.info(f"회원가입 요청: email={request.email}, name={request.name}")

        # Firebase에서 사용자 생성
        user_record = auth.create_user(
            email=request.email,
            password=request.password,
            display_name=request.name,
            email_verified=False
        )

        logging.info(f"Firebase 사용자 생성 완료: uid={user_record.uid}")

        # 이메일 인증 링크 생성
        try:
            verification_link = auth.generate_email_verification_link(request.email)
            logging.info(f"이메일 인증 링크 생성 완료")
        except Exception as e:
            logging.error(f"이메일 인증 링크 생성 실패: {e}")
            verification_link = None

        # 사용자 정보 저장 (DB)
        try:
            await save_user(
                uid=user_record.uid,
                email=request.email,
                name=request.name
            )
            logging.info(f"사용자 정보 DB 저장 완료")
        except Exception as e:
            logging.error(f"사용자 정보 저장 실패: {e}")
            # DB 저장 실패해도 Firebase 계정은 생성되었으므로 계속 진행

        return JSONResponse(
            status_code=201,
            content={
                "message": "회원가입이 완료되었습니다. 이메일 인증 링크를 확인해주세요.",
                "uid": user_record.uid,
                "email": request.email,
                "email_sent": False,  # 실제 이메일 발송은 구현되지 않음
                "verification_link": verification_link,
                "note": "iOS 앱에서 Firebase 클라이언트 SDK로 이메일 인증을 처리하세요."
            }
        )

    except auth.EmailAlreadyExistsError:
        logging.warning(f"이미 존재하는 이메일: {request.email}")
        raise HTTPException(
            status_code=400,
            detail="이미 존재하는 이메일 주소입니다."
        )
    except auth.WeakPasswordError as e:
        logging.warning(f"약한 비밀번호: {e}")
        raise HTTPException(
            status_code=400,
            detail="비밀번호가 너무 약합니다. 최소 6자 이상의 비밀번호를 사용해주세요."
        )
    except auth.InvalidEmailError:
        logging.warning(f"잘못된 이메일 형식: {request.email}")
        raise HTTPException(
            status_code=400,
            detail="올바른 이메일 형식이 아닙니다."
        )
    except Exception as e:
        logging.error(f"회원가입 중 예외 발생: {str(e)}")
        logging.error(f"Exception type: {type(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"회원가입 중 오류가 발생했습니다: {str(e)}"
        )


# 🆕 이메일 인증 재발송 API
@router.post("/resend-verification", summary="이메일 인증 재발송")
async def resend_verification_email(request: VerifyEmailRequest):
    try:
        # ID 토큰에서 사용자 정보 추출
        decoded_token = auth.verify_id_token(request.id_token)
        email = decoded_token.get("email")
        name = decoded_token.get("name", "사용자")

        if not email:
            raise HTTPException(
                status_code=400,
                detail="토큰에서 이메일 정보를 찾을 수 없습니다."
            )

        # 이메일 인증 링크 생성
        verification_link = auth.generate_email_verification_link(email)

        return JSONResponse(
            content={
                "message": "인증 링크가 생성되었습니다. iOS 앱에서 이메일 인증을 처리하세요.",
                "email": email,
                "verification_link": verification_link,
                "note": "Firebase 클라이언트 SDK를 사용하여 이메일을 발송하세요."
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
            detail="이메일 인증 재발송 중 오류가 발생했습니다."
        )


# 🆕 이메일/비밀번호 로그인
@router.post("/login", summary="이메일/비밀번호 로그인")
async def login_with_email(request: EmailPasswordLogin):
    try:
        user_record = auth.get_user_by_email(request.email)

        return JSONResponse(
            content={
                "message": "사용자 확인 완료. iOS 앱에서 Firebase 로그인을 진행하세요.",
                "user_exists": True,
                "email_verified": user_record.email_verified,
                "uid": user_record.uid
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
        decoded_token = auth.verify_id_token(request.id_token)

        uid = decoded_token["uid"]
        email = decoded_token.get("email")
        name = decoded_token.get("name")
        picture = decoded_token.get("picture")

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
                "message": "비밀번호 재설정 링크가 생성되었습니다.",
                "reset_link": reset_link,
                "note": "iOS 앱에서 Firebase 클라이언트 SDK로 이메일을 발송하세요."
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


# 🆕 로그아웃 (토큰 무효화)
@router.post("/logout", summary="로그아웃")
async def logout(user=Depends(verify_firebase_token_optional)):
    try:
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


# ✅ 디버깅용 API - 요청 데이터 확인
@router.post("/debug-register", summary="회원가입 디버깅", include_in_schema=False)
async def debug_register(request: dict):
    """디버깅용 API - 실제 처리 없이 요청 데이터만 확인"""
    return {
        "received_data": request,
        "data_types": {key: type(value).__name__ for key, value in request.items()},
        "message": "디버깅용 API - 데이터 수신 확인"
    }