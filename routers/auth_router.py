# routers/auth_router.py - 구글 로그인 원래대로 유지, JWT는 별도 API

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr, Field, validator
from utils.auth_utils import verify_firebase_token_optional, get_firebase_status, verify_jwt_only
from utils.email_sender import email_sender
from models.user_model import save_user
from firebase_admin import auth
import logging
import os
from datetime import datetime

router = APIRouter(prefix="/auth", tags=["Auth"])

# 로거 설정
logger = logging.getLogger(__name__)


# ✅ 요청/응답 스키마 (기존 유지)
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


class EmailPasswordLogin(BaseModel):
    email: EmailStr = Field(..., description="로그인 이메일")
    password: str = Field(..., description="로그인 비밀번호")


class GoogleLoginRequest(BaseModel):
    id_token: str = Field(..., description="Google ID 토큰")


class ForgotPasswordRequest(BaseModel):
    email: EmailStr = Field(..., description="비밀번호 재설정할 이메일")


class VerifyEmailRequest(BaseModel):
    id_token: str = Field(..., description="Firebase ID 토큰")


class RegisterResponse(BaseModel):
    message: str
    uid: str
    email: str
    email_sent: bool
    verification_link: str = None
    smtp_configured: bool = False
    email_error: str = None


# ─── JWT 토큰 import ──────────────────────────────────────────────
try:
    from utils.jwt_utils import create_access_token

    JWT_AVAILABLE = True
    logger.info("✅ JWT 유틸리티 로드 성공")
except ImportError:
    JWT_AVAILABLE = False
    logger.warning("⚠️ JWT 유틸리티 없음 - JWT 토큰 발급 불가")


    def create_access_token(user_data):
        return None


# ─── 로그인/회원가입 API들 ─────────────────────────────────────────

# 🆕 이메일/비밀번호 로그인 (JWT 토큰 발급)
@router.post("/login", summary="이메일/비밀번호 로그인")
async def login_with_email(request: EmailPasswordLogin):
    """이메일/비밀번호 로그인 후 JWT 토큰 발급"""
    logger.info(f"🔐 로그인 요청: {request.email}")

    try:
        # 1. 사용자 존재 확인
        user_record = auth.get_user_by_email(request.email)
        logger.info(f"✅ 사용자 확인 완료: {user_record.uid}")

        # 2. JWT 토큰 발급용 사용자 정보 구성
        user_data = {
            'uid': user_record.uid,
            'email': user_record.email,
            'name': user_record.display_name or request.email.split('@')[0],
            'picture': user_record.photo_url or '',
            'email_verified': user_record.email_verified
        }

        # 3. JWT 토큰 생성 (가능한 경우)
        jwt_token = None
        if JWT_AVAILABLE:
            jwt_token = create_access_token(user_data)
            logger.info(f"✅ JWT 토큰 생성 완료: {request.email}")

        # 4. 사용자 정보 DB 저장/업데이트
        await save_user(
            uid=user_record.uid,
            email=user_record.email,
            name=user_data['name']
        )

        # 5. 응답 구성
        response_content = {
            "message": "로그인이 완료되었습니다.",
            "user_exists": True,
            "email_verified": user_record.email_verified,
            "uid": user_record.uid,
            "user": {
                "uid": user_record.uid,
                "email": user_record.email,
                "name": user_data['name'],
                "picture": user_data['picture'],
                "email_verified": user_record.email_verified
            }
        }

        # JWT 토큰이 생성된 경우에만 추가
        if jwt_token:
            response_content.update({
                "token": jwt_token,
                "token_type": "Bearer",
                "expires_in": 86400
            })


# ─── JWT 토큰 검증 및 기타 API들 ──────────────────────────────────────

# 🆕 JWT 토큰 테스트 API
@router.get("/test-jwt", summary="JWT 토큰 테스트")
async def test_jwt_token(user=Depends(verify_jwt_only)):
    """JWT 토큰 유효성 테스트"""
    return {
        "message": f"{user.get('name', '알 수 없음')}님의 JWT 토큰이 유효합니다.",
        "user": {
            "uid": user["uid"],
            "email": user.get("email"),
            "name": user.get("name"),
            "picture": user.get("picture", "")
        },
        "token_info": {
            "type": "JWT",
            "email_verified": user.get("email_verified", False)
        }
    }


# 🆕 사용자 정보 조회 API (JWT 토큰 필요)
@router.get("/me", summary="내 정보 조회")
async def get_my_info(user=Depends(verify_jwt_only)):
    """JWT 토큰으로 사용자 정보 조회"""
    return {
        "uid": user["uid"],
        "email": user.get("email"),
        "name": user.get("name"),
        "picture": user.get("picture", ""),
        "email_verified": user.get("email_verified", False)
    }


# 🧪 테스트용 JWT 토큰 생성 API (개발 전용)
@router.post("/create-test-token", summary="테스트용 JWT 토큰 생성")
async def create_test_token(email: str = "test@whiff.com", name: str = "테스트 사용자"):
    """개발/테스트용 JWT 토큰 생성"""

    if not JWT_AVAILABLE:
        raise HTTPException(status_code=503, detail="JWT 기능이 비활성화되어 있습니다.")

    try:
        # 테스트용 사용자 데이터
        test_user_data = {
            'uid': f'test-user-{datetime.now().strftime("%Y%m%d%H%M%S")}',
            'email': email,
            'name': name,
            'picture': '',
            'email_verified': True
        }

        # JWT 토큰 생성
        jwt_token = create_access_token(test_user_data)

        logger.info(f"🧪 테스트 JWT 토큰 생성: {email}")

        return JSONResponse(content={
            "message": "테스트용 JWT 토큰이 생성되었습니다.",
            "token": jwt_token,
            "user": test_user_data,
            "token_type": "Bearer",
            "expires_in": 86400,
            "usage": {
                "header": f"Authorization: Bearer {jwt_token}",
                "test_endpoints": [
                    "GET /users/test-jwt",
                    "GET /users/me",
                    "GET /auth/test-jwt"
                ]
            },
            "note": "이 토큰은 테스트 전용입니다."
        })

    except Exception as e:
        logger.error(f"❌ 테스트 토큰 생성 실패: {e}")
        raise HTTPException(status_code=500, detail=f"테스트 토큰 생성 실패: {str(e)}")


# ─── 기존 API들 (Firebase 토큰 사용) ─────────────────────────────

@router.post("/logout", summary="로그아웃")
async def logout(user=Depends(verify_firebase_token_optional)):
    """로그아웃"""
    try:
        # JWT는 stateless이므로 서버에서 할 일은 없음
        # Firebase 토큰이면 revoke 처리
        if "firebase" in str(type(user)).lower():
            auth.revoke_refresh_tokens(user["uid"])

        return JSONResponse(content={
            "message": "로그아웃이 완료되었습니다.",
            "uid": user["uid"],
            "note": "JWT 토큰은 클라이언트에서 삭제해주세요."
        })
    except Exception as e:
        logging.error(f"Logout error: {e}")
        raise HTTPException(status_code=500, detail="로그아웃 중 오류가 발생했습니다.")


@router.get("/firebase-status", summary="Firebase 상태 확인")
async def check_firebase_status():
    """Firebase 상태 확인"""
    return get_firebase_status()


# ✅ 이메일 발송 상태 확인 API
@router.get("/email-status", summary="이메일 발송 상태 확인")
async def check_email_status():
    """SMTP 설정 상태와 이메일 발송 가능 여부를 확인합니다."""
    logger.info("📧 이메일 상태 확인 요청")

    # SMTP 설정 확인
    config_valid, config_message = email_sender.check_smtp_config()

    # 환경변수 상태
    env_status = {
        "SMTP_SERVER": os.getenv('SMTP_SERVER', 'smtp.gmail.com'),
        "SMTP_PORT": os.getenv('SMTP_PORT', '587'),
        "SMTP_USERNAME": "설정됨" if os.getenv('SMTP_USERNAME') else "❌ 없음",
        "SMTP_PASSWORD": "설정됨" if os.getenv('SMTP_PASSWORD') else "❌ 없음",
        "FROM_EMAIL": os.getenv('FROM_EMAIL', '기본값: SMTP_USERNAME 사용')
    }

    response = {
        "smtp_configured": config_valid,
        "config_message": config_message,
        "environment_variables": env_status,
        "email_sending_available": config_valid,
        "jwt_available": JWT_AVAILABLE
    }

    logger.info(f"📧 이메일 상태: {'✅ 사용 가능' if config_valid else '❌ 설정 필요'}")

    return JSONResponse(content=response)


@router.post("/test", summary="Firebase 토큰 유효성 테스트 (구버전)")
async def test_token(user=Depends(verify_firebase_token_optional)):
    """Firebase 토큰 테스트 (구버전 호환)"""
    return {
        "message": f"{user.get('name', '알 수 없음')}님, 인증되었습니다.",
        "uid": user["uid"],
        "email": user.get("email"),
        "note": "이 API는 구버전 호환용입니다. JWT API를 사용하세요."
    }
    content = response_content)

    except auth.UserNotFoundError:
    logger.warning(f"⚠️ 존재하지 않는 사용자: {request.email}")
    raise HTTPException(status_code=404, detail="존재하지 않는 사용자입니다.")

except Exception as e:
logger.error(f"❌ 로그인 처리 중 오류: {e}")
raise HTTPException(status_code=500, detail="로그인 처리 중 오류가 발생했습니다.")


# ✅ 구글 로그인 (★ 기존 방식 그대로 유지 ★)
@router.post("/google-login", summary="구글 로그인")
async def google_login(request: GoogleLoginRequest):
    """구글 로그인 (기존 연동 방식 유지 - JWT 토큰 없음)"""
    try:
        decoded_token = auth.verify_id_token(request.id_token)
        uid = decoded_token["uid"]
        email = decoded_token.get("email")
        name = decoded_token.get("name")
        picture = decoded_token.get("picture")

        await save_user(uid=uid, email=email, name=name, picture=picture)

        # ★ 기존 응답 형태 그대로 유지 (JWT 토큰 없음) ★
        return JSONResponse(
            content={
                "message": "구글 로그인이 완료되었습니다.",
                "user": {"uid": uid, "email": email, "name": name, "picture": picture}
            }
        )
    except auth.InvalidIdTokenError:
        raise HTTPException(status_code=401, detail="유효하지 않은 구글 토큰입니다.")
    except Exception as e:
        logging.error(f"Google login error: {e}")
        raise HTTPException(status_code=500, detail="구글 로그인 중 오류가 발생했습니다.")


# 🆕 이메일/비밀번호 회원가입 (JWT 토큰 발급)
@router.post("/register", summary="이메일/비밀번호 회원가입", response_model=RegisterResponse)
async def register_with_email(request: EmailPasswordRegister):
    """회원가입 후 JWT 토큰 발급"""
    logger.info(f"🚀 회원가입 요청 시작: {request.email}")

    try:
        # 1. Firebase 사용자 생성
        logger.info(f"🔥 Firebase 사용자 생성 시작...")
        user_record = auth.create_user(
            email=request.email,
            password=request.password,
            display_name=request.name,
            email_verified=False
        )
        logger.info(f"✅ Firebase 사용자 생성 완료: uid={user_record.uid}")

        # 2. JWT 토큰 생성용 사용자 정보
        user_data = {
            'uid': user_record.uid,
            'email': request.email,
            'name': request.name,
            'picture': '',
            'email_verified': False
        }

        # 3. JWT 토큰 생성 (가능한 경우)
        jwt_token = None
        if JWT_AVAILABLE:
            jwt_token = create_access_token(user_data)
            logger.info(f"✅ JWT 토큰 생성 완료: {request.email}")

        # 4. 사용자 정보 DB 저장
        await save_user(
            uid=user_record.uid,
            email=request.email,
            name=request.name
        )

        # 5. 이메일 인증 링크 생성 (선택적)
        verification_link = None
        email_sent = False
        email_error = None

        try:
            verification_link = auth.generate_email_verification_link(request.email)
            logger.info(f"✅ 이메일 인증 링크 생성 완료")

            # SMTP 설정 확인 및 이메일 발송
            smtp_configured, smtp_message = email_sender.check_smtp_config()
            if smtp_configured:
                email_sent, email_message = email_sender.send_verification_email(
                    to_email=request.email,
                    verification_link=verification_link,
                    user_name=request.name
                )
                if not email_sent:
                    email_error = email_message
            else:
                email_error = "SMTP 설정이 완료되지 않았습니다."

        except Exception as e:
            logger.warning(f"⚠️ 이메일 처리 실패: {e}")
            email_error = str(e)

        # 6. 응답 생성
        response_data = {
            "message": "회원가입이 완료되었습니다.",
            "uid": user_record.uid,
            "email": request.email,
            "user": {
                "uid": user_record.uid,
                "email": request.email,
                "name": request.name,
                "picture": "",
                "email_verified": False
            },
            "email_sent": email_sent,
            "smtp_configured": smtp_configured
        }

        # JWT 토큰이 생성된 경우에만 추가
        if jwt_token:
            response_data.update({
                "token": jwt_token,
                "token_type": "Bearer",
                "expires_in": 86400
            })

        if not email_sent and email_error:
            response_data["email_error"] = email_error
        if verification_link:
            response_data["verification_link"] = verification_link

        logger.info(f"🎉 회원가입 처리 완료 - JWT 토큰: {'✅' if jwt_token else '❌'}")

        return JSONResponse(
            status_code=201,
            content=response_data
        )

    except auth.EmailAlreadyExistsError:
        logger.warning(f"⚠️ 이미 존재하는 이메일: {request.email}")
        raise HTTPException(status_code=400, detail="이미 존재하는 이메일 주소입니다.")
    except Exception as e:
        logger.error(f"❌ 회원가입 중 예외 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=f"회원가입 중 오류가 발생했습니다: {str(e)}")