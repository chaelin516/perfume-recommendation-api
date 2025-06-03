# routers/auth_router.py - 회원 탈퇴 및 비밀번호 재설정 기능 추가

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr, Field, validator
from utils.auth_utils import verify_firebase_token_optional, get_firebase_status
from utils.email_sender import email_sender  # 새로 추가된 이메일 발송 기능
from models.user_model import save_user
from firebase_admin import auth
import logging
import os

router = APIRouter(prefix="/auth", tags=["Auth"])

# 로거 설정
logger = logging.getLogger(__name__)


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
    email_sent: bool
    verification_link: str = None
    smtp_configured: bool = False
    email_error: str = None


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
        "email_sending_available": config_valid
    }

    logger.info(f"📧 이메일 상태: {'✅ 사용 가능' if config_valid else '❌ 설정 필요'}")

    return JSONResponse(content=response)


# ✅ SMTP 연결 테스트 API
@router.post("/test-smtp", summary="SMTP 연결 테스트")
async def test_smtp_connection():
    """SMTP 서버 연결을 실제로 테스트합니다."""
    logger.info("🧪 SMTP 연결 테스트 시작")

    try:
        success, message = email_sender.test_smtp_connection()

        if success:
            logger.info(f"✅ SMTP 테스트 성공: {message}")
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "message": message,
                    "timestamp": logger.handlers[0].formatter.formatTime(
                        logger.makeRecord("", 0, "", 0, "", (), None)) if logger.handlers else None
                }
            )
        else:
            logger.error(f"❌ SMTP 테스트 실패: {message}")
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "error": message,
                    "suggestion": "SMTP 환경변수 설정을 확인하세요."
                }
            )

    except Exception as e:
        logger.error(f"❌ SMTP 테스트 중 예외: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": f"테스트 중 오류: {str(e)}"
            }
        )


# ✅ 기존 토큰 테스트 API
@router.post("/test", summary="Firebase 토큰 유효성 테스트")
async def test_token(user=Depends(verify_firebase_token_optional)):
    return {
        "message": f"{user.get('name', '알 수 없음')}님, 인증되었습니다.",
        "uid": user["uid"],
        "email": user.get("email")
    }


# 🆕 이메일/비밀번호 회원가입 (실제 이메일 발송 포함)
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
    logger.info(f"🚀 회원가입 요청 시작")
    logger.info(f"  - 이메일: {request.email}")
    logger.info(f"  - 이름: {request.name}")
    logger.info(f"  - 비밀번호 길이: {len(request.password)}자")

    try:
        # 1. Firebase에서 사용자 생성
        logger.info(f"🔥 Firebase 사용자 생성 시작...")
        user_record = auth.create_user(
            email=request.email,
            password=request.password,
            display_name=request.name,
            email_verified=False
        )
        logger.info(f"✅ Firebase 사용자 생성 완료: uid={user_record.uid}")

        # 2. 이메일 인증 링크 생성
        logger.info(f"📧 이메일 인증 링크 생성 시작...")
        try:
            verification_link = auth.generate_email_verification_link(request.email)
            logger.info(f"✅ 이메일 인증 링크 생성 완료")
            logger.info(f"  - 링크 길이: {len(verification_link)}자")
        except Exception as e:
            logger.error(f"❌ 이메일 인증 링크 생성 실패: {e}")
            verification_link = None

        # 3. 사용자 정보 DB 저장
        logger.info(f"💾 사용자 정보 DB 저장 시작...")
        try:
            await save_user(
                uid=user_record.uid,
                email=request.email,
                name=request.name
            )
            logger.info(f"✅ 사용자 정보 DB 저장 완료")
        except Exception as e:
            logger.error(f"❌ 사용자 정보 저장 실패: {e}")

        # 4. SMTP 설정 확인
        logger.info(f"📧 SMTP 설정 확인...")
        smtp_configured, smtp_message = email_sender.check_smtp_config()
        logger.info(f"  - SMTP 설정: {'✅ 완료' if smtp_configured else '❌ 미완료'}")
        logger.info(f"  - 메시지: {smtp_message}")

        # 5. 실제 이메일 발송 시도
        email_sent = False
        email_error = None

        if smtp_configured and verification_link:
            logger.info(f"📮 이메일 발송 시작...")
            try:
                email_sent, email_message = email_sender.send_verification_email(
                    to_email=request.email,
                    verification_link=verification_link,
                    user_name=request.name
                )

                if email_sent:
                    logger.info(f"✅ 이메일 발송 성공: {email_message}")
                else:
                    logger.error(f"❌ 이메일 발송 실패: {email_message}")
                    email_error = email_message

            except Exception as e:
                logger.error(f"❌ 이메일 발송 중 예외: {str(e)}")
                email_sent = False
                email_error = f"이메일 발송 예외: {str(e)}"
        else:
            if not smtp_configured:
                email_error = "SMTP 설정이 완료되지 않았습니다."
                logger.warning(f"⚠️ {email_error}")
            if not verification_link:
                email_error = "이메일 인증 링크 생성에 실패했습니다."
                logger.warning(f"⚠️ {email_error}")

        # 6. 응답 생성
        response_data = {
            "message": "회원가입이 완료되었습니다." + (
                " 이메일을 확인해서 인증을 완료해주세요." if email_sent
                else " 이메일 발송에 실패했습니다."
            ),
            "uid": user_record.uid,
            "email": request.email,
            "email_sent": email_sent,
            "smtp_configured": smtp_configured
        }

        # 이메일 발송 실패 시 추가 정보 제공
        if not email_sent:
            response_data["email_error"] = email_error
            if verification_link:
                response_data["verification_link"] = verification_link
                response_data["manual_verification_note"] = "위 링크를 브라우저에서 직접 열어 인증할 수 있습니다."

        logger.info(f"🎉 회원가입 처리 완료")
        logger.info(f"  - 사용자 생성: ✅")
        logger.info(f"  - DB 저장: ✅")
        logger.info(f"  - 이메일 발송: {'✅' if email_sent else '❌'}")

        return JSONResponse(
            status_code=201,
            content=response_data
        )

    except auth.EmailAlreadyExistsError:
        logger.warning(f"⚠️ 이미 존재하는 이메일: {request.email}")
        raise HTTPException(
            status_code=400,
            detail="이미 존재하는 이메일 주소입니다."
        )
    except auth.WeakPasswordError as e:
        logger.warning(f"⚠️ 약한 비밀번호: {e}")
        raise HTTPException(
            status_code=400,
            detail="비밀번호가 너무 약합니다. 최소 6자 이상의 비밀번호를 사용해주세요."
        )
    except auth.InvalidEmailError:
        logger.warning(f"⚠️ 잘못된 이메일 형식: {request.email}")
        raise HTTPException(
            status_code=400,
            detail="올바른 이메일 형식이 아닙니다."
        )
    except Exception as e:
        logger.error(f"❌ 회원가입 중 예외 발생: {str(e)}")
        logger.error(f"  - Exception Type: {type(e).__name__}")
        raise HTTPException(
            status_code=500,
            detail=f"회원가입 중 오류가 발생했습니다: {str(e)}"
        )


# 🆕 이메일 인증 재발송 API (실제 이메일 발송 포함)
@router.post("/resend-verification", summary="이메일 인증 재발송")
async def resend_verification_email(request: VerifyEmailRequest):
    logger.info(f"🔄 이메일 인증 재발송 요청")

    try:
        # ID 토큰에서 사용자 정보 추출
        decoded_token = auth.verify_id_token(request.id_token)
        email = decoded_token.get("email")
        name = decoded_token.get("name", "사용자")
        uid = decoded_token.get("uid")

        logger.info(f"  - 사용자: {name} ({email})")
        logger.info(f"  - UID: {uid}")

        if not email:
            raise HTTPException(
                status_code=400,
                detail="토큰에서 이메일 정보를 찾을 수 없습니다."
            )

        # 이메일 인증 링크 생성
        logger.info(f"📧 이메일 인증 링크 재생성...")
        verification_link = auth.generate_email_verification_link(email)
        logger.info(f"✅ 이메일 인증 링크 재생성 완료")

        # SMTP 설정 확인 및 이메일 발송
        smtp_configured, smtp_message = email_sender.check_smtp_config()

        if smtp_configured:
            logger.info(f"📮 이메일 재발송 시작...")
            email_sent, email_message = email_sender.send_verification_email(
                to_email=email,
                verification_link=verification_link,
                user_name=name
            )

            if email_sent:
                logger.info(f"✅ 이메일 재발송 성공")
                return JSONResponse(
                    content={
                        "message": "인증 이메일이 재발송되었습니다.",
                        "email": email,
                        "email_sent": True
                    }
                )
            else:
                logger.error(f"❌ 이메일 재발송 실패: {email_message}")
                return JSONResponse(
                    status_code=500,
                    content={
                        "message": "이메일 발송에 실패했습니다.",
                        "error": email_message,
                        "verification_link": verification_link,
                        "manual_note": "위 링크를 브라우저에서 직접 열어 인증할 수 있습니다."
                    }
                )
        else:
            logger.warning(f"⚠️ SMTP 설정 미완료: {smtp_message}")
            return JSONResponse(
                content={
                    "message": "SMTP 설정이 완료되지 않아 이메일을 발송할 수 없습니다.",
                    "smtp_error": smtp_message,
                    "verification_link": verification_link,
                    "note": "위 링크를 브라우저에서 직접 열어 인증할 수 있습니다."
                }
            )

    except auth.InvalidIdTokenError:
        logger.error(f"❌ 유효하지 않은 토큰")
        raise HTTPException(
            status_code=401,
            detail="유효하지 않은 토큰입니다."
        )
    except Exception as e:
        logger.error(f"❌ 이메일 재발송 중 오류: {e}")
        raise HTTPException(
            status_code=500,
            detail="이메일 인증 재발송 중 오류가 발생했습니다."
        )


# 🆕 비밀번호 재설정 이메일 발송 API
@router.post(
    "/forgot-password",
    summary="비밀번호 재설정 이메일 발송",
    description="비밀번호를 잊어버린 사용자에게 재설정 이메일을 발송합니다.",
    responses={
        200: {"description": "비밀번호 재설정 이메일 발송 성공"},
        404: {"description": "존재하지 않는 이메일"},
        500: {"description": "서버 내부 오류"}
    }
)
async def forgot_password(request: ForgotPasswordRequest):
    """비밀번호 재설정 이메일 발송"""
    logger.info(f"🔑 비밀번호 재설정 요청")
    logger.info(f"  - 이메일: {request.email}")

    try:
        # 1. 사용자 존재 확인
        logger.info(f"👤 사용자 존재 확인...")
        try:
            user_record = auth.get_user_by_email(request.email)
            logger.info(f"✅ 사용자 확인 완료: {user_record.uid}")
        except auth.UserNotFoundError:
            logger.warning(f"⚠️ 존재하지 않는 사용자: {request.email}")
            raise HTTPException(
                status_code=404,
                detail="해당 이메일로 가입된 계정을 찾을 수 없습니다."
            )

        # 2. 비밀번호 재설정 링크 생성
        logger.info(f"🔗 비밀번호 재설정 링크 생성...")
        try:
            reset_link = auth.generate_password_reset_link(request.email)
            logger.info(f"✅ 비밀번호 재설정 링크 생성 완료")
            logger.info(f"  - 링크 길이: {len(reset_link)}자")
        except Exception as e:
            logger.error(f"❌ 비밀번호 재설정 링크 생성 실패: {e}")
            raise HTTPException(
                status_code=500,
                detail="비밀번호 재설정 링크 생성에 실패했습니다."
            )

        # 3. SMTP 설정 확인
        logger.info(f"📧 SMTP 설정 확인...")
        smtp_configured, smtp_message = email_sender.check_smtp_config()
        logger.info(f"  - SMTP 설정: {'✅ 완료' if smtp_configured else '❌ 미완료'}")

        # 4. 이메일 발송
        email_sent = False
        email_error = None

        if smtp_configured:
            logger.info(f"📮 비밀번호 재설정 이메일 발송 시작...")
            try:
                email_sent, email_message = email_sender.send_password_reset_email(
                    to_email=request.email,
                    reset_link=reset_link,
                    user_name=user_record.display_name or "사용자"
                )

                if email_sent:
                    logger.info(f"✅ 비밀번호 재설정 이메일 발송 성공")
                else:
                    logger.error(f"❌ 비밀번호 재설정 이메일 발송 실패: {email_message}")
                    email_error = email_message

            except Exception as e:
                logger.error(f"❌ 비밀번호 재설정 이메일 발송 중 예외: {str(e)}")
                email_sent = False
                email_error = f"이메일 발송 예외: {str(e)}"
        else:
            email_error = f"SMTP 설정 미완료: {smtp_message}"
            logger.warning(f"⚠️ {email_error}")

        # 5. 응답 생성
        if email_sent:
            return JSONResponse(
                content={
                    "message": "비밀번호 재설정 이메일이 발송되었습니다. 이메일을 확인해주세요.",
                    "email": request.email,
                    "email_sent": True,
                    "note": "이메일이 도착하지 않으면 스팸 폴더를 확인해주세요."
                }
            )
        else:
            return JSONResponse(
                status_code=500,
                content={
                    "message": "비밀번호 재설정 이메일 발송에 실패했습니다.",
                    "email": request.email,
                    "email_sent": False,
                    "error": email_error,
                    "reset_link": reset_link,
                    "manual_note": "위 링크를 브라우저에서 직접 열어 비밀번호를 재설정할 수 있습니다."
                }
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 비밀번호 재설정 처리 중 오류: {e}")
        logger.error(f"  - Exception Type: {type(e).__name__}")
        raise HTTPException(
            status_code=500,
            detail=f"비밀번호 재설정 처리 중 오류가 발생했습니다: {str(e)}"
        )


# 🆕 테스트 이메일 발송 API
@router.post("/send-test-email", summary="테스트 이메일 발송")
async def send_test_email(email: EmailStr):
    """개발/디버깅용 테스트 이메일을 발송합니다."""
    logger.info(f"🧪 테스트 이메일 발송 요청: {email}")

    try:
        # 테스트 링크 생성
        test_link = "https://example.com/test-verification"

        # 이메일 발송
        email_sent, message = email_sender.send_verification_email(
            to_email=email,
            verification_link=test_link,
            user_name="테스트 사용자"
        )

        return JSONResponse(
            content={
                "message": "테스트 이메일 발송 완료" if email_sent else "테스트 이메일 발송 실패",
                "email": email,
                "success": email_sent,
                "details": message
            }
        )

    except Exception as e:
        logger.error(f"❌ 테스트 이메일 발송 중 오류: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "message": "테스트 이메일 발송 중 오류가 발생했습니다.",
                "error": str(e)
            }
        )


# 기존 API들...
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
        raise HTTPException(status_code=404, detail="존재하지 않는 사용자입니다.")
    except Exception as e:
        logging.error(f"Login error: {e}")
        raise HTTPException(status_code=500, detail="로그인 확인 중 오류가 발생했습니다.")


@router.post("/google-login", summary="구글 로그인")
async def google_login(request: GoogleLoginRequest):
    try:
        decoded_token = auth.verify_id_token(request.id_token)
        uid = decoded_token["uid"]
        email = decoded_token.get("email")
        name = decoded_token.get("name")
        picture = decoded_token.get("picture")

        await save_user(uid=uid, email=email, name=name, picture=picture)

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


@router.post("/logout", summary="로그아웃")
async def logout(user=Depends(verify_firebase_token_optional)):
    try:
        auth.revoke_refresh_tokens(user["uid"])
        return JSONResponse(content={"message": "로그아웃이 완료되었습니다.", "uid": user["uid"]})
    except Exception as e:
        logging.error(f"Logout error: {e}")
        raise HTTPException(status_code=500, detail="로그아웃 중 오류가 발생했습니다.")


@router.get("/firebase-status", summary="Firebase 상태 확인")
async def check_firebase_status():
    return get_firebase_status()