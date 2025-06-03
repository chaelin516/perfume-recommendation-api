# routers/temp_auth_router.py - Firebase 없이 테스트하기 위한 임시 API

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr
import json
import os
import uuid
import hashlib
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/temp-auth", tags=["Temporary Auth"])

# 임시 사용자 데이터 파일 경로
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_USERS_PATH = os.path.join(BASE_DIR, "../data/temp_users.json")


class TempRegisterRequest(BaseModel):
    email: EmailStr
    password: str
    name: str


def load_temp_users():
    """임시 사용자 데이터 로딩"""
    if os.path.exists(TEMP_USERS_PATH):
        try:
            with open(TEMP_USERS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"❌ 임시 사용자 데이터 로딩 실패: {e}")
            return []
    return []


def save_temp_users(users_data):
    """임시 사용자 데이터 저장"""
    try:
        os.makedirs(os.path.dirname(TEMP_USERS_PATH), exist_ok=True)
        with open(TEMP_USERS_PATH, "w", encoding="utf-8") as f:
            json.dump(users_data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        logger.error(f"❌ 임시 사용자 데이터 저장 실패: {e}")
        return False


def hash_password(password: str) -> str:
    """비밀번호 해시화"""
    return hashlib.sha256(password.encode()).hexdigest()


@router.post(
    "/register",
    summary="임시 회원가입 (Firebase 없이)",
    description="Firebase 설정 없이 테스트하기 위한 임시 회원가입 API입니다.",
)
async def temp_register(request: TempRegisterRequest):
    """Firebase 없이 임시로 회원가입 테스트"""
    logger.info(f"🧪 임시 회원가입 요청: {request.email}")

    try:
        # 기존 사용자 확인
        users = load_temp_users()

        for user in users:
            if user.get("email") == request.email:
                raise HTTPException(
                    status_code=400,
                    detail="이미 존재하는 이메일 주소입니다."
                )

        # 새 사용자 생성
        new_user = {
            "uid": f"temp-{str(uuid.uuid4())[:8]}",
            "email": request.email,
            "name": request.name,
            "password_hash": hash_password(request.password),
            "created_at": datetime.now().isoformat(),
            "email_verified": False,
            "is_temp_user": True
        }

        # 사용자 목록에 추가
        users.append(new_user)

        # 파일에 저장
        if save_temp_users(users):
            logger.info(f"✅ 임시 사용자 저장 완료: {request.email}")

            return JSONResponse(
                status_code=201,
                content={
                    "message": "임시 회원가입이 완료되었습니다. (Firebase 미사용)",
                    "uid": new_user["uid"],
                    "email": request.email,
                    "name": request.name,
                    "note": "이것은 Firebase 없이 테스트하기 위한 임시 계정입니다.",
                    "temp_user": True
                }
            )
        else:
            raise HTTPException(
                status_code=500,
                detail="사용자 데이터 저장에 실패했습니다."
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 임시 회원가입 중 오류: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"임시 회원가입 중 오류가 발생했습니다: {str(e)}"
        )


@router.post(
    "/login",
    summary="임시 로그인 (Firebase 없이)",
    description="임시 사용자로 로그인하여 테스트 토큰을 발급받습니다."
)
async def temp_login(email: EmailStr, password: str):
    """임시 로그인"""
    logger.info(f"🧪 임시 로그인 요청: {email}")

    try:
        users = load_temp_users()
        password_hash = hash_password(password)

        for user in users:
            if user.get("email") == email and user.get("password_hash") == password_hash:
                # 간단한 토큰 생성 (실제 JWT는 아님)
                temp_token = f"temp-token-{user['uid']}-{int(datetime.now().timestamp())}"

                logger.info(f"✅ 임시 로그인 성공: {email}")

                return JSONResponse(
                    content={
                        "message": "임시 로그인 성공",
                        "temp_token": temp_token,
                        "user": {
                            "uid": user["uid"],
                            "email": user["email"],
                            "name": user["name"],
                            "is_temp_user": True
                        },
                        "note": "이것은 Firebase 없이 테스트하기 위한 임시 토큰입니다."
                    }
                )

        raise HTTPException(
            status_code=401,
            detail="이메일 또는 비밀번호가 올바르지 않습니다."
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 임시 로그인 중 오류: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"임시 로그인 중 오류가 발생했습니다: {str(e)}"
        )


@router.get(
    "/users",
    summary="임시 사용자 목록",
    description="저장된 임시 사용자 목록을 확인합니다."
)
async def get_temp_users():
    """임시 사용자 목록 조회"""
    try:
        users = load_temp_users()

        # 비밀번호 해시 제거
        safe_users = []
        for user in users:
            safe_user = user.copy()
            safe_user.pop("password_hash", None)
            safe_users.append(safe_user)

        return JSONResponse(
            content={
                "message": f"임시 사용자 {len(safe_users)}명 조회 완료",
                "users": safe_users,
                "total": len(safe_users)
            }
        )

    except Exception as e:
        logger.error(f"❌ 임시 사용자 목록 조회 오류: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"사용자 목록 조회 중 오류: {str(e)}"
        )


@router.delete(
    "/clear",
    summary="임시 사용자 데이터 정리",
    description="모든 임시 사용자 데이터를 삭제합니다."
)
async def clear_temp_users():
    """임시 사용자 데이터 정리"""
    try:
        if os.path.exists(TEMP_USERS_PATH):
            os.remove(TEMP_USERS_PATH)
            logger.info("🧹 임시 사용자 데이터 파일 삭제 완료")

        return JSONResponse(
            content={
                "message": "임시 사용자 데이터가 모두 정리되었습니다.",
                "cleared": True
            }
        )

    except Exception as e:
        logger.error(f"❌ 임시 사용자 데이터 정리 오류: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"데이터 정리 중 오류: {str(e)}"
        )