# routers/user_router.py - JWT 토큰 지원 추가 버전

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional
import logging
from datetime import datetime
import os

# 기존 import
from utils.auth_utils import verify_firebase_token_optional, get_firebase_status

# 🆕 JWT 지원을 위한 import 추가
from utils.auth_utils import verify_jwt_token, verify_token_flexible

# 파일 유틸리티 import (기존 코드 유지)
from utils.file_utils import load_json_file, save_json_file

router = APIRouter(prefix="/users", tags=["Users"])

# 로거 설정
logger = logging.getLogger(__name__)

# 데이터 파일 경로
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
USER_DATA_PATH = os.path.join(DATA_DIR, "users.json")
DIARY_DATA_PATH = os.path.join(DATA_DIR, "diary_data.json")
TEMP_USERS_PATH = os.path.join(DATA_DIR, "temp_users.json")


# 요청/응답 스키마
class WithdrawRequest(BaseModel):
    reason: Optional[str] = Field(None, description="탈퇴 사유")
    feedback: Optional[str] = Field(None, description="피드백")


class WithdrawResponse(BaseModel):
    message: str
    deleted_data: dict
    withdraw_date: str
    note: str


# 사용자 데이터 삭제 함수
async def delete_user_data(user_id: str) -> dict:
    """사용자 관련 모든 데이터 삭제"""
    deleted_data = {
        "user_profile": 0,
        "diaries": 0,
        "temp_users": 0,
        "recommendations": 0
    }

    try:
        # 1. 사용자 프로필 데이터 삭제
        user_data = load_json_file(USER_DATA_PATH)
        original_user_count = len(user_data)
        user_data = [user for user in user_data if user.get("uid") != user_id]
        deleted_data["user_profile"] = original_user_count - len(user_data)

        if save_json_file(USER_DATA_PATH, user_data):
            logger.info(f"✅ 사용자 프로필 데이터 삭제 완료: {deleted_data['user_profile']}건")

        # 2. 시향 일기 데이터 삭제
        diary_data = load_json_file(DIARY_DATA_PATH)
        original_diary_count = len(diary_data)
        diary_data = [diary for diary in diary_data if diary.get("user_id") != user_id]
        deleted_data["diaries"] = original_diary_count - len(diary_data)

        if save_json_file(DIARY_DATA_PATH, diary_data):
            logger.info(f"✅ 시향 일기 데이터 삭제 완료: {deleted_data['diaries']}건")

        # 3. 임시 사용자 데이터 삭제 (있는 경우)
        temp_users = load_json_file(TEMP_USERS_PATH)
        original_temp_count = len(temp_users)
        temp_users = [user for user in temp_users if user.get("uid") != user_id]
        deleted_data["temp_users"] = original_temp_count - len(temp_users)

        if deleted_data["temp_users"] > 0:
            save_json_file(TEMP_USERS_PATH, temp_users)
            logger.info(f"✅ 임시 사용자 데이터 삭제 완료: {deleted_data['temp_users']}건")

        # 4. SQLite 추천 데이터 삭제 (추천 기록)
        try:
            from sqlmodel import Session
            from db.session import get_session
            from models.recommendation import RecommendedPerfume

            session = Session(get_session().bind)
            recommendations = session.query(RecommendedPerfume).filter(
                RecommendedPerfume.user_id == user_id
            ).all()

            deleted_data["recommendations"] = len(recommendations)

            for recommendation in recommendations:
                session.delete(recommendation)

            session.commit()
            session.close()

            logger.info(f"✅ 추천 기록 데이터 삭제 완료: {deleted_data['recommendations']}건")

        except Exception as e:
            logger.error(f"❌ 추천 기록 삭제 중 오류: {e}")
            deleted_data["recommendations"] = 0

        return deleted_data

    except Exception as e:
        logger.error(f"❌ 사용자 데이터 삭제 중 오류: {e}")
        raise e


# ✅ 로그인된 사용자 정보 조회 - JWT 토큰 지원
@router.get(
    "/me",
    summary="내 정보 조회",
    description="현재 로그인한 사용자의 정보를 반환합니다. JWT 토큰 또는 Firebase ID 토큰 모두 지원합니다."
)
async def get_my_info(user=Depends(verify_token_flexible)):  # 🔄 JWT + Firebase 모두 지원
    uid = user["uid"]

    # 사용자 정보 구성
    user_info = {
        "uid": uid,
        "email": user.get("email", ""),
        "name": user.get("name", ""),
        "picture": user.get("picture", ""),
        "is_test_user": uid.startswith("test-"),  # 테스트 사용자 여부
        "token_type": "jwt" if "iss" in user and user.get("iss") == "whiff-api" else "firebase"  # 토큰 타입 구분
    }

    return {
        "message": "사용자 정보 조회 성공",
        "data": user_info,
        "firebase_status": get_firebase_status()
    }


# ✅ 사용자 설정 정보 조회 - JWT 토큰 지원
@router.get(
    "/settings",
    summary="사용자 설정 조회",
    description="사용자의 설정 정보를 반환합니다."
)
async def get_user_settings(user=Depends(verify_token_flexible)):  # 🔄 JWT 지원
    # 더미 설정 데이터
    settings = {
        "notification_enabled": True,
        "public_profile": True,
        "preferred_language": "ko",
        "theme": "light",
        "marketing_consent": False,
        "auth_method": "jwt" if "iss" in user and user.get("iss") == "whiff-api" else "firebase"
    }

    return {
        "message": "사용자 설정 조회 성공",
        "data": settings
    }


# ✅ 사용자 프로필 업데이트 - JWT 토큰 지원
@router.put(
    "/profile",
    summary="프로필 업데이트",
    description="사용자 프로필 정보를 업데이트합니다."
)
async def update_user_profile(
        name: str = None,
        bio: str = None,
        user=Depends(verify_token_flexible)  # 🔄 JWT 지원
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
            "updated_at": datetime.now().isoformat(),
            "auth_method": "jwt" if "iss" in user and user.get("iss") == "whiff-api" else "firebase"
        }
    }


# ✅ 사용자 통계 정보 - JWT 토큰 지원
@router.get(
    "/stats",
    summary="사용자 통계",
    description="사용자의 활동 통계를 반환합니다."
)
async def get_user_stats(user=Depends(verify_token_flexible)):  # 🔄 JWT 지원
    # 더미 통계 데이터
    stats = {
        "total_diaries": 5,
        "total_likes_received": 12,
        "total_comments": 3,
        "favorite_perfume_brands": ["Dior", "Chanel", "Tom Ford"],
        "most_used_emotions": ["elegant", "fresh", "romantic"],
        "joined_date": "2025-01-15",
        "days_active": 45,
        "auth_method": "jwt" if "iss" in user and user.get("iss") == "whiff-api" else "firebase"
    }

    return {
        "message": "사용자 통계 조회 성공",
        "data": stats
    }


# 🆕 회원 탈퇴 API - JWT 토큰 지원
@router.delete(
    "/me/withdraw",
    summary="회원 탈퇴",
    description="현재 로그인한 사용자의 계정을 완전히 삭제합니다. 이 작업은 되돌릴 수 없습니다.",
    response_model=WithdrawResponse,
    responses={
        200: {"description": "회원 탈퇴 성공"},
        401: {"description": "인증되지 않은 사용자"},
        403: {"description": "Firebase에서 사용자 삭제 권한 없음"},
        500: {"description": "서버 내부 오류"}
    }
)
async def withdraw_user(
        request: WithdrawRequest,
        user=Depends(verify_token_flexible)  # 🔄 JWT 지원
):
    """회원 탈퇴 API"""
    uid = user["uid"]
    email = user.get("email", "")
    name = user.get("name", "익명 사용자")
    auth_method = "jwt" if "iss" in user and user.get("iss") == "whiff-api" else "firebase"

    logger.info(f"🚪 회원 탈퇴 요청 시작")
    logger.info(f"  - 사용자: {name} ({email})")
    logger.info(f"  - UID: {uid}")
    logger.info(f"  - 인증 방식: {auth_method}")
    logger.info(f"  - 탈퇴 사유: {request.reason or '미제공'}")

    try:
        # 1. 사용자 관련 데이터 삭제
        logger.info("🗑️ 사용자 데이터 삭제 시작...")
        deleted_data = await delete_user_data(uid)
        logger.info(f"✅ 사용자 데이터 삭제 완료: {deleted_data}")

        # 2. Firebase에서 사용자 삭제
        firebase_deleted = False
        try:
            from firebase_admin import auth
            auth.delete_user(uid)
            firebase_deleted = True
            logger.info(f"✅ Firebase 사용자 삭제 완료")
        except Exception as e:
            logger.warning(f"⚠️ Firebase 사용자 삭제 실패: {e}")
            logger.warning("  - 이는 정상적인 상황일 수 있습니다 (테스트 사용자 등)")

        # 3. 응답 반환
        response = WithdrawResponse(
            message="회원 탈퇴가 성공적으로 완료되었습니다. 그동안 Whiff를 이용해주셔서 감사했습니다.",
            deleted_data=deleted_data,
            withdraw_date=datetime.now().isoformat(),
            note="모든 개인 데이터가 영구적으로 삭제되었습니다. 이 작업은 되돌릴 수 없습니다."
        )

        logger.info(f"🎉 회원 탈퇴 처리 완료")
        logger.info(f"  - 삭제된 데이터: {deleted_data}")
        logger.info(f"  - Firebase 삭제: {'✅' if firebase_deleted else '❌'}")
        logger.info(f"  - 인증 방식: {auth_method}")

        return JSONResponse(
            status_code=200,
            content=response.dict()
        )

    except Exception as e:
        logger.error(f"❌ 회원 탈퇴 처리 중 오류: {e}")
        logger.error(f"  - Exception Type: {type(e).__name__}")

        raise HTTPException(
            status_code=500,
            detail=f"회원 탈퇴 처리 중 오류가 발생했습니다: {str(e)}"
        )


# 🆕 회원 탈퇴 사전 확인 API
@router.get(
    "/me/withdraw-preview",
    summary="회원 탈퇴 사전 확인",
    description="회원 탈퇴 시 삭제될 데이터를 미리 확인합니다."
)
async def preview_withdraw(user=Depends(verify_token_flexible)):  # 🔄 JWT 지원
    """회원 탈퇴 전 삭제될 데이터 미리보기"""
    uid = user["uid"]
    email = user.get("email", "")
    name = user.get("name", "익명 사용자")
    auth_method = "jwt" if "iss" in user and user.get("iss") == "whiff-api" else "firebase"

    logger.info(f"🔍 회원 탈퇴 사전 확인 요청: {name} ({email}) - {auth_method}")

    try:
        # 삭제될 데이터 카운트
        preview_data = {
            "user_profile": 0,
            "diaries": 0,
            "temp_users": 0,
            "recommendations": 0
        }

        # 1. 사용자 프로필 확인
        user_data = load_json_file(USER_DATA_PATH)
        for user_item in user_data:
            if user_item.get("uid") == uid:
                preview_data["user_profile"] = 1
                break

        # 2. 시향 일기 확인
        diary_data = load_json_file(DIARY_DATA_PATH)
        preview_data["diaries"] = len([
            diary for diary in diary_data
            if diary.get("user_id") == uid
        ])

        # 3. 임시 사용자 확인
        temp_users = load_json_file(TEMP_USERS_PATH)
        preview_data["temp_users"] = len([
            user_item for user_item in temp_users
            if user_item.get("uid") == uid
        ])

        # 4. 추천 기록 확인
        try:
            from sqlmodel import Session
            from db.session import get_session
            from models.recommendation import RecommendedPerfume

            session = Session(get_session().bind)
            recommendation_count = session.query(RecommendedPerfume).filter(
                RecommendedPerfume.user_id == uid
            ).count()
            preview_data["recommendations"] = recommendation_count
            session.close()

        except Exception as e:
            logger.warning(f"⚠️ 추천 기록 확인 중 오류: {e}")
            preview_data["recommendations"] = 0

        return JSONResponse(
            content={
                "message": "회원 탈퇴 시 삭제될 데이터 정보입니다.",
                "user": {
                    "uid": uid,
                    "name": name,
                    "email": email,
                    "auth_method": auth_method
                },
                "data_to_delete": preview_data,
                "total_items": sum(preview_data.values()),
                "warning": "탈퇴 후에는 모든 데이터가 영구적으로 삭제되며, 복구할 수 없습니다."
            }
        )

    except Exception as e:
        logger.error(f"❌ 회원 탈퇴 사전 확인 중 오류: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"회원 탈퇴 사전 확인 중 오류가 발생했습니다: {str(e)}"
        )


# 🆕 JWT 전용 테스트 API (Flutter 앱 테스트용)
@router.get(
    "/test-jwt",
    summary="JWT 토큰 테스트",
    description="JWT 토큰 전용 테스트 API - Flutter 앱에서 JWT 토큰이 제대로 작동하는지 확인"
)
async def test_jwt_auth(user=Depends(verify_jwt_token)):  # 🔄 JWT 전용
    """JWT 토큰 전용 테스트 API"""
    return {
        "message": f"✅ JWT 인증 성공! {user.get('name')}님 환영합니다.",
        "user": {
            "uid": user["uid"],
            "email": user.get("email", ""),
            "name": user.get("name", ""),
            "token_type": "jwt"
        },
        "test_timestamp": datetime.now().isoformat(),
        "jwt_info": {
            "issued_by": user.get("iss"),
            "token_type": user.get("type"),
            "expires_at": datetime.fromtimestamp(user.get("exp")).isoformat() if user.get("exp") else None
        }
    }


# 🆕 유연한 인증 테스트 API (Firebase + JWT 모두 지원)
@router.get(
    "/test-auth-flexible",
    summary="유연한 인증 테스트",
    description="Firebase ID 토큰과 JWT 토큰 모두 지원하는 테스트 API"
)
async def test_flexible_auth(user=Depends(verify_token_flexible)):  # 🔄 모두 지원
    """유연한 인증 테스트 API"""
    token_type = "jwt" if "iss" in user and user.get("iss") == "whiff-api" else "firebase"

    return {
        "message": f"✅ {token_type.upper()} 인증 성공! {user.get('name')}님 환영합니다.",
        "user": {
            "uid": user["uid"],
            "email": user.get("email", ""),
            "name": user.get("name", ""),
            "token_type": token_type
        },
        "test_timestamp": datetime.now().isoformat(),
        "auth_details": {
            "detected_token_type": token_type,
            "supports_both": True,
            "jwt_iss": user.get("iss") if token_type == "jwt" else None
        }
    }