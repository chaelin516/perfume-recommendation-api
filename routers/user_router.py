from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from utils.auth_utils import verify_firebase_token_optional, get_firebase_status
from firebase_admin import auth
import logging
import json
import os
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/users", tags=["User"])


# 회원 탈퇴 요청 스키마
class WithdrawRequest(BaseModel):
    reason: Optional[str] = None
    feedback: Optional[str] = None
    confirm_password: Optional[str] = None  # 추가 보안용 (선택사항)


class WithdrawResponse(BaseModel):
    message: str
    deleted_data: dict
    withdraw_date: str
    note: str


# 데이터 파일 경로들
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USER_DATA_PATH = os.path.join(BASE_DIR, "../data/user_data.json")
DIARY_DATA_PATH = os.path.join(BASE_DIR, "../data/diary_data.json")
TEMP_USERS_PATH = os.path.join(BASE_DIR, "../data/temp_users.json")


def load_json_file(file_path: str) -> list:
    """JSON 파일 로딩"""
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"❌ {file_path} 로딩 실패: {e}")
            return []
    return []


def save_json_file(file_path: str, data: list) -> bool:
    """JSON 파일 저장"""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        logger.error(f"❌ {file_path} 저장 실패: {e}")
        return False


async def delete_user_data(user_id: str) -> dict:
    """사용자 관련 모든 데이터 삭제"""
    deleted_data = {
        "user_profile": 0,
        "diaries": 0,
        "temp_users": 0,
        "recommendations": 0  # SQLite 데이터는 별도 처리 필요
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


# ✅ 로그인된 사용자 정보 조회
@router.get(
    "/me",
    summary="내 정보 조회",
    description="현재 로그인한 사용자의 정보를 반환합니다."
)
async def get_my_info(user=Depends(verify_firebase_token_optional)):
    uid = user["uid"]

    # 🔁 사용자 정보 조회 함수 정의 (간단한 목업)
    # 추후 실제 DB 연동이 필요하면 models.user_model 쪽으로 분리 가능
    user_info = {
        "uid": uid,
        "email": user.get("email", ""),
        "name": user.get("name", ""),
        "picture": user.get("picture", ""),
        "is_test_user": uid.startswith("test-")  # 테스트 사용자 여부
    }

    return {
        "message": "사용자 정보 조회 성공",
        "data": user_info,
        "firebase_status": get_firebase_status()
    }


# ✅ 사용자 설정 정보 조회 (더미 데이터)
@router.get(
    "/settings",
    summary="사용자 설정 조회",
    description="사용자의 설정 정보를 반환합니다."
)
async def get_user_settings(user=Depends(verify_firebase_token_optional)):
    # 더미 설정 데이터
    settings = {
        "notification_enabled": True,
        "public_profile": True,
        "preferred_language": "ko",
        "theme": "light",
        "marketing_consent": False
    }

    return {
        "message": "사용자 설정 조회 성공",
        "data": settings
    }


# ✅ 사용자 프로필 업데이트 (더미 기능)
@router.put(
    "/profile",
    summary="프로필 업데이트",
    description="사용자 프로필 정보를 업데이트합니다."
)
async def update_user_profile(
        name: str = None,
        bio: str = None,
        user=Depends(verify_firebase_token_optional)
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
            "updated_at": "2025-05-30T12:00:00Z"
        }
    }


# ✅ 사용자 통계 정보
@router.get(
    "/stats",
    summary="사용자 통계",
    description="사용자의 활동 통계를 반환합니다."
)
async def get_user_stats(user=Depends(verify_firebase_token_optional)):
    # 더미 통계 데이터
    stats = {
        "total_diaries": 5,
        "total_likes_received": 12,
        "total_comments": 3,
        "favorite_perfume_brands": ["Dior", "Chanel", "Tom Ford"],
        "most_used_emotions": ["elegant", "fresh", "romantic"],
        "joined_date": "2025-01-15",
        "days_active": 45
    }

    return {
        "message": "사용자 통계 조회 성공",
        "data": stats
    }


# 🆕 회원 탈퇴 API
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
        user=Depends(verify_firebase_token_optional)
):
    """회원 탈퇴 API"""
    uid = user["uid"]
    email = user.get("email", "")
    name = user.get("name", "익명 사용자")

    logger.info(f"🚪 회원 탈퇴 요청 시작")
    logger.info(f"  - 사용자: {name} ({email})")
    logger.info(f"  - UID: {uid}")
    logger.info(f"  - 탈퇴 사유: {request.reason or '미제공'}")

    try:
        # 1. 사용자 관련 데이터 삭제
        logger.info("🗑️ 사용자 데이터 삭제 시작...")
        deleted_data = await delete_user_data(uid)
        logger.info(f"✅ 사용자 데이터 삭제 완료: {deleted_data}")

        # 2. Firebase에서 사용자 삭제
        logger.info("🔥 Firebase 사용자 삭제 시작...")

        # 테스트 사용자인 경우 Firebase 삭제 건너뛰기
        if uid.startswith("test-") or uid.startswith("temp-"):
            logger.info("🧪 테스트/임시 사용자 - Firebase 삭제 건너뛰기")
            firebase_deleted = False
        else:
            try:
                auth.delete_user(uid)
                firebase_deleted = True
                logger.info(f"✅ Firebase 사용자 삭제 완료: {uid}")
            except Exception as e:
                logger.error(f"❌ Firebase 사용자 삭제 실패: {e}")
                firebase_deleted = False

        # 3. 탈퇴 로그 기록 (선택사항)
        withdraw_log = {
            "uid": uid,
            "email": email,
            "name": name,
            "reason": request.reason,
            "feedback": request.feedback,
            "deleted_data": deleted_data,
            "firebase_deleted": firebase_deleted,
            "withdraw_date": datetime.now().isoformat(),
            "withdraw_ip": "unknown"  # 실제로는 request에서 가져올 수 있음
        }

        # 탈퇴 로그 저장 (선택사항)
        try:
            withdraw_log_path = os.path.join(BASE_DIR, "../data/withdraw_logs.json")
            withdraw_logs = load_json_file(withdraw_log_path)
            withdraw_logs.append(withdraw_log)
            save_json_file(withdraw_log_path, withdraw_logs)
            logger.info("📝 탈퇴 로그 기록 완료")
        except Exception as e:
            logger.warning(f"⚠️ 탈퇴 로그 기록 실패: {e}")

        # 4. 탈퇴 확인 이메일 발송 (선택사항)
        try:
            from utils.email_sender import email_sender

            if email and email != "test@example.com":
                smtp_configured, _ = email_sender.check_smtp_config()

                if smtp_configured:
                    logger.info("📧 탈퇴 확인 이메일 발송 시작...")

                    # 간단한 탈퇴 확인 이메일 (실제로는 별도 함수로 분리)
                    subject = "Whiff - 회원 탈퇴가 완료되었습니다"

                    html_body = f"""
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <meta charset="UTF-8">
                        <title>회원 탈퇴 완료</title>
                        <style>
                            body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                            .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                            .header {{ color: #ff6b6b; text-align: center; }}
                            .footer {{ font-size: 12px; color: #666; margin-top: 30px; }}
                        </style>
                    </head>
                    <body>
                        <div class="container">
                            <h2 class="header">👋 Whiff 회원 탈퇴 완료</h2>
                            <p>안녕하세요, <strong>{name}</strong>님</p>
                            <p>Whiff 서비스 회원 탈퇴가 정상적으로 완료되었습니다.</p>
                            <p><strong>탈퇴 일시:</strong> {datetime.now().strftime('%Y년 %m월 %d일 %H:%M')}</p>
                            <p>그동안 Whiff를 이용해주셔서 감사했습니다.</p>
                            <div class="footer">
                                <hr>
                                <p>더 나은 서비스로 다시 만날 수 있기를 기대합니다.</p>
                                <p><strong>Whiff 팀</strong> 드림</p>
                            </div>
                        </div>
                    </body>
                    </html>
                    """

                    text_body = f"""
Whiff 회원 탈퇴 완료

안녕하세요, {name}님

Whiff 서비스 회원 탈퇴가 정상적으로 완료되었습니다.

탈퇴 일시: {datetime.now().strftime('%Y년 %m월 %d일 %H:%M')}

그동안 Whiff를 이용해주셔서 감사했습니다.
더 나은 서비스로 다시 만날 수 있기를 기대합니다.

Whiff 팀 드림
                    """

                    email_sent, email_message = email_sender._send_email(
                        to_email=email,
                        subject=subject,
                        html_body=html_body,
                        text_body=text_body
                    )

                    if email_sent:
                        logger.info("✅ 탈퇴 확인 이메일 발송 성공")
                    else:
                        logger.warning(f"⚠️ 탈퇴 확인 이메일 발송 실패: {email_message}")

        except Exception as e:
            logger.warning(f"⚠️ 탈퇴 확인 이메일 발송 중 오류: {e}")

        # 5. 응답 반환
        response = WithdrawResponse(
            message="회원 탈퇴가 성공적으로 완료되었습니다. 그동안 Whiff를 이용해주셔서 감사했습니다.",
            deleted_data=deleted_data,
            withdraw_date=datetime.now().isoformat(),
            note="모든 개인 데이터가 영구적으로 삭제되었습니다. 이 작업은 되돌릴 수 없습니다."
        )

        logger.info(f"🎉 회원 탈퇴 처리 완료")
        logger.info(f"  - 삭제된 데이터: {deleted_data}")
        logger.info(f"  - Firebase 삭제: {'✅' if firebase_deleted else '❌'}")

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
async def preview_withdraw(user=Depends(verify_firebase_token_optional)):
    """회원 탈퇴 전 삭제될 데이터 미리보기"""
    uid = user["uid"]
    email = user.get("email", "")
    name = user.get("name", "익명 사용자")

    logger.info(f"🔍 회원 탈퇴 사전 확인 요청: {name} ({email})")

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
                "user_info": {
                    "uid": uid,
                    "email": email,
                    "name": name
                },
                "data_to_delete": preview_data,
                "total_items": sum(preview_data.values()),
                "warning": "회원 탈퇴 시 모든 데이터가 영구적으로 삭제되며, 이 작업은 되돌릴 수 없습니다.",
                "note": "탈퇴 전에 필요한 데이터가 있다면 미리 백업해주세요."
            }
        )

    except Exception as e:
        logger.error(f"❌ 회원 탈퇴 사전 확인 중 오류: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"사전 확인 중 오류가 발생했습니다: {str(e)}"
        )