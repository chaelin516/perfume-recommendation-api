# models/user_model.py
import json
import os
import logging
from datetime import datetime
from typing import Optional, Dict, List
from utils.file_utils import load_json_file, save_json_file

logger = logging.getLogger(__name__)

# 데이터 파일 경로
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
USER_DATA_PATH = os.path.join(DATA_DIR, "users.json")


async def save_user(
        uid: str,
        email: str,
        name: str,
        picture: Optional[str] = None,
        provider: str = "firebase"
) -> bool:
    """사용자 정보를 JSON 파일에 저장"""
    try:
        logger.info(f"💾 사용자 저장 시작: {email} ({name})")

        # 기존 사용자 데이터 로드
        users = load_json_file(USER_DATA_PATH)

        # 사용자 정보 구성
        user_data = {
            "uid": uid,
            "email": email,
            "name": name,
            "picture": picture or "",
            "provider": provider,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "is_active": True
        }

        # 기존 사용자 확인 및 업데이트
        existing_user_index = None
        for i, user in enumerate(users):
            if user.get("uid") == uid:
                existing_user_index = i
                break

        if existing_user_index is not None:
            # 기존 사용자 업데이트 (created_at 유지)
            user_data["created_at"] = users[existing_user_index].get("created_at", user_data["created_at"])
            users[existing_user_index] = user_data
            logger.info(f"🔄 기존 사용자 정보 업데이트: {email}")
        else:
            # 새 사용자 추가
            users.append(user_data)
            logger.info(f"🆕 새 사용자 추가: {email}")

        # 파일 저장
        success = save_json_file(USER_DATA_PATH, users)

        if success:
            logger.info(f"✅ 사용자 저장 완료: {email}")
            return True
        else:
            logger.error(f"❌ 사용자 저장 실패: {email}")
            return False

    except Exception as e:
        logger.error(f"❌ 사용자 저장 중 오류: {e}")
        return False


async def get_user_by_uid(uid: str) -> Optional[Dict]:
    """UID로 사용자 정보 조회"""
    try:
        users = load_json_file(USER_DATA_PATH)

        for user in users:
            if user.get("uid") == uid:
                logger.info(f"✅ 사용자 조회 성공: {user.get('email')}")
                return user

        logger.warning(f"⚠️ 사용자를 찾을 수 없음: {uid}")
        return None

    except Exception as e:
        logger.error(f"❌ 사용자 조회 중 오류: {e}")
        return None


async def get_user_by_email(email: str) -> Optional[Dict]:
    """이메일로 사용자 정보 조회"""
    try:
        users = load_json_file(USER_DATA_PATH)

        for user in users:
            if user.get("email") == email:
                logger.info(f"✅ 사용자 조회 성공: {email}")
                return user

        logger.warning(f"⚠️ 사용자를 찾을 수 없음: {email}")
        return None

    except Exception as e:
        logger.error(f"❌ 사용자 조회 중 오류: {e}")
        return None


async def update_user(uid: str, **kwargs) -> bool:
    """사용자 정보 업데이트"""
    try:
        users = load_json_file(USER_DATA_PATH)

        for i, user in enumerate(users):
            if user.get("uid") == uid:
                # 업데이트 가능한 필드들
                updatable_fields = ["name", "picture", "email", "is_active"]

                for field, value in kwargs.items():
                    if field in updatable_fields:
                        user[field] = value

                user["updated_at"] = datetime.now().isoformat()
                users[i] = user

                success = save_json_file(USER_DATA_PATH, users)

                if success:
                    logger.info(f"✅ 사용자 업데이트 완료: {uid}")
                    return True
                else:
                    logger.error(f"❌ 사용자 업데이트 저장 실패: {uid}")
                    return False

        logger.warning(f"⚠️ 업데이트할 사용자를 찾을 수 없음: {uid}")
        return False

    except Exception as e:
        logger.error(f"❌ 사용자 업데이트 중 오류: {e}")
        return False


async def delete_user(uid: str) -> bool:
    """사용자 정보 삭제"""
    try:
        users = load_json_file(USER_DATA_PATH)

        original_count = len(users)
        users = [user for user in users if user.get("uid") != uid]

        if len(users) < original_count:
            success = save_json_file(USER_DATA_PATH, users)

            if success:
                logger.info(f"✅ 사용자 삭제 완료: {uid}")
                return True
            else:
                logger.error(f"❌ 사용자 삭제 저장 실패: {uid}")
                return False
        else:
            logger.warning(f"⚠️ 삭제할 사용자를 찾을 수 없음: {uid}")
            return False

    except Exception as e:
        logger.error(f"❌ 사용자 삭제 중 오류: {e}")
        return False


async def get_all_users() -> List[Dict]:
    """모든 사용자 목록 조회 (관리자용)"""
    try:
        users = load_json_file(USER_DATA_PATH)
        logger.info(f"✅ 전체 사용자 조회: {len(users)}명")
        return users

    except Exception as e:
        logger.error(f"❌ 전체 사용자 조회 중 오류: {e}")
        return []


async def get_user_stats() -> Dict:
    """사용자 통계 조회"""
    try:
        users = load_json_file(USER_DATA_PATH)

        total_users = len(users)
        active_users = len([user for user in users if user.get("is_active", True)])
        firebase_users = len([user for user in users if user.get("provider") == "firebase"])
        google_users = len([user for user in users if user.get("provider") == "google"])

        stats = {
            "total_users": total_users,
            "active_users": active_users,
            "inactive_users": total_users - active_users,
            "firebase_users": firebase_users,
            "google_users": google_users,
            "last_updated": datetime.now().isoformat()
        }

        logger.info(f"✅ 사용자 통계 조회 완료: {total_users}명 (활성: {active_users}명)")
        return stats

    except Exception as e:
        logger.error(f"❌ 사용자 통계 조회 중 오류: {e}")
        return {
            "total_users": 0,
            "active_users": 0,
            "inactive_users": 0,
            "firebase_users": 0,
            "google_users": 0,
            "last_updated": datetime.now().isoformat(),
            "error": str(e)
        }