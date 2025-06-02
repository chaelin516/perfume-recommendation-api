from typing import Optional
from datetime import datetime
import json
import os

# 사용자 데이터 파일 경로
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USER_DATA_PATH = os.path.join(BASE_DIR, "..", "data", "user_data.json")


def load_users():
    """사용자 데이터 로딩"""
    if os.path.exists(USER_DATA_PATH):
        try:
            with open(USER_DATA_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ 사용자 데이터 로딩 실패: {e}")
            return []
    return []


def save_users(users_data):
    """사용자 데이터 저장"""
    try:
        os.makedirs(os.path.dirname(USER_DATA_PATH), exist_ok=True)
        with open(USER_DATA_PATH, "w", encoding="utf-8") as f:
            json.dump(users_data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"❌ 사용자 데이터 저장 실패: {e}")
        return False


async def save_user(uid: str, email: str, name: str, picture: Optional[str] = None):
    """Firebase 로그인 시 사용자 정보를 저장하는 함수"""
    print("[USER SAVE] Firebase user info received:")
    print(f"  UID: {uid}")
    print(f"  Email: {email}")
    print(f"  Name: {name}")
    print(f"  Picture: {picture}")

    try:
        # 기존 사용자 데이터 로딩
        users_data = load_users()

        # 기존 사용자 찾기
        existing_user = None
        for i, user in enumerate(users_data):
            if user.get("uid") == uid:
                existing_user = i
                break

        # 사용자 정보 구성
        user_info = {
            "uid": uid,
            "email": email,
            "name": name,
            "picture": picture or "",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "login_count": 1,
            "last_login": datetime.now().isoformat()
        }

        if existing_user is not None:
            # 기존 사용자 업데이트
            old_user = users_data[existing_user]
            user_info["created_at"] = old_user.get("created_at", user_info["created_at"])
            user_info["login_count"] = old_user.get("login_count", 0) + 1
            users_data[existing_user] = user_info
            print(f"✅ 기존 사용자 정보 업데이트: {name}")
        else:
            # 새 사용자 추가
            users_data.append(user_info)
            print(f"✅ 새 사용자 정보 저장: {name}")

        # 파일에 저장
        if save_users(users_data):
            print(f"✅ 사용자 데이터 파일 저장 완료")
        else:
            print(f"❌ 사용자 데이터 파일 저장 실패")

    except Exception as e:
        print(f"❌ 사용자 정보 저장 중 오류: {e}")


async def get_user_by_uid(uid: str):
    """UID로 사용자 정보 조회"""
    try:
        users_data = load_users()
        for user in users_data:
            if user.get("uid") == uid:
                return user
        return None
    except Exception as e:
        print(f"❌ 사용자 조회 중 오류: {e}")
        return None


async def get_user_by_email(email: str):
    """이메일로 사용자 정보 조회"""
    try:
        users_data = load_users()
        for user in users_data:
            if user.get("email") == email:
                return user
        return None
    except Exception as e:
        print(f"❌ 사용자 조회 중 오류: {e}")
        return None


async def update_user_profile(uid: str, name: Optional[str] = None, picture: Optional[str] = None):
    """사용자 프로필 업데이트"""
    try:
        users_data = load_users()

        for i, user in enumerate(users_data):
            if user.get("uid") == uid:
                if name:
                    users_data[i]["name"] = name
                if picture:
                    users_data[i]["picture"] = picture
                users_data[i]["updated_at"] = datetime.now().isoformat()

                if save_users(users_data):
                    print(f"✅ 사용자 프로필 업데이트 완료: {uid}")
                    return users_data[i]
                else:
                    print(f"❌ 사용자 프로필 저장 실패: {uid}")
                    return None

        print(f"❌ 사용자를 찾을 수 없습니다: {uid}")
        return None

    except Exception as e:
        print(f"❌ 사용자 프로필 업데이트 중 오류: {e}")
        return None


async def get_all_users():
    """모든 사용자 목록 조회 (관리자용)"""
    try:
        return load_users()
    except Exception as e:
        print(f"❌ 전체 사용자 조회 중 오류: {e}")
        return []


async def delete_user(uid: str):
    """사용자 삭제"""
    try:
        users_data = load_users()

        for i, user in enumerate(users_data):
            if user.get("uid") == uid:
                deleted_user = users_data.pop(i)

                if save_users(users_data):
                    print(f"✅ 사용자 삭제 완료: {uid}")
                    return deleted_user
                else:
                    print(f"❌ 사용자 삭제 저장 실패: {uid}")
                    return None

        print(f"❌ 삭제할 사용자를 찾을 수 없습니다: {uid}")
        return None

    except Exception as e:
        print(f"❌ 사용자 삭제 중 오류: {e}")
        return None