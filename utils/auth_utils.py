# utils/auth_utils.py

import firebase_admin
from firebase_admin import credentials, auth
from fastapi import Header, HTTPException
from models.user_model import save_user
import os

# 🔐 Firebase 서비스 계정 키 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SERVICE_ACCOUNT_PATH = os.path.join(BASE_DIR, "../config/firebase-service-account.json")

# ⚙️ Firebase Admin SDK 초기화 (중복 방지)
if not firebase_admin._apps:
    cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
    firebase_admin.initialize_app(cred)

# ✅ Firebase ID 토큰 검증
def verify_firebase_token(id_token: str = Header(..., alias="Authorization")):
    """
    Authorization 헤더에 담긴 Firebase ID 토큰을 검증합니다.
    예: Authorization: Bearer <ID_TOKEN>
    """

    # Bearer 토큰 파싱
    if not id_token.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization 헤더 형식이 올바르지 않습니다.")

    id_token = id_token.replace("Bearer ", "").strip()

    try:
        decoded_token = auth.verify_id_token(id_token)

        # 🔐 사용자 정보 저장 (최초 로그인 시)
        uid = decoded_token["uid"]
        email = decoded_token.get("email", "")
        name = decoded_token.get("name", "")
        picture = decoded_token.get("picture", "")
        save_user(uid, email, name, picture)

        return decoded_token

    except auth.ExpiredIdTokenError:
        raise HTTPException(status_code=401, detail="토큰이 만료되었습니다. 다시 로그인 해주세요.")
    except auth.RevokedIdTokenError:
        raise HTTPException(status_code=401, detail="이 토큰은 더 이상 유효하지 않습니다. 로그아웃 후 재로그인 해주세요.")
    except auth.InvalidIdTokenError:
        raise HTTPException(status_code=401, detail="유효하지 않은 토큰입니다. 다시 로그인 해주세요.")
    except Exception as e:
        print(f"[AUTH ERROR] {e}")
        raise HTTPException(status_code=401, detail="Firebase 인증에 실패했습니다.")
