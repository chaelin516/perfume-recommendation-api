# utils/auth_utils.py

import firebase_admin
from firebase_admin import credentials, auth
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from models.user_model import save_user  # ✅ 사용자 저장
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SERVICE_ACCOUNT_PATH = os.path.join(BASE_DIR, "../config/firebase-service-account.json")

if not firebase_admin._apps:
    cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
    firebase_admin.initialize_app(cred)

security = HTTPBearer()

async def verify_firebase_token(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    try:
        id_token = credentials.credentials
        decoded_token = auth.verify_id_token(id_token)

        uid = decoded_token["uid"]
        email = decoded_token.get("email", "")
        name = decoded_token.get("name", "")
        picture = decoded_token.get("picture", "")

        # ✅ 사용자 정보 저장 (최초 로그인 시)
        save_user(uid, email, name, picture)

        return decoded_token
    except Exception as e:
        print(f"[AUTH ERROR] {e}")  # ✅ 인증 실패 로그
        raise HTTPException(status_code=401, detail="유효하지 않은 Firebase 인증 토큰입니다.")
