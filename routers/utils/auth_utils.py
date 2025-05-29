# utils/auth_utils.py

import firebase_admin
from firebase_admin import credentials, auth
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

import os

# 🔐 Firebase 서비스 계정 키 경로
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SERVICE_ACCOUNT_PATH = os.path.join(BASE_DIR, "../config/firebase-service-account.json")

# ⚙️ Firebase Admin 초기화 (이미 초기화된 경우 예외 발생 방지)
if not firebase_admin._apps:
    cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
    firebase_admin.initialize_app(cred)

# 🔐 HTTP Authorization 헤더 처리 (Bearer 토큰 방식)
security = HTTPBearer()

# ✅ 토큰 검증 함수 (FastAPI Depends에서 사용)
async def verify_firebase_token(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    try:
        id_token = credentials.credentials
        decoded_token = auth.verify_id_token(id_token)
        return decoded_token  # uid, email, name 등 포함
    except Exception as e:
        raise HTTPException(status_code=401, detail="유효하지 않은 Firebase 인증 토큰입니다.")
