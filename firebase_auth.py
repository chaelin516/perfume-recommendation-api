# app/services/firebase_auth.py

import firebase_admin
from firebase_admin import credentials, auth
from fastapi import HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# Firebase 서비스 계정 키 (JSON) 경로 지정
cred = credentials.Certificate("path/to/your/firebase-service-account.json")

# Firebase 초기화 (이미 초기화 되어있지 않으면)
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)

# 인증 헤더 보안 정의
security = HTTPBearer()

# 사용자 인증 함수
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    try:
        token = credentials.credentials
        decoded_token = auth.verify_id_token(token)
        return decoded_token  # 사용자 정보 리턴
    except Exception as e:
        raise HTTPException(status_code=401, detail="유효하지 않은 인증 토큰입니다.")
