from fastapi import Header, HTTPException
from firebase_admin import auth


def verify_firebase_token(id_token: str = Header(...)):
    """
    Firebase ID 토큰을 검증하고 디코딩된 사용자 정보를 반환합니다.
    유효하지 않으면 401 에러를 발생시킵니다.
    """
    try:
        decoded_token = auth.verify_id_token(id_token)
        return decoded_token  # 예: {"uid": "...", "email": "..."}
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid Firebase ID token")
