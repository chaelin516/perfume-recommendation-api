# utils/jwt_utils.py
import jwt
import os
from datetime import datetime, timedelta
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

# JWT 설정
JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', 'whiff-secret-key-2025-change-in-production')
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24  # 24시간


def create_access_token(user_data: Dict) -> str:
    """사용자 정보로부터 JWT 액세스 토큰 생성"""
    try:
        # 토큰 페이로드 구성
        payload = {
            "uid": user_data.get("uid"),
            "email": user_data.get("email"),
            "name": user_data.get("name"),
            "picture": user_data.get("picture", ""),
            "exp": datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS),
            "iat": datetime.utcnow(),
            "iss": "whiff-api",  # 토큰 발급자
            "type": "access_token"
        }

        # JWT 토큰 생성
        token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

        logger.info(f"✅ JWT 토큰 생성 성공: {user_data.get('email')}")
        return token

    except Exception as e:
        logger.error(f"❌ JWT 토큰 생성 실패: {e}")
        raise Exception(f"JWT 토큰 생성 실패: {str(e)}")


def verify_access_token(token: str) -> Optional[Dict]:
    """JWT 액세스 토큰 검증 및 사용자 정보 반환"""
    try:
        # 토큰 검증 및 디코딩
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])

        # 필수 필드 확인
        if not payload.get("uid"):
            logger.error("❌ JWT 토큰에 uid가 없습니다")
            return None

        if payload.get("type") != "access_token":
            logger.error("❌ 잘못된 토큰 타입입니다")
            return None

        logger.info(f"✅ JWT 토큰 검증 성공: {payload.get('email')}")
        return payload

    except jwt.ExpiredSignatureError:
        logger.error("❌ JWT 토큰이 만료되었습니다")
        return None
    except jwt.InvalidTokenError as e:
        logger.error(f"❌ 유효하지 않은 JWT 토큰: {e}")
        return None
    except Exception as e:
        logger.error(f"❌ JWT 토큰 검증 중 오류: {e}")
        return None


def get_token_info(token: str) -> Optional[Dict]:
    """토큰 정보 조회 (만료 시간 등)"""
    try:
        # 토큰 디코딩 (검증 없이)
        payload = jwt.decode(token, options={"verify_signature": False})

        exp_timestamp = payload.get("exp")
        iat_timestamp = payload.get("iat")

        result = {
            "uid": payload.get("uid"),
            "email": payload.get("email"),
            "issued_at": datetime.utcfromtimestamp(iat_timestamp).isoformat() if iat_timestamp else None,
            "expires_at": datetime.utcfromtimestamp(exp_timestamp).isoformat() if exp_timestamp else None,
        }

        if exp_timestamp:
            expiry_time = datetime.utcfromtimestamp(exp_timestamp)
            remaining_time = expiry_time - datetime.utcnow()
            result.update({
                "remaining_seconds": int(remaining_time.total_seconds()),
                "is_expired": remaining_time.total_seconds() <= 0
            })

        return result

    except Exception as e:
        logger.error(f"❌ 토큰 정보 조회 실패: {e}")
        return None