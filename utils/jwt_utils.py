# utils/jwt_utils.py - JWT 토큰 처리 유틸리티

import os
import jwt
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# JWT 설정
JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', 'your-secret-key-change-in-production')
JWT_ALGORITHM = 'HS256'
JWT_EXPIRE_HOURS = int(os.getenv('JWT_EXPIRE_HOURS', '24'))


def create_access_token(user_data: Dict[str, Any]) -> str:
    """JWT 액세스 토큰 생성

    Args:
        user_data: 사용자 정보 딕셔너리
        - uid: 사용자 ID
        - email: 이메일
        - name: 이름 (선택적)

    Returns:
        JWT 토큰 문자열
    """
    try:
        # 토큰 만료 시간 설정
        expire = datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRE_HOURS)

        # JWT 페이로드 구성
        payload = {
            'uid': user_data.get('uid'),
            'email': user_data.get('email'),
            'name': user_data.get('name', ''),
            'picture': user_data.get('picture', ''),
            'exp': expire,
            'iat': datetime.now(timezone.utc),
            'iss': 'whiff-api'  # 발급자
        }

        # JWT 토큰 생성
        token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

        logger.info(f"✅ JWT 토큰 생성 성공: {user_data.get('email')}")
        return token

    except Exception as e:
        logger.error(f"❌ JWT 토큰 생성 실패: {e}")
        raise ValueError(f"토큰 생성 실패: {str(e)}")


def verify_access_token(token: str) -> Optional[Dict[str, Any]]:
    """JWT 액세스 토큰 검증

    Args:
        token: JWT 토큰 문자열

    Returns:
        검증된 페이로드 딕셔너리 또는 None
    """
    try:
        # JWT 토큰 디코딩 및 검증
        payload = jwt.decode(
            token,
            JWT_SECRET_KEY,
            algorithms=[JWT_ALGORITHM],
            options={"verify_exp": True}
        )

        # 필수 필드 확인
        if not payload.get('uid') or not payload.get('email'):
            logger.warning("⚠️ JWT 토큰에 필수 필드 누락")
            return None

        logger.debug(f"✅ JWT 토큰 검증 성공: {payload.get('email')}")
        return payload

    except jwt.ExpiredSignatureError:
        logger.warning("⏰ JWT 토큰 만료")
        return None
    except jwt.InvalidTokenError as e:
        logger.warning(f"❌ 유효하지 않은 JWT 토큰: {e}")
        return None
    except Exception as e:
        logger.error(f"💥 JWT 토큰 검증 중 오류: {e}")
        return None


def refresh_access_token(old_token: str) -> Optional[str]:
    """JWT 토큰 갱신

    Args:
        old_token: 기존 JWT 토큰

    Returns:
        새로운 JWT 토큰 또는 None
    """
    try:
        # 만료된 토큰도 페이로드만 확인 (exp 검증 스킵)
        payload = jwt.decode(
            old_token,
            JWT_SECRET_KEY,
            algorithms=[JWT_ALGORITHM],
            options={"verify_exp": False}  # 만료 시간 검증 스킵
        )

        # 토큰이 너무 오래된 경우 갱신 거부 (예: 7일 초과)
        iat = payload.get('iat')
        if iat:
            issued_time = datetime.fromtimestamp(iat, tz=timezone.utc)
            if datetime.now(timezone.utc) - issued_time > timedelta(days=7):
                logger.warning("⚠️ 토큰이 너무 오래됨 - 갱신 거부")
                return None

        # 새로운 토큰 생성
        user_data = {
            'uid': payload.get('uid'),
            'email': payload.get('email'),
            'name': payload.get('name', ''),
            'picture': payload.get('picture', '')
        }

        new_token = create_access_token(user_data)
        logger.info(f"🔄 JWT 토큰 갱신 성공: {user_data.get('email')}")
        return new_token

    except Exception as e:
        logger.error(f"❌ JWT 토큰 갱신 실패: {e}")
        return None


def decode_token_without_verification(token: str) -> Optional[Dict[str, Any]]:
    """검증 없이 JWT 토큰 디코딩 (디버깅용)

    Args:
        token: JWT 토큰 문자열

    Returns:
        디코딩된 페이로드 또는 None
    """
    try:
        payload = jwt.decode(
            token,
            options={"verify_signature": False, "verify_exp": False}
        )
        return payload
    except Exception as e:
        logger.error(f"❌ JWT 토큰 디코딩 실패: {e}")
        return None


def get_jwt_status() -> Dict[str, Any]:
    """JWT 설정 상태 확인

    Returns:
        JWT 설정 정보 딕셔너리
    """
    return {
        "jwt_configured": bool(JWT_SECRET_KEY and JWT_SECRET_KEY != 'your-secret-key-change-in-production'),
        "jwt_algorithm": JWT_ALGORITHM,
        "jwt_expire_hours": JWT_EXPIRE_HOURS,
        "jwt_secret_set": "설정됨" if JWT_SECRET_KEY else "없음",
        "security_warning": "프로덕션에서는 강력한 시크릿 키 사용 필요" if JWT_SECRET_KEY == 'your-secret-key-change-in-production' else None
    }


# Flutter 앱 전용 헬퍼 함수들
def create_flutter_token(firebase_user: Dict[str, Any]) -> str:
    """Firebase 사용자 정보를 기반으로 Flutter 앱용 JWT 토큰 생성

    Args:
        firebase_user: Firebase 사용자 정보

    Returns:
        Flutter 앱용 JWT 토큰
    """
    try:
        user_data = {
            'uid': firebase_user.get('uid'),
            'email': firebase_user.get('email'),
            'name': firebase_user.get('name') or firebase_user.get('display_name', ''),
            'picture': firebase_user.get('picture') or firebase_user.get('photo_url', ''),
            'email_verified': firebase_user.get('email_verified', False)
        }

        token = create_access_token(user_data)
        logger.info(f"📱 Flutter 앱용 JWT 토큰 생성: {user_data.get('email')}")
        return token

    except Exception as e:
        logger.error(f"❌ Flutter JWT 토큰 생성 실패: {e}")
        raise


def validate_flutter_token(token: str) -> bool:
    """Flutter 앱에서 온 토큰인지 간단히 검증

    Args:
        token: JWT 토큰

    Returns:
        유효성 여부
    """
    payload = verify_access_token(token)
    if not payload:
        return False

    # Flutter 앱 토큰 특성 확인
    required_fields = ['uid', 'email']
    return all(payload.get(field) for field in required_fields)


# 개발/디버깅용 함수들
def create_test_token(test_user_email: str = "test@whiff.com") -> str:
    """테스트용 JWT 토큰 생성

    Args:
        test_user_email: 테스트 사용자 이메일

    Returns:
        테스트용 JWT 토큰
    """
    test_user = {
        'uid': 'test-user-123',
        'email': test_user_email,
        'name': '테스트 사용자',
        'picture': ''
    }

    return create_access_token(test_user)


if __name__ == "__main__":
    # 간단한 테스트
    print("🧪 JWT 유틸리티 테스트")

    # 상태 확인
    status = get_jwt_status()
    print(f"JWT 설정 상태: {status}")

    # 테스트 토큰 생성
    test_token = create_test_token()
    print(f"테스트 토큰: {test_token[:50]}...")

    # 토큰 검증
    payload = verify_access_token(test_token)
    print(f"토큰 검증 결과: {payload}")

    print("✅ JWT 유틸리티 테스트 완료")