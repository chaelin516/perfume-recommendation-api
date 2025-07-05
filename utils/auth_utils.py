# utils/auth_utils.py - 안전한 JWT 처리 버전
import firebase_admin
from firebase_admin import credentials, auth
from fastapi import Header, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import os
import json
import logging

logger = logging.getLogger(__name__)

# Firebase 초기화 상태
FIREBASE_AVAILABLE = False
firebase_app = None

# 🆕 JWT 기능 가용성 확인
JWT_AVAILABLE = False
try:
    from utils.jwt_utils import verify_access_token

    JWT_AVAILABLE = True
    logger.info("✅ JWT 유틸리티 로드 성공")
except ImportError:
    logger.info("⚠️ JWT 유틸리티 없음 - Firebase 인증만 사용")


    # JWT 없을 때를 위한 더미 함수
    def verify_access_token(token):
        return None
except Exception as e:
    logger.warning(f"⚠️ JWT 유틸리티 로드 실패: {e}")


    def verify_access_token(token):
        return None


def get_firebase_credentials():
    """환경변수 또는 파일에서 Firebase credentials 가져오기"""

    # 1. 환경변수에서 Firebase JSON 읽기 (우선순위 1)
    firebase_json_env = os.getenv('FIREBASE_CREDENTIAL_JSON')
    if firebase_json_env:
        try:
            firebase_config = json.loads(firebase_json_env)
            logger.info("✅ Firebase 환경변수에서 설정 로드 성공")
            return credentials.Certificate(firebase_config)
        except json.JSONDecodeError as e:
            logger.error(f"❌ Firebase 환경변수 JSON 파싱 오류: {e}")
        except Exception as e:
            logger.error(f"❌ Firebase 환경변수 credential 생성 오류: {e}")

    # 2. 개별 환경변수에서 Firebase 설정 구성 (우선순위 2)
    project_id = os.getenv('FIREBASE_PROJECT_ID')
    private_key = os.getenv('FIREBASE_PRIVATE_KEY')
    client_email = os.getenv('FIREBASE_CLIENT_EMAIL')

    if project_id and private_key and client_email:
        try:
            firebase_config = {
                "type": "service_account",
                "project_id": project_id,
                "private_key": private_key.replace('\\n', '\n'),
                "client_email": client_email,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                "client_x509_cert_url": f"https://www.googleapis.com/robot/v1/metadata/x509/{client_email}"
            }
            logger.info("✅ Firebase 개별 환경변수에서 설정 구성 성공")
            return credentials.Certificate(firebase_config)
        except Exception as e:
            logger.error(f"❌ Firebase 개별 환경변수 credential 생성 오류: {e}")

    # 3. 파일에서 Firebase 설정 읽기 (우선순위 3 - 로컬 개발용)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    FIREBASE_KEY_PATH = os.path.join(BASE_DIR, "..", "firebase_key.json")

    if os.path.exists(FIREBASE_KEY_PATH):
        try:
            logger.info(f"🔐 Firebase 키 파일 로딩: {FIREBASE_KEY_PATH}")
            return credentials.Certificate(FIREBASE_KEY_PATH)
        except Exception as e:
            logger.error(f"❌ Firebase 키 파일 로딩 오류: {e}")

    logger.warning("⚠️ Firebase 설정을 찾을 수 없습니다.")
    return None


def initialize_firebase():
    """Firebase 초기화 (안전한 방식)"""
    global FIREBASE_AVAILABLE, firebase_app

    if firebase_admin._apps:
        FIREBASE_AVAILABLE = True
        logger.info("✅ Firebase Admin SDK 이미 초기화됨")
        return True

    try:
        cred = get_firebase_credentials()
        if cred:
            firebase_app = firebase_admin.initialize_app(cred)
            FIREBASE_AVAILABLE = True
            logger.info("✅ Firebase Admin SDK 초기화 완료")
            return True
        else:
            FIREBASE_AVAILABLE = False
            logger.warning("⚠️ Firebase credentials를 찾을 수 없어 초기화를 건너뜁니다.")
            return False
    except Exception as e:
        FIREBASE_AVAILABLE = False
        logger.error(f"❌ Firebase 초기화 실패: {e}")
        return False


# Firebase 초기화 시도
initialize_firebase()

# HTTP Authorization 헤더 처리
security = HTTPBearer(auto_error=False)


async def verify_firebase_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Firebase ID 토큰 검증"""
    if not FIREBASE_AVAILABLE:
        logger.error("Firebase가 초기화되지 않았습니다.")
        raise HTTPException(status_code=503, detail="Firebase 인증 서비스를 사용할 수 없습니다.")

    if not credentials:
        raise HTTPException(status_code=401, detail="인증 토큰이 제공되지 않았습니다.")

    try:
        id_token = credentials.credentials
        decoded_token = auth.verify_id_token(id_token)

        uid = decoded_token["uid"]
        email = decoded_token.get("email", "")
        name = decoded_token.get("name", "")

        logger.info(f"[AUTH SUCCESS] 사용자 인증 완료: {name} ({email})")
        return decoded_token

    except auth.ExpiredIdTokenError:
        raise HTTPException(status_code=401, detail="토큰이 만료되었습니다.")
    except auth.RevokedIdTokenError:
        raise HTTPException(status_code=401, detail="토큰이 취소되었습니다.")
    except auth.InvalidIdTokenError:
        raise HTTPException(status_code=401, detail="유효하지 않은 토큰입니다.")
    except Exception as e:
        logger.error(f"[AUTH ERROR] {e}")
        raise HTTPException(status_code=401, detail="Firebase 인증에 실패했습니다.")


async def get_dummy_user():
    """테스트용 더미 사용자"""
    return {
        "uid": "test-user-123",
        "email": "test@example.com",
        "name": "테스트 사용자",
        "picture": ""
    }


# 🆕 JWT 토큰 검증 함수 (안전한 버전)
async def verify_jwt_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """JWT 토큰 검증 (Flutter 앱용) - JWT 라이브러리가 없어도 안전"""
    if not JWT_AVAILABLE:
        logger.warning("⚠️ JWT 기능이 비활성화됨 - Firebase 인증 사용")
        return await verify_firebase_token(credentials)

    if not credentials:
        raise HTTPException(status_code=401, detail="인증 토큰이 제공되지 않았습니다.")

    try:
        token = credentials.credentials
        payload = verify_access_token(token)

        if not payload:
            # JWT 실패시 Firebase로 대체 시도
            if FIREBASE_AVAILABLE:
                logger.info("🔄 JWT 실패 - Firebase 인증으로 대체 시도")
                return await verify_firebase_token(credentials)
            else:
                raise HTTPException(status_code=401, detail="유효하지 않은 토큰입니다.")

        logger.info(f"[JWT AUTH SUCCESS] 사용자 인증 완료: {payload.get('name')} ({payload.get('email')})")
        return payload

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[JWT AUTH ERROR] {e}")
        # JWT 처리 실패시 Firebase로 대체 시도
        if FIREBASE_AVAILABLE:
            logger.info("🔄 JWT 오류 - Firebase 인증으로 대체 시도")
            return await verify_firebase_token(credentials)
        else:
            raise HTTPException(status_code=401, detail="JWT 토큰 검증에 실패했습니다.")


# 🆕 유연한 인증 함수 (Firebase ID 토큰 또는 JWT 토큰 모두 지원)
async def verify_token_flexible(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Firebase ID 토큰 또는 JWT 토큰을 모두 지원하는 검증"""
    if not credentials:
        raise HTTPException(status_code=401, detail="인증 토큰이 제공되지 않았습니다.")

    token = credentials.credentials

    # 1. JWT 기능이 활성화된 경우 먼저 JWT 토큰으로 시도
    if JWT_AVAILABLE:
        jwt_payload = verify_access_token(token)
        if jwt_payload:
            logger.info(f"[JWT AUTH] 사용자 인증 완료: {jwt_payload.get('email')}")
            return jwt_payload

    # 2. JWT 실패하거나 비활성화시 Firebase ID 토큰으로 시도
    if FIREBASE_AVAILABLE:
        try:
            decoded_token = auth.verify_id_token(token)
            logger.info(f"[FIREBASE AUTH] 사용자 인증 완료: {decoded_token.get('email')}")
            return decoded_token
        except Exception as e:
            logger.error(f"[FIREBASE AUTH ERROR] {e}")

    # 3. 둘 다 실패시
    raise HTTPException(status_code=401, detail="유효하지 않은 토큰입니다.")


# 🆕 JWT 전용 인증 함수 (권장 - Flutter 앱용)
async def verify_jwt_only(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """JWT 토큰만 검증 (Flutter 앱 전용) - 안전한 대체 처리 포함"""
    return await verify_jwt_token(credentials)


async def verify_firebase_token_optional(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """선택적 Firebase 인증 (Firebase 없이도 작동) + JWT 지원"""
    if not credentials:
        if not FIREBASE_AVAILABLE and not JWT_AVAILABLE:
            logger.warning("⚠️ Firebase/JWT 모두 비활성화 - 더미 사용자로 인증 우회")
            return await get_dummy_user()
        raise HTTPException(status_code=401, detail="인증 토큰이 제공되지 않았습니다.")

    token = credentials.credentials

    # 1. JWT 토큰 먼저 시도 (활성화된 경우)
    if JWT_AVAILABLE:
        jwt_payload = verify_access_token(token)
        if jwt_payload:
            logger.info(f"[JWT AUTH] 사용자 인증 완료: {jwt_payload.get('email')}")
            return jwt_payload

    # 2. Firebase 사용 가능하면 Firebase ID 토큰 시도
    if FIREBASE_AVAILABLE:
        try:
            return await verify_firebase_token(credentials)
        except HTTPException:
            pass  # Firebase 실패해도 더미 사용자로 진행

    # 3. 모든 인증 실패시 더미 사용자 반환 (개발 환경용)
    logger.warning("⚠️ 모든 인증 방법 실패 - 더미 사용자로 인증 우회")
    return await get_dummy_user()


# 🆕 상세한 에러 처리가 포함된 인증 함수
async def verify_token_with_details(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """상세한 로깅과 에러 처리가 포함된 토큰 검증"""

    if not credentials:
        logger.warning("🔐 인증 헤더가 없습니다")
        raise HTTPException(
            status_code=401,
            detail={
                "error": "missing_token",
                "message": "인증 토큰이 제공되지 않았습니다",
                "hint": "Authorization: Bearer <token> 헤더를 포함해주세요"
            }
        )

    token = credentials.credentials
    logger.info(f"🔍 토큰 검증 시도 (길이: {len(token)}자)")

    # 1. JWT 토큰 시도 (가능한 경우)
    if JWT_AVAILABLE:
        try:
            jwt_payload = verify_access_token(token)
            if jwt_payload:
                logger.info(f"✅ JWT 인증 성공: {jwt_payload.get('email')}")
                return jwt_payload
        except Exception as e:
            logger.warning(f"⚠️ JWT 인증 실패: {e}")

    # 2. Firebase 인증 시도 (가능한 경우)
    if FIREBASE_AVAILABLE:
        try:
            decoded_token = auth.verify_id_token(token)
            uid = decoded_token["uid"]
            email = decoded_token.get("email", "")
            name = decoded_token.get("name", "")

            logger.info(f"✅ Firebase 인증 성공: {name} ({email}) - {uid}")
            return decoded_token

        except auth.ExpiredIdTokenError:
            logger.warning("⏰ Firebase 토큰 만료")
            raise HTTPException(
                status_code=401,
                detail={
                    "error": "token_expired",
                    "message": "토큰이 만료되었습니다",
                    "hint": "앱에서 새로운 토큰을 발급받아주세요"
                }
            )
        except auth.InvalidIdTokenError:
            logger.warning("❌ Firebase 토큰 형식 오류")
            raise HTTPException(
                status_code=401,
                detail={
                    "error": "invalid_token",
                    "message": "유효하지 않은 토큰입니다",
                    "hint": "Firebase 로그인을 다시 시도해주세요"
                }
            )
        except Exception as e:
            logger.error(f"💥 Firebase 인증 처리 중 예외: {e}")

    # 3. 모든 인증 방법 실패
    logger.error("❌ 모든 인증 방법 실패")
    raise HTTPException(
        status_code=401,
        detail={
            "error": "auth_failed",
            "message": "인증에 실패했습니다",
            "available_methods": {
                "jwt": JWT_AVAILABLE,
                "firebase": FIREBASE_AVAILABLE
            },
            "hint": "유효한 인증 토큰을 제공해주세요"
        }
    )


def get_firebase_status():
    """Firebase 상태 확인"""
    env_status = {
        "firebase_credential_json_env": "설정됨" if os.getenv('FIREBASE_CREDENTIAL_JSON') else "없음",
        "firebase_project_id_env": "설정됨" if os.getenv('FIREBASE_PROJECT_ID') else "없음",
        "firebase_private_key_env": "설정됨" if os.getenv('FIREBASE_PRIVATE_KEY') else "없음",
        "firebase_client_email_env": "설정됨" if os.getenv('FIREBASE_CLIENT_EMAIL') else "없음"
    }

    return {
        "firebase_available": FIREBASE_AVAILABLE,
        "jwt_available": JWT_AVAILABLE,
        "firebase_apps_count": len(firebase_admin._apps) if firebase_admin._apps else 0,
        "environment_config": env_status,
        "auth_methods": {
            "firebase_id_token": FIREBASE_AVAILABLE,
            "jwt_token": JWT_AVAILABLE,
            "fallback_dummy": not FIREBASE_AVAILABLE and not JWT_AVAILABLE
        }
    }