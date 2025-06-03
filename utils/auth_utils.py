# utils/auth_utils.py - 개선된 버전
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


async def verify_firebase_token_optional(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """선택적 Firebase 인증 (Firebase 없이도 작동)"""
    if not FIREBASE_AVAILABLE:
        logger.warning("⚠️ Firebase 없이 더미 사용자로 인증 우회")
        return await get_dummy_user()

    return await verify_firebase_token(credentials)


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
        "firebase_apps_count": len(firebase_admin._apps) if firebase_admin._apps else 0,
        "environment_config": env_status
    }