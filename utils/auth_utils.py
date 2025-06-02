# utils/auth_utils.py
# 환경변수 Firebase 지원 버전

import firebase_admin
from firebase_admin import credentials, auth
from fastapi import Header, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import os
import json
import tempfile

# 🔐 Firebase 설정 로딩 (환경변수 우선 지원)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIREBASE_KEY_PATH = os.path.join(BASE_DIR, "..", "firebase_key.json")


def get_firebase_credentials():
    """환경변수 또는 파일에서 Firebase credentials 가져오기"""

    # 1. 환경변수에서 Firebase JSON 읽기 시도
    firebase_json_env = os.getenv('FIREBASE_CREDENTIAL_JSON')
    if firebase_json_env:
        try:
            # JSON 문자열을 파싱하여 credential 객체 생성
            firebase_config = json.loads(firebase_json_env)
            print(f"✅ Firebase 환경변수에서 설정 로드 성공")
            return credentials.Certificate(firebase_config)
        except json.JSONDecodeError as e:
            print(f"❌ Firebase 환경변수 JSON 파싱 오류: {e}")
        except Exception as e:
            print(f"❌ Firebase 환경변수 credential 생성 오류: {e}")

    # 2. 파일에서 Firebase 설정 읽기 시도
    if os.path.exists(FIREBASE_KEY_PATH):
        try:
            print(f"🔐 Firebase 키 파일 로딩: {FIREBASE_KEY_PATH}")
            return credentials.Certificate(FIREBASE_KEY_PATH)
        except Exception as e:
            print(f"❌ Firebase 키 파일 로딩 오류: {e}")

    # 3. 개별 환경변수에서 Firebase 설정 구성 시도
    project_id = os.getenv('FIREBASE_PROJECT_ID')
    private_key = os.getenv('FIREBASE_PRIVATE_KEY')
    client_email = os.getenv('FIREBASE_CLIENT_EMAIL')

    if project_id and private_key and client_email:
        try:
            firebase_config = {
                "type": "service_account",
                "project_id": project_id,
                "private_key": private_key.replace('\\n', '\n'),  # 개행 문자 처리
                "client_email": client_email,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token"
            }
            print(f"✅ Firebase 개별 환경변수에서 설정 구성 성공")
            return credentials.Certificate(firebase_config)
        except Exception as e:
            print(f"❌ Firebase 개별 환경변수 credential 생성 오류: {e}")

    print(f"⚠️ Firebase 설정을 찾을 수 없습니다.")
    print(f"  - 환경변수 FIREBASE_CREDENTIAL_JSON: {'있음' if firebase_json_env else '없음'}")
    print(f"  - 파일 {FIREBASE_KEY_PATH}: {'있음' if os.path.exists(FIREBASE_KEY_PATH) else '없음'}")
    print(
        f"  - 개별 환경변수: PROJECT_ID={'있음' if project_id else '없음'}, PRIVATE_KEY={'있음' if private_key else '없음'}, CLIENT_EMAIL={'있음' if client_email else '없음'}")
    return None


# ⚙️ Firebase Admin 초기화 (이미 초기화된 경우 예외 발생 방지)
try:
    if not firebase_admin._apps:
        cred = get_firebase_credentials()
        if cred:
            firebase_admin.initialize_app(cred)
            print("✅ Firebase Admin SDK 초기화 완료")
            FIREBASE_AVAILABLE = True
        else:
            print("⚠️ Firebase credentials를 찾을 수 없어 초기화를 건너뜁니다.")
            FIREBASE_AVAILABLE = False
    else:
        FIREBASE_AVAILABLE = True
        print("✅ Firebase Admin SDK 이미 초기화됨")
except Exception as e:
    print(f"❌ Firebase 초기화 실패: {e}")
    FIREBASE_AVAILABLE = False

# 🔐 HTTP Authorization 헤더 처리 (Bearer 토큰 방식)
security = HTTPBearer(auto_error=False)


# ✅ Firebase ID 토큰 검증 (Header 방식)
def verify_firebase_token_header(authorization: str = Header(..., alias="Authorization")):
    """
    Authorization 헤더에 담긴 Firebase ID 토큰을 검증합니다.
    예: Authorization: Bearer <ID_TOKEN>
    """
    if not FIREBASE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Firebase 인증 서비스를 사용할 수 없습니다.")

    # Bearer 토큰 파싱
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization 헤더 형식이 올바르지 않습니다.")

    id_token = authorization.replace("Bearer ", "").strip()

    try:
        decoded_token = auth.verify_id_token(id_token)

        # 🔐 사용자 정보 저장 (최초 로그인 시) - 현재는 로그만 출력
        uid = decoded_token["uid"]
        email = decoded_token.get("email", "")
        name = decoded_token.get("name", "")
        picture = decoded_token.get("picture", "")

        print(f"[AUTH SUCCESS] 사용자 인증 완료: {name} ({email})")

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


# ✅ Firebase ID 토큰 검증 (Bearer 방식)
async def verify_firebase_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    HTTP Bearer 토큰으로 Firebase ID 토큰을 검증합니다.
    FastAPI의 Depends와 함께 사용됩니다.
    """
    if not FIREBASE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Firebase 인증 서비스를 사용할 수 없습니다.")

    if not credentials:
        raise HTTPException(status_code=401, detail="인증 토큰이 제공되지 않았습니다.")

    try:
        id_token = credentials.credentials
        decoded_token = auth.verify_id_token(id_token)

        # 사용자 정보 로그
        uid = decoded_token["uid"]
        email = decoded_token.get("email", "")
        name = decoded_token.get("name", "")

        print(f"[AUTH SUCCESS] 사용자 인증 완료: {name} ({email})")

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


# ✅ 테스트용 더미 사용자 (Firebase 없이 테스트 가능)
async def get_dummy_user():
    """
    Firebase가 없을 때 테스트용 더미 사용자를 반환합니다.
    """
    return {
        "uid": "test-user-123",
        "email": "test@example.com",
        "name": "테스트 사용자",
        "picture": ""
    }


# ✅ 선택적 Firebase 인증 (Firebase 없이도 작동)
async def verify_firebase_token_optional(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Firebase가 사용 가능하면 실제 인증을, 그렇지 않으면 더미 사용자를 반환합니다.
    개발/테스트 환경에서 유용합니다.
    """
    if not FIREBASE_AVAILABLE:
        print("⚠️ Firebase 없이 더미 사용자로 인증 우회")
        return await get_dummy_user()

    return await verify_firebase_token(credentials)


# ✅ Firebase 상태 확인 (환경변수 정보 포함)
def get_firebase_status():
    """Firebase 초기화 상태를 반환합니다."""
    firebase_json_env = os.getenv('FIREBASE_CREDENTIAL_JSON')

    return {
        "firebase_available": FIREBASE_AVAILABLE,
        "firebase_key_path": FIREBASE_KEY_PATH,
        "firebase_key_exists": os.path.exists(FIREBASE_KEY_PATH),
        "firebase_apps_count": len(firebase_admin._apps) if firebase_admin._apps else 0,
        "environment_config": {
            "firebase_credential_json_env": "설정됨" if firebase_json_env else "없음",
            "firebase_project_id_env": "설정됨" if os.getenv('FIREBASE_PROJECT_ID') else "없음",
            "firebase_private_key_env": "설정됨" if os.getenv('FIREBASE_PRIVATE_KEY') else "없음",
            "firebase_client_email_env": "설정됨" if os.getenv('FIREBASE_CLIENT_EMAIL') else "없음"
        }
    }