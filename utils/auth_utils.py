import firebase_admin
from firebase_admin import credentials, auth
from fastapi import Header, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import os
import json

# 🔐 Firebase 서비스 계정 키 파일 경로
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIREBASE_KEY_PATH = os.path.join(BASE_DIR, "..", "firebase_key.json")

# ⚙️ Firebase Admin 초기화 (이미 초기화된 경우 예외 발생 방지)
try:
    if not firebase_admin._apps:
        if os.path.exists(FIREBASE_KEY_PATH):
            print(f"🔐 Firebase 키 파일 로딩: {FIREBASE_KEY_PATH}")
            cred = credentials.Certificate(FIREBASE_KEY_PATH)
            firebase_admin.initialize_app(cred)
            print("✅ Firebase Admin SDK 초기화 완료")
            FIREBASE_AVAILABLE = True
        else:
            print(f"⚠️ Firebase 키 파일을 찾을 수 없습니다: {FIREBASE_KEY_PATH}")
            print("⚠️ Firebase 인증 기능이 비활성화됩니다.")
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

# ✅ Firebase 상태 확인
def get_firebase_status():
    """Firebase 초기화 상태를 반환합니다."""
    return {
        "firebase_available": FIREBASE_AVAILABLE,
        "firebase_key_path": FIREBASE_KEY_PATH,
        "firebase_key_exists": os.path.exists(FIREBASE_KEY_PATH),
        "firebase_apps_count": len(firebase_admin._apps) if firebase_admin._apps else 0
    }