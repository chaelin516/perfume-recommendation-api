# perfume_backend/firebase.py
# 환경변수 우선 지원 버전

import firebase_admin
from firebase_admin import credentials
import os
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIREBASE_KEY_PATH = os.path.join(BASE_DIR, "firebase_key.json")


def get_firebase_credentials():
    """환경변수 또는 파일에서 Firebase credentials 가져오기"""

    # 1. 환경변수에서 Firebase JSON 읽기 시도
    firebase_json_env = os.getenv('FIREBASE_CREDENTIAL_JSON')
    if firebase_json_env:
        try:
            firebase_config = json.loads(firebase_json_env)
            print(f"✅ Firebase 환경변수에서 설정 로드 성공")
            return credentials.Certificate(firebase_config)
        except json.JSONDecodeError as e:
            print(f"❌ Firebase 환경변수 JSON 파싱 오류: {e}")
        except Exception as e:
            print(f"❌ Firebase 환경변수 credential 생성 오류: {e}")

    # 2. 파일에서 Firebase 설정 읽기 시도 (fallback)
    if os.path.exists(FIREBASE_KEY_PATH):
        try:
            print(f"🔐 Firebase 키 파일 로딩: {FIREBASE_KEY_PATH}")
            return credentials.Certificate(FIREBASE_KEY_PATH)
        except Exception as e:
            print(f"❌ Firebase 키 파일 로딩 오류: {e}")

    print(f"⚠️ Firebase 설정을 찾을 수 없습니다.")
    return None


# Firebase Admin SDK 초기화
if not firebase_admin._apps:
    try:
        cred = get_firebase_credentials()
        if cred:
            firebase_admin.initialize_app(cred)
            print("✅ Firebase Admin SDK 초기화 완료")
        else:
            print("⚠️ Firebase credentials를 찾을 수 없어 초기화를 건너뜁니다.")
    except Exception as e:
        print(f"❌ Firebase 초기화 실패: {e}")
else:
    print("✅ Firebase Admin SDK 이미 초기화됨")