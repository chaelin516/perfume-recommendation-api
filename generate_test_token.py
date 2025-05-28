import firebase_admin
from firebase_admin import credentials, auth
import datetime
import os

# 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIREBASE_KEY_PATH = os.path.join(BASE_DIR, "firebase_key.json")

# Firebase 초기화
if not firebase_admin._apps:
    cred = credentials.Certificate(FIREBASE_KEY_PATH)
    firebase_admin.initialize_app(cred)

# 테스트 토큰 생성
def create_test_token():
    user_id = "test-user-id"
    expires_at = datetime.datetime.utcnow() + datetime.timedelta(hours=1)
    custom_token = auth.create_custom_token(user_id)
    return custom_token.decode()

if __name__ == "__main__":
    token = create_test_token()
    print("🧪 테스트용 Firebase 커스텀 토큰:\n")
    print(token)
