# perfume_backend/generate_test_token.py

import jwt
import datetime

# Firebase 프로젝트의 발급자 (예시로 작성한 값)
ISSUER = "test-firebase-adminsdk@example.com"

def create_test_token(uid: str = "testuser123"):
    now = datetime.datetime.utcnow()
    payload = {
        "iss": ISSUER,
        "aud": "https://identitytoolkit.googleapis.com/google.identity.identitytoolkit.v1.IdentityToolkit",
        "iat": now,
        "exp": now + datetime.timedelta(hours=1),
        "uid": uid,
        "email": f"{uid}@example.com"
    }

    # 개발용 시크릿 키 (실제 Firebase와는 무관한 로컬 테스트용)
    secret = "secret-for-testing"
    token = jwt.encode(payload, secret, algorithm="HS256")
    return token

if __name__ == "__main__":
    print("🔥 테스트용 Firebase ID 토큰:")
    print(create_test_token())
