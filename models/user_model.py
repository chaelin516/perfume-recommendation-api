from typing import Optional

# ✅ Firebase 로그인 시 최초 사용자 정보를 저장하는 함수
# 현재는 실제 DB 저장 없이 로그 출력만 수행합니다.
# 추후 Firestore, Supabase, PostgreSQL 등에 저장 가능

async def save_user(uid: str, email: str, name: str, picture: Optional[str] = None):
    print("[USER SAVE] Firebase user info received:")
    print(f"  UID: {uid}")
    print(f"  Email: {email}")
    print(f"  Name: {name}")
    print(f"  Picture: {picture}")

    # TODO: 추후 DB 저장 로직 구현
    # 예: Firestore에 document 생성, Supabase에 insert 등
    pass
