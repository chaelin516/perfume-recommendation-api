# create_database.py - Whiff 데이터베이스 초기화 스크립트

import os
import sqlite3
from sqlmodel import SQLModel, create_engine
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_whiff_database():
    """Whiff 데이터베이스 생성 및 초기화"""

    # 데이터베이스 파일 경로들
    db_paths = ["./whiff.db", "./perfume.db"]

    print("🌸 Whiff 데이터베이스 초기화 시작")
    print("=" * 50)

    for db_path in db_paths:
        print(f"\n📁 데이터베이스 생성: {db_path}")

        try:
            # SQLite 엔진 생성
            engine = create_engine(f"sqlite:///{db_path}", echo=True)

            # 기본 테이블들 생성
            create_basic_tables(db_path)

            print(f"✅ {db_path} 생성 완료!")

            # 파일 크기 확인
            if os.path.exists(db_path):
                size = os.path.getsize(db_path)
                print(f"📊 파일 크기: {size} bytes")

        except Exception as e:
            print(f"❌ {db_path} 생성 실패: {e}")


def create_basic_tables(db_path):
    """기본 테이블들 생성"""

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 추천 향수 테이블
    cursor.execute("""
                   CREATE TABLE IF NOT EXISTS recommendedperfume
                   (
                       id
                       INTEGER
                       PRIMARY
                       KEY
                       AUTOINCREMENT,
                       user_id
                       TEXT
                       NOT
                       NULL,
                       perfume_name
                       TEXT
                       NOT
                       NULL,
                       perfume_brand
                       TEXT,
                       cluster
                       INTEGER,
                       score
                       REAL,
                       recommendation_type
                       TEXT
                       DEFAULT
                       '1st',
                       created_at
                       DATETIME
                       DEFAULT
                       CURRENT_TIMESTAMP,
                       updated_at
                       DATETIME
                       DEFAULT
                       CURRENT_TIMESTAMP
                   )
                   """)

    # 사용자 테이블 (기본)
    cursor.execute("""
                   CREATE TABLE IF NOT EXISTS users
                   (
                       id
                       INTEGER
                       PRIMARY
                       KEY
                       AUTOINCREMENT,
                       uid
                       TEXT
                       UNIQUE
                       NOT
                       NULL,
                       email
                       TEXT
                       UNIQUE,
                       name
                       TEXT,
                       picture
                       TEXT,
                       created_at
                       DATETIME
                       DEFAULT
                       CURRENT_TIMESTAMP,
                       updated_at
                       DATETIME
                       DEFAULT
                       CURRENT_TIMESTAMP
                   )
                   """)

    # 신고 테이블
    cursor.execute("""
                   CREATE TABLE IF NOT EXISTS reports
                   (
                       id
                       INTEGER
                       PRIMARY
                       KEY
                       AUTOINCREMENT,
                       diary_id
                       TEXT
                       NOT
                       NULL,
                       reporter_id
                       TEXT
                       NOT
                       NULL,
                       reason
                       TEXT
                       NOT
                       NULL,
                       description
                       TEXT,
                       status
                       TEXT
                       DEFAULT
                       'pending',
                       created_at
                       DATETIME
                       DEFAULT
                       CURRENT_TIMESTAMP,
                       updated_at
                       DATETIME
                       DEFAULT
                       CURRENT_TIMESTAMP
                   )
                   """)

    # 샘플 데이터 삽입
    sample_data = [
        ("user123", "Chanel No.5", "Chanel", 2, 0.85, "1st"),
        ("user456", "Dior Sauvage", "Dior", 1, 0.92, "1st"),
        ("user789", "Tom Ford Black Orchid", "Tom Ford", 3, 0.78, "2nd")
    ]

    cursor.executemany("""
                       INSERT
                       OR IGNORE INTO recommendedperfume 
        (user_id, perfume_name, perfume_brand, cluster, score, recommendation_type)
        VALUES (?, ?, ?, ?, ?, ?)
                       """, sample_data)

    conn.commit()
    conn.close()

    print(f"  ✅ 기본 테이블 생성 완료")
    print(f"  📊 샘플 데이터 {len(sample_data)}개 삽입")


def create_json_files():
    """필요한 JSON 파일들 생성"""

    print("\n📁 JSON 파일들 생성")

    # data 디렉토리 생성
    os.makedirs("data", exist_ok=True)

    json_files = {
        "data/users.json": [
            {
                "uid": "user123",
                "email": "user1@example.com",
                "name": "홍길동",
                "picture": "",
                "created_at": "2025-07-01T10:00:00"
            },
            {
                "uid": "user456",
                "email": "user2@example.com",
                "name": "김철수",
                "picture": "",
                "created_at": "2025-07-02T11:00:00"
            }
        ],
        "data/diaries.json": [
            {
                "id": "diary001",
                "user_id": "user123",
                "perfume_name": "Chanel No.5",
                "content": "오늘 시향한 향수가 정말 좋았다.",
                "emotion": "happy",
                "is_public": True,
                "created_at": "2025-07-06T14:30:00"
            },
            {
                "id": "diary002",
                "user_id": "user456",
                "perfume_name": "Dior Sauvage",
                "content": "상쾌한 느낌의 향수",
                "emotion": "excited",
                "is_public": True,
                "created_at": "2025-07-06T15:45:00"
            }
        ],
        "data/temp_users.json": []
    }

    import json

    for file_path, data in json_files.items():
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"  ✅ {file_path} 생성 완료 ({len(data)}개 항목)")
        except Exception as e:
            print(f"  ❌ {file_path} 생성 실패: {e}")


def verify_database():
    """데이터베이스 생성 확인"""

    print("\n🔍 데이터베이스 생성 확인")

    db_files = ["./whiff.db", "./perfume.db"]

    for db_path in db_files:
        if os.path.exists(db_path):
            print(f"  ✅ {db_path} 존재")

            # 테이블 목록 확인
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()

            print(f"    📋 테이블 수: {len(tables)}")
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table[0]}")
                count = cursor.fetchone()[0]
                print(f"      - {table[0]}: {count}개 레코드")

            conn.close()
        else:
            print(f"  ❌ {db_path} 없음")


if __name__ == "__main__":
    print("🌸 Whiff 프로젝트 데이터베이스 초기화")
    print("이 스크립트는 필요한 데이터베이스 파일들을 생성합니다.\n")

    try:
        # 1. 데이터베이스 생성
        create_whiff_database()

        # 2. JSON 파일 생성
        create_json_files()

        # 3. 생성 확인
        verify_database()

        print("\n🎉 데이터베이스 초기화 완료!")
        print("이제 whiff_db_browser.py를 실행할 수 있습니다.")

    except Exception as e:
        print(f"\n❌ 초기화 중 오류 발생: {e}")
        import traceback

        traceback.print_exc()