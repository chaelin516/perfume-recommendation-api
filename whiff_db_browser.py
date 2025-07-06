# whiff_db_browser.py - Whiff SQLite 데이터베이스 브라우저

import sqlite3
import json
import os
from datetime import datetime
from typing import List, Dict, Any
import pandas as pd


class WhiffDBBrowser:
    """Whiff 프로젝트 데이터베이스 브라우저"""

    def __init__(self, db_path: str = "./whiff.db"):
        self.db_path = db_path
        self.json_files = {
            "users": "data/users.json",
            "diaries": "data/diaries.json",
            "temp_users": "data/temp_users.json"
        }

        # 데이터베이스가 없으면 생성 시도
        if not os.path.exists(self.db_path):
            print(f"⚠️ 데이터베이스 파일이 없습니다: {self.db_path}")
            self.create_database_if_missing()

    def create_database_if_missing(self):
        """데이터베이스가 없으면 기본 구조로 생성"""
        try:
            print(f"🔧 데이터베이스 생성 중: {self.db_path}")

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # 기본 테이블 생성
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
                               INSERT INTO recommendedperfume
                               (user_id, perfume_name, perfume_brand, cluster, score, recommendation_type)
                               VALUES (?, ?, ?, ?, ?, ?)
                               """, sample_data)

            conn.commit()
            conn.close()

            print(f"✅ 데이터베이스 생성 완료 (샘플 데이터 {len(sample_data)}개 포함)")

        except Exception as e:
            print(f"❌ 데이터베이스 생성 실패: {e}")

    def connect(self):
        """데이터베이스 연결"""
        try:
            return sqlite3.connect(self.db_path)
        except Exception as e:
            print(f"❌ DB 연결 실패: {e}")
            return None

    def get_table_list(self) -> List[str]:
        """모든 테이블 목록 조회"""
        conn = self.connect()
        if not conn:
            return []

        try:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            conn.close()
            return tables
        except Exception as e:
            print(f"❌ 테이블 목록 조회 실패: {e}")
            return []

    def get_table_schema(self, table_name: str) -> List[Dict]:
        """테이블 스키마 조회"""
        conn = self.connect()
        if not conn:
            return []

        try:
            cursor = conn.cursor()
            cursor.execute(f"PRAGMA table_info({table_name});")
            schema = []
            for row in cursor.fetchall():
                schema.append({
                    "column_id": row[0],
                    "name": row[1],
                    "type": row[2],
                    "not_null": bool(row[3]),
                    "default_value": row[4],
                    "primary_key": bool(row[5])
                })
            conn.close()
            return schema
        except Exception as e:
            print(f"❌ 스키마 조회 실패: {e}")
            return []

    def query_table(self, table_name: str, limit: int = 100) -> pd.DataFrame:
        """테이블 데이터 조회"""
        conn = self.connect()
        if not conn:
            return pd.DataFrame()

        try:
            query = f"SELECT * FROM {table_name} LIMIT {limit};"
            df = pd.read_sql_query(query, conn)
            conn.close()
            return df
        except Exception as e:
            print(f"❌ 데이터 조회 실패: {e}")
            return pd.DataFrame()

    def execute_query(self, query: str) -> pd.DataFrame:
        """사용자 정의 쿼리 실행"""
        conn = self.connect()
        if not conn:
            return pd.DataFrame()

        try:
            df = pd.read_sql_query(query, conn)
            conn.close()
            return df
        except Exception as e:
            print(f"❌ 쿼리 실행 실패: {e}")
            return pd.DataFrame()

    def get_table_count(self, table_name: str) -> int:
        """테이블 레코드 수 조회"""
        conn = self.connect()
        if not conn:
            return 0

        try:
            cursor = conn.cursor()
            cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
            count = cursor.fetchone()[0]
            conn.close()
            return count
        except Exception as e:
            print(f"❌ 레코드 수 조회 실패: {e}")
            return 0

    def load_json_file(self, file_path: str) -> List[Dict]:
        """JSON 파일 로드"""
        if not os.path.exists(file_path):
            print(f"⚠️ 파일이 존재하지 않음: {file_path}")
            return []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ JSON 파일 로드 실패: {e}")
            return []

    def show_database_overview(self):
        """데이터베이스 전체 개요 출력"""
        print("🗄️ " + "=" * 60)
        print("🗄️  WHIFF 데이터베이스 개요")
        print("🗄️ " + "=" * 60)

        # DB 파일 정보
        if os.path.exists(self.db_path):
            file_size = os.path.getsize(self.db_path)
            file_modified = datetime.fromtimestamp(os.path.getmtime(self.db_path))
            print(f"📁 DB 파일: {self.db_path}")
            print(f"📊 파일 크기: {file_size:,} bytes ({file_size / 1024:.1f} KB)")
            print(f"🕒 수정 시간: {file_modified.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print(f"❌ DB 파일이 존재하지 않음: {self.db_path}")
            return

        print("\n📋 SQLite 테이블 목록:")
        tables = self.get_table_list()

        if not tables:
            print("  테이블이 없습니다.")
        else:
            for table in tables:
                count = self.get_table_count(table)
                print(f"  📄 {table:20} ({count:,} 레코드)")

        print("\n📁 JSON 파일 현황:")
        for name, path in self.json_files.items():
            if os.path.exists(path):
                data = self.load_json_file(path)
                print(f"  📄 {name:15} ({path:20}) - {len(data):,} 항목")
            else:
                print(f"  ❌ {name:15} ({path:20}) - 파일 없음")

    def show_table_details(self, table_name: str):
        """특정 테이블 상세 정보 출력"""
        print(f"\n📄 테이블 '{table_name}' 상세 정보")
        print("=" * 50)

        # 스키마 정보
        schema = self.get_table_schema(table_name)
        if schema:
            print("\n🔧 테이블 스키마:")
            for col in schema:
                pk_mark = " (PK)" if col["primary_key"] else ""
                null_mark = " NOT NULL" if col["not_null"] else ""
                default_mark = f" DEFAULT {col['default_value']}" if col["default_value"] else ""
                print(f"  {col['name']:15} {col['type']:10}{pk_mark}{null_mark}{default_mark}")

        # 데이터 미리보기
        print(f"\n📊 데이터 미리보기 (상위 5개):")
        df = self.query_table(table_name, limit=5)
        if not df.empty:
            print(df.to_string(index=False))
        else:
            print("  데이터가 없습니다.")

        # 레코드 수
        count = self.get_table_count(table_name)
        print(f"\n📈 총 레코드 수: {count:,}개")

    def show_json_details(self, file_name: str):
        """JSON 파일 상세 정보 출력"""
        if file_name not in self.json_files:
            print(f"❌ 알 수 없는 JSON 파일: {file_name}")
            return

        file_path = self.json_files[file_name]
        print(f"\n📁 JSON 파일 '{file_name}' 상세 정보")
        print("=" * 50)
        print(f"📍 파일 경로: {file_path}")

        if not os.path.exists(file_path):
            print("❌ 파일이 존재하지 않습니다.")
            return

        data = self.load_json_file(file_path)
        if not data:
            print("📄 파일이 비어있습니다.")
            return

        print(f"📊 총 항목 수: {len(data):,}개")

        # 첫 번째 항목의 구조 표시
        if data:
            print("\n🔧 데이터 구조 (첫 번째 항목):")
            first_item = data[0]
            for key, value in first_item.items():
                value_type = type(value).__name__
                value_preview = str(value)[:50] + "..." if len(str(value)) > 50 else str(value)
                print(f"  {key:15} ({value_type:10}): {value_preview}")

        # 최근 5개 항목 미리보기
        print(f"\n📋 최근 {min(5, len(data))}개 항목:")
        for i, item in enumerate(data[-5:], 1):
            print(f"  {i}. {item}")

    def interactive_query(self):
        """대화형 쿼리 모드"""
        print("\n🔍 대화형 쿼리 모드 (종료: 'exit')")
        print("예시: SELECT * FROM recommendedperfume LIMIT 10;")

        while True:
            query = input("\n💬 SQL 쿼리 입력: ").strip()

            if query.lower() in ['exit', 'quit', '종료']:
                print("👋 쿼리 모드를 종료합니다.")
                break

            if not query:
                continue

            try:
                df = self.execute_query(query)
                if not df.empty:
                    print(f"\n📊 쿼리 결과 ({len(df)}개 행):")
                    print(df.to_string(index=False))
                else:
                    print("📄 결과가 없습니다.")
            except Exception as e:
                print(f"❌ 쿼리 실행 오류: {e}")


def main():
    """메인 함수"""
    print("🌸 Whiff 데이터베이스 브라우저")
    print("=" * 60)

    # DB 경로 확인 및 자동 생성
    db_paths = ["./perfume.db", "./whiff.db"]  # perfume.db를 먼저 찾도록 변경
    db_path = None

    # 기존 파일 확인
    for path in db_paths:
        if os.path.exists(path):
            db_path = path
            print(f"✅ 기존 데이터베이스 발견: {path}")
            break

    # 없으면 기본 파일로 생성
    if not db_path:
        db_path = db_paths[0]  # ./whiff.db
        print(f"📁 데이터베이스가 없어서 새로 생성합니다: {db_path}")

        # 사용자 확인
        create = input("데이터베이스를 생성하시겠습니까? (y/n): ").lower()
        if create != 'y':
            print("👋 프로그램을 종료합니다.")
            return

    browser = WhiffDBBrowser(db_path)

    while True:
        print("\n🎯 메뉴:")
        print("1. 데이터베이스 전체 개요")
        print("2. 특정 테이블 조회")
        print("3. JSON 파일 조회")
        print("4. 사용자 정의 쿼리")
        print("5. 대화형 쿼리 모드")
        print("0. 종료")

        choice = input("\n선택하세요: ").strip()

        if choice == "1":
            browser.show_database_overview()

        elif choice == "2":
            tables = browser.get_table_list()
            if tables:
                print("\n📋 사용 가능한 테이블:")
                for i, table in enumerate(tables, 1):
                    print(f"  {i}. {table}")

                try:
                    idx = int(input("\n테이블 번호 선택: ")) - 1
                    if 0 <= idx < len(tables):
                        browser.show_table_details(tables[idx])
                    else:
                        print("❌ 잘못된 번호입니다.")
                except ValueError:
                    print("❌ 숫자를 입력하세요.")
            else:
                print("❌ 테이블이 없습니다.")

        elif choice == "3":
            print("\n📁 JSON 파일 목록:")
            json_names = list(browser.json_files.keys())
            for i, name in enumerate(json_names, 1):
                print(f"  {i}. {name}")

            try:
                idx = int(input("\nJSON 파일 번호 선택: ")) - 1
                if 0 <= idx < len(json_names):
                    browser.show_json_details(json_names[idx])
                else:
                    print("❌ 잘못된 번호입니다.")
            except ValueError:
                print("❌ 숫자를 입력하세요.")

        elif choice == "4":
            query = input("\n💬 SQL 쿼리 입력: ").strip()
            if query:
                df = browser.execute_query(query)
                if not df.empty:
                    print(f"\n📊 쿼리 결과 ({len(df)}개 행):")
                    print(df.to_string(index=False))
                else:
                    print("📄 결과가 없습니다.")

        elif choice == "5":
            browser.interactive_query()

        elif choice == "0":
            print("👋 데이터베이스 브라우저를 종료합니다.")
            break

        else:
            print("❌ 잘못된 선택입니다.")


if __name__ == "__main__":
    main()