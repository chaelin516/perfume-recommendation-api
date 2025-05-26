# init_db.py

from sqlmodel import SQLModel
models.recommendation import RecommendedPerfume
db.session import engine  # ← session.py에 있는 engine

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

if __name__ == "__main__":
    create_db_and_tables()
    print("✅ DB 테이블 생성 완료")
