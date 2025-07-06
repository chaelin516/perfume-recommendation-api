# db/session.py - 수정된 버전
import os
from sqlmodel import SQLModel, create_engine, Session
import logging

logger = logging.getLogger(__name__)

# 데이터베이스 URL 설정 (환경변수 우선, 없으면 기본값)
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./whiff.db")

# SQLite의 경우 특별 처리
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        echo=False  # SQL 로그 출력 비활성화 (필요시 True로 변경)
    )
else:
    engine = create_engine(DATABASE_URL, echo=False)

logger.info(f"📊 데이터베이스 엔진 생성: {DATABASE_URL}")


def create_db_and_tables():
    """데이터베이스 테이블 생성"""
    try:
        SQLModel.metadata.create_all(engine)
        logger.info("✅ 데이터베이스 테이블 생성 완료")
    except Exception as e:
        logger.error(f"❌ 데이터베이스 테이블 생성 실패: {e}")


def get_session():
    """데이터베이스 세션 반환"""
    try:
        with Session(engine) as session:
            yield session
    except Exception as e:
        logger.error(f"❌ 데이터베이스 세션 생성 실패: {e}")
        raise


# 앱 시작 시 테이블 생성
def init_db():
    """데이터베이스 초기화"""
    try:
        create_db_and_tables()
        logger.info("✅ 데이터베이스 초기화 완료")
    except Exception as e:
        logger.error(f"❌ 데이터베이스 초기화 실패: {e}")


# 모듈 로드 시 자동 초기화
try:
    init_db()
except Exception as e:
    logger.warning(f"⚠️ 데이터베이스 자동 초기화 실패: {e}")


# 데이터베이스 상태 확인 함수
def check_database_status():
    """데이터베이스 상태 확인"""
    try:
        # 데이터베이스 파일 존재 확인
        if DATABASE_URL.startswith("sqlite"):
            db_file = DATABASE_URL.replace("sqlite:///", "").replace("sqlite://", "")
            file_exists = os.path.exists(db_file)
            file_size = os.path.getsize(db_file) if file_exists else 0

            return {
                "database_url": DATABASE_URL,
                "file_exists": file_exists,
                "file_size": file_size,
                "file_path": db_file
            }
        else:
            return {
                "database_url": DATABASE_URL,
                "type": "non-sqlite"
            }
    except Exception as e:
        return {
            "error": str(e),
            "database_url": DATABASE_URL
        }