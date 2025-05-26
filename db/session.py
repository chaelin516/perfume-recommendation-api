from sqlmodel import create_engine, Session

DATABASE_URL = "sqlite:///./perfume.db"  # 또는 사용 중인 DB URL
engine = create_engine(DATABASE_URL, echo=True)

def get_session():
    return Session(engine)
