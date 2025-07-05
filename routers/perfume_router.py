# Whiff API Server - 모든 라우터를 위한 완전한 라이브러리

# ===== 기본 FastAPI 및 서버 =====
fastapi==0.104.1
uvicorn==0.24.0
python-multipart==0.0.6

# ===== Firebase 및 JWT 인증 =====
firebase-admin==6.2.0
PyJWT==2.8.0
pydantic[email]==2.5.0
email-validator==2.1.0

# ===== 데이터 처리 (향수 라우터용) =====
pandas==2.0.3
numpy==1.24.3

# ===== 이미지 처리 (시향 일기, 신고 라우터용) =====
Pillow==10.0.0

# ===== 데이터베이스 (추천 저장 라우터용) =====
sqlmodel==0.0.14
sqlalchemy==1.4.41

# ===== 머신러닝 (추천 시스템용) =====
scikit-learn==1.3.0
tensorflow==2.15.0

# ===== 기본 유틸리티 =====
requests==2.31.0
python-dotenv==1.0.0
typing-extensions==4.8.0

# ===== 추가 데이터 처리 =====
python-dateutil==2.8.2
openpyxl==3.1.2

# ===== 수학 및 과학 계산 =====
scipy==1.11.3

# ===== JSON 및 성능 최적화 =====
ujson==5.8.0