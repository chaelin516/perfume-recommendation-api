FROM python:3.9-slim

# 컨테이너 내부 작업 디렉토리 설정
WORKDIR /project

# requirements.txt 복사 및 패키지 설치
COPY perfume_docker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 모델, 데이터, main.py 복사
COPY perfume_docker/models /project/models
COPY perfume_docker/data /project/data
COPY perfume_docker/main.py .

# 포트 오픈
EXPOSE 8080

# FastAPI 실행
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
