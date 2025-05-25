FROM python:3.9-slim

WORKDIR /project

# 필요한 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 모델 및 데이터 파일 복사
COPY /models /project/models
COPY /data /project/data
COPY main.py .

# 포트 설정
EXPOSE 8080

# 애플리케이션 실행
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
