FROM python:3.9-slim

WORKDIR /project

COPY perfume_docker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY perfume_docker/models /project/models
COPY perfume_docker/data /project/data
COPY perfume_docker/perfume_backend /project/perfume_backend

# 🔥 수정된 실행 명령
CMD ["uvicorn", "perfume_backend.main:app", "--host", "0.0.0.0", "--port", "8080"]
