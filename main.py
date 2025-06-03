# main.py - 수정된 버전

import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.exception_handlers import request_validation_exception_handler

# ✅ 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 기존 import들...
from routers.perfume_router import router as perfume_router
from routers.store_router import router as store_router
from routers.course_router import router as course_router
from routers.recommend_router import router as recommend_router
from routers.diary_router import router as diary_router
from routers.auth_router import router as auth_router
from routers.recommendation_save_router import router as recommendation_save_router
from routers.user_router import router as user_router

app = FastAPI(
    title="ScentRoute API",
    description="AI 기반 향수 추천 및 시향 코스 추천 서비스의 백엔드 API입니다.",
    version="1.0.6"
)

# 모든 라우터 등록
app.include_router(perfume_router)
app.include_router(store_router)
app.include_router(course_router)
app.include_router(recommend_router)
app.include_router(diary_router)
app.include_router(auth_router)
app.include_router(user_router)
app.include_router(recommendation_save_router)


# ✅ 서버 시작 이벤트
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 ScentRoute API 서버가 시작되었습니다.")
    logger.info("📋 등록된 라우터: perfume, store, course, recommend, diary, auth, user, recommendation_save")


# ✅ 서버 종료 이벤트
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("🔚 ScentRoute API 서버가 종료됩니다.")


# ✅ 유효성 검사 에러 커스텀 응답 처리
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    error_details = []
    for error in exc.errors():
        error_details.append({
            "field": " -> ".join(str(x) for x in error["loc"]),
            "message": error["msg"],
            "type": error["type"],
            "input": error.get("input")
        })

    # 에러 로깅
    logger.error(f"Validation error on {request.method} {request.url}")
    logger.error(f"Error details: {error_details}")

    return JSONResponse(
        status_code=422,
        content={
            "message": "입력값이 유효하지 않습니다.",
            "errors": error_details,
            "detail": "요청 데이터의 형식을 확인해주세요.",
            "path": str(request.url.path)
        }
    )


# ✅ 루트 엔드포인트 (operation_id 명시적 지정)
@app.get(
    "/",
    summary="루트",
    description="ScentRoute API 서버가 정상 작동 중인지 확인합니다.",
    response_description="서버 상태 메시지",
    operation_id="get_root"
)
def read_root():
    return {
        "message": "✅ ScentRoute API is running!",
        "status": "ok",
        "version": "1.0.3"
    }


# ✅ HEAD 요청 지원 (별도 operation_id)
@app.head(
    "/",
    summary="루트 헤드 체크",
    description="서버 상태를 헤더로만 확인",
    operation_id="head_root"
)
def head_root():
    return JSONResponse(content={})


# ✅ 헬스 체크 엔드포인트 (operation_id 명시적 지정)
@app.get(
    "/health",
    summary="헬스 체크",
    description="서버 상태 확인용 경량 엔드포인트",
    operation_id="get_health_check"
)
def health_check():
    return {"status": "ok", "service": "ScentRoute API"}


# ✅ HEAD 요청 지원 (별도 operation_id)
@app.head(
    "/health",
    summary="헬스 체크 헤드",
    description="서버 상태를 헤더로만 확인",
    operation_id="head_health_check"
)
def head_health_check():
    return JSONResponse(content={})


# ✅ 상세 상태 정보 엔드포인트
@app.get(
    "/status",
    summary="서버 상태 정보",
    description="서버의 상세 상태 정보를 반환합니다.",
    operation_id="get_server_status"
)
def get_server_status():
    return {
        "service": "ScentRoute API",
        "version": "1.0.3",
        "status": "running",
        "features": {
            "auth": "Firebase",
            "database": "SQLite + JSON",
            "ml_model": "TensorFlow",
            "deployment": "Render.com"
        }
    }