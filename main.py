# main.py - 개선된 버전
import logging
import sys
import traceback
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
import os

# ✅ 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)

# FastAPI 앱 생성
app = FastAPI(
    title="Whiff API",
    description="AI 기반 향수 추천 및 시향 코스 추천 서비스의 백엔드 API입니다.",
    version="1.0.8"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 구체적인 도메인 설정
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ✅ 전역 예외 핸들러
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """모든 예외를 잡아서 로깅하고 적절한 응답을 반환"""
    logger.error(f"Unhandled exception on {request.method} {request.url}")
    logger.error(f"Exception type: {type(exc).__name__}")
    logger.error(f"Exception message: {str(exc)}")
    logger.error(f"Traceback: {traceback.format_exc()}")

    return JSONResponse(
        status_code=500,
        content={
            "message": "서버 내부 오류가 발생했습니다.",
            "error": str(exc),
            "path": str(request.url.path),
            "method": request.method
        }
    )


# ✅ 유효성 검사 에러 핸들러
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

    logger.error(f"Validation error on {request.method} {request.url}")
    logger.error(f"Error details: {error_details}")

    return JSONResponse(
        status_code=422,
        content={
            "message": "입력값이 유효하지 않습니다.",
            "errors": error_details,
            "path": str(request.url.path)
        }
    )


# ✅ HTTP 예외 핸들러
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.error(f"HTTP {exc.status_code} error on {request.method} {request.url}: {exc.detail}")

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "message": exc.detail,
            "status_code": exc.status_code,
            "path": str(request.url.path)
        }
    )


# ✅ 서버 시작 이벤트
@app.on_event("startup")
async def startup_event():
    try:
        logger.info("🚀 Whiff API 서버 시작 중...")

        # 환경변수 확인
        env_vars = {
            "FIREBASE_PROJECT_ID": os.getenv('FIREBASE_PROJECT_ID'),
            "FIREBASE_PRIVATE_KEY": "설정됨" if os.getenv('FIREBASE_PRIVATE_KEY') else "없음",
            "FIREBASE_CLIENT_EMAIL": os.getenv('FIREBASE_CLIENT_EMAIL'),
            "SMTP_USERNAME": os.getenv('SMTP_USERNAME'),
            "SMTP_PASSWORD": "설정됨" if os.getenv('SMTP_PASSWORD') else "없음"
        }

        logger.info("📋 환경변수 확인:")
        for key, value in env_vars.items():
            if "PASSWORD" in key or "PRIVATE_KEY" in key:
                logger.info(f"  - {key}: {value}")
            else:
                logger.info(f"  - {key}: {value}")

        # Firebase 초기화 확인
        try:
            from utils.auth_utils import get_firebase_status
            firebase_status = get_firebase_status()
            logger.info(f"🔥 Firebase 상태: {'✅ 사용 가능' if firebase_status['firebase_available'] else '❌ 사용 불가'}")
        except Exception as e:
            logger.error(f"❌ Firebase 상태 확인 실패: {e}")

        # SMTP 설정 확인
        try:
            from utils.email_sender import email_sender
            smtp_valid, smtp_message = email_sender.check_smtp_config()
            logger.info(f"📧 SMTP 상태: {'✅ ' + smtp_message if smtp_valid else '❌ ' + smtp_message}")
        except Exception as e:
            logger.error(f"❌ SMTP 상태 확인 실패: {e}")

        logger.info("✅ Whiff API 서버가 성공적으로 시작되었습니다!")

    except Exception as e:
        logger.error(f"❌ 서버 시작 중 오류: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")


# ✅ 서버 종료 이벤트
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("🔚 Whiff API 서버가 종료됩니다.")


# 라우터 임포트 및 등록 (예외 처리 포함)
try:
    from routers.perfume_router import router as perfume_router
    from routers.store_router import router as store_router
    from routers.course_router import router as course_router
    from routers.recommend_router import router as recommend_router
    from routers.diary_router import router as diary_router
    from routers.auth_router import router as auth_router
    from routers.recommendation_save_router import router as recommendation_save_router
    from routers.user_router import router as user_router

    # Firebase 없이 테스트하기 위한 임시 라우터
    try:
        from routers.temp_auth_router import router as temp_auth_router

        app.include_router(temp_auth_router)
        logger.info("🧪 임시 인증 라우터 등록 완료")
    except Exception as e:
        logger.warning(f"⚠️ 임시 인증 라우터 등록 실패: {e}")

    # 라우터 등록
    app.include_router(perfume_router)
    app.include_router(store_router)
    app.include_router(course_router)
    app.include_router(recommend_router)
    app.include_router(diary_router)
    app.include_router(auth_router)
    app.include_router(user_router)
    app.include_router(recommendation_save_router)

    logger.info("📋 모든 라우터가 성공적으로 등록되었습니다.")

except Exception as e:
    logger.error(f"❌ 라우터 등록 중 오류: {e}")
    logger.error(f"Traceback: {traceback.format_exc()}")


# ✅ 루트 엔드포인트
@app.get("/", summary="루트", operation_id="get_root")
def read_root():
    return {
        "message": "✅ Whiff API is running!",
        "status": "ok",
        "version": "1.0.7",
        "environment": "production" if os.getenv("RENDER") else "development"
    }


@app.head("/", operation_id="head_root")
def head_root():
    return JSONResponse(content={})


# ✅ 헬스 체크
@app.get("/health", summary="헬스 체크", operation_id="get_health_check")
def health_check():
    try:
        # 간단한 헬스 체크
        return {
            "status": "ok",
            "service": "Whiff API",
            "version": "1.0.7",
            "timestamp": logger.handlers[0].formatter.formatTime(
                logger.makeRecord("", 0, "", 0, "", (), None)
            ) if logger.handlers else None
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={"status": "error", "message": str(e)}
        )


@app.head("/health", operation_id="head_health_check")
def head_health_check():
    return JSONResponse(content={})


# ✅ 상세 상태 정보
@app.get("/status", summary="서버 상태 정보", operation_id="get_server_status")
def get_server_status():
    try:
        # Firebase 상태 확인
        firebase_status = None
        try:
            from utils.auth_utils import get_firebase_status
            firebase_status = get_firebase_status()
        except Exception as e:
            logger.error(f"Firebase 상태 확인 실패: {e}")

        # SMTP 상태 확인
        smtp_status = None
        try:
            from utils.email_sender import email_sender
            smtp_valid, smtp_message = email_sender.check_smtp_config()
            smtp_status = {"configured": smtp_valid, "message": smtp_message}
        except Exception as e:
            logger.error(f"SMTP 상태 확인 실패: {e}")

        return {
            "service": "Whiff API",
            "version": "1.0.7",
            "status": "running",
            "environment": "production" if os.getenv("RENDER") else "development",
            "firebase": firebase_status,
            "smtp": smtp_status,
            "features": {
                "auth": "Firebase",
                "database": "SQLite + JSON",
                "ml_model": "TensorFlow",
                "deployment": "Render.com"
            }
        }
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )