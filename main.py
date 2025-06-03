# main.py - 감정 분석 시스템 포함 버전

import logging
import sys
import traceback
import os
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware

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
    description="AI 기반 향수 추천, 시향 코스 추천 및 감정 분석 서비스의 백엔드 API입니다.",
    version="1.2.0"  # 🔄 버전 업데이트 (감정 분석 추가)
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


# ✅ 서버 시작 이벤트 (감정 분석 시스템 포함)
@app.on_event("startup")
async def startup_event():
    try:
        logger.info("🚀 Whiff API 서버 시작 중...")

        # 환경변수 확인 (필수만)
        port = os.getenv('PORT', '8000')
        environment = "production" if os.getenv("RENDER") else "development"

        logger.info(f"📋 기본 설정:")
        logger.info(f"  - 포트: {port}")
        logger.info(f"  - 환경: {environment}")

        # Firebase 초기화 확인 (빠른 체크)
        try:
            from utils.auth_utils import get_firebase_status
            firebase_status = get_firebase_status()
            logger.info(f"🔥 Firebase: {'✅ 사용 가능' if firebase_status['firebase_available'] else '❌ 사용 불가'}")
        except Exception as e:
            logger.warning(f"⚠️ Firebase 상태 확인 건너뜀: {e}")

        # 🎭 감정 분석 시스템 초기화
        try:
            from utils.emotion_analyzer import emotion_analyzer
            stats = emotion_analyzer.get_analysis_stats()
            logger.info(f"🎭 감정 분석 시스템: ✅ 초기화 완료")
            logger.info(f"  - 지원 감정: {stats['supported_emotions']}개")
            logger.info(f"  - 모델 상태: {'✅ 로딩됨' if stats['model_loaded'] else '⚠️ 룰 기반만'}")
            logger.info(f"  - 버전: {stats['model_version']}")
        except Exception as e:
            logger.warning(f"⚠️ 감정 분석 시스템 초기화 건너뜀: {e}")

        # ML 모델은 lazy loading으로 처리 (시작 시 로딩하지 않음)
        logger.info("🤖 ML 모델: Lazy Loading 설정 완료")

        logger.info("✅ Whiff API 서버가 성공적으로 시작되었습니다!")

    except Exception as e:
        logger.error(f"❌ 서버 시작 중 오류: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")


# ✅ 서버 종료 이벤트
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("🔚 Whiff API 서버가 종료됩니다.")


# 🎯 모든 라우터 등록 (감정 분석 라우터 추가)
try:
    logger.info("📋 라우터 등록 시작...")

    # 기존 핵심 라우터들
    from routers.perfume_router import router as perfume_router
    from routers.store_router import router as store_router
    from routers.course_router import router as course_router
    from routers.recommend_router import router as recommend_router
    from routers.diary_router import router as diary_router
    from routers.auth_router import router as auth_router
    from routers.recommendation_save_router import router as recommendation_save_router
    from routers.user_router import router as user_router

    # 🆕 감정 분석 라우터 추가
    from routers.emotion_router import router as emotion_router

    # 라우터 등록 (순서 중요 - 의존성 고려)
    app.include_router(perfume_router)
    app.include_router(store_router)
    app.include_router(course_router)
    app.include_router(recommend_router)
    app.include_router(diary_router)  # 감정 분석 기능이 통합된 버전
    app.include_router(emotion_router)  # 🆕 감정 분석 전용 API
    app.include_router(auth_router)
    app.include_router(user_router)
    app.include_router(recommendation_save_router)

    logger.info("✅ 모든 라우터 등록 완료")
    logger.info("  - 기존 라우터: 8개")
    logger.info("  - 새 라우터: 1개 (감정 분석)")
    logger.info("  - 총 라우터: 9개")

except Exception as e:
    logger.error(f"❌ 라우터 등록 중 오류: {e}")
    logger.error(f"Traceback: {traceback.format_exc()}")


# ✅ 루트 엔드포인트 (기능 업데이트)
@app.get("/", summary="루트", operation_id="get_root")
def read_root():
    return {
        "message": "✅ Whiff API is running!",
        "status": "ok",
        "version": "1.2.0",  # 🔄 버전 업데이트
        "environment": "production" if os.getenv("RENDER") else "development",
        "port": os.getenv("PORT", "8000"),
        "features": [
            "향수 추천",
            "시향 일기",
            "매장 정보",
            "코스 추천",
            "사용자 인증",
            "회원 관리",
            "🆕 감정 분석"  # 새 기능 추가
        ],
        "new_endpoints": [
            "/emotions/analyze - 감정 분석",
            "/emotions/analyze-batch - 배치 감정 분석",
            "/emotions/emotions - 지원 감정 조회",
            "/emotions/status - 감정 시스템 상태"
        ]
    }


@app.head("/", operation_id="head_root")
def head_root():
    return JSONResponse(content={})


# ✅ 헬스 체크 (감정 분석 상태 포함)
@app.get("/health", summary="헬스 체크", operation_id="get_health_check")
def health_check():
    try:
        # 🎭 감정 분석 시스템 상태 체크
        emotion_status = "unknown"
        try:
            from utils.emotion_analyzer import emotion_analyzer
            stats = emotion_analyzer.get_analysis_stats()
            emotion_status = "available" if stats else "error"
        except Exception:
            emotion_status = "unavailable"

        return {
            "status": "ok",
            "service": "Whiff API",
            "version": "1.2.0",
            "environment": "production" if os.getenv("RENDER") else "development",
            "port": os.getenv("PORT", "8000"),
            "uptime": "running",
            "systems": {
                "core_api": "ok",
                "emotion_analysis": emotion_status,  # 🆕 감정 분석 상태
                "firebase_auth": "ok",
                "database": "ok"
            }
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


# ✅ 상태 정보 (감정 분석 정보 포함)
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

        # 🎭 감정 분석 시스템 상태 확인
        emotion_status = None
        try:
            from utils.emotion_analyzer import emotion_analyzer
            emotion_status = emotion_analyzer.get_analysis_stats()
        except Exception as e:
            logger.error(f"감정 분석 상태 확인 실패: {e}")

        return {
            "service": "Whiff API",
            "version": "1.2.0",
            "status": "running",
            "environment": "production" if os.getenv("RENDER") else "development",
            "systems": {
                "firebase": firebase_status,
                "smtp": smtp_status,
                "emotion_analysis": emotion_status  # 🆕 감정 분석 상태
            },
            "features": {
                "auth": "Firebase Authentication",
                "database": "SQLite + JSON Files",
                "ml_model": "TensorFlow (Lazy Loading)",
                "deployment": "Render.com",
                "email": "SMTP (Gmail)",
                "emotion_analysis": "Rule-based + AI Model (준비중)"  # 🆕 감정 분석
            },
            "endpoints": {
                "perfumes": "향수 정보 및 추천",
                "stores": "매장 정보",
                "courses": "시향 코스 추천",
                "diaries": "시향 일기 (감정 분석 통합)",  # 🔄 업데이트됨
                "emotions": "🆕 감정 분석",  # 새 엔드포인트
                "auth": "사용자 인증",
                "users": "사용자 관리"
            }
        }
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )


# 🆕 감정 분석 시스템 전용 상태 확인
@app.get("/emotion-system-info", summary="감정 분석 시스템 정보")
def get_emotion_system_info():
    """감정 분석 시스템의 상세 정보를 제공하는 별도 엔드포인트"""
    try:
        from utils.emotion_analyzer import emotion_analyzer

        stats = emotion_analyzer.get_analysis_stats()
        supported_emotions = emotion_analyzer.get_supported_emotions()
        emotion_mapping = emotion_analyzer.emotion_to_tags

        return {
            "system": "Whiff 감정 분석 시스템",
            "version": "1.0.0",
            "model_version": stats.get("model_version", "unknown"),
            "status": "operational",
            "capabilities": {
                "text_analysis": True,
                "batch_analysis": True,
                "emotion_tagging": True,
                "ai_model": stats.get("model_loaded", False),
                "rule_based": True,
                "korean_language": True
            },
            "supported_emotions": {
                "count": len(supported_emotions),
                "list": supported_emotions,
                "mapping": emotion_mapping
            },
            "limits": {
                "max_text_length": 2000,
                "max_batch_size": 10,
                "supported_languages": ["한국어"]
            },
            "statistics": stats,
            "endpoints": [
                "/emotions/analyze",
                "/emotions/analyze-batch",
                "/emotions/emotions",
                "/emotions/status",
                "/diaries/ (감정 분석 통합)"
            ]
        }

    except Exception as e:
        logger.error(f"Emotion system info failed: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "감정 분석 시스템 정보 조회 실패",
                "message": str(e),
                "fallback_info": {
                    "system": "Whiff 감정 분석 시스템",
                    "status": "error"
                }
            }
        )


# ✅ Render.com을 위한 메인 실행 부분
if __name__ == "__main__":
    import uvicorn

    # Render.com에서 제공하는 PORT 환경변수 사용 (중요!)
    port = int(os.getenv("PORT", 8000))

    logger.info(f"🚀 서버 시작: 포트 {port}")
    logger.info(f"🎭 감정 분석 시스템 포함")

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,  # 프로덕션에서는 reload 비활성화
        access_log=True,
        log_level="info"
    )