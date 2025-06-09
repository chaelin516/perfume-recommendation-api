# main.py - 2차 추천 라우터 추가 및 Google Drive 감정 모델 연동 버전
import logging
import sys
import traceback
import os
import asyncio
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
    description="AI 기반 향수 추천 및 시향 코스 추천 서비스의 백엔드 API입니다.",
    version="1.3.0"  # Google Drive 감정 모델 연동으로 버전 업데이트
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


# ✅ 백그라운드 작업 함수들
async def initialize_emotion_model_async():
    """백그라운드에서 감정 모델 초기화"""
    try:
        logger.info("🤖 백그라운드 감정 모델 초기화 시작...")

        # 감정 분석기 import (지연 로딩)
        from utils.emotion_analyzer import emotion_analyzer
        from utils.model_downloader import model_downloader

        # 모델 다운로드 및 로딩 시도
        logger.info("📦 Google Drive에서 감정 모델 확인 중...")
        results = model_downloader.ensure_models_available()

        if results.get("emotion_model", False):
            logger.info("✅ 감정 모델 다운로드 완료, AI 모델 로딩 시도...")

            # AI 모델 초기화
            ai_loaded = await emotion_analyzer.initialize_ai_model()

            if ai_loaded:
                logger.info("🎉 감정 태깅 AI 모델 준비 완료!")
            else:
                logger.warning("⚠️ AI 모델 로딩 실패 - 키워드 기반으로 동작")
        else:
            logger.warning("⚠️ 감정 모델 다운로드 실패 - 키워드 기반으로 동작")

    except Exception as e:
        logger.error(f"❌ 백그라운드 감정 모델 초기화 실패: {e}")
        logger.info("📋 키워드 기반 감정 분석으로 폴백")


async def download_models_async():
    """백그라운드에서 모델 다운로드 (빠른 확인용)"""
    try:
        from utils.model_downloader import model_downloader

        # 빠른 상태 확인 (다운로드는 실제 사용 시점에)
        logger.info("🔍 모델 파일 상태 빠른 확인...")
        status = model_downloader.get_download_status()

        emotion_model = status["models"].get("emotion_model", {})
        if emotion_model.get("is_valid", False):
            logger.info("✅ 감정 모델 파일 로컬에 준비됨")
        else:
            logger.info("📥 감정 모델 파일 없음 - 첫 사용 시 다운로드 예정")

    except Exception as e:
        logger.warning(f"⚠️ 모델 상태 확인 중 오류: {e}")


# ✅ 서버 시작 이벤트 (최적화된 버전)
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
        logger.info(f"  - API 버전: 1.3.0 (Google Drive 감정 모델 연동)")

        # Firebase 초기화 확인 (빠른 체크)
        try:
            from utils.auth_utils import get_firebase_status
            firebase_status = get_firebase_status()
            logger.info(f"🔥 Firebase: {'✅ 사용 가능' if firebase_status['firebase_available'] else '❌ 사용 불가'}")
        except Exception as e:
            logger.warning(f"⚠️ Firebase 상태 확인 건너뜀: {e}")

        # 🆕 Google Drive 감정 모델 백그라운드 초기화
        try:
            logger.info("🎭 감정 분석 시스템 초기화...")

            # 감정 분석기 기본 초기화 (키워드 시스템)
            from utils.emotion_analyzer import emotion_analyzer
            logger.info("✅ 키워드 기반 감정 분석 시스템 준비 완료")

            # AI 모델은 백그라운드에서 로딩 (서버 시작 지연 방지)
            use_ai_model = os.getenv('USE_EMOTION_AI_MODEL', 'true').lower() == 'true'

            if use_ai_model:
                logger.info("🤖 AI 감정 모델 백그라운드 초기화 예약...")
                asyncio.create_task(initialize_emotion_model_async())
            else:
                logger.info("🎭 AI 감정 모델 비활성화 (환경변수 설정)")

        except Exception as e:
            logger.warning(f"⚠️ 감정 분석 시스템 초기화 건너뜀: {e}")

        # ML 향수 추천 모델은 lazy loading으로 처리
        logger.info("🤖 향수 추천 ML 모델: Lazy Loading 설정 완료")

        logger.info("✅ Whiff API 서버가 빠르게 시작되었습니다!")
        logger.info("🎉 Google Drive 연동 감정 모델 지원!")

    except Exception as e:
        logger.error(f"❌ 서버 시작 중 오류: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")


# ✅ 서버 종료 이벤트
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("🔚 Whiff API 서버가 종료됩니다.")


# 🎯 모든 라우터 등록 (감정 태깅 라우터 포함)
try:
    logger.info("📋 라우터 등록 시작...")

    # 기존 라우터들
    from routers.perfume_router import router as perfume_router
    from routers.store_router import router as store_router
    from routers.course_router import router as course_router
    from routers.recommend_router import router as recommend_router
    from routers.diary_router import router as diary_router
    from routers.auth_router import router as auth_router
    from routers.recommendation_save_router import router as recommendation_save_router
    from routers.user_router import router as user_router

    # 🆕 2차 추천 라우터
    from routers.recommend_2nd_router import router as recommend_2nd_router

    # 🆕 감정 태깅 라우터 (Google Drive 연동)
    from routers.emotion_tagging_router import router as emotion_tagging_router

    # 라우터 등록 (등록 순서 중요)
    app.include_router(perfume_router)  # 기본 향수 정보
    app.include_router(store_router)  # 매장 정보
    app.include_router(course_router)  # 시향 코스
    app.include_router(recommend_router)  # 1차 추천 (기존)
    app.include_router(recommend_2nd_router)  # 🆕 2차 추천 (노트 기반)
    app.include_router(diary_router)  # 시향 일기
    app.include_router(emotion_tagging_router)  # 🆕 감정 태깅 (Google Drive)
    app.include_router(auth_router)  # 인증
    app.include_router(user_router)  # 사용자 관리
    app.include_router(recommendation_save_router)  # 추천 저장

    logger.info("✅ 모든 라우터 등록 완료")
    logger.info("🆕 2차 추천 라우터 (/perfumes/recommend-2nd) 추가됨")
    logger.info("🎭 감정 태깅 라우터 (/emotions/*) 추가됨 (Google Drive 연동)")

except Exception as e:
    logger.error(f"❌ 라우터 등록 중 오류: {e}")
    logger.error(f"Traceback: {traceback.format_exc()}")


# ✅ 루트 엔드포인트
@app.get("/", summary="루트", operation_id="get_root")
def read_root():
    return {
        "message": "✅ Whiff API is running!",
        "status": "ok",
        "version": "1.3.0",
        "environment": "production" if os.getenv("RENDER") else "development",
        "port": os.getenv("PORT", "8000"),
        "features": [
            "향수 추천 (1차)",
            "향수 추천 (2차 - 노트 기반)",
            "시향 일기",
            "감정 태깅 (AI + 키워드 기반)",  # 🆕 추가됨
            "매장 정보",
            "코스 추천",
            "사용자 인증",
            "회원 관리"
        ],
        "new_features": [
            "🆕 2차 추천 API (/perfumes/recommend-2nd)",
            "🎯 사용자 노트 선호도 기반 정밀 추천",
            "🧮 AI 감정 클러스터 + 노트 매칭 알고리즘",
            "🎭 Google Drive 연동 감정 태깅 (/emotions/*)",  # 🆕 추가됨
            "🤖 1.33GB AI 모델 자동 다운로드 시스템"  # 🆕 추가됨
        ]
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
            "version": "1.3.0",
            "environment": "production" if os.getenv("RENDER") else "development",
            "port": os.getenv("PORT", "8000"),
            "uptime": "running",
            "features_available": [
                "1차 추천",
                "2차 추천 (노트 기반)",
                "시향 일기",
                "감정 태깅 (Google Drive AI)",  # 🆕
                "매장 정보",
                "사용자 인증"
            ]
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


# ✅ 상태 정보 (Google Drive 연동 상태 포함)
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

        # 🆕 감정 모델 상태 확인
        emotion_model_status = None
        try:
            from utils.emotion_analyzer import emotion_analyzer
            from utils.model_downloader import model_downloader

            emotion_stats = emotion_analyzer.get_analysis_stats()
            download_status = model_downloader.get_download_status()

            emotion_model_status = {
                "ai_model_loaded": emotion_stats["model_loaded"],
                "use_ai_model": emotion_stats["use_ai_model"],
                "google_drive_models": download_status["models"],
                "analysis_count": emotion_stats["performance"]["total_analyses"],
                "success_rate": emotion_stats["performance"]["success_rate"]
            }
        except Exception as e:
            logger.error(f"감정 모델 상태 확인 실패: {e}")

        return {
            "service": "Whiff API",
            "version": "1.3.0",
            "status": "running",
            "environment": "production" if os.getenv("RENDER") else "development",
            "firebase": firebase_status,
            "smtp": smtp_status,
            "emotion_model": emotion_model_status,  # 🆕 추가
            "features": {
                "auth": "Firebase Authentication",
                "database": "SQLite + JSON Files",
                "ml_model": "TensorFlow (Lazy Loading)",
                "emotion_ai": "Google Drive + TensorFlow",  # 🆕 추가
                "deployment": "Render.com",
                "email": "SMTP (Gmail)"
            },
            "endpoints": {
                "perfumes": "향수 정보 및 1차 추천",
                "perfumes_2nd": "🆕 2차 추천 (노트 기반)",
                "emotions": "🆕 감정 태깅 (Google Drive AI)",  # 🆕 추가
                "stores": "매장 정보",
                "courses": "시향 코스 추천",
                "diaries": "시향 일기",
                "auth": "사용자 인증",
                "users": "사용자 관리"
            },
            "recommendation_system": {
                "primary_recommendation": {
                    "endpoint": "/perfumes/recommend-cluster",
                    "method": "AI 감정 클러스터 모델",
                    "input": "사용자 선호도 6개 특성",
                    "output": "클러스터 + 향수 인덱스"
                },
                "secondary_recommendation": {
                    "endpoint": "/perfumes/recommend-2nd",
                    "method": "노트 매칭 + 감정 가중치",
                    "input": "노트 선호도 + 감정 확률 + 선택 인덱스",
                    "output": "정밀 점수 기반 추천"
                },
                "emotion_tagging": {  # 🆕 추가
                    "endpoint": "/emotions/predict",
                    "method": "Google Drive AI 모델 + 키워드 기반",
                    "input": "시향 일기 텍스트",
                    "output": "감정 + 태그 + 신뢰도"
                }
            }
        }
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )


# ✅ API 문서 정보 (Google Drive 연동 포함)
@app.get("/api-info", summary="API 정보", operation_id="get_api_info")
def get_api_info():
    """API 기능 및 엔드포인트 정보 제공"""
    return {
        "api_name": "Whiff API",
        "version": "1.3.0",
        "description": "AI 기반 향수 추천 및 시향 코스 추천 서비스 (Google Drive 감정 모델 연동)",
        "documentation_url": "/docs",
        "redoc_url": "/redoc",

        "recommendation_flow": {
            "step_1": {
                "title": "1차 추천",
                "endpoint": "/perfumes/recommend-cluster",
                "description": "사용자 선호도 → AI 감정 클러스터 → 향수 인덱스 목록",
                "input_example": {
                    "gender": "women",
                    "season_tags": "spring",
                    "time_tags": "day",
                    "desired_impression": "confident, fresh",
                    "activity": "casual",
                    "weather": "hot"
                }
            },
            "step_2": {
                "title": "2차 추천",
                "endpoint": "/perfumes/recommend-2nd",
                "description": "노트 선호도 + 1차 결과 → 정밀 점수 계산 → 최종 추천",
                "input_example": {
                    "user_note_scores": {
                        "jasmine": 5,
                        "rose": 4,
                        "amber": 3,
                        "musk": 0,
                        "citrus": 2,
                        "vanilla": 1
                    },
                    "emotion_proba": [0.01, 0.03, 0.85, 0.02, 0.05, 0.04],
                    "selected_idx": [23, 45, 102, 200, 233, 305, 399, 410, 487, 512]
                }
            },
            "step_3": {  # 🆕 추가
                "title": "감정 태깅",
                "endpoint": "/emotions/predict",
                "description": "시향 일기 텍스트 → Google Drive AI 모델 → 감정 + 태그",
                "input_example": {
                    "text": "이 향수 정말 좋아요! 달콤하고 상큼해서 기분이 좋아져요.",
                    "use_ai_model": True
                }
            }
        },

        "main_features": [
            "🤖 AI 감정 클러스터 기반 1차 추천",
            "🎯 노트 선호도 기반 2차 정밀 추천",
            "🎭 Google Drive AI 감정 태깅 (1.33GB 모델)",  # 🆕 추가
            "📝 시향 일기 작성 및 관리",
            "🗺️ 위치 기반 시향 코스 추천",
            "🏪 매장 정보 및 검색",
            "🔐 Firebase 인증 시스템",
            "📧 이메일 발송 기능",
            "👥 사용자 관리 (회원가입/탈퇴)"
        ],

        "technical_stack": {
            "framework": "FastAPI",
            "ml_framework": "TensorFlow + scikit-learn",
            "emotion_ai": "Google Drive + TensorFlow (1.33GB 모델)",  # 🆕 추가
            "authentication": "Firebase Auth",
            "database": "SQLite + JSON Files",
            "deployment": "Render.com",
            "email": "SMTP (Gmail)"
        },

        "google_drive_integration": {  # 🆕 추가
            "model_size": "1.33GB",
            "auto_download": "서버 시작 시 자동 확인",
            "fallback": "키워드 기반 감정 분석",
            "supported_emotions": ["기쁨", "불안", "당황", "분노", "상처", "슬픔", "우울", "흥분"]
        }
    }


# ✅ Render.com을 위한 메인 실행 부분
if __name__ == "__main__":
    import uvicorn

    # Render.com에서 제공하는 PORT 환경변수 사용 (중요!)
    port = int(os.getenv("PORT", 8000))

    logger.info(f"🚀 서버 시작: 포트 {port}")
    logger.info(f"🆕 Google Drive 감정 모델 연동이 포함된 Whiff API v1.3.0")

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,  # 프로덕션에서는 reload 비활성화
        access_log=True,
        log_level="info"
    )