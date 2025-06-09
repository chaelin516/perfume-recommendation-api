# main.py - 감정 태깅 라우터 추가 및 Google Drive 모델 지원 버전
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
    description="AI 기반 향수 추천, 시향 코스 추천 및 감정 태깅 서비스의 백엔드 API입니다.",
    version="1.3.0"  # 🆕 감정 태깅 및 Google Drive 모델 지원 추가
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


# ✅ 서버 시작 이벤트 (최적화)
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
        logger.info(f"  - API 버전: 1.3.0 (감정 태깅 및 Google Drive 모델 지원)")

        # Firebase 초기화 확인 (빠른 체크)
        try:
            from utils.auth_utils import get_firebase_status
            firebase_status = get_firebase_status()
            logger.info(f"🔥 Firebase: {'✅ 사용 가능' if firebase_status['firebase_available'] else '❌ 사용 불가'}")
        except Exception as e:
            logger.warning(f"⚠️ Firebase 상태 확인 건너뜀: {e}")

        # 🆕 감정 분석기 초기화 확인
        try:
            from utils.emotion_analyzer import emotion_analyzer
            stats = emotion_analyzer.get_analysis_stats()
            logger.info(f"🎭 감정 분석기: {'✅ 준비 완료' if stats['model_loaded'] else '📋 룰 기반 준비'}")
            logger.info(f"  - 지원 감정: {stats['supported_emotions']}개")
            logger.info(f"  - 분석 방법: {stats['analysis_methods']}")
        except Exception as e:
            logger.warning(f"⚠️ 감정 분석기 상태 확인 건너뜀: {e}")

        # ML 모델은 lazy loading으로 처리 (시작 시 로딩하지 않음)
        logger.info("🤖 ML 모델: Lazy Loading 설정 완료")

        # 🆕 Google Drive 모델 확인
        try:
            from utils.emotion_analyzer import emotion_analyzer
            if hasattr(emotion_analyzer, 'check_google_drive_model'):
                gdrive_status = emotion_analyzer.check_google_drive_model()
                logger.info(f"☁️ Google Drive 모델: {'✅ 사용 가능' if gdrive_status else '📋 로컬 모델 사용'}")
        except Exception as e:
            logger.warning(f"⚠️ Google Drive 모델 확인 건너뜀: {e}")

        logger.info("✅ Whiff API 서버가 빠르게 시작되었습니다!")

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
    from routers.recommend_2nd_router import router as recommend_2nd_router

    # 🆕 감정 태깅 라우터 추가
    from routers.emotion_tagging_router import router as emotion_tagging_router

    # 라우터 등록 (등록 순서 중요)
    app.include_router(perfume_router)  # 기본 향수 정보
    app.include_router(store_router)  # 매장 정보
    app.include_router(course_router)  # 시향 코스
    app.include_router(recommend_router)  # 1차 추천 (기존)
    app.include_router(recommend_2nd_router)  # 2차 추천 (노트 기반)
    app.include_router(emotion_tagging_router)  # 🆕 감정 태깅
    app.include_router(diary_router)  # 시향 일기
    app.include_router(auth_router)  # 인증
    app.include_router(user_router)  # 사용자 관리
    app.include_router(recommendation_save_router)  # 추천 저장

    logger.info("✅ 모든 라우터 등록 완료")
    logger.info("🆕 감정 태깅 라우터 (/emotions) 추가됨")

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
            "감정 태깅 (🆕)",  # 새로 추가
            "시향 일기",
            "매장 정보",
            "코스 추천",
            "사용자 인증",
            "회원 관리"
        ],
        "new_features": [
            "🆕 감정 태깅 API (/emotions)",
            "🎭 AI 기반 텍스트 감정 분석",
            "☁️ Google Drive 모델 지원",
            "📊 감정 분석 성능 모니터링"
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
                "감정 태깅 (🆕)",  # 새로 추가
                "시향 일기",
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


# ✅ 상태 정보
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

        # 🆕 감정 분석기 상태 확인
        emotion_status = None
        try:
            from utils.emotion_analyzer import emotion_analyzer
            emotion_status = emotion_analyzer.get_analysis_stats()
        except Exception as e:
            logger.error(f"감정 분석기 상태 확인 실패: {e}")

        return {
            "service": "Whiff API",
            "version": "1.3.0",
            "status": "running",
            "environment": "production" if os.getenv("RENDER") else "development",
            "firebase": firebase_status,
            "smtp": smtp_status,
            "emotion_analyzer": emotion_status,  # 🆕 추가
            "features": {
                "auth": "Firebase Authentication",
                "database": "SQLite + JSON Files",
                "ml_model": "TensorFlow (Lazy Loading)",
                "emotion_analysis": "AI + Rule-based",  # 🆕 추가
                "deployment": "Render.com",
                "email": "SMTP (Gmail)"
            },
            "endpoints": {
                "perfumes": "향수 정보 및 1차 추천",
                "perfumes_2nd": "2차 추천 (노트 기반)",
                "emotions": "🆕 감정 태깅 및 분석",  # 🆕 추가
                "stores": "매장 정보",
                "courses": "시향 코스 추천",
                "diaries": "시향 일기",
                "auth": "사용자 인증",
                "users": "사용자 관리"
            },
            "analysis_systems": {  # 🆕 추가
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
                    }
                },
                "emotion_analysis": {  # 🆕 추가
                    "endpoint": "/emotions/analyze",
                    "method": "AI 모델 + 룰 기반",
                    "input": "텍스트 (시향 일기, 리뷰 등)",
                    "output": "감정 태그 + 신뢰도"
                }
            }
        }
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )


# ✅ API 문서 정보
@app.get("/api-info", summary="API 정보", operation_id="get_api_info")
def get_api_info():
    """API 기능 및 엔드포인트 정보 제공"""
    return {
        "api_name": "Whiff API",
        "version": "1.3.0",
        "description": "AI 기반 향수 추천, 시향 코스 추천 및 감정 태깅 서비스",
        "documentation_url": "/docs",
        "redoc_url": "/redoc",

        "main_workflows": {  # 🆕 수정
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
                }
            },
            "emotion_analysis_flow": {  # 🆕 추가
                "step_1": {
                    "title": "텍스트 감정 분석",
                    "endpoint": "/emotions/analyze",
                    "description": "시향 일기나 리뷰 텍스트 → AI 감정 분석 → 감정 태그",
                    "input_example": {
                        "text": "이 향수 정말 좋아요! 달콤하고 상큼해서 기분이 좋아져요."
                    }
                },
                "step_2": {
                    "title": "배치 감정 분석",
                    "endpoint": "/emotions/analyze-batch",
                    "description": "여러 텍스트 → 일괄 감정 분석 → 감정 통계",
                    "input_example": {
                        "texts": [
                            "향이 너무 진해서 별로예요",
                            "달콤하고 포근한 향기가 좋아요",
                            "상쾌하고 시원한 느낌이에요"
                        ]
                    }
                }
            }
        },

        "main_features": [
            "🤖 AI 감정 클러스터 기반 1차 추천",
            "🎯 노트 선호도 기반 2차 정밀 추천",
            "🎭 AI 기반 텍스트 감정 분석 (🆕)",  # 새로 추가
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
            "emotion_analysis": "AI Model + Rule-based",  # 🆕 추가
            "authentication": "Firebase Auth",
            "database": "SQLite + JSON Files",
            "deployment": "Render.com",
            "email": "SMTP (Gmail)",
            "model_storage": "Local + Google Drive"  # 🆕 추가
        }
    }


# ✅ Render.com을 위한 메인 실행 부분
if __name__ == "__main__":
    import uvicorn

    # Render.com에서 제공하는 PORT 환경변수 사용 (중요!)
    port = int(os.getenv("PORT", 8000))

    logger.info(f"🚀 서버 시작: 포트 {port}")
    logger.info(f"🆕 감정 태깅 및 Google Drive 모델 지원이 포함된 Whiff API v1.3.0")

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,  # 프로덕션에서는 reload 비활성화
        access_log=True,
        log_level="info"
    )