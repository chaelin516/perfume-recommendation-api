# main.py - 2차 추천 라우터 추가 및 감정 모델 초기화 최적화 버전
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
    description="AI 기반 향수 추천 및 감정 분석 서비스의 백엔드 API입니다.",
    version="1.3.0"  # 감정 모델 추가로 버전 업데이트
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


# ✅ 서버 시작 이벤트 (감정 모델 초기화 포함)
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
        logger.info(f"  - API 버전: 1.3.0 (감정 모델 + 2차 추천 기능 포함)")

        # Firebase 초기화 확인 (빠른 체크)
        try:
            from utils.auth_utils import get_firebase_status
            firebase_status = get_firebase_status()
            logger.info(f"🔥 Firebase: {'✅ 사용 가능' if firebase_status['firebase_available'] else '❌ 사용 불가'}")
        except Exception as e:
            logger.warning(f"⚠️ Firebase 상태 확인 건너뜀: {e}")

        # 🆕 감정 모델 초기화 (Google Drive 연동)
        try:
            from utils.emotion_model_loader import initialize_emotion_models, get_emotion_models_status
            logger.info("🎭 감정 태깅 모델 초기화 시작...")

            emotion_success = initialize_emotion_models()
            emotion_status = get_emotion_models_status()

            if emotion_success:
                logger.info("✅ 감정 태깅 모델 로딩 완료")
                logger.info(f"  - 벡터라이저: {'✅' if emotion_status['vectorizer_loaded'] else '❌'}")
                logger.info(f"  - 감정 모델: {'✅' if emotion_status['emotion_model_loaded'] else '❌'}")

                # 파일 크기 정보
                if emotion_status.get('vectorizer_size'):
                    logger.info(f"  - 벡터라이저 크기: {emotion_status['vectorizer_size']:,} bytes")
                if emotion_status.get('emotion_model_size'):
                    logger.info(f"  - 감정 모델 크기: {emotion_status['emotion_model_size']:,} bytes")

            else:
                logger.warning("⚠️ 감정 태깅 모델 로딩 실패 - 룰 기반으로 동작")
                logger.warning(f"  - 벡터라이저 파일 존재: {'✅' if emotion_status['vectorizer_exists'] else '❌'}")
                logger.warning(f"  - 감정 모델 파일 존재: {'✅' if emotion_status['emotion_model_exists'] else '❌'}")

        except Exception as e:
            logger.warning(f"⚠️ 감정 모델 초기화 건너뜀: {e}")

        # 🆕 감정 분석기 초기화 확인
        try:
            from utils.emotion_analyzer import emotion_analyzer
            stats = emotion_analyzer.get_analysis_stats()
            logger.info(f"🎭 감정 분석기 상태:")
            logger.info(f"  - 지원 감정: {stats['supported_emotions']}개")
            logger.info(f"  - 총 키워드: {stats['total_keywords']}개")
            logger.info(f"  - 모델 로드됨: {'✅' if stats['model_loaded'] else '❌ (룰 기반 동작)'}")
            logger.info(f"  - 분석 방법: {stats['analysis_methods']}")
        except Exception as e:
            logger.warning(f"⚠️ 감정 분석기 상태 확인 건너뜀: {e}")

        # ML 모델은 lazy loading으로 처리 (시작 시 로딩하지 않음)
        logger.info("🤖 향수 추천 모델: Lazy Loading 설정 완료")

        logger.info("✅ Whiff API 서버가 빠르게 시작되었습니다!")

    except Exception as e:
        logger.error(f"❌ 서버 시작 중 오류: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")


# ✅ 서버 종료 이벤트
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("🔚 Whiff API 서버가 종료됩니다.")

    # 🆕 감정 모델 정리
    try:
        from utils.emotion_model_loader import reset_emotion_models
        reset_emotion_models()
        logger.info("🎭 감정 모델 리소스 정리 완료")
    except Exception as e:
        logger.warning(f"⚠️ 감정 모델 정리 중 오류: {e}")


# 🎯 모든 라우터 등록 (2차 추천 라우터 포함)
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

    # 🆕 2차 추천 라우터 추가
    from routers.recommend_2nd_router import router as recommend_2nd_router

    # 라우터 등록 (등록 순서 중요)
    app.include_router(perfume_router)  # 기본 향수 정보
    app.include_router(store_router)  # 매장 정보
    app.include_router(course_router)  # 시향 코스
    app.include_router(recommend_router)  # 1차 추천 (기존)
    app.include_router(recommend_2nd_router)  # 🆕 2차 추천 (노트 기반)
    app.include_router(diary_router)  # 시향 일기 (감정 분석 포함)
    app.include_router(auth_router)  # 인증
    app.include_router(user_router)  # 사용자 관리
    app.include_router(recommendation_save_router)  # 추천 저장

    logger.info("✅ 모든 라우터 등록 완료")
    logger.info("🆕 2차 추천 라우터 (/perfumes/recommend-2nd) 추가됨")
    logger.info("🎭 시향 일기에 감정 분석 기능 연동됨")

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
            "향수 추천 (2차 - 노트 기반)",  # 🆕 추가됨
            "시향 일기 (감정 분석 포함)",  # 🆕 업데이트됨
            "감정 태깅 (AI + 룰 기반)",  # 🆕 추가됨
            "매장 정보",
            "코스 추천",
            "사용자 인증",
            "회원 관리"
        ],
        "new_features": [
            "🆕 2차 추천 API (/perfumes/recommend-2nd)",
            "🎯 사용자 노트 선호도 기반 정밀 추천",
            "🧮 AI 감정 클러스터 + 노트 매칭 알고리즘",
            "🎭 시향 일기 자동 감정 분석",
            "🤖 감정 태깅 모델 (vectorizer + 분류기)"
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
                "2차 추천 (노트 기반)",  # 🆕
                "감정 분석",  # 🆕
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

        # 🆕 감정 모델 상태 확인
        emotion_model_status = None
        try:
            from utils.emotion_model_loader import get_emotion_models_status
            emotion_model_status = get_emotion_models_status()
        except Exception as e:
            logger.error(f"감정 모델 상태 확인 실패: {e}")

        # 🆕 감정 분석기 상태 확인
        emotion_analyzer_status = None
        try:
            from utils.emotion_analyzer import emotion_analyzer
            emotion_analyzer_status = emotion_analyzer.get_analysis_stats()
        except Exception as e:
            logger.error(f"감정 분석기 상태 확인 실패: {e}")

        return {
            "service": "Whiff API",
            "version": "1.3.0",
            "status": "running",
            "environment": "production" if os.getenv("RENDER") else "development",
            "firebase": firebase_status,
            "smtp": smtp_status,
            "emotion_models": emotion_model_status,  # 🆕 추가
            "emotion_analyzer": emotion_analyzer_status,  # 🆕 추가
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
                "perfumes_2nd": "🆕 2차 추천 (노트 기반)",  # 🆕 추가
                "stores": "매장 정보",
                "courses": "시향 코스 추천",
                "diaries": "시향 일기 (감정 분석 포함)",  # 🆕 업데이트
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
                "secondary_recommendation": {  # 🆕 추가
                    "endpoint": "/perfumes/recommend-2nd",
                    "method": "노트 매칭 + 감정 가중치",
                    "input": "노트 선호도 + 감정 확률 + 선택 인덱스",
                    "output": "정밀 점수 기반 추천"
                }
            },
            "emotion_analysis_system": {  # 🆕 추가
                "emotion_tagging": {
                    "method": "AI 모델 (vectorizer + 분류기) + 룰 기반 폴백",
                    "supported_emotions": emotion_analyzer_status.get("emotion_list",
                                                                      []) if emotion_analyzer_status else [],
                    "model_loaded": emotion_model_status.get("models_initialized",
                                                             False) if emotion_model_status else False
                },
                "diary_integration": {
                    "endpoint": "/diaries/",
                    "auto_analysis": True,
                    "confidence_threshold": 0.7
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
        "description": "AI 기반 향수 추천 및 감정 분석 서비스",  # 🆕 업데이트
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
                "title": "2차 추천 (🆕 신규)",
                "endpoint": "/perfumes/recommend-2nd",
                "description": "노트 선호도 + 1차 결과 → 정밀 점수 계산 → 최종 추천",
                "input_example": {
                    "user_preferences": {
                        "gender": "women",
                        "season_tags": "spring",
                        "time_tags": "day",
                        "desired_impression": "confident, fresh",
                        "activity": "casual",
                        "weather": "hot"
                    },
                    "user_note_scores": {
                        "jasmine": 5,
                        "rose": 4,
                        "amber": 3,
                        "musk": 0,
                        "citrus": 2,
                        "vanilla": 1
                    }
                }
            }
        },

        "emotion_analysis_flow": {  # 🆕 추가
            "step_1": {
                "title": "텍스트 감정 분석",
                "method": "AI 모델 (우선) + 룰 기반 (폴백)",
                "description": "시향 일기 텍스트 → 감정 클러스터 + 태그",
                "supported_emotions": ["기쁨", "불안", "당황", "분노", "상처", "슬픔", "우울", "흥분"]
            },
            "step_2": {
                "title": "자동 태깅",
                "endpoint": "/diaries/",
                "description": "일기 작성 시 자동으로 감정 분석 및 태깅",
                "confidence_scoring": "0.0 ~ 1.0 범위의 신뢰도 점수"
            }
        },

        "main_features": [
            "🤖 AI 감정 클러스터 기반 1차 추천",
            "🎯 노트 선호도 기반 2차 정밀 추천",
            "🎭 AI + 룰 기반 감정 분석",  # 🆕 추가
            "📝 시향 일기 자동 감정 태깅",  # 🆕 추가
            "🗺️ 위치 기반 시향 코스 추천",
            "🏪 매장 정보 및 검색",
            "🔐 Firebase 인증 시스템",
            "📧 이메일 발송 기능",
            "👥 사용자 관리 (회원가입/탈퇴)"
        ],

        "technical_stack": {
            "framework": "FastAPI",
            "ml_framework": "TensorFlow + scikit-learn",
            "emotion_analysis": "TF-IDF Vectorizer + 분류 모델",  # 🆕 추가
            "authentication": "Firebase Auth",
            "database": "SQLite + JSON Files",
            "deployment": "Render.com",
            "email": "SMTP (Gmail)"
        }
    }


# ✅ 🆕 감정 분석 상태 확인 API
@app.get("/emotion-status", summary="감정 분석 시스템 상태", operation_id="get_emotion_status")
def get_emotion_status():
    """감정 분석 시스템의 상태를 확인합니다."""
    try:
        # 감정 모델 상태
        emotion_model_status = None
        try:
            from utils.emotion_model_loader import get_emotion_models_status
            emotion_model_status = get_emotion_models_status()
        except Exception as e:
            logger.error(f"감정 모델 상태 확인 실패: {e}")

        # 감정 분석기 상태
        emotion_analyzer_status = None
        try:
            from utils.emotion_analyzer import emotion_analyzer
            emotion_analyzer_status = emotion_analyzer.get_analysis_stats()
        except Exception as e:
            logger.error(f"감정 분석기 상태 확인 실패: {e}")

        return {
            "service": "Whiff 감정 분석 시스템",
            "status": "operational",
            "models": emotion_model_status,
            "analyzer": emotion_analyzer_status,
            "integration": {
                "diary_auto_analysis": True,
                "api_endpoint": "/diaries/",
                "analysis_methods": ["AI 모델", "룰 기반"],
                "fallback_enabled": True
            }
        }

    except Exception as e:
        logger.error(f"감정 분석 상태 확인 실패: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )


# ✅ Render.com을 위한 메인 실행 부분
if __name__ == "__main__":
    import uvicorn

    # Render.com에서 제공하는 PORT 환경변수 사용 (중요!)
    port = int(os.getenv("PORT", 8000))

    logger.info(f"🚀 서버 시작: 포트 {port}")
    logger.info(f"🆕 감정 분석 + 2차 추천 기능이 포함된 Whiff API v1.3.0")

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,  # 프로덕션에서는 reload 비활성화
        access_log=True,
        log_level="info"
    )