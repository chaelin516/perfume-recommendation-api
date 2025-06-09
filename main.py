# main.py - 라우터 등록 및 감정 모델 연동 수정 완전판

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
    description="AI 기반 향수 추천 및 시향 코스 추천 서비스의 백엔드 API입니다.",
    version="1.3.0",  # 버전 업데이트
    docs_url="/docs",
    redoc_url="/redoc"
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
        port = os.getenv('PORT', '8000')
        environment = "production" if os.getenv("RENDER") else "development"

        logger.info(f"📋 기본 설정:")
        logger.info(f"  - 포트: {port}")
        logger.info(f"  - 환경: {environment}")
        logger.info(f"  - API 버전: 1.3.0")

        # 🔥 Firebase 초기화 확인
        firebase_status = {"firebase_available": False}
        try:
            from utils.auth_utils import get_firebase_status
            firebase_status = get_firebase_status()
            logger.info(f"🔥 Firebase: {'✅ 사용 가능' if firebase_status['firebase_available'] else '❌ 사용 불가'}")
        except Exception as e:
            logger.warning(f"⚠️ Firebase 상태 확인 실패: {e}")

        # 🎭 감정 분석기 초기화 (안전한 방식)
        emotion_analyzer_status = {"available": False, "method": "none"}
        try:
            logger.info("🎭 감정 분석 시스템 초기화...")

            # emotion_analyzer 모듈 임포트 시도
            try:
                from utils.emotion_analyzer import emotion_analyzer
                # 간단한 테스트 수행
                test_result = await emotion_analyzer.analyze_emotion("테스트 텍스트입니다.", use_model=False)
                if test_result and test_result.get("success"):
                    emotion_analyzer_status = {
                        "available": True,
                        "method": "AI + Rule-based",
                        "supported_emotions": emotion_analyzer.get_supported_emotions()
                    }
                    logger.info("✅ 감정 분석기 초기화 완료 (AI + 룰 기반)")
                else:
                    raise Exception("감정 분석기 테스트 실패")
            except ImportError as e:
                logger.warning(f"⚠️ emotion_analyzer 모듈 없음: {e}")
                emotion_analyzer_status = {"available": False, "method": "fallback_only"}
            except Exception as e:
                logger.warning(f"⚠️ 감정 분석기 초기화 실패: {e}")
                emotion_analyzer_status = {"available": False, "method": "fallback_only"}

        except Exception as e:
            logger.error(f"❌ 감정 분석 시스템 초기화 실패: {e}")
            emotion_analyzer_status = {"available": False, "method": "none"}

        # 📊 추천 모델 상태 확인 (lazy loading)
        recommendation_status = {"ai_model_available": False, "fallback_available": True}
        try:
            # 추천 모델 파일 존재 여부만 확인
            model_paths = [
                "./models/final_model.keras",
                "./models/encoder.pkl"
            ]

            files_exist = all(os.path.exists(path) for path in model_paths)
            if files_exist:
                recommendation_status["ai_model_available"] = True
                logger.info("🤖 추천 AI 모델 파일 확인됨 (Lazy Loading)")
            else:
                logger.info("📋 추천 AI 모델 없음, 룰 기반으로 동작")

        except Exception as e:
            logger.warning(f"⚠️ 추천 모델 상태 확인 실패: {e}")

        logger.info("📊 시스템 상태 요약:")
        logger.info(f"  - Firebase: {'✅' if firebase_status['firebase_available'] else '❌'}")
        logger.info(
            f"  - 감정 분석: {'✅' if emotion_analyzer_status['available'] else '❌'} ({emotion_analyzer_status['method']})")
        logger.info(f"  - 추천 AI: {'✅' if recommendation_status['ai_model_available'] else '❌'}")
        logger.info(f"  - 추천 룰: ✅")

        logger.info("✅ Whiff API 서버 시작 완료!")

    except Exception as e:
        logger.error(f"❌ 서버 시작 중 오류: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("🔚 Whiff API 서버가 종료됩니다.")


# main.py - 라우터 등록 개선 버전

def register_routers():
    """라우터를 안전하게 등록 (각 라우터 독립적으로 처리)"""
    try:
        logger.info("📋 라우터 등록 시작...")

        # 📊 등록 성공/실패 추적
        router_status = {}

        # 1. 기본 라우터들 (필수)
        essential_routers = [
            ("perfume_router", "기본 향수 정보", "routers.perfume_router"),
            ("store_router", "매장 정보", "routers.store_router"),
            ("auth_router", "사용자 인증", "routers.auth_router"),
            ("user_router", "사용자 관리", "routers.user_router")
        ]

        for router_name, description, module_path in essential_routers:
            try:
                module = __import__(module_path, fromlist=['router'])
                app.include_router(module.router)
                router_status[router_name] = "✅ 성공"
                logger.info(f"  ✅ {description} 라우터 등록 완료")
            except Exception as e:
                router_status[router_name] = f"❌ 실패: {str(e)}"
                logger.error(f"  ❌ {description} 라우터 등록 실패: {e}")

        # 2. 추천 시스템 라우터들
        recommendation_routers = [
            ("course_router", "시향 코스 추천", "routers.course_router"),
            ("recommend_router", "1차 향수 추천", "routers.recommend_router"),
            ("recommend_2nd_router", "2차 향수 추천", "routers.recommend_2nd_router"),
            ("recommendation_save_router", "추천 결과 저장", "routers.recommendation_save_router")
        ]

        for router_name, description, module_path in recommendation_routers:
            try:
                module = __import__(module_path, fromlist=['router'])
                app.include_router(module.router)
                router_status[router_name] = "✅ 성공"
                logger.info(f"  ✅ {description} 라우터 등록 완료")
            except Exception as e:
                router_status[router_name] = f"❌ 실패: {str(e)}"
                logger.error(f"  ❌ {description} 라우터 등록 실패: {e}")

        # 3. 🎭 시향 일기 라우터 (특별 처리 - 감정 분석 의존성 있음)
        try:
            logger.info("🎭 시향 일기 라우터 등록 시도...")

            # diary_router 임포트 시도
            from routers.diary_router import router as diary_router
            app.include_router(diary_router)

            router_status["diary_router"] = "✅ 성공"
            logger.info("  ✅ 시향 일기 라우터 등록 완료 (감정 분석 포함)")

            # 감정 분석기 상태 확인
            try:
                from routers.diary_router import EMOTION_ANALYZER_AVAILABLE
                if EMOTION_ANALYZER_AVAILABLE:
                    logger.info("    ✅ AI 감정 분석기 사용 가능")
                else:
                    logger.info("    ⚠️ 기본 감정 분석 모드로 동작")
            except ImportError:
                logger.info("    ⚠️ 감정 분석 상태 확인 불가")

        except ImportError as e:
            router_status["diary_router"] = f"❌ ImportError: {str(e)}"
            logger.error(f"  ❌ 시향 일기 라우터 임포트 실패: {e}")
            logger.error("    💡 emotion_analyzer 모듈 관련 문제일 가능성이 높습니다")

        except Exception as e:
            router_status["diary_router"] = f"❌ Exception: {str(e)}"
            logger.error(f"  ❌ 시향 일기 라우터 등록 실패: {e}")

        # 4. 기타 라우터들 (선택적)
        optional_routers = [
            ("emotion_router", "감정 분석 전용", "routers.emotion_router"),
            ("emotion_tagging_router", "감정 태깅", "routers.emotion_tagging_router")
        ]

        for router_name, description, module_path in optional_routers:
            try:
                module = __import__(module_path, fromlist=['router'])
                app.include_router(module.router)
                router_status[router_name] = "✅ 성공"
                logger.info(f"  ✅ {description} 라우터 등록 완료")
            except ImportError:
                router_status[router_name] = "⚠️ 모듈 없음 (선택적)"
                logger.info(f"  ⚠️ {description} 라우터 없음 (선택적 기능)")
            except Exception as e:
                router_status[router_name] = f"❌ 실패: {str(e)}"
                logger.warning(f"  ❌ {description} 라우터 등록 실패: {e}")

        logger.info("✅ 라우터 등록 완료")

        # 📊 등록 결과 요약
        success_count = sum(1 for status in router_status.values() if "✅" in status)
        total_count = len(router_status)

        logger.info(f"📊 라우터 등록 결과: {success_count}/{total_count} 성공")

        for router_name, status in router_status.items():
            logger.info(f"  - {router_name}: {status}")

        # 등록된 라우트 확인
        registered_routes = [route.path for route in app.routes if hasattr(route, 'path')]
        logger.info(f"📋 등록된 총 라우트 수: {len(registered_routes)}")

        # 주요 엔드포인트 확인
        key_endpoints = [
            "/perfumes/recommend-cluster",
            "/perfumes/recommend-2nd",
            "/diaries/",
            "/diaries/emotion-status",
            "/auth/register",
            "/stores/",
            "/courses/recommend"
        ]

        logger.info("🎯 주요 엔드포인트 확인:")
        for endpoint in key_endpoints:
            if any(endpoint in route for route in registered_routes):
                logger.info(f"  ✅ {endpoint}")
            else:
                logger.warning(f"  ❌ {endpoint} - 누락됨")

        # 🎭 시향 일기 API 특별 확인
        diary_endpoints = [ep for ep in registered_routes if "/diaries" in ep]
        if diary_endpoints:
            logger.info(f"🎭 시향 일기 API 엔드포인트 ({len(diary_endpoints)}개):")
            for endpoint in diary_endpoints[:5]:  # 처음 5개만 표시
                logger.info(f"  ✅ {endpoint}")
            if len(diary_endpoints) > 5:
                logger.info(f"  ... 외 {len(diary_endpoints) - 5}개")
        else:
            logger.error("❌ 시향 일기 API 엔드포인트가 전혀 등록되지 않았습니다!")
            logger.error("💡 diary_router.py의 의존성 문제를 확인해주세요")

    except Exception as e:
        logger.error(f"❌ 라우터 등록 중 치명적 오류: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")


# main.py의 기존 register_routers() 함수를 위 코드로 교체

# 라우터 등록 실행
register_routers()


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
            "향수 추천 (1차 - AI 감정 클러스터)",
            "향수 추천 (2차 - 노트 기반 정밀 추천)",
            "시향 일기 (AI 감정 분석 포함)",
            "매장 정보 및 위치 기반 검색",
            "코스 추천",
            "사용자 인증 (Firebase)",
            "회원 관리 (가입/탈퇴)"
        ],
        "new_features_v1_3": [
            "🎭 AI 감정 분석 자동 태깅",
            "🎯 노트 선호도 기반 2차 정밀 추천",
            "🔄 안전한 폴백 메커니즘",
            "📊 실시간 감정 통계 분석"
        ],
        "docs_url": "/docs",
        "redoc_url": "/redoc"
    }


@app.head("/", operation_id="head_root")
def head_root():
    return JSONResponse(content={})


# ✅ 헬스 체크
@app.get("/health", summary="헬스 체크", operation_id="get_health_check")
def health_check():
    try:
        return {
            "status": "ok",
            "service": "Whiff API",
            "version": "1.3.0",
            "environment": "production" if os.getenv("RENDER") else "development",
            "port": os.getenv("PORT", "8000"),
            "uptime": "running",
            "features_available": [
                "1차 추천 (AI 감정 클러스터)",
                "2차 추천 (노트 기반 정밀)",
                "시향 일기 (AI 감정 분석)",
                "매장 정보",
                "사용자 인증",
                "실시간 통계"
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

        # 감정 분석 상태 확인
        emotion_status = None
        try:
            from utils.emotion_analyzer import emotion_analyzer
            emotion_status = {
                "available": True,
                "supported_emotions": emotion_analyzer.get_supported_emotions(),
                "stats": emotion_analyzer.get_analysis_stats()
            }
        except Exception as e:
            emotion_status = {"available": False, "error": str(e)}

        return {
            "service": "Whiff API",
            "version": "1.3.0",
            "status": "running",
            "environment": "production" if os.getenv("RENDER") else "development",
            "firebase": firebase_status,
            "smtp": smtp_status,
            "emotion_analysis": emotion_status,
            "features": {
                "auth": "Firebase Authentication",
                "database": "SQLite + JSON Files",
                "ml_model": "TensorFlow (Lazy Loading)",
                "emotion_ai": "Custom Emotion Analyzer",
                "deployment": "Render.com",
                "email": "SMTP (Gmail)"
            },
            "endpoints": {
                "perfumes": "향수 정보 및 1차 추천",
                "perfumes_2nd": "2차 추천 (노트 기반)",
                "perfumes_cluster": "클러스터 기반 추천",
                "diaries": "시향 일기 (감정 분석 포함)",
                "stores": "매장 정보",
                "courses": "시향 코스 추천",
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
                }
            },
            "emotion_analysis_system": {
                "auto_tagging": {
                    "endpoint": "/diaries/",
                    "method": "AI + 룰 기반 하이브리드",
                    "input": "시향 일기 텍스트",
                    "output": "감정 태그 + 신뢰도"
                },
                "statistics": {
                    "endpoint": "/diaries/emotion-statistics",
                    "method": "실시간 통계 분석",
                    "features": ["감정 분포", "신뢰도 추적", "트렌드 분석"]
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
        "description": "AI 기반 향수 추천 및 시향 코스 추천 서비스",
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
                "title": "2차 추천 (정밀)",
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

        "emotion_analysis_flow": {
            "auto_tagging": {
                "title": "자동 감정 태깅",
                "endpoint": "/diaries/",
                "description": "시향 일기 작성 시 자동으로 감정 분석 및 태깅",
                "features": [
                    "AI 모델 우선 분석",
                    "룰 기반 폴백",
                    "백그라운드 정밀 분석",
                    "수동 태그 수정 지원"
                ]
            },
            "statistics": {
                "title": "감정 통계 분석",
                "endpoint": "/diaries/emotion-statistics",
                "description": "전체 일기의 감정 분포 및 트렌드 분석",
                "features": [
                    "실시간 감정 분포",
                    "신뢰도 통계",
                    "인기 감정 순위",
                    "분석 성공률"
                ]
            }
        },

        "main_features": [
            "🤖 AI 감정 클러스터 기반 1차 추천",
            "🎯 노트 선호도 기반 2차 정밀 추천",
            "🎭 AI 자동 감정 태깅 (시향 일기)",
            "📊 실시간 감정 통계 및 트렌드 분석",
            "🗺️ 위치 기반 시향 코스 추천",
            "🏪 매장 정보 및 검색",
            "🔐 Firebase 인증 시스템",
            "📧 이메일 발송 기능",
            "👥 사용자 관리 (회원가입/탈퇴)"
        ],

        "technical_stack": {
            "framework": "FastAPI",
            "ml_framework": "TensorFlow + scikit-learn",
            "emotion_ai": "Custom Emotion Analyzer",
            "authentication": "Firebase Auth",
            "database": "SQLite + JSON Files",
            "deployment": "Render.com",
            "email": "SMTP (Gmail)"
        }
    }


# ✅ Render.com을 위한 메인 실행 부분
if __name__ == "__main__":
    import uvicorn

    # Render.com에서 제공하는 PORT 환경변수 사용
    port = int(os.getenv("PORT", 8000))

    logger.info(f"🚀 서버 시작: 포트 {port}")
    logger.info(f"🎭 감정 분석 포함 Whiff API v1.3.0")

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        access_log=True,
        log_level="info"
    )