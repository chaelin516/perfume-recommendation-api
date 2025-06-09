# main.py - Whiff API Server (API 삭제 반영 버전)

import os
import logging
import traceback
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ─── 로깅 설정 ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("whiff_main")

# ─── FastAPI 앱 생성 ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Whiff API",
    description="""
    🌸 **Whiff - 취향 맞춤 향수 추천 서비스**

    고객의 취향에 맞는 향수를 AI 기반으로 추천해주는 서비스입니다.

    ## 🎯 주요 기능
    - **1차 추천**: AI 감정 클러스터 모델 기반 향수 추천
    - **2차 추천**: 노트 선호도 기반 정밀 추천  
    - **시향 일기**: AI 감정 분석 포함 일기 작성
    - **사용자 인증**: Firebase 기반 회원 관리

    ## 🚀 기술 스택
    - **Backend**: FastAPI + Python
    - **AI/ML**: TensorFlow + Custom Emotion Analyzer
    - **Database**: SQLite + JSON Files
    - **Authentication**: Firebase
    - **Deployment**: Render.com

    ## 📋 API 버전 정보
    - **Version**: 1.3.0
    - **Environment**: Production
    - **Last Updated**: 2025-06-10
    """,
    version="1.3.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ─── CORS 설정 ────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React 개발 서버
        "http://localhost:8000",  # FastAPI 개발 서버
        "https://whiff-api-9nd8.onrender.com",  # 프로덕션 API
        "*"  # 개발 단계에서는 모든 origin 허용
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
)


# ─── 서버 시작/종료 이벤트 ─────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    """서버 시작 시 초기화"""
    try:
        logger.info("🚀 Whiff API 서버 시작 중...")
        logger.info(f"📍 Environment: {'Production' if os.getenv('RENDER') else 'Development'}")
        logger.info(f"📍 Port: {os.getenv('PORT', '8000')}")

        # 📊 Firebase 상태 확인
        firebase_status = {"firebase_available": False, "error": None}
        try:
            from utils.auth_utils import get_firebase_status
            firebase_status = get_firebase_status()
            logger.info(f"🔥 Firebase 상태: {'✅ 연결됨' if firebase_status['firebase_available'] else '❌ 연결 실패'}")
        except Exception as e:
            logger.error(f"Firebase 상태 확인 실패: {e}")
            firebase_status = {"firebase_available": False, "error": str(e)}

        # 🎭 감정 분석 시스템 상태 확인 (lazy loading)
        emotion_analyzer_status = {"available": False, "method": "none"}
        try:
            logger.info("🎭 감정 분석 시스템 확인 중...")
            try:
                from emotion.emotion_analyzer import EmotionAnalyzer
                emotion_analyzer = EmotionAnalyzer()

                # 간단한 테스트로 초기화 확인
                test_result = emotion_analyzer.analyze_emotion("테스트 텍스트", use_model=False)
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


# ─── 라우터 등록 함수 (API 삭제 반영) ────────────────────────────────────────────
def register_routers():
    """라우터를 안전하게 등록 (삭제된 API 반영)"""
    try:
        logger.info("📋 라우터 등록 시작...")

        # 📊 등록 성공/실패 추적
        router_status = {}

        # 1. 기본 라우터들 (필수) - store_router 제거됨
        essential_routers = [
            ("perfume_router", "기본 향수 정보", "routers.perfume_router"),
            # ("store_router", "매장 정보", "routers.store_router"),  # ✅ 삭제됨
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

        # 2. 추천 시스템 라우터들 - course_router 제거됨
        recommendation_routers = [
            # ("course_router", "시향 코스 추천", "routers.course_router"),  # ✅ 삭제됨
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

        # 3. 🎭 시향 일기 라우터 (특별 처리)
        # ✅ 옵션 1: 전체 diary_router 유지 (개별 API만 삭제)
        try:
            logger.info("🎭 시향 일기 라우터 등록 시도...")

            from routers.diary_router import router as diary_router
            app.include_router(diary_router)

            router_status["diary_router"] = "✅ 성공"
            logger.info("  ✅ 시향 일기 라우터 등록 완료")
            logger.info("  📝 주의: diary_router.py에서 개별 API 함수 삭제 필요:")
            logger.info("    - get_diary_detail() 함수 삭제 (/diaries/{diary_id})")
            logger.info("    - get_emotion_stats() 함수 삭제 (/diaries/stats/emotions)")

        except ImportError as e:
            router_status["diary_router"] = f"❌ ImportError: {str(e)}"
            logger.error(f"  ❌ 시향 일기 라우터 임포트 실패: {e}")
            logger.error("    💡 emotion_analyzer 모듈 관련 문제일 가능성이 높습니다")

        except Exception as e:
            router_status["diary_router"] = f"❌ Exception: {str(e)}"
            logger.error(f"  ❌ 시향 일기 라우터 등록 실패: {e}")

        # ✅ 옵션 2: 전체 diary_router 비활성화 (아래 주석 해제 시 사용)
        # router_status["diary_router"] = "⚠️ 수동 비활성화"
        # logger.info("  ⚠️ 시향 일기 라우터 수동 비활성화됨")

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
            "/auth/register"
        ]

        logger.info("🎯 주요 엔드포인트 확인:")
        for endpoint in key_endpoints:
            if any(endpoint in route for route in registered_routes):
                logger.info(f"  ✅ {endpoint}")
            else:
                logger.warning(f"  ❌ {endpoint} - 누락됨")

        # 🗑️ 삭제된 엔드포인트 확인
        deleted_endpoints = [
            "/courses/recommend",
            "/stores/",
            "/stores/{brand}"
        ]

        logger.info("🗑️ 삭제된 엔드포인트 확인:")
        for endpoint in deleted_endpoints:
            if any(endpoint in route for route in registered_routes):
                logger.warning(f"  ⚠️ {endpoint} - 아직 존재함 (추가 삭제 필요)")
            else:
                logger.info(f"  ✅ {endpoint} - 성공적으로 삭제됨")

        # 🎭 시향 일기 API 특별 확인
        diary_endpoints = [ep for ep in registered_routes if "/diaries" in ep]
        if diary_endpoints:
            logger.info(f"🎭 시향 일기 API 엔드포인트 ({len(diary_endpoints)}개):")
            for endpoint in diary_endpoints[:5]:  # 처음 5개만 표시
                logger.info(f"  📝 {endpoint}")
            if len(diary_endpoints) > 5:
                logger.info(f"  ... 외 {len(diary_endpoints) - 5}개")

            # 삭제되어야 할 diary 엔드포인트 확인
            should_be_deleted = [ep for ep in diary_endpoints
                                 if "/{diary_id}" in ep or "/stats/emotions" in ep]
            if should_be_deleted:
                logger.warning("  ⚠️ 다음 diary 엔드포인트들이 아직 존재합니다:")
                for ep in should_be_deleted:
                    logger.warning(f"    🗑️ {ep} - diary_router.py에서 수동 삭제 필요")
        else:
            logger.info("🎭 시향 일기 API 엔드포인트가 등록되지 않음")

    except Exception as e:
        logger.error(f"❌ 라우터 등록 중 치명적 오류: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")


# 라우터 등록 실행
register_routers()


# ─── 기본 엔드포인트들 ────────────────────────────────────────────────────────────
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
            "사용자 인증 (Firebase)",
            "회원 관리 (가입/탈퇴)"
        ],
        "deleted_apis": [
            "❌ /courses/recommend (시향 코스 추천)",
            "❌ /stores/ (전체 매장 목록)",
            "❌ /stores/{brand} (브랜드별 매장)",
            "❌ /diaries/{diary_id} (특정 일기 조회)",
            "❌ /diaries/stats/emotions (감정 통계)"
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
                "사용자 인증",
                "실시간 통계"
            ],
            "deleted_features": [
                "시향 코스 추천",
                "매장 정보 조회",
                "특정 일기 상세 조회",
                "감정 통계 조회"
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
            "version": "1.3.0",
            "status": "running",
            "environment": "production" if os.getenv("RENDER") else "development",
            "firebase": firebase_status,
            "smtp": smtp_status,
            "features": {
                "auth": "Firebase Authentication",
                "database": "SQLite + JSON Files",
                "ml_model": "TensorFlow (Lazy Loading)",
                "emotion_ai": "Custom Emotion Analyzer",
                "deployment": "Render.com",
                "email": "SMTP (Gmail)"
            },
            "active_endpoints": {
                "perfumes": "향수 정보 및 1차 추천",
                "perfumes_2nd": "2차 추천 (노트 기반)",
                "perfumes_cluster": "클러스터 기반 추천",
                "diaries": "시향 일기 (일부 기능)",
                "auth": "사용자 인증",
                "users": "사용자 관리"
            },
            "deleted_endpoints": {
                "courses": "시향 코스 추천 (완전 삭제)",
                "stores": "매장 정보 (완전 삭제)",
                "diary_detail": "특정 일기 조회 (개별 삭제)",
                "emotion_stats": "감정 통계 (개별 삭제)"
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
                    "method": "노트 기반 정밀 매칭",
                    "input": "노트 선호도 + 1차 추천 결과",
                    "output": "정밀 점수 기반 향수 순위"
                }
            }
        }

    except Exception as e:
        logger.error(f"Status check failed: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )


@app.head("/status", operation_id="head_server_status")
def head_server_status():
    return JSONResponse(content={})


# ─── 개발 환경에서만 실행 ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    logger.info(f"🚀 개발 서버 시작: http://localhost:{port}")
    logger.info("📚 API 문서: http://localhost:{port}/docs")

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )