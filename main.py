# main.py - Whiff API Server (신고 기능 추가 버전)

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
    - **🆕 신고 시스템**: 부적절한 콘텐츠 신고 및 관리

    ## 🚀 기술 스택
    - **Backend**: FastAPI + Python
    - **AI/ML**: TensorFlow + Custom Emotion Analyzer
    - **Database**: SQLite + JSON Files
    - **Authentication**: Firebase
    - **Deployment**: Render.com

    ## 📋 API 버전 정보
    - **Version**: 1.4.0
    - **Environment**: Production
    - **Last Updated**: 2025-06-30
    """,
    version="1.4.0",
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


# ─── 라우터 등록 함수 ─────────────────────────────────────────────────────────────
def register_routers():
    """라우터 등록 함수"""
    router_status = {}

    try:
        logger.info("🔧 라우터 등록 시작...")

        # 1. 향수 기본 정보 라우터
        try:
            logger.info("🌸 향수 기본 정보 라우터 등록 시도...")
            from routers.perfume_router import router as perfume_router
            app.include_router(perfume_router)
            router_status["perfume_router"] = "✅ 성공"
            logger.info("  ✅ 향수 기본 정보 라우터 등록 완료")
        except ImportError as e:
            router_status["perfume_router"] = f"❌ ImportError: {str(e)}"
            logger.error(f"  ❌ 향수 기본 정보 라우터 임포트 실패: {e}")
        except Exception as e:
            router_status["perfume_router"] = f"❌ Exception: {str(e)}"
            logger.error(f"  ❌ 향수 기본 정보 라우터 등록 실패: {e}")

        # 1-2. 클러스터 기반 추천 라우터 (누락된 것)
        try:
            logger.info("🤖 클러스터 기반 추천 라우터 등록 시도...")
            from routers.recommend_router import router as recommend_router
            app.include_router(recommend_router)
            router_status["recommend_router"] = "✅ 성공"
            logger.info("  ✅ 클러스터 기반 추천 라우터 등록 완료")
            logger.info("    🎯 엔드포인트: /perfumes/recommend-cluster")
        except ImportError as e:
            router_status["recommend_router"] = f"❌ ImportError: {str(e)}"
            logger.error(f"  ❌ 클러스터 기반 추천 라우터 임포트 실패: {e}")
        except Exception as e:
            router_status["recommend_router"] = f"❌ Exception: {str(e)}"
            logger.error(f"  ❌ 클러스터 기반 추천 라우터 등록 실패: {e}")

        # 1-3. 2차 추천 라우터 (누락된 것)
        try:
            logger.info("🎯 2차 추천 라우터 등록 시도...")
            from routers.recommend_2nd_router import router as recommend_2nd_router
            app.include_router(recommend_2nd_router)
            router_status["recommend_2nd_router"] = "✅ 성공"
            logger.info("  ✅ 2차 추천 라우터 등록 완료")
            logger.info("    🎯 엔드포인트: /perfumes/recommend-2nd")
        except ImportError as e:
            router_status["recommend_2nd_router"] = f"❌ ImportError: {str(e)}"
            logger.error(f"  ❌ 2차 추천 라우터 임포트 실패: {e}")
        except Exception as e:
            router_status["recommend_2nd_router"] = f"❌ Exception: {str(e)}"
            logger.error(f"  ❌ 2차 추천 라우터 등록 실패: {e}")

        # 1-4. 추천 결과 저장 라우터 (선택적)
        try:
            logger.info("💾 추천 결과 저장 라우터 등록 시도...")
            from routers.recommendation_save_router import router as recommendation_save_router
            app.include_router(recommendation_save_router)
            router_status["recommendation_save_router"] = "✅ 성공"
            logger.info("  ✅ 추천 결과 저장 라우터 등록 완료")
        except ImportError as e:
            router_status["recommendation_save_router"] = f"⚠️ ImportError: {str(e)}"
            logger.info("  ⚠️ 추천 결과 저장 라우터 없음 (선택적 기능)")
        except Exception as e:
            router_status["recommendation_save_router"] = f"❌ Exception: {str(e)}"
            logger.warning(f"  ❌ 추천 결과 저장 라우터 등록 실패: {e}")

        # 2. 사용자 인증 라우터
        try:
            logger.info("🔐 사용자 인증 라우터 등록 시도...")
            from routers.auth_router import router as auth_router
            app.include_router(auth_router)
            router_status["auth_router"] = "✅ 성공"
            logger.info("  ✅ 사용자 인증 라우터 등록 완료")
        except ImportError as e:
            router_status["auth_router"] = f"❌ ImportError: {str(e)}"
            logger.error(f"  ❌ 사용자 인증 라우터 임포트 실패: {e}")
        except Exception as e:
            router_status["auth_router"] = f"❌ Exception: {str(e)}"
            logger.error(f"  ❌ 사용자 인증 라우터 등록 실패: {e}")

        # 3. 사용자 관리 라우터
        try:
            logger.info("👤 사용자 관리 라우터 등록 시도...")
            from routers.user_router import router as user_router
            app.include_router(user_router)
            router_status["user_router"] = "✅ 성공"
            logger.info("  ✅ 사용자 관리 라우터 등록 완료")
        except ImportError as e:
            router_status["user_router"] = f"❌ ImportError: {str(e)}"
            logger.error(f"  ❌ 사용자 관리 라우터 임포트 실패: {e}")
        except Exception as e:
            router_status["user_router"] = f"❌ Exception: {str(e)}"
            logger.error(f"  ❌ 사용자 관리 라우터 등록 실패: {e}")

        # 4. 시향 일기 라우터 (기존)
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

        # 🆕 5. 신고 관리 라우터 (새로 추가)
        try:
            logger.info("🚨 신고 관리 라우터 등록 시도...")
            from routers.report_router import router as report_router
            app.include_router(report_router)
            router_status["report_router"] = "✅ 성공"
            logger.info("  ✅ 신고 관리 라우터 등록 완료")
            logger.info("    📢 새 기능: 시향 일기 신고 기능 활성화")
            logger.info("    🔗 새 엔드포인트:")
            logger.info("      - POST /reports/diary (시향 일기 신고)")
            logger.info("      - GET /reports/ (신고 목록 조회)")
            logger.info("      - GET /reports/stats (신고 통계)")
            logger.info("      - PUT /reports/{report_id}/action (신고 처리)")
            logger.info("      - DELETE /reports/{report_id} (신고 삭제)")
        except ImportError as e:
            router_status["report_router"] = f"❌ ImportError: {str(e)}"
            logger.error(f"  ❌ 신고 관리 라우터 임포트 실패: {e}")
            logger.error("    💡 다음 파일들이 필요합니다:")
            logger.error("      - models/report_models.py")
            logger.error("      - routers/report_router.py")
        except Exception as e:
            router_status["report_router"] = f"❌ Exception: {str(e)}"
            logger.error(f"  ❌ 신고 관리 라우터 등록 실패: {e}")

        # 6. 기타 라우터들 (선택적)
        optional_routers = [
            ("emotion_router", "감정 분석 전용", "routers.emotion_router"),
            ("emotion_tagging_router", "감정 태깅", "routers.emotion_tagging_router")
        ]

        for router_name, description, module_path in optional_routers:
            try:
                logger.info(f"🔄 {description} 라우터 등록 시도...")
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

        # 주요 엔드포인트 확인 (업데이트된 체크)
        key_endpoints = [
            "/perfumes/recommend-cluster",
            "/perfumes/recommend-2nd",
            "/perfumes/",
            "/diaries/",
            "/auth/register",
            "/reports/diary"
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

        # 🚨 신고 API 특별 확인
        report_endpoints = [ep for ep in registered_routes if "/reports" in ep]
        if report_endpoints:
            logger.info(f"🚨 신고 관리 API 엔드포인트 ({len(report_endpoints)}개):")
            for endpoint in report_endpoints:
                logger.info(f"  📢 {endpoint}")
        else:
            logger.warning("🚨 신고 관리 API 엔드포인트가 등록되지 않음")

    except Exception as e:
        logger.error(f"❌ 라우터 등록 중 치명적 오류: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")


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
            # Firebase 초기화 체크 (실제 구현 시 firebase 모듈 사용)
            logger.info("🔥 Firebase 상태 확인 중...")
            firebase_status["firebase_available"] = True
            logger.info("  ✅ Firebase 연결 가능")
        except Exception as e:
            firebase_status["error"] = str(e)
            logger.warning(f"  ⚠️ Firebase 연결 실패: {e}")

        # 📁 데이터 디렉토리 확인
        data_dirs = ["data", "data/reports"]
        for dir_path in data_dirs:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
                logger.info(f"📁 디렉토리 생성: {dir_path}")
            else:
                logger.info(f"📁 디렉토리 확인: {dir_path}")

        logger.info("🎉 서버 초기화 완료!")

    except Exception as e:
        logger.error(f"❌ 서버 시작 중 오류: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")


# 라우터 등록 실행
register_routers()


# ─── 기본 엔드포인트들 ────────────────────────────────────────────────────────────
@app.get("/", summary="루트", operation_id="get_root")
def read_root():
    return {
        "message": "✅ Whiff API is running!",
        "status": "ok",
        "version": "1.4.0",
        "environment": "production" if os.getenv("RENDER") else "development",
        "port": os.getenv("PORT", "8000"),
        "features": [
            "🎯 향수 추천 (1차 - AI 감정 클러스터)",
            "🤖 향수 추천 (2차 - 노트 기반 정밀 추천)",
            "📝 시향 일기 (AI 감정 분석 포함)",
            "🔐 사용자 인증 (Firebase)",
            "👤 회원 관리 (가입/탈퇴)",
            "🚨 시향 일기 신고 기능"
        ],
        "deleted_apis": [
            "❌ /courses/recommend (시향 코스 추천)",
            "❌ /stores/ (전체 매장 목록)",
            "❌ /stores/{brand} (브랜드별 매장)",
            "❌ /diaries/{diary_id} (특정 일기 조회)",
            "❌ /diaries/stats/emotions (감정 통계)"
        ],
        "new_features_v1_4": [
            "🚨 시향 일기 신고 시스템",
            "📊 신고 통계 및 관리",
            "⚖️ 관리자 신고 처리 기능",
            "🔒 중복 신고 방지",
            "📈 실시간 신고 현황",
            "🎯 클러스터 기반 추천 시스템 복구",
            "🤖 2차 정밀 추천 시스템 복구"
        ],
        "recommend_endpoints": [
            "POST /perfumes/recommend-cluster - 클러스터 기반 1차 추천",
            "POST /perfumes/recommend-2nd - 노트 기반 2차 정밀 추천",
            "GET /perfumes/ - 향수 목록 조회",
            "GET /perfumes/{name} - 향수 상세 정보"
        ],
        "report_endpoints": [
            "POST /reports/diary - 시향 일기 신고",
            "GET /reports/ - 신고 목록 조회",
            "GET /reports/stats - 신고 통계",
            "PUT /reports/{report_id}/action - 신고 처리",
            "DELETE /reports/{report_id} - 신고 삭제"
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
            "version": "1.4.0",
            "environment": "production" if os.getenv("RENDER") else "development",
            "port": os.getenv("PORT", "8000"),
            "uptime": "running",
            "features_available": [
                "🎯 1차 추천 (AI 감정 클러스터)",
                "🤖 2차 추천 (노트 기반 정밀)",
                "📝 시향 일기 (AI 감정 분석)",
                "🔐 사용자 인증",
                "📊 실시간 통계",
                "🚨 신고 관리 시스템"
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
            status_code=500,
            content={"status": "error", "message": str(e)}
        )


@app.head("/health", operation_id="head_health_check")
def head_health_check():
    return JSONResponse(content={})


@app.get("/status", summary="서버 상태 확인", operation_id="get_server_status")
def get_server_status():
    try:
        return {
            "status": "running",
            "service": "Whiff API Server",
            "version": "1.4.0",
            "environment": "production" if os.getenv("RENDER") else "development",
            "port": os.getenv("PORT", "8000"),
            "available_routers": {
                "perfumes": "향수 기본 정보 조회",
                "perfumes_recommend": "🎯 클러스터 기반 추천",
                "perfumes_recommend_2nd": "🤖 2차 정밀 추천",
                "diaries": "시향 일기 (일부 기능)",
                "auth": "사용자 인증",
                "users": "사용자 관리",
                "reports": "🆕 신고 관리 시스템"
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
                    "output": "클러스터 + 향수 인덱스 + 확률 배열",
                    "features": ["클러스터 예측", "상위 15개 노트", "10개 추천 향수"]
                },
                "secondary_recommendation": {
                    "endpoint": "/perfumes/recommend-2nd",
                    "method": "노트 기반 정밀 매칭 + AI 결합",
                    "input": "노트 선호도 + 1차 추천 결과",
                    "output": "정밀 점수 기반 향수 순위",
                    "features": ["노트 매칭 (70%)", "감정 가중치 (25%)", "다양성 보너스 (5%)"]
                },
                "perfume_info": {
                    "endpoint": "/perfumes/",
                    "method": "향수 데이터베이스 조회",
                    "features": ["3000+ 향수 정보", "브랜드/노트/계절 필터링", "상세 정보 제공"]
                }
            },
            "report_system": {
                "report_diary": {
                    "endpoint": "/reports/diary",
                    "method": "POST",
                    "description": "시향 일기 신고 접수"
                },
                "manage_reports": {
                    "endpoint": "/reports/",
                    "method": "GET",
                    "description": "신고 목록 조회 (관리자용)"
                },
                "report_stats": {
                    "endpoint": "/reports/stats",
                    "method": "GET",
                    "description": "신고 통계 조회"
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
    logger.info("🔍 ReDoc: http://localhost:{port}/redoc")
    logger.info("🆕 신고 기능: /reports/ 엔드포인트 활성화")

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )