# main.py - 이미지 업로드 및 서빙 기능 추가된 버전

import os
import logging
import traceback
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles  # 🆕 정적 파일 서빙용
from fastapi import HTTPException

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
    - **📸 이미지 업로드**: 시향 일기에 사진 첨부 기능
    - **사용자 인증**: Firebase 기반 회원 관리

    ## 🚀 기술 스택
    - **Backend**: FastAPI + Python
    - **AI/ML**: TensorFlow + Custom Emotion Analyzer
    - **Database**: SQLite + JSON Files
    - **Image Processing**: Pillow (PIL)
    - **Authentication**: Firebase
    - **Deployment**: Render.com

    ## 📋 API 버전 정보
    - **Version**: 1.4.0
    - **Environment**: Production
    - **Last Updated**: 2025-06-10
    - **New Features**: 이미지 업로드 및 처리 기능 추가
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


# ─── 🆕 업로드 디렉토리 설정 및 정적 파일 서빙 ──────────────────────────────────────
def setup_upload_directories():
    """업로드 디렉토리 생성 및 정적 파일 마운트"""
    try:
        # 업로드 디렉토리 경로
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
        DIARY_IMAGES_DIR = os.path.join(UPLOAD_DIR, "diary_images")
        THUMBNAILS_DIR = os.path.join(DIARY_IMAGES_DIR, "thumbnails")

        # 디렉토리 생성
        os.makedirs(DIARY_IMAGES_DIR, exist_ok=True)
        os.makedirs(THUMBNAILS_DIR, exist_ok=True)

        logger.info(f"✅ 업로드 디렉토리 생성: {UPLOAD_DIR}")

        # 정적 파일 마운트 (업로드된 이미지들을 웹에서 접근 가능하게)
        if os.path.exists(UPLOAD_DIR):
            app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
            logger.info(f"📁 정적 파일 마운트: /uploads -> {UPLOAD_DIR}")
        else:
            logger.warning(f"⚠️ 업로드 디렉토리가 존재하지 않습니다: {UPLOAD_DIR}")

        return True

    except Exception as e:
        logger.error(f"❌ 업로드 디렉토리 설정 실패: {e}")
        return False


# ─── 서버 시작/종료 이벤트 ─────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    """서버 시작 시 초기화"""
    try:
        logger.info("🚀 Whiff API 서버 시작 중...")
        logger.info(f"📍 Environment: {'Production' if os.getenv('RENDER') else 'Development'}")
        logger.info(f"📍 Port: {os.getenv('PORT', '8000')}")

        # 🆕 업로드 디렉토리 설정
        upload_setup_success = setup_upload_directories()
        if upload_setup_success:
            logger.info("📸 이미지 업로드 기능 활성화")
        else:
            logger.warning("⚠️ 이미지 업로드 기능 비활성화")

        # 📊 Firebase 상태 확인
        firebase_status = {"firebase_available": False, "error": None}
        try:
            from utils.auth_utils import get_firebase_status
            firebase_status = get_firebase_status()
        except Exception as e:
            firebase_status["error"] = str(e)
            logger.warning(f"⚠️ Firebase 상태 확인 실패: {e}")

        if firebase_status["firebase_available"]:
            logger.info("🔥 Firebase 인증 시스템 활성화")
        else:
            logger.warning("⚠️ Firebase 인증 시스템 비활성화")

        logger.info("✅ Whiff API 서버 시작 완료!")

    except Exception as e:
        logger.error(f"❌ 서버 시작 중 오류: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """서버 종료 시 정리"""
    logger.info("🛑 Whiff API 서버 종료 중...")


# ─── 라우터 등록 함수 ─────────────────────────────────────────────────────────────
def register_routers():
    """모든 라우터를 등록합니다"""
    try:
        logger.info("🔌 라우터 등록 시작...")

        router_status = {}

        # 1. 향수 관련 라우터들
        main_routers = [
            ("perfume_router", "향수 데이터", "routers.perfume_router"),
            ("recommend_router", "1차 추천 (감정 클러스터)", "routers.recommend_router"),
            ("recommend_2nd_router", "2차 추천 (노트 기반)", "routers.recommend_2nd_router"),
        ]

        for router_name, description, module_path in main_routers:
            try:
                module = __import__(module_path, fromlist=['router'])
                app.include_router(module.router)
                router_status[router_name] = "✅ 성공"
                logger.info(f"  ✅ {description} 라우터 등록 완료")
            except Exception as e:
                router_status[router_name] = f"❌ 실패: {str(e)}"
                logger.error(f"  ❌ {description} 라우터 등록 실패: {e}")

        # 2. 사용자 인증 라우터
        try:
            from routers.auth_router import router as auth_router
            app.include_router(auth_router)
            router_status["auth_router"] = "✅ 성공"
            logger.info("  ✅ 사용자 인증 라우터 등록 완료")
        except Exception as e:
            router_status["auth_router"] = f"❌ 실패: {str(e)}"
            logger.error(f"  ❌ 사용자 인증 라우터 등록 실패: {e}")

        # 3. 사용자 관리 라우터
        try:
            from routers.user_router import router as user_router
            app.include_router(user_router)
            router_status["user_router"] = "✅ 성공"
            logger.info("  ✅ 사용자 관리 라우터 등록 완료")
        except Exception as e:
            router_status["user_router"] = f"❌ 실패: {str(e)}"
            logger.error(f"  ❌ 사용자 관리 라우터 등록 실패: {e}")

        # 4. 🎭 시향 일기 라우터 (이미지 업로드 기능 포함)
        try:
            logger.info("🎭 시향 일기 라우터 등록 시도...")

            from routers.diary_router import router as diary_router
            app.include_router(diary_router)

            router_status["diary_router"] = "✅ 성공"
            logger.info("  ✅ 시향 일기 라우터 등록 완료")
            logger.info("  📸 이미지 업로드 기능 포함")

        except ImportError as e:
            router_status["diary_router"] = f"❌ ImportError: {str(e)}"
            logger.error(f"  ❌ 시향 일기 라우터 임포트 실패: {e}")

        except Exception as e:
            router_status["diary_router"] = f"❌ Exception: {str(e)}"
            logger.error(f"  ❌ 시향 일기 라우터 등록 실패: {e}")

        # 5. 기타 라우터들 (선택적)
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
            "/diaries/upload-image",  # 🆕 이미지 업로드
            "/diaries/with-image",  # 🆕 일기+이미지 동시 작성
            "/auth/register"
        ]

        logger.info("🎯 주요 엔드포인트 확인:")
        for endpoint in key_endpoints:
            if any(endpoint in route for route in registered_routes):
                logger.info(f"  ✅ {endpoint}")
            else:
                logger.warning(f"  ❌ {endpoint} - 누락됨")

        # 🆕 이미지 관련 엔드포인트 특별 확인
        image_endpoints = [ep for ep in registered_routes if "/image" in ep or "/upload" in ep]
        if image_endpoints:
            logger.info(f"📸 이미지 관련 엔드포인트 ({len(image_endpoints)}개):")
            for endpoint in image_endpoints:
                logger.info(f"  📸 {endpoint}")
        else:
            logger.warning("📸 이미지 관련 엔드포인트가 등록되지 않음")

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
        "version": "1.4.0",
        "environment": "production" if os.getenv("RENDER") else "development",
        "port": os.getenv("PORT", "8000"),
        "features": [
            "향수 추천 (1차 - AI 감정 클러스터)",
            "향수 추천 (2차 - 노트 기반 정밀 추천)",
            "시향 일기 (AI 감정 분석 포함)",
            "📸 이미지 업로드 및 처리 기능",  # 🆕 추가
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
        "new_features_v1_4": [  # 🆕 버전 정보 업데이트
            "📸 시향 일기 이미지 업로드 기능",
            "🖼️ 자동 이미지 리사이징 및 썸네일 생성",
            "🔒 이미지 파일 검증 및 보안",
            "📁 정적 파일 서빙 (/uploads 경로)",
            "🎭 일기+이미지 통합 작성 API"
        ],
        "image_features": {  # 🆕 이미지 기능 상세 정보
            "supported_formats": ["JPG", "JPEG", "PNG", "WEBP"],
            "max_file_size": "10MB",
            "auto_resize": "1920x1920",
            "thumbnail_size": "400x400",
            "upload_endpoint": "/diaries/upload-image",
            "combined_endpoint": "/diaries/with-image"
        },
        "docs_url": "/docs",
        "redoc_url": "/redoc"
    }


@app.head("/", operation_id="head_root")
def head_root():
    return JSONResponse(content={})


@app.get("/health", summary="헬스 체크", operation_id="get_health_check")
def health_check():
    try:
        # 🆕 업로드 디렉토리 상태 확인
        upload_dir_status = "unknown"
        try:
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            UPLOAD_DIR = os.path.join(BASE_DIR, "uploads", "diary_images")
            if os.path.exists(UPLOAD_DIR):
                upload_dir_status = "available"
            else:
                upload_dir_status = "not_found"
        except:
            upload_dir_status = "error"

        return {
            "status": "ok",
            "service": "Whiff API",
            "version": "1.4.0",
            "environment": "production" if os.getenv("RENDER") else "development",
            "port": os.getenv("PORT", "8000"),
            "uptime": "running",
            "features_available": [
                "1차 추천 (AI 감정 클러스터)",
                "2차 추천 (노트 기반 정밀)",
                "시향 일기 (AI 감정 분석)",
                "📸 이미지 업로드 및 처리",  # 🆕 추가
                "사용자 인증",
                "실시간 통계"
            ],
            "deleted_features": [
                "시향 코스 추천",
                "매장 정보 조회",
                "특정 일기 상세 조회",
                "감정 통계 조회"
            ],
            "image_system": {  # 🆕 이미지 시스템 상태
                "upload_dir_status": upload_dir_status,
                "static_mount": "/uploads",
                "supported_formats": ["jpg", "jpeg", "png", "webp"],
                "max_size_mb": 10
            }
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "service": "Whiff API",
                "error": str(e)
            }
        )


# 🆕 이미지 업로드 관련 정보 엔드포인트
@app.get("/image-info", summary="이미지 업로드 기능 정보")
def get_image_info():
    """이미지 업로드 기능의 상세 정보 제공"""
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        UPLOAD_DIR = os.path.join(BASE_DIR, "uploads", "diary_images")

        return {
            "image_upload_enabled": True,
            "upload_endpoints": {
                "image_only": "POST /diaries/upload-image",
                "diary_with_image": "POST /diaries/with-image",
                "add_to_existing": "PUT /diaries/{diary_id}/add-image"
            },
            "supported_formats": ["JPG", "JPEG", "PNG", "WEBP"],
            "file_size_limit": "10MB",
            "processing_features": [
                "자동 리사이징 (최대 1920x1920)",
                "썸네일 생성 (400x400)",
                "EXIF 회전 보정",
                "이미지 최적화"
            ],
            "upload_directory": UPLOAD_DIR,
            "static_url_base": "/uploads/diary_images/",
            "directory_exists": os.path.exists(UPLOAD_DIR),
            "security_features": [
                "파일 확장자 검증",
                "MIME 타입 검증",
                "파일 크기 제한",
                "사용자별 파일 접근 제어"
            ]
        }
    except Exception as e:
        return {
            "image_upload_enabled": False,
            "error": str(e)
        }


# 🆕 업로드 디렉토리 수동 생성 엔드포인트 (관리용)
@app.post("/admin/setup-upload-dirs", summary="업로드 디렉토리 설정 (관리자용)")
def setup_upload_dirs_manual():
    """업로드 디렉토리를 수동으로 생성합니다 (관리자용)"""
    try:
        success = setup_upload_directories()
        if success:
            return {
                "status": "success",
                "message": "업로드 디렉토리 설정 완료",
                "timestamp": datetime.now().isoformat()
            }
        else:
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "message": "업로드 디렉토리 설정 실패"
                }
            )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"설정 중 오류: {str(e)}"
            }
        )


# ─── 예외 처리 ──────────────────────────────────────────────────────────────────
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "message": "요청하신 리소스를 찾을 수 없습니다.",
            "path": str(request.url.path),
            "method": request.method
        }
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"❌ 내부 서버 오류: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "message": "내부 서버 오류가 발생했습니다.",
            "error": "서버에서 요청을 처리하는 중 문제가 발생했습니다."
        }
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)