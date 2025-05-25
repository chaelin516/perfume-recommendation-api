from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.exception_handlers import request_validation_exception_handler

# 각 기능별 라우터 임포트 (경로 수정됨)
from routers.perfume_router import router as perfume_router
from routers.store_router import router as store_router
from routers.course_router import router as course_router
from routers.recommend_router import router as recommend_router
from routers.diary_router import router as diary_router
from routers.auth_router import router as auth_router
from routers.recommendation_save_router import router as recommendation_save_router

# ✅ FastAPI 인스턴스 정의
app = FastAPI(
    title="ScentRoute API",
    description="AI 기반 향수 추천 및 시향 코스 추천 서비스의 백엔드 API입니다.",
    version="1.0.0"
)

# ✅ 라우터 등록
app.include_router(perfume_router)      # 향수 목록, 상세
app.include_router(store_router)        # 매장 전체, 브랜드별 조회
app.include_router(course_router)       # 향수 코스 추천 (AI 미사용)
app.include_router(recommend_router)    # 향수 추천 (성별/감정/계절/시간 기반)
app.include_router(diary_router)        # 시향 일기 저장 및 조회
app.include_router(auth_router)         # Firebase 인증
app.include_router(recommendation_save_router)  # 추천 저장

# ✅ 유효성 검사 에러 응답 커스터마이징
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"message": "입력값이 유효하지 않습니다. 예를 들어 notes는 문자열 리스트여야 합니다."},
    )

# ✅ 루트 확인용 엔드포인트
@app.get("/", summary="루트", description="서버 동작 확인용", response_description="작동 메시지 반환")
def read_root():
    return {"message": "Hello, FastAPI is working!"}

# ✅ 헬스 체크 엔드포인트
@app.get("/health", summary="헬스 체크", description="서버 상태 확인용")
def health_check():
    return {"status": "ok"}
