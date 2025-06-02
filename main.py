# main.py에서 루트 엔드포인트 수정

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.exception_handlers import request_validation_exception_handler

# 기존 import들...
from routers.perfume_router import router as perfume_router
from routers.store_router import router as store_router
from routers.course_router import router as course_router
from routers.recommend_router import router as recommend_router
from routers.diary_router import router as diary_router
from routers.auth_router import router as auth_router
from routers.recommendation_save_router import router as recommendation_save_router
from routers.user_router import router as user_router

app = FastAPI(
    title="ScentRoute API",
    description="AI 기반 향수 추천 및 시향 코스 추천 서비스의 백엔드 API입니다.",
    version="1.0.2"
)

# 모든 라우터 등록
app.include_router(perfume_router)
app.include_router(store_router)
app.include_router(course_router)
app.include_router(recommend_router)
app.include_router(diary_router)
app.include_router(auth_router)
app.include_router(user_router)
app.include_router(recommendation_save_router)

# ✅ 유효성 검사 에러 커스텀 응답 처리
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"message": "입력값이 유효하지 않습니다. 예를 들어 notes는 문자열 리스트여야 합니다."},
    )

# ✅ GET과 HEAD 요청 모두 지원하는 루트 엔드포인트
@app.api_route(
    "/",
    methods=["GET", "HEAD"],  # GET과 HEAD 요청 모두 허용
    summary="루트",
    description="ScentRoute API 서버가 정상 작동 중인지 확인합니다.",
    response_description="서버 상태 메시지"
)
def read_root():
    return {
        "message": "✅ ScentRoute API is running!",
        "status": "ok",
        "version": "1.0.0"
    }

# ✅ 헬스 체크 엔드포인트도 GET/HEAD 지원
@app.api_route("/health", methods=["GET", "HEAD"], summary="헬스 체크", description="서버 상태 확인용 경량 엔드포인트")
def health_check():
    return {"status": "ok"}