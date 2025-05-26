# schemas/base.py
from pydantic import BaseModel
from typing import Any

# 모든 API 응답에 공통으로 사용할 BaseResponse
class BaseResponse(BaseModel):
    code: int               # 0: 성공, 그 외는 에러 코드
    message: str            # 처리 결과 메시지
    data: Any               # 실제 응답 데이터 (향수 리스트, 리뷰 목록 등)
