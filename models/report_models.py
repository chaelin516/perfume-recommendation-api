# models/report_models.py
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum

class ReportType(str, Enum):
    """신고 유형"""
    INAPPROPRIATE_CONTENT = "inappropriate_content"  # 부적절한 내용
    SPAM = "spam"  # 스팸
    HARASSMENT = "harassment"  # 괴롭힘
    FAKE_INFO = "fake_info"  # 거짓 정보
    COPYRIGHT = "copyright"  # 저작권 침해
    OTHER = "other"  # 기타

class ReportStatus(str, Enum):
    """신고 상태"""
    PENDING = "pending"  # 대기중
    REVIEWING = "reviewing"  # 검토중
    RESOLVED = "resolved"  # 해결됨
    REJECTED = "rejected"  # 기각됨

class DiaryReportRequest(BaseModel):
    """시향 일기 신고 요청 모델"""
    diary_id: str = Field(..., description="신고할 일기 ID")
    reporter_id: str = Field(..., description="신고자 ID")
    report_type: ReportType = Field(..., description="신고 유형")
    reason: str = Field(..., min_length=10, max_length=500, description="신고 사유 (10-500자)")
    additional_info: Optional[str] = Field(None, max_length=1000, description="추가 정보")

class DiaryReportResponse(BaseModel):
    """시향 일기 신고 응답 모델"""
    report_id: str
    diary_id: str
    reporter_id: str
    report_type: ReportType
    reason: str
    status: ReportStatus
    created_at: datetime
    message: str

class AdminReportAction(BaseModel):
    """관리자 신고 처리 모델"""
    action: str = Field(..., description="처리 결과 (approve/reject/delete_diary)")
    admin_note: Optional[str] = Field(None, description="관리자 메모")
    notify_reporter: bool = Field(True, description="신고자에게 알림 여부")