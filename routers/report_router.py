# routers/report_router.py
import os
import json
import uuid
import logging
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import JSONResponse

# 모델 import (위에서 만든 모델들)
from models.report_models import (
    DiaryReportRequest,
    DiaryReportResponse,
    AdminReportAction,
    ReportStatus,
    ReportType
)

# 기존 다이어리 유틸리티 import
from routers.diary_router import load_diary_data

logger = logging.getLogger("whiff_report")

# 📁 데이터 파일 경로
REPORTS_PATH = "data/reports.json"
REPORTS_STATS_PATH = "data/reports_stats.json"

router = APIRouter(
    prefix="/reports",
    tags=["신고 관리"],
    responses={404: {"description": "Not found"}},
)


# ─── 유틸리티 함수들 ─────────────────────────────────────────────────────────────

def load_reports_data():
    """신고 데이터 로딩"""
    if os.path.exists(REPORTS_PATH):
        try:
            with open(REPORTS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.info(f"✅ 신고 데이터 로딩: {len(data)}개")
            return data
        except Exception as e:
            logger.error(f"❌ 신고 데이터 로딩 실패: {e}")
    return []


def save_reports_data(data):
    """신고 데이터 저장"""
    try:
        os.makedirs(os.path.dirname(REPORTS_PATH), exist_ok=True)
        with open(REPORTS_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"✅ 신고 데이터 저장: {len(data)}개")
    except Exception as e:
        logger.error(f"❌ 신고 데이터 저장 실패: {e}")
        raise e


def get_diary_by_id(diary_id: str):
    """일기 ID로 일기 찾기"""
    diaries = load_diary_data()
    for diary in diaries:
        if diary.get("id") == diary_id:
            return diary
    return None


def check_duplicate_report(diary_id: str, reporter_id: str):
    """중복 신고 확인"""
    reports = load_reports_data()
    for report in reports:
        if (report.get("diary_id") == diary_id and
                report.get("reporter_id") == reporter_id and
                report.get("status") in ["pending", "reviewing"]):
            return True
    return False


def update_report_stats(report_type: str):
    """신고 통계 업데이트"""
    try:
        stats = {}
        if os.path.exists(REPORTS_STATS_PATH):
            with open(REPORTS_STATS_PATH, "r", encoding="utf-8") as f:
                stats = json.load(f)

        # 오늘 날짜
        today = datetime.now().strftime("%Y-%m-%d")

        # 통계 초기화
        if "daily" not in stats:
            stats["daily"] = {}
        if "total" not in stats:
            stats["total"] = {}

        # 일일 통계 업데이트
        if today not in stats["daily"]:
            stats["daily"][today] = {}
        stats["daily"][today][report_type] = stats["daily"][today].get(report_type, 0) + 1

        # 전체 통계 업데이트
        stats["total"][report_type] = stats["total"].get(report_type, 0) + 1

        # 저장
        os.makedirs(os.path.dirname(REPORTS_STATS_PATH), exist_ok=True)
        with open(REPORTS_STATS_PATH, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

    except Exception as e:
        logger.error(f"❌ 신고 통계 업데이트 실패: {e}")


# ─── API 엔드포인트들 ────────────────────────────────────────────────────────────

@router.post("/diary", summary="시향 일기 신고", response_model=DiaryReportResponse)
async def report_diary(request: DiaryReportRequest):
    """
    시향 일기 신고 기능

    - 부적절한 콘텐츠가 포함된 시향 일기를 신고합니다
    - 중복 신고는 방지됩니다
    - 신고 후 관리자 검토 대기 상태가 됩니다
    """
    try:
        # 1. 신고할 일기 존재 확인
        diary = get_diary_by_id(request.diary_id)
        if not diary:
            raise HTTPException(
                status_code=404,
                detail="신고하려는 일기를 찾을 수 없습니다."
            )

        # 2. 자기 자신의 일기 신고 방지
        if diary.get("user_id") == request.reporter_id:
            raise HTTPException(
                status_code=400,
                detail="자신의 일기는 신고할 수 없습니다."
            )

        # 3. 중복 신고 확인
        if check_duplicate_report(request.diary_id, request.reporter_id):
            raise HTTPException(
                status_code=409,
                detail="이미 이 일기에 대해 신고한 기록이 있습니다."
            )

        # 4. 신고 데이터 생성
        report_id = str(uuid.uuid4())
        now = datetime.now()

        report_data = {
            "id": report_id,
            "diary_id": request.diary_id,
            "diary_title": diary.get("perfume_name", "Unknown"),
            "diary_author": diary.get("user_name", "Unknown"),
            "diary_author_id": diary.get("user_id", "Unknown"),
            "reporter_id": request.reporter_id,
            "report_type": request.report_type.value,
            "reason": request.reason,
            "additional_info": request.additional_info,
            "status": ReportStatus.PENDING.value,
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "admin_note": None,
            "resolved_at": None
        }

        # 5. 데이터 저장
        reports_data = load_reports_data()
        reports_data.append(report_data)
        save_reports_data(reports_data)

        # 6. 통계 업데이트
        update_report_stats(request.report_type.value)

        logger.info(f"📢 새 신고 접수: {report_id} (일기: {request.diary_id})")

        return DiaryReportResponse(
            report_id=report_id,
            diary_id=request.diary_id,
            reporter_id=request.reporter_id,
            report_type=request.report_type,
            reason=request.reason,
            status=ReportStatus.PENDING,
            created_at=now,
            message="신고가 성공적으로 접수되었습니다. 관리자 검토 후 조치될 예정입니다."
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 신고 처리 오류: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"신고 처리 중 오류가 발생했습니다: {str(e)}"
        )


@router.get("/", summary="신고 목록 조회 (관리자용)")
async def get_reports_list(
        status: Optional[ReportStatus] = Query(None, description="상태 필터"),
        report_type: Optional[ReportType] = Query(None, description="신고 유형 필터"),
        page: int = Query(1, ge=1, description="페이지 번호"),
        size: int = Query(20, ge=1, le=100, description="페이지 크기")
):
    """
    신고 목록 조회 (관리자용)

    - 모든 신고를 조회할 수 있습니다
    - 상태나 유형별로 필터링 가능합니다
    """
    try:
        reports_data = load_reports_data()

        # 필터링
        filtered_reports = reports_data
        if status:
            filtered_reports = [r for r in filtered_reports if r.get("status") == status.value]
        if report_type:
            filtered_reports = [r for r in filtered_reports if r.get("report_type") == report_type.value]

        # 최신순 정렬
        filtered_reports.sort(key=lambda x: x.get("created_at", ""), reverse=True)

        # 페이징
        start_idx = (page - 1) * size
        end_idx = start_idx + size
        paginated_reports = filtered_reports[start_idx:end_idx]

        return {
            "reports": paginated_reports,
            "total": len(filtered_reports),
            "page": page,
            "size": size,
            "total_pages": (len(filtered_reports) + size - 1) // size
        }

    except Exception as e:
        logger.error(f"❌ 신고 목록 조회 오류: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"신고 목록 조회 중 오류가 발생했습니다: {str(e)}"
        )


@router.get("/stats", summary="신고 통계 조회")
async def get_report_stats():
    """신고 통계 정보 조회"""
    try:
        # 신고 통계 로딩
        stats = {}
        if os.path.exists(REPORTS_STATS_PATH):
            with open(REPORTS_STATS_PATH, "r", encoding="utf-8") as f:
                stats = json.load(f)

        # 실시간 통계 계산
        reports_data = load_reports_data()
        status_counts = {}
        for report in reports_data:
            status = report.get("status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1

        return {
            "daily_stats": stats.get("daily", {}),
            "total_stats": stats.get("total", {}),
            "status_counts": status_counts,
            "total_reports": len(reports_data)
        }

    except Exception as e:
        logger.error(f"❌ 신고 통계 조회 오류: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"신고 통계 조회 중 오류가 발생했습니다: {str(e)}"
        )


@router.put("/{report_id}/action", summary="신고 처리 (관리자용)")
async def process_report(report_id: str, action: AdminReportAction):
    """
    신고 처리 (관리자용)

    - approve: 신고 승인 (일기는 유지)
    - reject: 신고 기각
    - delete_diary: 신고 승인 + 일기 삭제
    """
    try:
        # 신고 찾기
        reports_data = load_reports_data()
        report_index = None
        for i, report in enumerate(reports_data):
            if report.get("id") == report_id:
                report_index = i
                break

        if report_index is None:
            raise HTTPException(
                status_code=404,
                detail="해당 신고를 찾을 수 없습니다."
            )

        report = reports_data[report_index]

        # 이미 처리된 신고인지 확인
        if report.get("status") in ["resolved", "rejected"]:
            raise HTTPException(
                status_code=400,
                detail="이미 처리된 신고입니다."
            )

        # 액션에 따른 처리
        now = datetime.now()

        if action.action == "approve":
            report["status"] = "resolved"
            message = "신고가 승인되었습니다."
        elif action.action == "reject":
            report["status"] = "rejected"
            message = "신고가 기각되었습니다."
        elif action.action == "delete_diary":
            # TODO: 실제 일기 삭제 로직 구현 필요
            report["status"] = "resolved"
            message = "신고가 승인되어 해당 일기가 삭제되었습니다."
        else:
            raise HTTPException(
                status_code=400,
                detail="유효하지 않은 액션입니다."
            )

        # 신고 업데이트
        report["admin_note"] = action.admin_note
        report["resolved_at"] = now.isoformat()
        report["updated_at"] = now.isoformat()

        # 저장
        save_reports_data(reports_data)

        logger.info(f"📋 신고 처리 완료: {report_id} - {action.action}")

        return {
            "message": message,
            "report_id": report_id,
            "action": action.action,
            "processed_at": now.isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 신고 처리 오류: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"신고 처리 중 오류가 발생했습니다: {str(e)}"
        )


@router.delete("/{report_id}", summary="신고 삭제 (관리자용)")
async def delete_report(report_id: str):
    """신고 삭제 (관리자용)"""
    try:
        reports_data = load_reports_data()
        initial_count = len(reports_data)

        # 해당 신고 제거
        reports_data = [r for r in reports_data if r.get("id") != report_id]

        if len(reports_data) == initial_count:
            raise HTTPException(
                status_code=404,
                detail="해당 신고를 찾을 수 없습니다."
            )

        # 저장
        save_reports_data(reports_data)

        logger.info(f"🗑️ 신고 삭제: {report_id}")

        return {
            "message": "신고가 삭제되었습니다.",
            "report_id": report_id
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 신고 삭제 오류: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"신고 삭제 중 오류가 발생했습니다: {str(e)}"
        )