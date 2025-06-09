# routers/emotion_tagging_router.py
# 🆕 감정 태깅 및 분석 API 라우터

import logging
import asyncio
from datetime import datetime
from typing import List, Dict, Optional, Any
from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

# 감정 분석기 임포트
from utils.emotion_analyzer import emotion_analyzer
from utils.auth_utils import verify_firebase_token_optional

# ─── 로거 설정 ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("emotion_tagging_router")


# ─── 요청/응답 스키마 정의 ─────────────────────────────────────────────────────────────
class EmotionAnalysisRequest(BaseModel):
    """단일 텍스트 감정 분석 요청"""

    text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="분석할 텍스트 (1-2000자)",
        example="이 향수 정말 좋아요! 달콤하고 상큼해서 기분이 좋아져요."
    )

    use_ai_model: bool = Field(
        True,
        description="AI 모델 사용 여부 (False면 룰 기반만 사용)"
    )

    @validator('text')
    def validate_text(cls, v):
        if not v or not v.strip():
            raise ValueError('텍스트는 비어있을 수 없습니다.')
        return v.strip()

    class Config:
        schema_extra = {
            "example": {
                "text": "이 향수 정말 좋아요! 달콤하고 상큼해서 기분이 좋아져요.",
                "use_ai_model": True
            }
        }


class BatchEmotionAnalysisRequest(BaseModel):
    """배치 감정 분석 요청"""

    texts: List[str] = Field(
        ...,
        min_items=1,
        max_items=50,
        description="분석할 텍스트 목록 (최대 50개)"
    )

    use_ai_model: bool = Field(
        True,
        description="AI 모델 사용 여부"
    )

    @validator('texts')
    def validate_texts(cls, v):
        for i, text in enumerate(v):
            if not text or not text.strip():
                raise ValueError(f'텍스트 {i + 1}번이 비어있습니다.')
            if len(text) > 2000:
                raise ValueError(f'텍스트 {i + 1}번이 2000자를 초과합니다.')
        return [text.strip() for text in v]

    class Config:
        schema_extra = {
            "example": {
                "texts": [
                    "향이 너무 진해서 별로예요",
                    "달콤하고 포근한 향기가 좋아요",
                    "상쾌하고 시원한 느낌이에요"
                ],
                "use_ai_model": True
            }
        }


class EmotionAnalysisResult(BaseModel):
    """단일 감정 분석 결과"""

    text: str = Field(..., description="분석된 텍스트")
    primary_emotion: str = Field(..., description="주요 감정")
    confidence: float = Field(..., description="신뢰도 (0.0-1.0)", ge=0.0, le=1.0)
    emotion_tags: List[str] = Field(..., description="감정 태그 목록")
    method: str = Field(..., description="분석 방법")
    processing_time_ms: float = Field(..., description="처리 시간 (밀리초)")


class BatchEmotionAnalysisResult(BaseModel):
    """배치 감정 분석 결과"""

    results: List[EmotionAnalysisResult] = Field(..., description="각 텍스트별 분석 결과")
    summary: Dict[str, Any] = Field(..., description="전체 분석 요약")
    total_processing_time_ms: float = Field(..., description="전체 처리 시간 (밀리초)")


class EmotionSystemStatus(BaseModel):
    """감정 분석 시스템 상태"""

    system_status: str = Field(..., description="시스템 상태")
    supported_emotions: List[str] = Field(..., description="지원하는 감정 목록")
    analysis_methods: List[str] = Field(..., description="사용 가능한 분석 방법")
    performance_stats: Dict[str, Any] = Field(..., description="성능 통계")
    google_drive_model: Dict[str, Any] = Field(..., description="Google Drive 모델 정보")


# ─── 라우터 설정 ────────────────────────────────────────────────
router = APIRouter(prefix="/emotions", tags=["Emotion Analysis"])

# 시작 시 감정 분석기 초기화 확인
logger.info("🎭 감정 태깅 라우터 초기화 시작...")
try:
    stats = emotion_analyzer.get_analysis_stats()
    logger.info(f"✅ 감정 분석기 준비 완료: {stats['supported_emotions']}개 감정 지원")
    logger.info(f"  - 분석 방법: {stats['analysis_methods']}")
    logger.info(f"  - Google Drive: {'사용 가능' if stats['google_drive']['enabled'] else '사용 불가'}")
except Exception as e:
    logger.error(f"❌ 감정 분석기 초기화 확인 실패: {e}")


# ─── API 엔드포인트들 ────────────────────────────────────────────────

@router.post(
    "/analyze",
    response_model=EmotionAnalysisResult,
    summary="텍스트 감정 분석",
    description=(
            "🎭 **텍스트 감정 분석 API**\n\n"
            "시향 일기, 리뷰, 댓글 등의 텍스트를 분석하여 감정을 태깅합니다.\n\n"
            "**🤖 분석 방법:**\n"
            "1. **Google Drive AI 모델**: 클라우드 기반 고성능 모델 (우선순위 1)\n"
            "2. **로컬 AI 모델**: 서버 내장 모델 (우선순위 2)\n"
            "3. **룰 기반 분석**: 향수 도메인 특화 키워드 매칭 (폴백)\n\n"
            "**🎯 지원 감정:**\n"
            "- 기쁨, 불안, 당황, 분노, 상처, 슬픔, 우울, 흥분\n"
            "- 각 감정별 전용 태그 시스템\n\n"
            "**📝 입력 제한:**\n"
            "- 텍스트 길이: 1-2000자\n"
            "- 지원 언어: 한국어\n"
            "- 도메인: 향수/화장품 리뷰 특화\n\n"
            "**📊 출력 정보:**\n"
            "- 주요 감정 + 신뢰도\n"
            "- 감정별 해시태그\n"
            "- 분석 방법 및 처리 시간"
    )
)
async def analyze_emotion(request: EmotionAnalysisRequest):
    """단일 텍스트 감정 분석"""

    start_time = datetime.now()

    logger.info(f"🎭 단일 텍스트 감정 분석 요청")
    logger.info(f"  - 텍스트 길이: {len(request.text)}자")
    logger.info(f"  - AI 모델 사용: {'✅' if request.use_ai_model else '❌'}")

    try:
        # 감정 분석 수행
        analysis_result = await emotion_analyzer.analyze_emotion(
            text=request.text,
            use_model=request.use_ai_model
        )

        # 처리 시간 계산
        processing_time = (datetime.now() - start_time).total_seconds() * 1000  # 밀리초

        if not analysis_result.get("success", False):
            # 분석 실패 시
            error_message = analysis_result.get("message", "감정 분석에 실패했습니다.")
            logger.error(f"❌ 감정 분석 실패: {error_message}")
            raise HTTPException(
                status_code=500,
                detail=f"감정 분석 실패: {error_message}"
            )

        # 성공 응답 구성
        result = EmotionAnalysisResult(
            text=request.text,
            primary_emotion=analysis_result.get("primary_emotion", "중립"),
            confidence=analysis_result.get("confidence", 0.0),
            emotion_tags=analysis_result.get("emotion_tags", ["#neutral"]),
            method=analysis_result.get("method", "unknown"),
            processing_time_ms=round(processing_time, 2)
        )

        logger.info(f"✅ 감정 분석 완료: {result.primary_emotion} (신뢰도: {result.confidence:.3f})")
        logger.info(f"⏱️ 처리 시간: {result.processing_time_ms:.2f}ms")

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 감정 분석 API 처리 중 예외: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"감정 분석 중 서버 오류가 발생했습니다: {str(e)}"
        )


@router.post(
    "/analyze-batch",
    response_model=BatchEmotionAnalysisResult,
    summary="배치 텍스트 감정 분석",
    description=(
            "🎭 **배치 텍스트 감정 분석 API**\n\n"
            "여러 텍스트를 한 번에 분석하여 감정 통계를 제공합니다.\n\n"
            "**📥 입력:**\n"
            "- 텍스트 목록 (최대 50개)\n"
            "- 각 텍스트 최대 2000자\n\n"
            "**📊 출력:**\n"
            "- 각 텍스트별 감정 분석 결과\n"
            "- 전체 감정 분포 통계\n"
            "- 평균 신뢰도 및 처리 성능\n\n"
            "**🚀 최적화:**\n"
            "- 비동기 병렬 처리\n"
            "- 에러 발생 시 개별 텍스트만 제외\n"
            "- 전체 통계 제공"
    )
)
async def analyze_batch_emotion(request: BatchEmotionAnalysisRequest):
    """배치 텍스트 감정 분석"""

    start_time = datetime.now()

    logger.info(f"🎭 배치 감정 분석 요청")
    logger.info(f"  - 텍스트 개수: {len(request.texts)}개")
    logger.info(f"  - AI 모델 사용: {'✅' if request.use_ai_model else '❌'}")

    try:
        # 비동기 병렬 처리를 위한 태스크 생성
        async def analyze_single_text(text: str, index: int) -> Dict[str, Any]:
            try:
                text_start = datetime.now()
                result = await emotion_analyzer.analyze_emotion(
                    text=text,
                    use_model=request.use_ai_model
                )
                text_time = (datetime.now() - text_start).total_seconds() * 1000

                return {
                    "index": index,
                    "text": text,
                    "success": result.get("success", False),
                    "result": result,
                    "processing_time_ms": text_time
                }
            except Exception as e:
                logger.warning(f"⚠️ 텍스트 {index + 1} 분석 실패: {e}")
                return {
                    "index": index,
                    "text": text,
                    "success": False,
                    "error": str(e),
                    "processing_time_ms": 0
                }

        # 모든 텍스트 병렬 분석
        tasks = [
            analyze_single_text(text, i)
            for i, text in enumerate(request.texts)
        ]

        batch_results = await asyncio.gather(*tasks)

        # 성공한 결과와 실패한 결과 분리
        successful_results = []
        failed_count = 0
        emotion_distribution = {}
        total_confidence = 0.0
        method_distribution = {}

        for batch_result in batch_results:
            if batch_result["success"]:
                result = batch_result["result"]

                # 성공한 분석 결과 저장
                emotion_result = EmotionAnalysisResult(
                    text=batch_result["text"],
                    primary_emotion=result.get("primary_emotion", "중립"),
                    confidence=result.get("confidence", 0.0),
                    emotion_tags=result.get("emotion_tags", ["#neutral"]),
                    method=result.get("method", "unknown"),
                    processing_time_ms=round(batch_result["processing_time_ms"], 2)
                )
                successful_results.append(emotion_result)

                # 통계 집계
                emotion = emotion_result.primary_emotion
                emotion_distribution[emotion] = emotion_distribution.get(emotion, 0) + 1
                total_confidence += emotion_result.confidence

                method = emotion_result.method
                method_distribution[method] = method_distribution.get(method, 0) + 1

            else:
                failed_count += 1

        # 전체 처리 시간 계산
        total_processing_time = (datetime.now() - start_time).total_seconds() * 1000

        # 요약 통계 생성
        success_count = len(successful_results)
        summary = {
            "total_texts": len(request.texts),
            "successful_analyses": success_count,
            "failed_analyses": failed_count,
            "success_rate": round((success_count / len(request.texts)) * 100, 2) if request.texts else 0,
            "emotion_distribution": emotion_distribution,
            "method_distribution": method_distribution,
            "average_confidence": round(total_confidence / success_count, 3) if success_count > 0 else 0.0,
            "average_processing_time_ms": round(
                sum(r.processing_time_ms for r in successful_results) / success_count, 2
            ) if success_count > 0 else 0.0
        }

        # 응답 구성
        response = BatchEmotionAnalysisResult(
            results=successful_results,
            summary=summary,
            total_processing_time_ms=round(total_processing_time, 2)
        )

        logger.info(f"✅ 배치 감정 분석 완료")
        logger.info(f"  - 성공: {success_count}/{len(request.texts)}개")
        logger.info(f"  - 성공률: {summary['success_rate']}%")
        logger.info(f"  - 평균 신뢰도: {summary['average_confidence']:.3f}")
        logger.info(f"  - 전체 처리시간: {total_processing_time:.2f}ms")
        logger.info(f"  - 감정 분포: {emotion_distribution}")

        return response

    except Exception as e:
        logger.error(f"❌ 배치 감정 분석 API 처리 중 예외: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"배치 감정 분석 중 서버 오류가 발생했습니다: {str(e)}"
        )


@router.get(
    "/system-status",
    response_model=EmotionSystemStatus,
    summary="감정 분석 시스템 상태",
    description=(
            "🎭 **감정 분석 시스템 상태 확인**\n\n"
            "감정 분석 시스템의 현재 상태와 성능 통계를 제공합니다.\n\n"
            "**📊 제공 정보:**\n"
            "- 시스템 동작 상태\n"
            "- 지원하는 감정 목록\n"
            "- 사용 가능한 분석 방법\n"
            "- 성능 통계 (분석 횟수, 성공률, 응답시간)\n"
            "- Google Drive 모델 상태\n\n"
            "**🔧 용도:**\n"
            "- 시스템 모니터링\n"
            "- 성능 분석\n"
            "- 디버깅 지원"
    )
)
async def get_system_status():
    """감정 분석 시스템 상태 확인"""

    logger.info("📊 감정 분석 시스템 상태 요청")

    try:
        # 감정 분석기 상태 가져오기
        stats = emotion_analyzer.get_analysis_stats()

        # 시스템 상태 판단
        performance = stats["performance"]
        success_rate = performance["success_rate"]

        if success_rate >= 95:
            system_status = "excellent"
        elif success_rate >= 90:
            system_status = "good"
        elif success_rate >= 80:
            system_status = "fair"
        else:
            system_status = "poor"

        # 응답 구성
        response = EmotionSystemStatus(
            system_status=system_status,
            supported_emotions=emotion_analyzer.get_supported_emotions(),
            analysis_methods=stats["analysis_methods"],
            performance_stats=stats["performance"],
            google_drive_model=stats["google_drive"]
        )

        logger.info(f"✅ 시스템 상태 조회 완료: {system_status}")
        logger.info(f"  - 성공률: {success_rate}%")
        logger.info(f"  - 총 분석: {performance['total_analyses']}회")
        logger.info(f"  - Google Drive: {'사용' if stats['google_drive']['enabled'] else '미사용'}")

        return response

    except Exception as e:
        logger.error(f"❌ 시스템 상태 확인 중 예외: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"시스템 상태 확인 중 오류가 발생했습니다: {str(e)}"
        )


@router.get(
    "/emotions",
    summary="지원하는 감정 목록",
    description="감정 분석 시스템에서 지원하는 모든 감정과 해당 태그를 반환합니다."
)
async def get_supported_emotions():
    """지원하는 감정 목록 조회"""

    try:
        emotions = emotion_analyzer.get_supported_emotions()
        emotion_info = {}

        for emotion in emotions:
            tags = emotion_analyzer.get_emotion_tags(emotion)
            emotion_info[emotion] = {
                "tags": tags,
                "tag_count": len(tags)
            }

        return {
            "supported_emotions": emotions,
            "emotion_count": len(emotions),
            "emotion_details": emotion_info,
            "total_tags": sum(len(tags) for tags in emotion_info.values())
        }

    except Exception as e:
        logger.error(f"❌ 지원 감정 목록 조회 중 예외: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"지원 감정 목록 조회 중 오류가 발생했습니다: {str(e)}"
        )


@router.get(
    "/performance-report",
    summary="성능 분석 리포트",
    description="감정 분석 시스템의 상세 성능 분석 리포트를 제공합니다."
)
async def get_performance_report():
    """감정 분석 성능 리포트"""

    try:
        report = emotion_analyzer.get_performance_report()

        logger.info("📋 성능 리포트 생성 완료")
        logger.info(f"  - 시스템 상태: {report['system_overview']['status']}")
        logger.info(f"  - 권장사항: {len(report['recommendations'])}개")

        return report

    except Exception as e:
        logger.error(f"❌ 성능 리포트 생성 중 예외: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"성능 리포트 생성 중 오류가 발생했습니다: {str(e)}"
        )


# ─── 🆕 Google Drive 모델 관리 API들 ────────────────────────────────────────────────

@router.get(
    "/google-drive/status",
    summary="Google Drive 모델 상태",
    description="Google Drive 기반 감정 분석 모델의 상태를 확인합니다."
)
async def get_google_drive_model_status():
    """Google Drive 모델 상태 확인"""

    try:
        gdrive_info = emotion_analyzer.get_google_drive_model_info()

        # Google Drive 모델 사용 가능 여부 확인
        available = emotion_analyzer.check_google_drive_model()

        response = {
            "google_drive_model": gdrive_info,
            "model_available": available,
            "can_download": gdrive_info["enabled"],
            "recommendations": []
        }

        # 권장사항 추가
        if not gdrive_info["enabled"]:
            response["recommendations"].append("GOOGLE_DRIVE_MODEL_ID 환경변수를 설정하여 Google Drive 모델을 활성화하세요.")
        elif not available:
            response["recommendations"].append("Google Drive에서 모델을 다운로드하여 성능을 향상시키세요.")
        elif gdrive_info["cached_model"]["exists"]:
            response["recommendations"].append("Google Drive 모델이 준비되어 최적의 성능을 제공합니다.")

        logger.info(f"📊 Google Drive 모델 상태: {'사용 가능' if available else '사용 불가'}")

        return response

    except Exception as e:
        logger.error(f"❌ Google Drive 모델 상태 확인 중 예외: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Google Drive 모델 상태 확인 중 오류가 발생했습니다: {str(e)}"
        )


@router.post(
    "/google-drive/download",
    summary="Google Drive 모델 다운로드",
    description="Google Drive에서 최신 감정 분석 모델을 다운로드합니다."
)
async def download_google_drive_model():
    """Google Drive 모델 다운로드"""

    logger.info("📥 Google Drive 모델 다운로드 요청")

    try:
        if not emotion_analyzer.google_drive_enabled:
            raise HTTPException(
                status_code=400,
                detail="Google Drive 모델이 활성화되지 않았습니다. GOOGLE_DRIVE_MODEL_ID 환경변수를 설정하세요."
            )

        # 모델 다운로드 시도
        download_success = await emotion_analyzer.download_google_drive_model()

        if download_success:
            # 다운로드 성공 시 모델 정보 확인
            gdrive_info = emotion_analyzer.get_google_drive_model_info()

            return {
                "success": True,
                "message": "Google Drive 모델 다운로드가 완료되었습니다.",
                "model_info": gdrive_info["cached_model"],
                "next_step": "이제 감정 분석에서 Google Drive 모델이 우선적으로 사용됩니다."
            }
        else:
            raise HTTPException(
                status_code=500,
                detail="Google Drive 모델 다운로드에 실패했습니다."
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Google Drive 모델 다운로드 중 예외: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Google Drive 모델 다운로드 중 오류가 발생했습니다: {str(e)}"
        )


@router.post(
    "/google-drive/force-download",
    summary="Google Drive 모델 강제 다운로드",
    description="기존 캐시를 삭제하고 Google Drive에서 모델을 강제로 다시 다운로드합니다."
)
async def force_download_google_drive_model():
    """Google Drive 모델 강제 다운로드"""

    logger.info("🔄 Google Drive 모델 강제 다운로드 요청")

    try:
        if not emotion_analyzer.google_drive_enabled:
            raise HTTPException(
                status_code=400,
                detail="Google Drive 모델이 활성화되지 않았습니다."
            )

        # 강제 다운로드 시도
        download_success = await emotion_analyzer.force_download_google_drive_model()

        if download_success:
            gdrive_info = emotion_analyzer.get_google_drive_model_info()

            return {
                "success": True,
                "message": "Google Drive 모델 강제 다운로드가 완료되었습니다.",
                "model_info": gdrive_info["cached_model"],
                "note": "기존 캐시가 삭제되고 새로운 모델이 다운로드되었습니다."
            }
        else:
            raise HTTPException(
                status_code=500,
                detail="Google Drive 모델 강제 다운로드에 실패했습니다."
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Google Drive 모델 강제 다운로드 중 예외: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Google Drive 모델 강제 다운로드 중 오류가 발생했습니다: {str(e)}"
        )


# ─── 🧪 테스트 및 디버깅 API들 ────────────────────────────────────────────────

@router.post(
    "/test",
    summary="감정 분석 테스트",
    description="감정 분석 시스템을 테스트하기 위한 샘플 텍스트들을 분석합니다."
)
async def test_emotion_analysis():
    """감정 분석 테스트"""

    logger.info("🧪 감정 분석 테스트 시작")

    test_texts = [
        "이 향수 정말 좋아요! 달콤하고 상큼해서 기분이 좋아져요.",
        "향이 너무 진해서 별로예요. 좀 부담스럽네요.",
        "처음 맡았을 때 놀랐어요. 예상과 완전 달라서 당황스러웠어요.",
        "이 향수를 맡으면 옛날 생각이 나서 슬퍼져요.",
        "향수가 너무 자극적이어서 화가 나요. 최악이에요.",
        "새로운 향수를 발견해서 너무 신나요! 에너지가 넘쳐요."
    ]

    try:
        # 배치 분석 수행
        batch_request = BatchEmotionAnalysisRequest(
            texts=test_texts,
            use_ai_model=True
        )

        result = await analyze_batch_emotion(batch_request)

        logger.info(f"✅ 감정 분석 테스트 완료: {len(result.results)}개 결과")

        return {
            "test_completed": True,
            "test_texts_count": len(test_texts),
            "analysis_results": result,
            "test_summary": {
                "success_rate": result.summary["success_rate"],
                "average_confidence": result.summary["average_confidence"],
                "emotion_distribution": result.summary["emotion_distribution"]
            }
        }

    except Exception as e:
        logger.error(f"❌ 감정 분석 테스트 중 예외: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"감정 분석 테스트 중 오류가 발생했습니다: {str(e)}"
        )


@router.post(
    "/reset-stats",
    summary="성능 통계 리셋",
    description="감정 분석 시스템의 성능 통계를 초기화합니다. (관리자용)"
)
async def reset_performance_stats():
    """성능 통계 리셋"""

    logger.info("🔄 성능 통계 리셋 요청")

    try:
        # 통계 리셋 전 현재 상태 저장
        old_stats = emotion_analyzer.get_analysis_stats()["performance"]

        # 통계 리셋 수행
        emotion_analyzer.reset_performance_stats()

        logger.info("✅ 성능 통계 리셋 완료")

        return {
            "success": True,
            "message": "성능 통계가 초기화되었습니다.",
            "previous_stats": old_stats,
            "reset_time": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"❌ 성능 통계 리셋 중 예외: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"성능 통계 리셋 중 오류가 발생했습니다: {str(e)}"
        )