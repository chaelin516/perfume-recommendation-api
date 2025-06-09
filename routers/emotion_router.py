# routers/emotion_router.py
# 🎭 감정 분석 전용 API 라우터

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

# 감정 분석기 import
from utils.emotion_analyzer import emotion_analyzer
from utils.emotion_model_loader import get_emotion_models_status, is_emotion_models_available

logger = logging.getLogger("emotion_router")

router = APIRouter(prefix="/emotions", tags=["Emotion Analysis"])


# ─── 스키마 정의 ─────────────────────────────────────────────────────────
class EmotionAnalysisRequest(BaseModel):
    """감정 분석 요청 스키마"""
    text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="분석할 텍스트 (최대 2000자)",
        example="이 향수는 정말 좋아요! 달콤하고 상큼해서 기분이 좋아져요."
    )
    use_vectorizer: Optional[bool] = Field(
        True,
        description="벡터라이저 사용 여부 (기본값: True)"
    )
    include_details: Optional[bool] = Field(
        False,
        description="상세 분석 결과 포함 여부 (기본값: False)"
    )

    class Config:
        schema_extra = {
            "example": {
                "text": "이 향수는 정말 좋아요! 달콤하고 상큼해서 기분이 좋아져요.",
                "use_vectorizer": True,
                "include_details": False
            }
        }


class EmotionAnalysisResponse(BaseModel):
    """감정 분석 응답 스키마"""
    success: bool = Field(..., description="분석 성공 여부")
    primary_emotion: str = Field(..., description="주요 감정")
    confidence: float = Field(..., description="신뢰도 (0.0-1.0)", ge=0.0, le=1.0)
    emotion_tags: List[str] = Field(..., description="감정 태그 목록")
    method: str = Field(..., description="분석 방법")
    analyzed_at: str = Field(..., description="분석 시간")
    analysis_details: Optional[Dict[str, Any]] = Field(None, description="상세 분석 결과")
    message: Optional[str] = Field(None, description="메시지 (에러 시)")

    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "primary_emotion": "기쁨",
                "confidence": 0.857,
                "emotion_tags": ["#joyful", "#bright", "#citrus", "#happy", "#cheerful"],
                "method": "vectorizer_based",
                "analyzed_at": "2025-06-09T10:30:45.123456",
                "analysis_details": None,
                "message": None
            }
        }


class BatchEmotionRequest(BaseModel):
    """배치 감정 분석 요청 스키마"""
    texts: List[str] = Field(
        ...,
        min_items=1,
        max_items=50,
        description="분석할 텍스트 목록 (최대 50개)"
    )
    use_vectorizer: Optional[bool] = Field(True, description="벡터라이저 사용 여부")

    class Config:
        schema_extra = {
            "example": {
                "texts": [
                    "이 향수는 정말 좋아요!",
                    "향이 너무 진해서 별로예요.",
                    "예상과 달라서 당황스러웠어요."
                ],
                "use_vectorizer": True
            }
        }


class EmotionSystemStatus(BaseModel):
    """감정 분석 시스템 상태 스키마"""
    system_available: bool = Field(..., description="시스템 사용 가능 여부")
    vectorizer_available: bool = Field(..., description="벡터라이저 사용 가능 여부")
    emotion_model_available: bool = Field(..., description="감정 모델 사용 가능 여부")
    supported_emotions: List[str] = Field(..., description="지원하는 감정 목록")
    total_analyses: int = Field(..., description="총 분석 횟수")
    average_response_time: float = Field(..., description="평균 응답 시간")
    method_distribution: Dict[str, int] = Field(..., description="방법별 분석 분포")


# ─── API 엔드포인트들 ─────────────────────────────────────────────────────
@router.post(
    "/analyze",
    response_model=EmotionAnalysisResponse,
    summary="텍스트 감정 분석",
    description=(
            "🎭 **텍스트 감정 분석 API**\n\n"
            "입력된 텍스트의 감정을 분석하여 주요 감정과 관련 태그를 반환합니다.\n\n"
            "**🤖 분석 방법:**\n"
            "1. **벡터라이저 기반**: 머신러닝 모델을 사용한 감정 분류 (우선순위)\n"
            "2. **룰 기반**: 키워드 매칭 및 향수 도메인 특화 규칙 (폴백)\n\n"
            "**🎯 지원 감정:**\n"
            "- 기쁨, 불안, 당황, 분노, 상처, 슬픔, 우울, 흥분, 중립\n\n"
            "**💡 활용 방법:**\n"
            "- 시향 일기 자동 감정 태깅\n"
            "- 리뷰 감정 분석\n"
            "- 사용자 만족도 분석"
    )
)
async def analyze_emotion(request: EmotionAnalysisRequest):
    """텍스트 감정 분석"""
    try:
        logger.info(f"🎭 감정 분석 요청: 텍스트 길이 {len(request.text)}자, 벡터라이저 사용: {request.use_vectorizer}")

        # 감정 분석 수행
        result = await emotion_analyzer.analyze_emotion(
            text=request.text,
            use_vectorizer=request.use_vectorizer
        )

        # 응답 구성
        response_data = {
            "success": result.get("success", False),
            "primary_emotion": result.get("primary_emotion", "오류"),
            "confidence": result.get("confidence", 0.0),
            "emotion_tags": result.get("emotion_tags", ["#error"]),
            "method": result.get("method", "unknown"),
            "analyzed_at": result.get("analyzed_at", datetime.now().isoformat())
        }

        # 상세 정보 포함 여부
        if request.include_details:
            response_data["analysis_details"] = result.get("analysis_details")

        # 에러 메시지 포함 (실패 시)
        if not result.get("success"):
            response_data["message"] = result.get("message", "분석에 실패했습니다.")

        logger.info(f"✅ 감정 분석 완료: {response_data['primary_emotion']} (신뢰도: {response_data['confidence']:.3f})")

        return EmotionAnalysisResponse(**response_data)

    except Exception as e:
        logger.error(f"❌ 감정 분석 중 오류: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"감정 분석 중 오류가 발생했습니다: {str(e)}"
        )


@router.post(
    "/batch-analyze",
    summary="배치 감정 분석",
    description="여러 텍스트의 감정을 한 번에 분석합니다. (최대 50개)"
)
async def batch_analyze_emotions(request: BatchEmotionRequest):
    """배치 감정 분석"""
    try:
        logger.info(f"🎭 배치 감정 분석 요청: {len(request.texts)}개 텍스트")

        results = []

        for i, text in enumerate(request.texts):
            try:
                result = await emotion_analyzer.analyze_emotion(
                    text=text,
                    use_vectorizer=request.use_vectorizer
                )

                # 간단한 형태로 변환
                simple_result = {
                    "index": i,
                    "text": text[:50] + "..." if len(text) > 50 else text,
                    "success": result.get("success", False),
                    "primary_emotion": result.get("primary_emotion", "오류"),
                    "confidence": result.get("confidence", 0.0),
                    "emotion_tags": result.get("emotion_tags", ["#error"]),
                    "method": result.get("method", "unknown")
                }

                results.append(simple_result)

            except Exception as e:
                logger.error(f"❌ 텍스트 {i} 분석 실패: {e}")
                results.append({
                    "index": i,
                    "text": text[:50] + "..." if len(text) > 50 else text,
                    "success": False,
                    "primary_emotion": "오류",
                    "confidence": 0.0,
                    "emotion_tags": ["#error"],
                    "method": "error",
                    "error": str(e)
                })

        # 성공률 계산
        successful_analyses = sum(1 for r in results if r["success"])
        success_rate = successful_analyses / len(results) * 100

        logger.info(f"✅ 배치 감정 분석 완료: {successful_analyses}/{len(results)}개 성공 ({success_rate:.1f}%)")

        return {
            "message": f"배치 감정 분석 완료: {successful_analyses}/{len(results)}개 성공",
            "total_texts": len(request.texts),
            "successful_analyses": successful_analyses,
            "success_rate": round(success_rate, 1),
            "results": results
        }

    except Exception as e:
        logger.error(f"❌ 배치 감정 분석 중 오류: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"배치 감정 분석 중 오류가 발생했습니다: {str(e)}"
        )


@router.get(
    "/status",
    response_model=EmotionSystemStatus,
    summary="감정 분석 시스템 상태",
    description="감정 분석 시스템의 현재 상태와 통계를 반환합니다."
)
async def get_emotion_system_status():
    """감정 분석 시스템 상태 조회"""
    try:
        # 감정 분석기 통계
        analyzer_stats = emotion_analyzer.get_analysis_stats()

        # 모델 상태
        model_status = get_emotion_models_status()

        # 응답 구성
        status_data = {
            "system_available": True,
            "vectorizer_available": is_emotion_models_available(),
            "emotion_model_available": model_status.get("emotion_model_loaded", False),
            "supported_emotions": emotion_analyzer.get_supported_emotions(),
            "total_analyses": analyzer_stats["performance"]["total_analyses"],
            "average_response_time": analyzer_stats["performance"]["average_response_time"],
            "method_distribution": analyzer_stats["performance"]["method_distribution"]
        }

        return EmotionSystemStatus(**status_data)

    except Exception as e:
        logger.error(f"❌ 상태 조회 중 오류: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"상태 조회 중 오류가 발생했습니다: {str(e)}"
        )


@router.get(
    "/supported-emotions",
    summary="지원하는 감정 목록",
    description="시스템에서 지원하는 모든 감정과 관련 태그를 반환합니다."
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
            "message": "지원하는 감정 목록입니다.",
            "total_emotions": len(emotions),
            "emotions": emotion_info,
            "emotion_list": emotions
        }

    except Exception as e:
        logger.error(f"❌ 지원 감정 조회 중 오류: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"지원 감정 조회 중 오류가 발생했습니다: {str(e)}"
        )


@router.get(
    "/performance-report",
    summary="성능 리포트",
    description="감정 분석 시스템의 상세 성능 리포트를 반환합니다."
)
async def get_performance_report():
    """성능 리포트 조회"""
    try:
        report = emotion_analyzer.get_performance_report()

        # 모델 상태 추가
        model_status = get_emotion_models_status()
        report["model_status"] = model_status

        return report

    except Exception as e:
        logger.error(f"❌ 성능 리포트 조회 중 오류: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"성능 리포트 조회 중 오류가 발생했습니다: {str(e)}"
        )


@router.post(
    "/reset-stats",
    summary="통계 리셋",
    description="감정 분석 시스템의 성능 통계를 리셋합니다. (개발/디버깅용)"
)
async def reset_performance_stats():
    """성능 통계 리셋"""
    try:
        emotion_analyzer.reset_performance_stats()

        return {
            "message": "성능 통계가 리셋되었습니다.",
            "reset_at": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"❌ 통계 리셋 중 오류: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"통계 리셋 중 오류가 발생했습니다: {str(e)}"
        )


@router.get(
    "/test",
    summary="감정 분석 테스트",
    description="미리 정의된 테스트 케이스로 감정 분석을 테스트합니다."
)
async def test_emotion_analysis():
    """감정 분석 테스트"""
    try:
        test_cases = [
            "이 향수 정말 좋아요! 달콤하고 상큼해서 기분이 좋아져요.",
            "향이 너무 진해서 별로예요. 좀 부담스럽네요.",
            "처음 맡았을 때 놀랐어요. 예상과 완전 달라서 당황스러웠어요.",
            "이 향수를 맡으면 옛날 생각이 나서 슬퍼져요.",
            "향수가 너무 자극적이어서 화가 나요. 최악이에요.",
            "새로운 향수를 발견해서 너무 신나요! 에너지가 넘쳐요."
        ]

        test_results = []

        for i, text in enumerate(test_cases):
            # 벡터라이저 기반 테스트
            vec_result = await emotion_analyzer.analyze_emotion(text, use_vectorizer=True)

            # 룰 기반 테스트
            rule_result = await emotion_analyzer.analyze_emotion(text, use_vectorizer=False)

            test_results.append({
                "test_case": i + 1,
                "text": text,
                "vectorizer_result": {
                    "emotion": vec_result.get("primary_emotion"),
                    "confidence": vec_result.get("confidence"),
                    "method": vec_result.get("method")
                },
                "rule_result": {
                    "emotion": rule_result.get("primary_emotion"),
                    "confidence": rule_result.get("confidence"),
                    "method": rule_result.get("method")
                },
                "same_emotion": vec_result.get("primary_emotion") == rule_result.get("primary_emotion")
            })

        # 일치율 계산
        same_count = sum(1 for r in test_results if r["same_emotion"])
        agreement_rate = same_count / len(test_results) * 100

        return {
            "message": "감정 분석 테스트 완료",
            "total_tests": len(test_cases),
            "vectorizer_rule_agreement": f"{same_count}/{len(test_results)} ({agreement_rate:.1f}%)",
            "test_results": test_results
        }

    except Exception as e:
        logger.error(f"❌ 테스트 중 오류: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"테스트 중 오류가 발생했습니다: {str(e)}"
        )