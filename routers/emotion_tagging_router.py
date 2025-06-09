# routers/emotion_tagging_router.py
# 🎭 감정 태깅 및 분석 API 라우터 (Google Drive 연동)

import os
import json
import logging
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Union
from fastapi import APIRouter, HTTPException, Depends, Query, Body, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from utils.emotion_analyzer import emotion_analyzer
from utils.auth_utils import verify_firebase_token_optional

# ─── 로거 설정 ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("emotion_tagging_router")


# ─── 스키마 정의 ─────────────────────────────────────────────────────────────────

class EmotionAnalysisRequest(BaseModel):
    """감정 분석 요청 스키마"""
    text: str = Field(..., min_length=1, max_length=2000, description="분석할 텍스트 (최대 2000자)")
    use_ai_model: bool = Field(True, description="AI 모델 사용 여부")
    save_to_gdrive: bool = Field(False, description="Google Drive에 결과 저장 여부")
    analysis_type: str = Field("diary", description="분석 유형 (diary, review, comment)")

    @validator('text')
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError("분석할 텍스트가 비어있습니다.")
        return v.strip()

    class Config:
        schema_extra = {
            "example": {
                "text": "이 향수 정말 좋아요! 달콤하고 상큼해서 기분이 좋아져요.",
                "use_ai_model": True,
                "save_to_gdrive": False,
                "analysis_type": "diary"
            }
        }


class BatchAnalysisRequest(BaseModel):
    """일괄 감정 분석 요청 스키마"""
    texts: List[str] = Field(..., min_items=1, max_items=50, description="분석할 텍스트 목록 (최대 50개)")
    use_ai_model: bool = Field(True, description="AI 모델 사용 여부")
    save_to_gdrive: bool = Field(False, description="Google Drive에 결과 저장 여부")

    @validator('texts')
    def validate_texts(cls, v):
        if not v:
            raise ValueError("분석할 텍스트가 없습니다.")

        valid_texts = []
        for text in v:
            if text and text.strip() and len(text.strip()) <= 2000:
                valid_texts.append(text.strip())

        if not valid_texts:
            raise ValueError("유효한 텍스트가 없습니다.")

        return valid_texts

    class Config:
        schema_extra = {
            "example": {
                "texts": [
                    "이 향수 정말 좋아요!",
                    "향이 너무 진해서 별로예요.",
                    "처음 맡았을 때 놀랐어요."
                ],
                "use_ai_model": True,
                "save_to_gdrive": False
            }
        }


class GDriveAnalysisRequest(BaseModel):
    """Google Drive 문서 분석 요청 스키마"""
    file_id: str = Field(..., description="Google Drive 파일 ID")
    analysis_type: str = Field("document", description="분석 유형")

    class Config:
        schema_extra = {
            "example": {
                "file_id": "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                "analysis_type": "document"
            }
        }


class GDriveBatchRequest(BaseModel):
    """Google Drive 폴더 일괄 분석 요청 스키마"""
    folder_id: str = Field(..., description="Google Drive 폴더 ID")
    max_files: int = Field(10, ge=1, le=50, description="최대 분석 파일 수 (1-50)")
    file_type_filter: Optional[str] = Field(None, description="파일 유형 필터 (text, doc 등)")

    class Config:
        schema_extra = {
            "example": {
                "folder_id": "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                "max_files": 10,
                "file_type_filter": "text"
            }
        }


class EmotionTagUpdateRequest(BaseModel):
    """감정 태그 업데이트 요청 스키마"""
    emotion: str = Field(..., description="감정 이름")
    new_tags: List[str] = Field(..., min_items=1, description="새로운 태그 목록")

    @validator('emotion')
    def validate_emotion(cls, v):
        valid_emotions = ["기쁨", "불안", "당황", "분노", "상처", "슬픔", "우울", "흥분"]
        if v not in valid_emotions:
            raise ValueError(f"지원되지 않는 감정입니다. 지원 감정: {valid_emotions}")
        return v

    class Config:
        schema_extra = {
            "example": {
                "emotion": "기쁨",
                "new_tags": ["#joyful", "#bright", "#citrus", "#happy"]
            }
        }


class EmotionKeywordAddRequest(BaseModel):
    """감정 키워드 추가 요청 스키마"""
    emotion: str = Field(..., description="감정 이름")
    keywords: List[str] = Field(..., min_items=1, max_items=20, description="추가할 키워드 목록")

    @validator('keywords')
    def validate_keywords(cls, v):
        valid_keywords = []
        for keyword in v:
            if keyword and keyword.strip() and len(keyword.strip()) <= 50:
                valid_keywords.append(keyword.strip())

        if not valid_keywords:
            raise ValueError("유효한 키워드가 없습니다.")

        return valid_keywords

    class Config:
        schema_extra = {
            "example": {
                "emotion": "기쁨",
                "keywords": ["환상적", "멋져", "대박"]
            }
        }


# ─── 응답 스키마 ─────────────────────────────────────────────────────────────────

class EmotionAnalysisResponse(BaseModel):
    """감정 분석 응답 스키마"""
    success: bool
    primary_emotion: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    emotion_tags: List[str]
    method: str
    processing_time: Optional[float] = None
    analysis_details: Optional[Dict] = None
    analyzed_at: str

    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "primary_emotion": "기쁨",
                "confidence": 0.875,
                "emotion_tags": ["#joyful", "#bright", "#citrus"],
                "method": "rule_based",
                "processing_time": 0.045,
                "analyzed_at": "2025-06-09T12:30:45"
            }
        }


class BatchAnalysisResponse(BaseModel):
    """일괄 분석 응답 스키마"""
    total_analyzed: int
    successful_analyses: int
    failed_analyses: int
    processing_time: float
    results: List[Dict[str, Any]]

    class Config:
        schema_extra = {
            "example": {
                "total_analyzed": 3,
                "successful_analyses": 3,
                "failed_analyses": 0,
                "processing_time": 0.156,
                "results": [
                    {
                        "text": "이 향수 정말 좋아요!",
                        "emotion": "기쁨",
                        "confidence": 0.875,
                        "tags": ["#joyful", "#bright"]
                    }
                ]
            }
        }


# ─── 라우터 설정 ─────────────────────────────────────────────────────────────────
router = APIRouter(prefix="/emotion", tags=["Emotion Tagging"])


# ─── 기본 감정 분석 API ─────────────────────────────────────────────────────────────

@router.post(
    "/analyze",
    response_model=EmotionAnalysisResponse,
    summary="텍스트 감정 분석",
    description=(
            "🎭 **텍스트 감정 분석 API**\n\n"
            "향수 리뷰, 시향 일기 등의 텍스트를 분석하여 감정을 파악하고 태그를 생성합니다.\n\n"
            "**🤖 분석 방법:**\n"
            "- AI 모델 (개발 중): 딥러닝 기반 감정 분류\n"
            "- 룰 기반: 향수 도메인 특화 키워드 매칭\n\n"
            "**🎯 지원 감정:**\n"
            "기쁨, 불안, 당황, 분노, 상처, 슬픔, 우울, 흥분\n\n"
            "**✨ 특징:**\n"
            "- 향수 도메인 특화 분석\n"
            "- Google Drive 연동 저장\n"
            "- 실시간 성능 모니터링\n"
            "- 학습 데이터 자동 수집"
    )
)
async def analyze_emotion(
        request: EmotionAnalysisRequest,
        user=Depends(verify_firebase_token_optional)
):
    """단일 텍스트 감정 분석"""

    logger.info(f"🎭 감정 분석 요청 접수")
    logger.info(f"  - 사용자: {user.get('name', '익명')}")
    logger.info(f"  - 텍스트 길이: {len(request.text)}자")
    logger.info(f"  - 분석 유형: {request.analysis_type}")

    try:
        # 감정 분석 수행
        result = await emotion_analyzer.analyze_emotion(
            text=request.text,
            use_model=request.use_ai_model,
            save_to_gdrive=request.save_to_gdrive
        )

        if not result.get("success"):
            logger.error(f"❌ 감정 분석 실패: {result.get('message')}")
            raise HTTPException(
                status_code=400,
                detail=f"감정 분석에 실패했습니다: {result.get('message', '알 수 없는 오류')}"
            )

        # 응답 데이터 구성
        response_data = {
            "success": True,
            "primary_emotion": result.get("primary_emotion", "중립"),
            "confidence": result.get("confidence", 0.0),
            "emotion_tags": result.get("emotion_tags", ["#neutral"]),
            "method": result.get("method", "unknown"),
            "processing_time": result.get("processing_time"),
            "analysis_details": result.get("analysis_details"),
            "analyzed_at": result.get("analyzed_at", datetime.now().isoformat())
        }

        logger.info(f"✅ 감정 분석 완료: {response_data['primary_emotion']} (신뢰도: {response_data['confidence']:.3f})")

        return EmotionAnalysisResponse(**response_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 감정 분석 처리 중 예외: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"감정 분석 처리 중 오류가 발생했습니다: {str(e)}"
        )


@router.post(
    "/batch-analyze",
    response_model=BatchAnalysisResponse,
    summary="일괄 텍스트 감정 분석",
    description=(
            "📊 **일괄 텍스트 감정 분석 API**\n\n"
            "여러 텍스트를 한 번에 분석하여 효율적인 감정 분석을 제공합니다.\n\n"
            "**📥 입력:**\n"
            "- 최대 50개 텍스트 동시 분석\n"
            "- 각 텍스트 최대 2000자\n\n"
            "**📤 출력:**\n"
            "- 개별 분석 결과\n"
            "- 성공/실패 통계\n"
            "- 전체 처리 시간\n\n"
            "**⚡ 성능:**\n"
            "- 병렬 처리로 빠른 분석\n"
            "- 실패한 텍스트 별도 처리\n"
            "- 진행률 실시간 모니터링"
    )
)
async def batch_analyze_emotions(
        request: BatchAnalysisRequest,
        background_tasks: BackgroundTasks,
        user=Depends(verify_firebase_token_optional)
):
    """여러 텍스트 일괄 감정 분석"""

    start_time = datetime.now()

    logger.info(f"📊 일괄 감정 분석 요청 접수")
    logger.info(f"  - 사용자: {user.get('name', '익명')}")
    logger.info(f"  - 텍스트 개수: {len(request.texts)}개")

    try:
        results = []
        successful_count = 0
        failed_count = 0

        # 각 텍스트에 대해 감정 분석 수행
        for i, text in enumerate(request.texts, 1):
            try:
                logger.info(f"📝 분석 중: {i}/{len(request.texts)} - {text[:30]}...")

                # 개별 텍스트 분석
                analysis_result = await emotion_analyzer.analyze_emotion(
                    text=text,
                    use_model=request.use_ai_model,
                    save_to_gdrive=request.save_to_gdrive
                )

                if analysis_result.get("success"):
                    result_item = {
                        "index": i - 1,
                        "text": text[:100],  # 응답 크기 최적화를 위해 텍스트 제한
                        "success": True,
                        "emotion": analysis_result.get("primary_emotion"),
                        "confidence": analysis_result.get("confidence"),
                        "tags": analysis_result.get("emotion_tags"),
                        "method": analysis_result.get("method"),
                        "processing_time": analysis_result.get("processing_time")
                    }
                    successful_count += 1
                else:
                    result_item = {
                        "index": i - 1,
                        "text": text[:100],
                        "success": False,
                        "error": analysis_result.get("message", "분석 실패"),
                        "error_type": analysis_result.get("error_type", "unknown")
                    }
                    failed_count += 1

                results.append(result_item)

                # 과부하 방지를 위한 짧은 대기
                if i % 10 == 0:
                    await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"❌ 텍스트 {i} 분석 실패: {e}")
                results.append({
                    "index": i - 1,
                    "text": text[:100],
                    "success": False,
                    "error": str(e),
                    "error_type": "processing_error"
                })
                failed_count += 1

        # 전체 처리 시간 계산
        total_processing_time = (datetime.now() - start_time).total_seconds()

        # 응답 데이터 구성
        response_data = {
            "total_analyzed": len(request.texts),
            "successful_analyses": successful_count,
            "failed_analyses": failed_count,
            "processing_time": round(total_processing_time, 3),
            "results": results
        }

        logger.info(f"✅ 일괄 분석 완료: {successful_count}개 성공, {failed_count}개 실패")
        logger.info(f"⏱️ 총 처리 시간: {total_processing_time:.3f}초")

        # 백그라운드에서 통계 업데이트
        if request.save_to_gdrive:
            background_tasks.add_task(
                save_batch_analysis_summary,
                user.get('uid', 'anonymous'),
                response_data
            )

        return BatchAnalysisResponse(**response_data)

    except Exception as e:
        logger.error(f"❌ 일괄 분석 처리 중 예외: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"일괄 분석 처리 중 오류가 발생했습니다: {str(e)}"
        )


# ─── Google Drive 연동 API ─────────────────────────────────────────────────────────

@router.post(
    "/analyze-gdrive-document",
    summary="Google Drive 문서 감정 분석",
    description=(
            "📁 **Google Drive 문서 감정 분석 API**\n\n"
            "Google Drive에 저장된 문서를 직접 분석합니다.\n\n"
            "**📋 지원 파일:**\n"
            "- 텍스트 파일 (.txt)\n"
            "- 문서 파일 (.doc, .docx) - 향후 지원\n\n"
            "**🔐 권한:**\n"
            "- Google Drive 읽기 권한 필요\n"
            "- 서비스 계정 인증 사용\n\n"
            "**💾 자동 저장:**\n"
            "- 분석 결과 Google Drive 백업\n"
            "- 분석 이력 추적 가능"
    )
)
async def analyze_gdrive_document(
        request: GDriveAnalysisRequest,
        user=Depends(verify_firebase_token_optional)
):
    """Google Drive 문서 감정 분석"""

    logger.info(f"📁 Google Drive 문서 분석 요청")
    logger.info(f"  - 사용자: {user.get('name', '익명')}")
    logger.info(f"  - 파일 ID: {request.file_id}")

    try:
        # Google Drive 연결 확인
        gdrive_status = emotion_analyzer.get_gdrive_status()
        if not gdrive_status["connected"]:
            raise HTTPException(
                status_code=503,
                detail="Google Drive에 연결되지 않았습니다. 관리자에게 문의하세요."
            )

        # Google Drive 문서 분석
        result = await emotion_analyzer.analyze_gdrive_document(request.file_id)

        if not result.get("success"):
            logger.error(f"❌ Google Drive 문서 분석 실패: {result.get('message')}")
            raise HTTPException(
                status_code=400,
                detail=f"Google Drive 문서 분석에 실패했습니다: {result.get('message', '알 수 없는 오류')}"
            )

        logger.info(f"✅ Google Drive 문서 분석 완료: {result.get('primary_emotion')}")

        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Google Drive 문서 분석 중 예외: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Google Drive 문서 분석 중 오류가 발생했습니다: {str(e)}"
        )


@router.post(
    "/batch-analyze-gdrive",
    summary="Google Drive 폴더 일괄 분석",
    description=(
            "📂 **Google Drive 폴더 일괄 분석 API**\n\n"
            "Google Drive 폴더 내 모든 문서를 일괄 분석합니다.\n\n"
            "**📊 처리 방식:**\n"
            "- 폴더 내 파일 자동 탐지\n"
            "- 순차적 안전 처리\n"
            "- 실시간 진행률 추적\n\n"
            "**⚙️ 설정 옵션:**\n"
            "- 최대 파일 수 제한 (1-50)\n"
            "- 파일 유형 필터링\n"
            "- 과부하 방지 대기\n\n"
            "**💾 결과 저장:**\n"
            "- 개별 분석 결과 저장\n"
            "- 일괄 분석 요약 생성\n"
            "- Google Drive 백업"
    )
)
async def batch_analyze_gdrive_folder(
        request: GDriveBatchRequest,
        background_tasks: BackgroundTasks,
        user=Depends(verify_firebase_token_optional)
):
    """Google Drive 폴더 일괄 감정 분석"""

    logger.info(f"📂 Google Drive 폴더 일괄 분석 요청")
    logger.info(f"  - 사용자: {user.get('name', '익명')}")
    logger.info(f"  - 폴더 ID: {request.folder_id}")
    logger.info(f"  - 최대 파일 수: {request.max_files}")

    try:
        # Google Drive 연결 확인
        gdrive_status = emotion_analyzer.get_gdrive_status()
        if not gdrive_status["connected"]:
            raise HTTPException(
                status_code=503,
                detail="Google Drive에 연결되지 않았습니다. 관리자에게 문의하세요."
            )

        # Google Drive 폴더 일괄 분석
        results = await emotion_analyzer.batch_analyze_gdrive_folder(
            folder_id=request.folder_id,
            max_files=request.max_files
        )

        if not results:
            return JSONResponse(content={
                "message": "분석할 파일이 없습니다.",
                "folder_id": request.folder_id,
                "results": [],
                "total_analyzed": 0
            })

        # 성공/실패 통계 계산
        successful_count = sum(1 for r in results if r.get("success"))
        failed_count = len(results) - successful_count

        response_data = {
            "message": f"Google Drive 폴더 일괄 분석 완료",
            "folder_id": request.folder_id,
            "total_analyzed": len(results),
            "successful_analyses": successful_count,
            "failed_analyses": failed_count,
            "results": results
        }

        logger.info(f"✅ Google Drive 폴더 분석 완료: {successful_count}개 성공, {failed_count}개 실패")

        # 백그라운드에서 결과 요약 저장
        background_tasks.add_task(
            save_gdrive_batch_summary,
            user.get('uid', 'anonymous'),
            request.folder_id,
            response_data
        )

        return JSONResponse(content=response_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Google Drive 폴더 분석 중 예외: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Google Drive 폴더 분석 중 오류가 발생했습니다: {str(e)}"
        )


# ─── 감정 태그 관리 API ─────────────────────────────────────────────────────────────

@router.get(
    "/emotions",
    summary="지원 감정 목록 조회",
    description="시스템에서 지원하는 모든 감정과 해당 태그를 조회합니다."
)
async def get_supported_emotions():
    """지원하는 감정 목록 반환"""

    try:
        emotions = emotion_analyzer.get_supported_emotions()
        emotion_details = {}

        for emotion in emotions:
            tags = emotion_analyzer.get_emotion_tags(emotion)
            emotion_details[emotion] = {
                "tags": tags,
                "tag_count": len(tags)
            }

        return JSONResponse(content={
            "supported_emotions": emotions,
            "emotion_count": len(emotions),
            "emotion_details": emotion_details,
            "total_tags": sum(len(tags) for tags in emotion_details.values())
        })

    except Exception as e:
        logger.error(f"❌ 지원 감정 조회 중 오류: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"지원 감정 조회 중 오류가 발생했습니다: {str(e)}"
        )


@router.post(
    "/emotions/{emotion}/tags",
    summary="감정 태그 업데이트",
    description=(
            "🏷️ **감정 태그 업데이트 API**\n\n"
            "특정 감정의 태그를 업데이트합니다.\n\n"
            "**⚠️ 주의사항:**\n"
            "- 기존 태그가 완전히 대체됩니다\n"
            "- 관리자 권한이 필요할 수 있습니다\n"
            "- 변경사항은 Google Drive에 백업됩니다"
    )
)
async def update_emotion_tags(
        emotion: str,
        request: EmotionTagUpdateRequest,
        user=Depends(verify_firebase_token_optional)
):
    """감정 태그 업데이트"""

    logger.info(f"🏷️ 감정 태그 업데이트 요청")
    logger.info(f"  - 사용자: {user.get('name', '익명')}")
    logger.info(f"  - 감정: {emotion}")
    logger.info(f"  - 새 태그: {request.new_tags}")

    try:
        # 감정 유효성 검사
        supported_emotions = emotion_analyzer.get_supported_emotions()
        if emotion not in supported_emotions:
            raise HTTPException(
                status_code=400,
                detail=f"지원되지 않는 감정입니다. 지원 감정: {supported_emotions}"
            )

        # 기존 태그 조회
        old_tags = emotion_analyzer.get_emotion_tags(emotion)

        # 태그 업데이트
        emotion_analyzer.emotion_to_tags[emotion] = request.new_tags

        # Google Drive 백업 (백그라운드)
        if emotion_analyzer.get_gdrive_status()["connected"]:
            asyncio.create_task(emotion_analyzer.sync_with_gdrive())

        logger.info(f"✅ 감정 태그 업데이트 완료: {emotion}")

        return JSONResponse(content={
            "message": f"'{emotion}' 감정의 태그가 성공적으로 업데이트되었습니다.",
            "emotion": emotion,
            "old_tags": old_tags,
            "new_tags": request.new_tags,
            "updated_by": user.get('name', '익명'),
            "updated_at": datetime.now().isoformat()
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 감정 태그 업데이트 중 오류: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"감정 태그 업데이트 중 오류가 발생했습니다: {str(e)}"
        )


@router.post(
    "/emotions/{emotion}/keywords",
    summary="감정 키워드 추가",
    description=(
            "📝 **감정 키워드 추가 API**\n\n"
            "특정 감정의 분석 키워드를 추가합니다.\n\n"
            "**🎯 용도:**\n"
            "- 분석 정확도 향상\n"
            "- 도메인 특화 키워드 확장\n"
            "- 사용자 피드백 반영\n\n"
            "**📊 효과:**\n"
            "- 즉시 분석에 반영\n"
            "- Google Drive 동기화\n"
            "- 성능 통계 업데이트"
    )
)
async def add_emotion_keywords(
        emotion: str,
        request: EmotionKeywordAddRequest,
        user=Depends(verify_firebase_token_optional)
):
    """감정 키워드 추가"""

    logger.info(f"📝 감정 키워드 추가 요청")
    logger.info(f"  - 사용자: {user.get('name', '익명')}")
    logger.info(f"  - 감정: {emotion}")
    logger.info(f"  - 새 키워드: {request.keywords}")

    try:
        # 감정 유효성 검사
        supported_emotions = emotion_analyzer.get_supported_emotions()
        if emotion not in supported_emotions:
            raise HTTPException(
                status_code=400,
                detail=f"지원되지 않는 감정입니다. 지원 감정: {supported_emotions}"
            )

        # 키워드 추가
        emotion_analyzer.add_custom_keywords(emotion, request.keywords)

        # 업데이트된 키워드 목록 조회
        updated_keywords = emotion_analyzer.emotion_keywords.get(emotion, [])

        logger.info(f"✅ 감정 키워드 추가 완료: {emotion}")

        return JSONResponse(content={
            "message": f"'{emotion}' 감정에 키워드가 성공적으로 추가되었습니다.",
            "emotion": emotion,
            "added_keywords": request.keywords,
            "total_keywords": len(updated_keywords),
            "updated_by": user.get('name', '익명'),
            "updated_at": datetime.now().isoformat()
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 감정 키워드 추가 중 오류: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"감정 키워드 추가 중 오류가 발생했습니다: {str(e)}"
        )


# ─── 시스템 상태 및 통계 API ─────────────────────────────────────────────────────────

@router.get(
    "/system/status",
    summary="감정 분석 시스템 상태",
    description=(
            "📊 **감정 분석 시스템 상태 API**\n\n"
            "시스템의 전반적인 상태와 성능 통계를 제공합니다.\n\n"
            "**📈 포함 정보:**\n"
            "- 분석 성능 통계\n"
            "- Google Drive 연결 상태\n"
            "- 지원 감정 및 키워드 수\n"
            "- 모델 로딩 상태\n\n"
            "**🔄 실시간 데이터:**\n"
            "- 총 분석 횟수\n"
            "- 성공률\n"
            "- 평균 응답 시간\n"
            "- 방법별 분포"
    )
)
async def get_system_status():
    """감정 분석 시스템 상태 조회"""

    try:
        # 기본 시스템 상태
        system_stats = emotion_analyzer.get_analysis_stats()

        # Google Drive 상태
        gdrive_status = emotion_analyzer.get_gdrive_status()

        # 추가 시스템 정보
        additional_info = {
            "service_name": "Whiff 감정 분석 시스템",
            "version": "v2.0 (Google Drive 연동)",
            "uptime": "실행 중",
            "last_updated": datetime.now().isoformat(),
            "features": [
                "룰 기반 감정 분석",
                "AI 모델 준비 중",
                "Google Drive 연동",
                "실시간 성능 모니터링",
                "자동 학습 데이터 수집"
            ]
        }

        return JSONResponse(content={
            "system_status": "operational",
            "system_info": additional_info,
            "analysis_stats": system_stats,
            "google_drive": gdrive_status,
            "health_check": {
                "analyzer_ready": True,
                "gdrive_ready": gdrive_status["connected"],
                "performance_good": system_stats["performance"]["success_rate"] > 80
            }
        })

    except Exception as e:
        logger.error(f"❌ 시스템 상태 조회 중 오류: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"시스템 상태 조회 중 오류가 발생했습니다: {str(e)}"
        )


@router.get(
    "/system/performance-report",
    summary="성능 리포트 조회",
    description="감정 분석 시스템의 상세 성능 리포트를 제공합니다."
)
async def get_performance_report():
    """성능 리포트 조회"""

    try:
        report = emotion_analyzer.get_performance_report()

        return JSONResponse(content=report)

    except Exception as e:
        logger.error(f"❌ 성능 리포트 조회 중 오류: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"성능 리포트 조회 중 오류가 발생했습니다: {str(e)}"
        )


@router.post(
    "/system/sync-gdrive",
    summary="Google Drive 수동 동기화",
    description=(
            "🔄 **Google Drive 수동 동기화 API**\n\n"
            "Google Drive와 수동으로 동기화를 수행합니다.\n\n"
            "**🔄 동기화 항목:**\n"
            "- 감정 키워드 사전\n"
            "- 성능 통계 백업\n"
            "- 학습 데이터 업로드\n\n"
            "**⏱️ 처리 시간:**\n"
            "- 일반적으로 5-10초 소요\n"
            "- 대용량 데이터 시 더 오래 걸릴 수 있음"
    )
)
async def manual_sync_gdrive(
        force: bool = Query(False, description="강제 동기화 여부"),
        user=Depends(verify_firebase_token_optional)
):
    """Google Drive 수동 동기화"""

    logger.info(f"🔄 Google Drive 수동 동기화 요청")
    logger.info(f"  - 사용자: {user.get('name', '익명')}")
    logger.info(f"  - 강제 동기화: {'✅' if force else '❌'}")

    try:
        # Google Drive 연결 확인
        gdrive_status = emotion_analyzer.get_gdrive_status()
        if not gdrive_status["connected"]:
            raise HTTPException(
                status_code=503,
                detail="Google Drive에 연결되지 않았습니다. 관리자에게 문의하세요."
            )

        # 동기화 수행
        sync_result = await emotion_analyzer.sync_with_gdrive(force=force)

        if sync_result["success"]:
            logger.info(f"✅ Google Drive 동기화 완료 (소요시간: {sync_result['sync_time']:.3f}초)")

            return JSONResponse(content={
                "message": "Google Drive 동기화가 성공적으로 완료되었습니다.",
                "sync_result": sync_result,
                "requested_by": user.get('name', '익명'),
                "sync_requested_at": datetime.now().isoformat()
            })
        else:
            logger.error(f"❌ Google Drive 동기화 실패: {sync_result['message']}")
            raise HTTPException(
                status_code=500,
                detail=f"Google Drive 동기화에 실패했습니다: {sync_result['message']}"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Google Drive 동기화 중 예외: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Google Drive 동기화 중 오류가 발생했습니다: {str(e)}"
        )


@router.post(
    "/system/reset-stats",
    summary="성능 통계 리셋",
    description="감정 분석 시스템의 성능 통계를 초기화합니다. (관리자 기능)"
)
async def reset_performance_stats(
        confirm: bool = Query(False, description="리셋 확인"),
        user=Depends(verify_firebase_token_optional)
):
    """성능 통계 리셋"""

    logger.info(f"🔄 성능 통계 리셋 요청")
    logger.info(f"  - 사용자: {user.get('name', '익명')}")
    logger.info(f"  - 확인: {'✅' if confirm else '❌'}")

    try:
        if not confirm:
            raise HTTPException(
                status_code=400,
                detail="성능 통계 리셋을 확인해주세요. confirm=true 파라미터를 추가하세요."
            )

        # 통계 백업 (선택사항)
        backup_stats = emotion_analyzer.get_analysis_stats()

        # 통계 리셋
        emotion_analyzer.reset_performance_stats()

        logger.info(f"✅ 성능 통계 리셋 완료")

        return JSONResponse(content={
            "message": "성능 통계가 성공적으로 리셋되었습니다.",
            "backup_stats": backup_stats,
            "reset_by": user.get('name', '익명'),
            "reset_at": datetime.now().isoformat()
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 성능 통계 리셋 중 오류: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"성능 통계 리셋 중 오류가 발생했습니다: {str(e)}"
        )


# ─── 백그라운드 작업 함수들 ─────────────────────────────────────────────────────────

async def save_batch_analysis_summary(user_id: str, analysis_data: dict):
    """일괄 분석 요약을 Google Drive에 저장"""
    try:
        if emotion_analyzer.get_gdrive_status()["connected"]:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"batch_analysis_summary_{user_id}_{timestamp}.json"

            summary_content = json.dumps({
                "user_id": user_id,
                "analysis_summary": analysis_data,
                "created_at": datetime.now().isoformat()
            }, ensure_ascii=False, indent=2)

            await emotion_analyzer.gdrive_manager.upload_analysis_result(
                summary_content, filename
            )

            logger.info(f"💾 일괄 분석 요약 저장 완료: {filename}")

    except Exception as e:
        logger.error(f"❌ 일괄 분석 요약 저장 실패: {e}")


async def save_gdrive_batch_summary(user_id: str, folder_id: str, analysis_data: dict):
    """Google Drive 폴더 분석 요약 저장"""
    try:
        if emotion_analyzer.get_gdrive_status()["connected"]:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"gdrive_batch_summary_{user_id}_{timestamp}.json"

            summary_content = json.dumps({
                "user_id": user_id,
                "folder_id": folder_id,
                "analysis_summary": analysis_data,
                "created_at": datetime.now().isoformat()
            }, ensure_ascii=False, indent=2)

            await emotion_analyzer.gdrive_manager.upload_analysis_result(
                summary_content, filename
            )

            logger.info(f"💾 Google Drive 폴더 분석 요약 저장 완료: {filename}")

    except Exception as e:
        logger.error(f"❌ Google Drive 폴더 분석 요약 저장 실패: {e}")


# 라우터 시작 시 초기화
logger.info("🎭 감정 태깅 라우터 초기화 완료")
logger.info("✨ Google Drive 연동 감정 분석 시스템 준비됨")