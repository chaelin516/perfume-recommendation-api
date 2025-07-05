# routers/recommendation_save_router.py
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlmodel import Session, select
from datetime import datetime
import logging
from typing import List

# 로컬 imports
from db.session import get_session
from models.recommendation import RecommendedPerfume, RecommendationHistory
from schemas.recommendation import (
    SaveRecommendationsRequest,
    SaveRecommendationsResponse,
    GetRecommendationsResponse,
    RecommendationHistoryResponse
)
from utils.auth_utils import verify_token_flexible

router = APIRouter(prefix="/recommendations", tags=["Recommendations"])
logger = logging.getLogger(__name__)


@router.post(
    "/save",
    summary="추천 향수 저장",
    description="1차 또는 2차 추천 향수 목록을 저장합니다.",
    response_model=SaveRecommendationsResponse
)
async def save_recommendations(
        request: SaveRecommendationsRequest,
        session: Session = Depends(get_session),
        user=Depends(verify_token_flexible)
):
    """추천 결과 저장"""
    try:
        logger.info(f"💾 추천 결과 저장 시작: {request.user_id} (라운드: {request.recommend_round})")

        # 토큰의 사용자와 요청의 사용자 ID 일치 확인
        if user["uid"] != request.user_id:
            raise HTTPException(
                status_code=403,
                detail="자신의 추천 결과만 저장할 수 있습니다."
            )

        saved_count = 0

        # 각 추천 향수를 저장
        for item in request.recommendations:
            record = RecommendedPerfume(
                user_id=request.user_id,
                recommend_round=request.recommend_round,
                perfume_name=item.perfume_name,
                perfume_brand=item.perfume_brand,
                score=item.score,
                created_at=datetime.utcnow()
            )
            session.add(record)
            saved_count += 1

        # 커밋
        session.commit()

        logger.info(f"✅ 추천 결과 저장 완료: {saved_count}건")

        response = SaveRecommendationsResponse(
            message="추천 결과가 성공적으로 저장되었습니다.",
            saved_count=saved_count,
            user_id=request.user_id,
            recommend_round=request.recommend_round,
            created_at=datetime.utcnow()
        )

        return JSONResponse(content=response.dict())

    except HTTPException:
        raise
    except Exception as e:
        session.rollback()
        logger.error(f"❌ 추천 결과 저장 실패: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"추천 결과 저장 중 오류가 발생했습니다: {str(e)}"
        )


@router.get(
    "/my-recommendations",
    summary="내 추천 조회",
    description="현재 사용자의 추천 히스토리를 조회합니다.",
    response_model=GetRecommendationsResponse
)
async def get_my_recommendations(
        recommend_round: int = None,
        limit: int = 50,
        session: Session = Depends(get_session),
        user=Depends(verify_token_flexible)
):
    """사용자의 추천 히스토리 조회"""
    try:
        user_id = user["uid"]
        logger.info(f"📋 추천 히스토리 조회: {user_id}")

        # 쿼리 구성
        query = select(RecommendedPerfume).where(RecommendedPerfume.user_id == user_id)

        if recommend_round:
            query = query.where(RecommendedPerfume.recommend_round == recommend_round)

        query = query.order_by(RecommendedPerfume.created_at.desc()).limit(limit)

        # 실행
        results = session.exec(query).all()

        # 응답 구성
        recommendations = [
            RecommendationHistoryResponse(
                id=rec.id,
                user_id=rec.user_id,
                perfume_name=rec.perfume_name,
                perfume_brand=rec.perfume_brand,
                score=rec.score,
                recommend_round=rec.recommend_round,
                created_at=rec.created_at
            )
            for rec in results
        ]

        response = GetRecommendationsResponse(
            message="추천 히스토리 조회 성공",
            total_count=len(recommendations),
            recommendations=recommendations,
            user_id=user_id
        )

        logger.info(f"✅ 추천 히스토리 조회 완료: {len(recommendations)}건")
        return JSONResponse(content=response.dict())

    except Exception as e:
        logger.error(f"❌ 추천 히스토리 조회 실패: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"추천 히스토리 조회 중 오류가 발생했습니다: {str(e)}"
        )


@router.delete(
    "/clear-my-recommendations",
    summary="내 추천 삭제",
    description="현재 사용자의 모든 추천 히스토리를 삭제합니다."
)
async def clear_my_recommendations(
        session: Session = Depends(get_session),
        user=Depends(verify_token_flexible)
):
    """사용자의 추천 히스토리 삭제"""
    try:
        user_id = user["uid"]
        logger.info(f"🗑️ 추천 히스토리 삭제 시작: {user_id}")

        # 삭제 쿼리
        query = select(RecommendedPerfume).where(RecommendedPerfume.user_id == user_id)
        results = session.exec(query).all()

        deleted_count = len(results)

        # 삭제 실행
        for rec in results:
            session.delete(rec)

        session.commit()

        logger.info(f"✅ 추천 히스토리 삭제 완료: {deleted_count}건")

        return JSONResponse(content={
            "message": "추천 히스토리가 성공적으로 삭제되었습니다.",
            "deleted_count": deleted_count,
            "user_id": user_id
        })

    except Exception as e:
        session.rollback()
        logger.error(f"❌ 추천 히스토리 삭제 실패: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"추천 히스토리 삭제 중 오류가 발생했습니다: {str(e)}"
        )


@router.get(
    "/stats",
    summary="추천 통계",
    description="현재 사용자의 추천 통계를 조회합니다."
)
async def get_recommendation_stats(
        session: Session = Depends(get_session),
        user=Depends(verify_token_flexible)
):
    """사용자의 추천 통계 조회"""
    try:
        user_id = user["uid"]
        logger.info(f"📊 추천 통계 조회: {user_id}")

        # 전체 추천 수
        total_query = select(RecommendedPerfume).where(RecommendedPerfume.user_id == user_id)
        total_results = session.exec(total_query).all()
        total_count = len(total_results)

        # 1차 추천 수
        round1_count = len([r for r in total_results if r.recommend_round == 1])

        # 2차 추천 수
        round2_count = len([r for r in total_results if r.recommend_round == 2])

        # 최고 점수 향수
        best_perfume = None
        if total_results:
            best_rec = max(total_results, key=lambda x: x.score)
            best_perfume = {
                "name": best_rec.perfume_name,
                "brand": best_rec.perfume_brand,
                "score": best_rec.score
            }

        # 브랜드별 통계
        brand_stats = {}
        for rec in total_results:
            brand = rec.perfume_brand
            if brand not in brand_stats:
                brand_stats[brand] = 0
            brand_stats[brand] += 1

        # 상위 3개 브랜드
        top_brands = sorted(brand_stats.items(), key=lambda x: x[1], reverse=True)[:3]

        stats = {
            "user_id": user_id,
            "total_recommendations": total_count,
            "round_1_count": round1_count,
            "round_2_count": round2_count,
            "best_perfume": best_perfume,
            "top_brands": [{"brand": brand, "count": count} for brand, count in top_brands],
            "last_updated": datetime.utcnow().isoformat()
        }

        logger.info(f"✅ 추천 통계 조회 완료: 총 {total_count}건")

        return JSONResponse(content={
            "message": "추천 통계 조회 성공",
            "stats": stats
        })

    except Exception as e:
        logger.error(f"❌ 추천 통계 조회 실패: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"추천 통계 조회 중 오류가 발생했습니다: {str(e)}"
        )